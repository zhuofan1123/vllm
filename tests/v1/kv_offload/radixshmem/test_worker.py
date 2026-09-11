# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU <-> shared-slot round trips through upstream's transfer handler.

Both TP ranks run in this one process against the same shared region, which is
exactly the aliasing the packed slot layout has to get right: rank 1 writing its
slice must not disturb rank 0's. Hybrid cases drive two pools through one
connector job. Needs a GPU.
"""

import pytest
import torch

from vllm.v1.kv_offload.base import (
    CanonicalKVCacheRef,
    CanonicalKVCaches,
    CanonicalKVCacheTensor,
    GPULoadStoreSpec,
)
from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec
from vllm.v1.kv_offload.radixshmem.geometry import PoolKind
from vllm.v1.kv_offload.radixshmem.worker import (
    RadixShmemOffloadingWorker,
    group_cpu_views,
    plan_groups,
)

from .utils import (
    geometry_for,
    group,
    make_offloading_config,
    open_test_client,
    unique_tag,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

PAGE_BYTES = 2048
NUM_GPU_BLOCKS = 32
TP_SIZE = 2


def make_kv_caches(num_tensors=2):
    """Layer-outermost: one canonical tensor per layer, all in group 0."""
    tensors = [
        torch.zeros(NUM_GPU_BLOCKS, PAGE_BYTES, dtype=torch.int8, device="cuda")
        for _ in range(num_tensors)
    ]
    return tensors, CanonicalKVCaches(
        tensors=[
            CanonicalKVCacheTensor(tensor=t, page_size_bytes=PAGE_BYTES)
            for t in tensors
        ],
        group_data_refs=[
            [
                CanonicalKVCacheRef(tensor_idx=i, page_size_bytes=PAGE_BYTES)
                for i in range(num_tensors)
            ]
        ],
    )


def make_hybrid_kv_caches():
    """Layer-outermost hybrid: tensors 0,1 -> FULL group 0; tensor 2 -> SWA group 1."""
    tensors = [
        torch.zeros(NUM_GPU_BLOCKS, PAGE_BYTES, dtype=torch.int8, device="cuda")
        for _ in range(3)
    ]
    return tensors, CanonicalKVCaches(
        tensors=[
            CanonicalKVCacheTensor(tensor=t, page_size_bytes=PAGE_BYTES)
            for t in tensors
        ],
        group_data_refs=[
            [CanonicalKVCacheRef(0, PAGE_BYTES), CanonicalKVCacheRef(1, PAGE_BYTES)],
            [CanonicalKVCacheRef(2, PAGE_BYTES)],
        ],
    )


def make_packed_kv_caches():
    """Block-outermost hybrid: one packed tensor per block holding
    [full layer 0 | full layer 1 | swa layer 0] pages."""
    block = torch.zeros(NUM_GPU_BLOCKS, 3 * PAGE_BYTES, dtype=torch.int8, device="cuda")
    caches = CanonicalKVCaches(
        tensors=[CanonicalKVCacheTensor(tensor=block, page_size_bytes=3 * PAGE_BYTES)],
        group_data_refs=[
            [CanonicalKVCacheRef(0, 3 * PAGE_BYTES)],
            [CanonicalKVCacheRef(0, 3 * PAGE_BYTES)],
        ],
    )
    return block, caches


def hybrid_config(tag, *, packed):
    layer_pages_full = ((0, PAGE_BYTES), (PAGE_BYTES, PAGE_BYTES)) if packed else ()
    layer_pages_swa = ((2 * PAGE_BYTES, PAGE_BYTES),) if packed else ()
    return make_offloading_config(
        tag=tag,
        tp_size=TP_SIZE,
        tokens_per_hash=8,
        groups=[
            group(
                16,
                kind="full",
                bytes_per_block=2 * PAGE_BYTES,
                layer_pages=layer_pages_full,
                num_layers=2,
                name="full",
            ),
            group(
                8,
                kind="swa",
                window=16,
                bytes_per_block=PAGE_BYTES,
                layer_pages=layer_pages_swa,
                num_layers=1,
                name="swa",
            ),
        ],
        cpu_bytes=16 << 20,
    )


def gpu_spec(block_ids, first_block_idx=0, num_groups=1, group_idx=0):
    sizes = [0] * num_groups
    indices = [0] * num_groups
    sizes[group_idx] = len(block_ids)
    indices[group_idx] = first_block_idx
    return GPULoadStoreSpec(block_ids, group_sizes=sizes, block_indices=indices)


def run(worker, gpu, cpu, *, store, job_id=0):
    ok = (
        worker.submit_store(job_id, gpu, cpu)
        if store
        else worker.submit_load(job_id, cpu, gpu)
    )
    assert ok
    worker.wait({job_id})
    torch.cuda.synchronize()
    assert [r.job_id for r in worker.get_finished()] == [job_id]


class Rig:
    def __init__(self, request, config, make_caches):
        # the server lives in this process; both "ranks" are read-only clients
        self.owner, self.server = open_test_client(config, read_only=True)
        self.peer, _ = open_test_client(config, may_start=False, read_only=True)
        self.geometry = geometry_for(config).adopt(self.owner)
        self.gpu, self.workers = [], []
        for tp_rank, client in enumerate((self.owner, self.peer)):
            tensors, caches = make_caches()
            self.gpu.append(tensors)
            self.workers.append(
                RadixShmemOffloadingWorker(
                    attach=lambda c=client, k=tp_rank: (c, self.geometry, k),
                    kv_caches=caches,
                )
            )

    def close(self):
        for w in reversed(self.workers):  # every client detaches before the server
            w.shutdown()
        self.server.close()


@pytest.fixture
def rig(request):
    param = getattr(request, "param", {}) or {}
    config = make_offloading_config(
        tag=unique_tag(request),
        tp_size=TP_SIZE,
        blocks_per_chunk=param.get("blocks_per_chunk", 1),
        groups=[group(bytes_per_block=2 * PAGE_BYTES)],
        cpu_bytes=16 << 20,
        extra={"replicated_kv": param.get("replicated", False)},
    )
    r = Rig(request, config, make_kv_caches)
    yield r
    r.close()


def test_group_views_address_this_ranks_region(rig):
    g = rig.geometry
    full = g.require_pool(PoolKind.FULL)
    _, caches = make_kv_caches()
    [plan] = plan_groups(g, caches)
    base, views = group_cpu_views(rig.owner.store, g, plan, writer_idx=1)
    assert len(views) == 2
    for t, v in enumerate(views):
        assert v.shape == (full.num_slots, PAGE_BYTES)
        assert v.stride() == (full.slot_stride, 1)
        assert v.data_ptr() == base.data_ptr() + full.slice_bytes + t * PAGE_BYTES


def test_round_trip_preserves_bytes(rig):
    w = rig.workers[0]
    src = rig.gpu[0]
    for t_idx, t in enumerate(src):
        for b in range(4):
            t[b].fill_(10 + t_idx * 4 + b)
    run(w, gpu_spec([0, 1, 2, 3]), CPULoadStoreSpec([5, 6, 7, 8]), store=True)
    for t in src:
        t.zero_()
    run(w, gpu_spec([0, 1, 2, 3]), CPULoadStoreSpec([5, 6, 7, 8]), store=False)
    for t_idx, t in enumerate(src):
        for b in range(4):
            assert torch.all(t[b] == 10 + t_idx * 4 + b), (t_idx, b)


def test_tp_ranks_do_not_alias(rig):
    w0, w1 = rig.workers
    for t in rig.gpu[0]:
        t[0].fill_(0x11)
    for t in rig.gpu[1]:
        t[0].fill_(0x22)
    # both ranks write block 0 into the *same* slot, each into its own slice
    run(w0, gpu_spec([0]), CPULoadStoreSpec([3]), store=True)
    run(w1, gpu_spec([0]), CPULoadStoreSpec([3]), store=True)
    for t in rig.gpu[0] + rig.gpu[1]:
        t[0].zero_()
    run(w0, gpu_spec([0]), CPULoadStoreSpec([3]), store=False)
    run(w1, gpu_spec([0]), CPULoadStoreSpec([3]), store=False)
    for t in rig.gpu[0]:
        assert torch.all(t[0] == 0x11)
    for t in rig.gpu[1]:
        assert torch.all(t[0] == 0x22)


def test_store_leaves_other_slots_untouched(rig):
    w = rig.workers[0]
    for t in rig.gpu[0]:
        t[0].fill_(0x5A)
    run(w, gpu_spec([0]), CPULoadStoreSpec([2]), store=True)
    for t in rig.gpu[0]:
        t[1].fill_(0x7F)
    run(w, gpu_spec([1]), CPULoadStoreSpec([9]), store=False)  # never written
    for t in rig.gpu[0]:
        assert torch.all(t[1] == 0)
        assert torch.all(t[0] == 0x5A)


@pytest.mark.parametrize("rig", [{"blocks_per_chunk": 4}], indirect=True)
def test_blocks_per_chunk_packs_sub_blocks(rig):
    """One slot holds ``blocks_per_chunk`` GPU blocks, in order, per tensor."""
    w = rig.workers[0]
    src = rig.gpu[0]
    for t in src:
        for b in range(8):
            t[b].fill_(b + 1)
    run(w, gpu_spec(list(range(8))), CPULoadStoreSpec([1, 2]), store=True)
    for t in src:
        t[:8].zero_()
    # load back only the second chunk -> GPU blocks 4..7 land in blocks 0..3
    run(
        w,
        gpu_spec(list(range(4)), first_block_idx=4),
        CPULoadStoreSpec([2]),
        store=False,
    )
    for t in src:
        for b in range(4):
            assert torch.all(t[b] == b + 5), b


@pytest.mark.parametrize("rig", [{"replicated": True}], indirect=True)
def test_replicated_layout_shares_one_slice(rig):
    """With replicated KV, rank 0 stores and every rank loads slice 0."""
    w0, w1 = rig.workers
    assert rig.geometry.require_pool(PoolKind.FULL).num_slices == 1
    for t in rig.gpu[0]:
        t[0].fill_(0x33)
    run(w0, gpu_spec([0]), CPULoadStoreSpec([4]), store=True)
    run(w1, gpu_spec([0]), CPULoadStoreSpec([4]), store=False)
    for t in rig.gpu[1]:
        assert torch.all(t[0] == 0x33)


def test_queued_transfers_complete_in_submission_order(rig):
    w = rig.workers[0]
    src = rig.gpu[0]
    for t in src:
        t[0].fill_(1)
        t[1].fill_(2)
    assert w.submit_store(0, gpu_spec([0]), CPULoadStoreSpec([1]))
    assert w.submit_store(1, gpu_spec([1]), CPULoadStoreSpec([2]))
    for t in src:
        t[:2].zero_()
    torch.cuda.synchronize()
    results = []
    while len(results) < 2:
        results += w.get_finished()
    assert [r.job_id for r in results] == [0, 1]
    assert all(r.success and r.transfer_size > 0 for r in results)
    run(w, gpu_spec([0, 1]), CPULoadStoreSpec([1, 2]), store=False)
    for t in src:
        assert torch.all(t[0] == 1) and torch.all(t[1] == 2)


# ---------------------------------------------------------------- two pools


@pytest.mark.parametrize("packed", [False, True])
def test_hybrid_job_splits_across_full_and_swa_pools(request, packed):
    """One connector job carrying a FULL group and a SWA group lands in two
    pools; the SWA slot packs two 8-token SWA blocks per 16-token position."""
    config = hybrid_config(
        unique_tag(request) + ("p" if packed else "l"), packed=packed
    )
    make_caches = make_packed_kv_caches if packed else make_hybrid_kv_caches
    r = Rig(request, config, make_caches)
    try:
        g = r.geometry
        assert [gr.sub_blocks for gr in g.groups if gr.kind == PoolKind.SWA] == [2]
        w = r.workers[0]
        if packed:
            block = r.gpu[0]
            full_view = [block[:, :PAGE_BYTES], block[:, PAGE_BYTES : 2 * PAGE_BYTES]]
            swa_view = block[:, 2 * PAGE_BYTES :]
        else:
            full_view, swa_view = r.gpu[0][:2], r.gpu[0][2]
        for t_idx, t in enumerate(full_view):
            for b in range(4):
                t[b].fill_(10 + t_idx * 4 + b)
        for b in range(4):
            swa_view[b].fill_(50 + b)

        # FULL positions 0..3 (blocks 0..3) and SWA blocks 0..3 (positions 0, 1)
        gpu = GPULoadStoreSpec(
            [0, 1, 2, 3, 0, 1, 2, 3], group_sizes=[4, 4], block_indices=[0, 0]
        )
        cpu = CPULoadStoreSpec([10, 11, 12, 13, 3, 4])  # 4 FULL slots + 2 SWA slots
        run(w, gpu, cpu, store=True)

        # the SWA pool holds exactly the SWA layer, nothing from the FULL layers
        swa_slot = r.owner.store.read_slot(3, PoolKind.SWA)
        assert swa_slot[:PAGE_BYTES] == bytes([50]) * PAGE_BYTES
        assert swa_slot[PAGE_BYTES : 2 * PAGE_BYTES] == bytes([51]) * PAGE_BYTES
        full_slot = r.owner.store.read_slot(10, PoolKind.FULL)
        assert full_slot[:PAGE_BYTES] == bytes([10]) * PAGE_BYTES
        assert full_slot[PAGE_BYTES : 2 * PAGE_BYTES] == bytes([14]) * PAGE_BYTES

        for t in full_view:
            t[:4].zero_()
        swa_view[:4].zero_()
        # load FULL position 2..3 and SWA blocks 2..3 (position 1) only
        gpu = GPULoadStoreSpec([2, 3, 2, 3], group_sizes=[2, 2], block_indices=[2, 2])
        run(w, gpu, CPULoadStoreSpec([12, 13, 4]), store=False, job_id=1)
        for t_idx, t in enumerate(full_view):
            assert torch.all(t[0] == 0) and torch.all(t[1] == 0)
            assert torch.all(t[2] == 12 + t_idx * 4) and torch.all(
                t[3] == 13 + t_idx * 4
            )
        assert torch.all(swa_view[0] == 0) and torch.all(swa_view[1] == 0)
        assert torch.all(swa_view[2] == 52) and torch.all(swa_view[3] == 53)
    finally:
        r.close()
