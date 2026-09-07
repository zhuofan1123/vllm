# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU <-> shared-slot round trips through upstream's transfer handler.

Both TP ranks run in this one process against the same shared region, which is
exactly the aliasing the packed slot layout has to get right: rank 1 writing its
slice must not disturb rank 0's. Needs a GPU.
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
from vllm.v1.kv_offload.radixshmem.bootstrap import attach_regions, create_regions
from vllm.v1.kv_offload.radixshmem.worker import (
    RadixShmemOffloadingWorker,
    slot_views,
)

from .utils import geometry_for, make_offloading_config, unique_tag

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

PAGE_BYTES = 2048
NUM_TENSORS = 2
NUM_GPU_BLOCKS = 32
TP_SIZE = 2


def make_kv_caches():
    tensors = [
        torch.zeros(NUM_GPU_BLOCKS, PAGE_BYTES, dtype=torch.int8, device="cuda")
        for _ in range(NUM_TENSORS)
    ]
    return tensors, CanonicalKVCaches(
        tensors=[
            CanonicalKVCacheTensor(tensor=t, page_size_bytes=PAGE_BYTES)
            for t in tensors
        ],
        group_data_refs=[
            [
                CanonicalKVCacheRef(tensor_idx=i, page_size_bytes=PAGE_BYTES)
                for i in range(NUM_TENSORS)
            ]
        ],
    )


def gpu_spec(block_ids, first_block_idx=0):
    return GPULoadStoreSpec(
        block_ids, group_sizes=(len(block_ids),), block_indices=(first_block_idx,)
    )


def run(worker, gpu_block_ids, slot_ids, *, store, first_block_idx=0):
    if store:
        ok = worker.submit_store(
            0, gpu_spec(gpu_block_ids, first_block_idx), CPULoadStoreSpec(slot_ids)
        )
    else:
        ok = worker.submit_load(
            0, CPULoadStoreSpec(slot_ids), gpu_spec(gpu_block_ids, first_block_idx)
        )
    assert ok
    worker.wait({0})
    torch.cuda.synchronize()
    assert [r.job_id for r in worker.get_finished()] == [0]


@pytest.fixture
def rig(request):
    """One worker per TP rank, each over its own mapping of one region."""
    replicated = getattr(request, "param", {}).get("replicated", False)
    blocks_per_chunk = getattr(request, "param", {}).get("blocks_per_chunk", 1)
    config = make_offloading_config(
        tag=unique_tag(request),
        tp_size=TP_SIZE,
        blocks_per_chunk=blocks_per_chunk,
        worker_bytes_per_block=PAGE_BYTES * NUM_TENSORS,
        cpu_bytes=16 << 20,
        extra={"replicated_kv": replicated},
    )
    geometry = geometry_for(config)
    owner = create_regions(geometry, dict(config.extra_config))
    peer = attach_regions(geometry, timeout_s=60, role="tp1")
    gpu_tensors, workers = [], []
    try:
        for tp_rank, regions in enumerate((owner, peer)):
            tensors, caches = make_kv_caches()
            gpu_tensors.append(tensors)
            workers.append(
                RadixShmemOffloadingWorker(
                    attach=lambda r=regions, k=tp_rank: (r, k),
                    kv_caches=caches,
                    blocks_per_chunk=blocks_per_chunk,
                )
            )
        yield {
            "geometry": geometry,
            "workers": workers,
            "gpu": gpu_tensors,
            "regions": (owner, peer),
        }
    finally:
        for w in reversed(workers):  # the peer detaches before the owner unlinks
            w.shutdown()


def test_slot_views_address_this_ranks_slice(rig):
    g = rig["geometry"]
    owner = rig["regions"][0]
    _, caches = make_kv_caches()
    base, views = slot_views(owner, caches, writer_idx=1)
    assert len(views) == NUM_TENSORS
    for t, v in enumerate(views):
        assert v.shape == (g.num_slots, PAGE_BYTES)
        assert v.stride() == (g.slot_stride, 1)
        expected = base.data_ptr() + g.slice_bytes + t * PAGE_BYTES
        assert v.data_ptr() == expected


def test_round_trip_preserves_bytes(rig):
    w = rig["workers"][0]
    src = rig["gpu"][0]
    for t_idx, t in enumerate(src):
        for b in range(4):
            t[b].fill_(10 + t_idx * 4 + b)
    run(w, [0, 1, 2, 3], [5, 6, 7, 8], store=True)
    for t in src:
        t.zero_()
    run(w, [0, 1, 2, 3], [5, 6, 7, 8], store=False)
    for t_idx, t in enumerate(src):
        for b in range(4):
            assert torch.all(t[b] == 10 + t_idx * 4 + b), (t_idx, b)


def test_tp_ranks_do_not_alias(rig):
    w0, w1 = rig["workers"]
    for t in rig["gpu"][0]:
        t[0].fill_(0x11)
    for t in rig["gpu"][1]:
        t[0].fill_(0x22)
    # both ranks write block 0 into the *same* slot, each into its own slice
    run(w0, [0], [3], store=True)
    run(w1, [0], [3], store=True)
    for t in rig["gpu"][0] + rig["gpu"][1]:
        t[0].zero_()
    run(w0, [0], [3], store=False)
    run(w1, [0], [3], store=False)
    for t in rig["gpu"][0]:
        assert torch.all(t[0] == 0x11)
    for t in rig["gpu"][1]:
        assert torch.all(t[0] == 0x22)


def test_store_leaves_other_slots_untouched(rig):
    w = rig["workers"][0]
    for t in rig["gpu"][0]:
        t[0].fill_(0x5A)
    run(w, [0], [2], store=True)
    for t in rig["gpu"][0]:
        t[1].fill_(0x7F)
    run(w, [1], [9], store=False)  # a slot nobody wrote is still zero
    for t in rig["gpu"][0]:
        assert torch.all(t[1] == 0)
        assert torch.all(t[0] == 0x5A)


@pytest.mark.parametrize("rig", [{"blocks_per_chunk": 4}], indirect=True)
def test_blocks_per_chunk_packs_sub_blocks(rig):
    """One slot holds ``blocks_per_chunk`` GPU blocks, in order, per tensor."""
    w = rig["workers"][0]
    src = rig["gpu"][0]
    for t in src:
        for b in range(8):
            t[b].fill_(b + 1)
    run(w, list(range(8)), [1, 2], store=True)
    for t in src:
        t[:8].zero_()
    # load back only the second chunk -> GPU blocks 4..7 land in blocks 0..3
    run(w, list(range(4)), [2], store=False, first_block_idx=4)
    for t in src:
        for b in range(4):
            assert torch.all(t[b] == b + 5), b


@pytest.mark.parametrize("rig", [{"replicated": True}], indirect=True)
def test_replicated_layout_shares_one_slice(rig):
    """With replicated KV, rank 0 stores and every rank loads slice 0."""
    w0, w1 = rig["workers"]
    assert rig["geometry"].num_slices == 1
    for t in rig["gpu"][0]:
        t[0].fill_(0x33)
    run(w0, [0], [4], store=True)
    run(w1, [0], [4], store=False)
    for t in rig["gpu"][1]:
        assert torch.all(t[0] == 0x33)


def test_queued_transfers_complete_in_submission_order(rig):
    w = rig["workers"][0]
    src = rig["gpu"][0]
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
    run(w, [0, 1], [1, 2], store=False)
    for t in src:
        assert torch.all(t[0] == 1) and torch.all(t[1] == 2)
