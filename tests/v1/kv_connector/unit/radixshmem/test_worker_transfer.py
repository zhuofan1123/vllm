# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU <-> shared-slot round trips through the real handlers.

Both TP ranks run in this one process against the same shared region, which is
exactly the aliasing the packed slot layout has to get right: rank 1 writing its
slice must not disturb rank 0's. Needs a GPU. Run inside the radixshmem_vllm
container with PYTHONPATH=/raid/zfl/vllm:/raid/zfl/RadixShmem/python.
"""

import os
from types import SimpleNamespace

import pytest
import torch

from vllm.distributed.kv_transfer.kv_connector.v1.radixshmem import (
    attach_regions,
    compute_geometry,
    create_regions,
)
from vllm.v1.kv_offload.base import (
    CanonicalKVCacheRef,
    CanonicalKVCaches,
    GPULoadStoreSpec,
)
from vllm.v1.kv_offload.base import CanonicalKVCacheTensor as Tensor
from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec
from vllm.v1.kv_offload.radixshmem.worker import RadixShmemOffloadingHandlers

pytest.importorskip("shmradix")
pytest.importorskip("shmradix._data")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

GPU_BLOCK_SIZE = 16
PAGE_BYTES = 2048
NUM_TENSORS = 2
NUM_GPU_BLOCKS = 32
TP_SIZE = 2


def make_configs(*, tag: str, factor: int = 1):
    extra = {
        "index_shm_name": f"/rs_wt_idx_{tag}",
        "data_shm_name": f"/rs_wt_dat_{tag}",
        "cpu_bytes_to_use": 16 << 20,
        "slot_align": 4096,
    }
    if factor != 1:
        extra["block_size"] = GPU_BLOCK_SIZE * factor
    vllm_config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(kv_connector_extra_config=extra),
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=1,
            tensor_parallel_size=TP_SIZE,
            world_size=TP_SIZE,
            data_parallel_index=0,
        ),
        kv_events_config=None,
    )
    kv_cache_config = SimpleNamespace(
        kv_cache_groups=[
            SimpleNamespace(
                kv_cache_spec=SimpleNamespace(
                    block_size=GPU_BLOCK_SIZE, page_size_bytes=PAGE_BYTES
                ),
                layer_names=[f"l{i}" for i in range(NUM_TENSORS)],
            )
        ]
    )
    return vllm_config, kv_cache_config, extra


def make_kv_caches():
    tensors = [
        torch.zeros(NUM_GPU_BLOCKS, PAGE_BYTES, dtype=torch.int8, device="cuda")
        for _ in range(NUM_TENSORS)
    ]
    return tensors, CanonicalKVCaches(
        tensors=[Tensor(tensor=t, page_size_bytes=PAGE_BYTES) for t in tensors],
        group_data_refs=[
            [CanonicalKVCacheRef(tensor_idx=i, page_size_bytes=PAGE_BYTES)]
            for i in range(NUM_TENSORS)
        ],
    )


def run(handler, gpu_block_ids, slot_ids, *, gpu_to_cpu):
    gpu = GPULoadStoreSpec(
        gpu_block_ids, group_sizes=(len(gpu_block_ids),), block_indices=(0,)
    )
    cpu = CPULoadStoreSpec(slot_ids)
    spec = (gpu, cpu) if gpu_to_cpu else (cpu, gpu)
    assert handler.transfer_async(0, spec)
    handler.drain()
    torch.cuda.synchronize()


@pytest.fixture
def rig(request):
    """One handler set per TP rank, each over its own mapping of one region.

    Separate mappings (rank 0 creates, rank 1 attaches) rather than a shared
    ``SharedRegions``: each ``host_register`` needs its own virtual range, and
    it is closer to what the real worker processes do anyway.
    """
    tag = f"{os.getpid()}_{request.node.name[:20]}"
    factor = getattr(request, "param", 1)
    vllm_config, kv_cache_config, extra = make_configs(tag=tag, factor=factor)
    geometry = compute_geometry(vllm_config, kv_cache_config)
    owner = create_regions(geometry, extra)
    peer = attach_regions(geometry, timeout_s=60, role="tp1")
    gpu_tensors, handlers = [], []
    try:
        for tp_rank, regions in enumerate((owner, peer)):
            tensors, caches = make_kv_caches()
            gpu_tensors.append(tensors)
            handlers.append(
                RadixShmemOffloadingHandlers(
                    regions=regions, kv_caches=caches, tp_rank=tp_rank
                )
            )
        yield SimpleNamespace(
            geometry=geometry,
            handlers=handlers,
            gpu_tensors=gpu_tensors,
            factor=factor,
        )
    finally:
        for h in reversed(handlers):  # the peer detaches before the owner unlinks
            h.close()
        peer.close()
        owner.close()


def test_round_trip_preserves_bytes(rig):
    h = rig.handlers[0]
    src = rig.gpu_tensors[0]
    for t_idx, t in enumerate(src):
        for b in range(4):
            t[b].fill_(10 + t_idx * 4 + b)

    run(h.gpu_to_cpu_handler, [0, 1, 2, 3], [5, 6, 7, 8], gpu_to_cpu=True)
    for t in src:
        t.zero_()
    run(h.cpu_to_gpu_handler, [0, 1, 2, 3], [5, 6, 7, 8], gpu_to_cpu=False)

    for t_idx, t in enumerate(src):
        for b in range(4):
            assert torch.all(t[b] == 10 + t_idx * 4 + b), (t_idx, b)


def test_tp_ranks_do_not_alias(rig):
    h0, h1 = rig.handlers
    for t in rig.gpu_tensors[0]:
        t[0].fill_(0x11)
    for t in rig.gpu_tensors[1]:
        t[0].fill_(0x22)

    # both ranks write block 0 into the *same* slot, each into its own slice
    run(h0.gpu_to_cpu_handler, [0], [3], gpu_to_cpu=True)
    run(h1.gpu_to_cpu_handler, [0], [3], gpu_to_cpu=True)

    for t in rig.gpu_tensors[0] + rig.gpu_tensors[1]:
        t[0].zero_()
    run(h0.cpu_to_gpu_handler, [0], [3], gpu_to_cpu=False)
    run(h1.cpu_to_gpu_handler, [0], [3], gpu_to_cpu=False)

    for t in rig.gpu_tensors[0]:
        assert torch.all(t[0] == 0x11)
    for t in rig.gpu_tensors[1]:
        assert torch.all(t[0] == 0x22)


def test_store_leaves_other_slots_untouched(rig):
    h = rig.handlers[0]
    for t in rig.gpu_tensors[0]:
        t[0].fill_(0x5A)
    run(h.gpu_to_cpu_handler, [0], [2], gpu_to_cpu=True)

    # read back a slot nobody wrote; it must still be zero
    for t in rig.gpu_tensors[0]:
        t[1].fill_(0x7F)
    run(h.cpu_to_gpu_handler, [1], [9], gpu_to_cpu=False)
    for t in rig.gpu_tensors[0]:
        assert torch.all(t[1] == 0)
        assert torch.all(t[0] == 0x5A)


@pytest.mark.parametrize("rig", [4], indirect=True)
def test_block_size_factor_packs_sub_blocks(rig):
    """One slot holds ``factor`` GPU blocks, in order, per tensor."""
    h = rig.handlers[0]
    src = rig.gpu_tensors[0]
    for t in src:
        for b in range(8):
            t[b].fill_(b + 1)

    run(h.gpu_to_cpu_handler, list(range(8)), [1, 2], gpu_to_cpu=True)
    for t in src:
        t[:8].zero_()
    # load back only the second slot -> GPU blocks 4..7
    run(h.cpu_to_gpu_handler, list(range(4)), [2], gpu_to_cpu=False)
    for t in src:
        for b in range(4):
            assert torch.all(t[b] == b + 5), b


def test_queued_transfers_complete_in_submission_order(rig):
    h = rig.handlers[0].gpu_to_cpu_handler
    src = rig.gpu_tensors[0]
    for t in src:
        t[0].fill_(1)
        t[1].fill_(2)

    gpu = lambda ids: GPULoadStoreSpec(  # noqa: E731
        ids, group_sizes=(len(ids),), block_indices=(0,)
    )
    assert h.transfer_async(0, (gpu([0]), CPULoadStoreSpec([1])))
    assert h.transfer_async(1, (gpu([1]), CPULoadStoreSpec([2])))
    for t in rig.gpu_tensors[0]:
        t[:2].zero_()
    torch.cuda.synchronize()

    results = []
    while len(results) < 2:
        results += h.get_finished()
    assert [r.job_id for r in results] == [0, 1]
    assert all(r.success and r.transfer_size > 0 for r in results)

    load = rig.handlers[0].cpu_to_gpu_handler
    run(load, [0, 1], [1, 2], gpu_to_cpu=False)
    for t in src:
        assert torch.all(t[0] == 1) and torch.all(t[1] == 2)


def test_empty_transfer_is_rejected(rig):
    h = rig.handlers[0].gpu_to_cpu_handler
    spec = (
        GPULoadStoreSpec([], group_sizes=(0,), block_indices=(0,)),
        CPULoadStoreSpec([]),
    )
    assert h.transfer_async(0, spec) is False
