# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Packed-slot addressing and the batched GPU<->SlotStore DMA."""

import numpy as np
import pytest
import torch

from vllm.v1.kv_offload.radixshmem.mem import PackedSlotAddresser

shmradix_data = pytest.importorskip("shmradix._data")

PAGE_BYTES = (256, 512)
FACTOR = 2
TP_SIZE = 2
NUM_SLOTS = 8
NUM_GPU_BLOCKS = 32

TP_SLICE = sum(PAGE_BYTES) * FACTOR  # 1536
SLOT_BYTES = TP_SLICE * TP_SIZE  # 3072
SLOT_STRIDE = 4096  # deliberately padded, to catch stride/size confusion
TENSOR_OFFSETS = (0, PAGE_BYTES[0] * FACTOR)  # (0, 512)


def make_addresser(tp_rank: int, data_base: int, gpu_bases: tuple[int, ...]):
    return PackedSlotAddresser(
        data_base=data_base,
        slot_stride=SLOT_STRIDE,
        num_slots=NUM_SLOTS,
        tp_rank=tp_rank,
        tp_slice_bytes=TP_SLICE,
        tensor_offsets=TENSOR_OFFSETS,
        page_bytes=PAGE_BYTES,
        gpu_bases=gpu_bases,
        gpu_num_blocks=NUM_GPU_BLOCKS,
        block_size_factor=FACTOR,
    )


# ------------------------------------------------------------------ addressing


def test_addresses_match_the_formula():
    a = make_addresser(tp_rank=1, data_base=0, gpu_bases=(0, 1 << 20))
    gpu_blocks = np.array([10, 11, 20, 21], dtype=np.int64)
    slots = np.array([3, 5], dtype=np.int64)
    gpu, host, sizes = a.build(gpu_blocks, slots)

    assert gpu.size == host.size == sizes.size == len(PAGE_BYTES) * 4

    def expect_host(t, slot, sub):
        return (
            slot * SLOT_STRIDE + 1 * TP_SLICE + TENSOR_OFFSETS[t] + sub * PAGE_BYTES[t]
        )

    # tensor-major, then block order
    assert host[0] == expect_host(0, 3, 0)
    assert host[1] == expect_host(0, 3, 1)
    assert host[2] == expect_host(0, 5, 0)
    assert host[3] == expect_host(0, 5, 1)
    assert host[4] == expect_host(1, 3, 0)
    assert host[7] == expect_host(1, 5, 1)

    assert gpu[0] == 10 * PAGE_BYTES[0]
    assert gpu[4] == (1 << 20) + 10 * PAGE_BYTES[1]
    assert list(sizes[:4]) == [PAGE_BYTES[0]] * 4
    assert list(sizes[4:]) == [PAGE_BYTES[1]] * 4


def test_load_skips_leading_sub_blocks():
    a = make_addresser(tp_rank=0, data_base=0, gpu_bases=(0, 0))
    # 3 GPU blocks over 2 slots of factor 2 -> the first sub-block is skipped
    gpu_blocks = np.array([7, 8, 9], dtype=np.int64)
    slots = np.array([1, 2], dtype=np.int64)
    _, host, _ = a.build(gpu_blocks, slots)
    assert host[0] == 1 * SLOT_STRIDE + PAGE_BYTES[0]  # slot 1, sub-block 1
    assert host[1] == 2 * SLOT_STRIDE  # slot 2, sub-block 0
    assert host[2] == 2 * SLOT_STRIDE + PAGE_BYTES[0]


def test_mismatched_counts_rejected():
    a = make_addresser(tp_rank=0, data_base=0, gpu_bases=(0, 0))
    with pytest.raises(ValueError, match="do not fill"):
        a.build(np.array([1, 2, 3, 4, 5]), np.array([0, 1]))


def test_out_of_range_ids_rejected():
    a = make_addresser(tp_rank=0, data_base=0, gpu_bases=(0, 0))
    with pytest.raises(ValueError, match="slot id"):
        a.build(np.array([0, 1]), np.array([NUM_SLOTS]))
    with pytest.raises(ValueError, match="GPU block id"):
        a.build(np.array([0, NUM_GPU_BLOCKS]), np.array([0]))


def test_no_overlap_across_ranks_tensors_and_slots():
    """Every byte written by any (rank, tensor, slot, sub-block) is distinct."""
    touched: dict[int, tuple] = {}
    slots = np.arange(NUM_SLOTS, dtype=np.int64)
    gpu_blocks = np.arange(NUM_SLOTS * FACTOR, dtype=np.int64)
    for rank in range(TP_SIZE):
        a = make_addresser(tp_rank=rank, data_base=0, gpu_bases=(0, 0))
        _, host, sizes = a.build(gpu_blocks, slots)
        for addr, size in zip(host.tolist(), sizes.tolist()):
            for b in range(addr, addr + size):
                key = (rank, addr)
                assert b not in touched, f"byte {b} written twice: {touched[b]} / {key}"
                touched[b] = key
            # never crosses the slot it belongs to
            slot = addr // SLOT_STRIDE
            assert (addr + size - 1) // SLOT_STRIDE == slot
            assert addr % SLOT_STRIDE + size <= SLOT_BYTES


# ------------------------------------------------------------------- real DMA


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("tp_rank", [0, 1])
def test_roundtrip_gpu_cpu_gpu(tp_rank):
    import os

    from vllm.v1.kv_offload.radixshmem.mem import (
        copy_addrs,
        host_register,
        host_unregister,
    )

    name = f"/rs_copy_{os.getpid()}_{tp_rank}"
    shmradix_data.SlotStore.destroy(name, "")
    cfg = shmradix_data.SlotStoreConfig()
    cfg.name, cfg.num_slots, cfg.slot_bytes = name, NUM_SLOTS, SLOT_BYTES
    cfg.slot_align, cfg.hugepage_path, cfg.prefault = 4096, "", True
    store = shmradix_data.SlotStore.create(cfg)
    assert store.slot_bytes == SLOT_STRIDE

    host_np = np.frombuffer(store.data_view(), dtype=np.uint8)
    data_base = host_np.ctypes.data
    host_np[:] = 0xEE  # guard fill

    torch.cuda.set_device(0)
    gpu_tensors = [
        torch.arange(NUM_GPU_BLOCKS * pb, dtype=torch.uint8, device="cuda")
        .remainder(251)
        .add(tp_rank + 1)
        .view(NUM_GPU_BLOCKS, pb)
        for pb in PAGE_BYTES
    ]
    golden = [t.cpu().clone() for t in gpu_tensors]

    host_register(data_base, host_np.nbytes)
    try:
        a = make_addresser(
            tp_rank=tp_rank,
            data_base=data_base,
            gpu_bases=tuple(t.data_ptr() for t in gpu_tensors),
        )
        slots = np.array([1, 3, 6], dtype=np.int64)
        gpu_blocks = np.array([4, 5, 12, 13, 30, 31], dtype=np.int64)
        stream = torch.cuda.Stream()

        gpu_addrs, host_addrs, sizes = a.build(gpu_blocks, slots)
        with torch.cuda.stream(stream):
            copy_addrs(gpu_addrs, host_addrs, sizes, stream)
        stream.synchronize()

        # every touched byte matches the GPU source
        for ti, pb in enumerate(PAGE_BYTES):
            for i, slot in enumerate(slots):
                for sub in range(FACTOR):
                    off = (
                        slot * SLOT_STRIDE
                        + tp_rank * TP_SLICE
                        + TENSOR_OFFSETS[ti]
                        + sub * pb
                    )
                    want = golden[ti][gpu_blocks[i * FACTOR + sub]].numpy()
                    assert np.array_equal(host_np[off : off + pb], want)

        # nothing else moved: other TP slice, other slots, slot tail padding
        other = 1 - tp_rank
        for slot in range(NUM_SLOTS):
            base = slot * SLOT_STRIDE
            assert np.all(
                host_np[base + other * TP_SLICE : base + other * TP_SLICE + TP_SLICE]
                == 0xEE
            ), f"TP slice {other} of slot {slot} was clobbered"
            assert np.all(host_np[base + SLOT_BYTES : base + SLOT_STRIDE] == 0xEE), (
                f"tail padding of slot {slot} was clobbered"
            )
            if slot not in slots:
                assert np.all(host_np[base : base + SLOT_BYTES] == 0xEE), (
                    f"untouched slot {slot} was written"
                )

        # load it back into different GPU blocks and compare
        for t in gpu_tensors:
            t.zero_()
        dst_blocks = np.array([0, 1, 2, 3, 20, 21], dtype=np.int64)
        gpu_addrs, host_addrs, sizes = a.build(dst_blocks, slots)
        with torch.cuda.stream(stream):
            copy_addrs(host_addrs, gpu_addrs, sizes, stream)
        stream.synchronize()

        for ti in range(len(PAGE_BYTES)):
            got = gpu_tensors[ti][dst_blocks].cpu()
            want = golden[ti][gpu_blocks]
            assert torch.equal(got, want)
    finally:
        host_unregister(data_base)
        del host_np
        store = None
        shmradix_data.SlotStore.destroy(name, "")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_load_skip_roundtrip():
    """A partial first slot: the skipped sub-block must stay untouched on GPU."""
    import os

    from vllm.v1.kv_offload.radixshmem.mem import (
        copy_addrs,
        host_register,
        host_unregister,
    )

    name = f"/rs_skip_{os.getpid()}"
    shmradix_data.SlotStore.destroy(name, "")
    cfg = shmradix_data.SlotStoreConfig()
    cfg.name, cfg.num_slots, cfg.slot_bytes = name, NUM_SLOTS, SLOT_BYTES
    cfg.slot_align, cfg.hugepage_path, cfg.prefault = 4096, "", True
    store = shmradix_data.SlotStore.create(cfg)
    host_np = np.frombuffer(store.data_view(), dtype=np.uint8)
    data_base = host_np.ctypes.data
    host_np[:] = 0x77

    torch.cuda.set_device(0)
    gpu_tensors = [
        torch.zeros(NUM_GPU_BLOCKS, pb, dtype=torch.uint8, device="cuda")
        for pb in PAGE_BYTES
    ]
    host_register(data_base, host_np.nbytes)
    try:
        a = make_addresser(
            tp_rank=0,
            data_base=data_base,
            gpu_bases=tuple(t.data_ptr() for t in gpu_tensors),
        )
        # 3 GPU blocks over 2 slots -> skip the first sub-block of slot 2
        gpu_blocks = np.array([1, 2, 3], dtype=np.int64)
        slots = np.array([2, 4], dtype=np.int64)
        gpu_addrs, host_addrs, sizes = a.build(gpu_blocks, slots)
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            copy_addrs(host_addrs, gpu_addrs, sizes, stream)
        stream.synchronize()

        for ti in range(len(PAGE_BYTES)):
            assert torch.all(gpu_tensors[ti][0] == 0), "GPU block 0 should be untouched"
            assert torch.all(gpu_tensors[ti][1] == 0x77)
            assert torch.all(gpu_tensors[ti][3] == 0x77)
    finally:
        host_unregister(data_base)
        del host_np
        store = None
        shmradix_data.SlotStore.destroy(name, "")
