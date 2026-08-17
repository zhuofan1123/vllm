# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Packed-slot DMA between GPU KV caches and the shared CPU SlotStore.

``swap_blocks`` can only express ``base + block_id * page_bytes`` on both sides,
which cannot reach a sub-block inside a TP slice inside a slot. So addresses are
computed here and handed to ``cuMemcpyBatchAsync`` directly -- the same driver
entry point ``vllm/v1/simple_kv_offload/cuda_mem_ops.py`` already uses, so no
custom CUDA op and no source build of vLLM is needed.

Host address of GPU sub-block ``j`` of canonical tensor ``t`` in slot ``s``:

    data_base + s * slot_stride + tp_rank * tp_slice_bytes
              + tensor_offsets[t] + j * page_bytes[t]
"""

import ctypes

import numpy as np
import torch

from vllm.logger import init_logger
from vllm.v1.simple_kv_offload.cuda_mem_ops import (
    _CUmemcpyAttributes,
    _resolve_batch_memcpy,
)

logger = init_logger(__name__)

_batch_memcpy_fn = None


def host_register(ptr: int, nbytes: int) -> None:
    """cudaHostRegister a raw range (the whole SlotStore mapping)."""
    err = torch.cuda.cudart().cudaHostRegister(ptr, nbytes, 0)
    if err.value != 0:
        raise RuntimeError(
            f"cudaHostRegister({ptr:#x}, {nbytes}) failed: {err}. Registering a "
            "multi-GB shared region can fail on a low RLIMIT_MEMLOCK."
        )


def host_unregister(ptr: int) -> None:
    err = torch.cuda.cudart().cudaHostUnregister(ptr)
    if err.value != 0:
        logger.warning("cudaHostUnregister(%#x) failed: %s", ptr, err)


def copy_addrs(
    src_addrs: np.ndarray,
    dst_addrs: np.ndarray,
    sizes: np.ndarray,
    stream: torch.cuda.Stream,
) -> None:
    """Submit an arbitrary-address batch copy on ``stream``."""
    global _batch_memcpy_fn
    n = src_addrs.size
    if n == 0:
        return
    if _batch_memcpy_fn is None:
        _batch_memcpy_fn = _resolve_batch_memcpy()

    attrs = _CUmemcpyAttributes(srcAccessOrder=3)  # ANY
    attrs_idx = ctypes.c_size_t(0)
    fail_idx = ctypes.c_size_t(0)
    err = _batch_memcpy_fn(
        dst_addrs.ctypes.data,
        src_addrs.ctypes.data,
        sizes.ctypes.data,
        n,
        ctypes.addressof(attrs),
        ctypes.byref(attrs_idx),
        1,
        ctypes.byref(fail_idx),
        stream.cuda_stream,
    )
    if err != 0:
        raise RuntimeError(
            f"cuMemcpyBatchAsync failed: err={err} failIdx={fail_idx.value}"
        )


class PackedSlotAddresser:
    """Turns (gpu block ids, slot ids) into GPU/host address pairs.

    Pure arithmetic, no CUDA -- so it is unit-testable on a CPU-only box.
    """

    def __init__(
        self,
        *,
        data_base: int,
        slot_stride: int,
        num_slots: int,
        tp_rank: int,
        tp_slice_bytes: int,
        tensor_offsets: tuple[int, ...],
        page_bytes: tuple[int, ...],
        gpu_bases: tuple[int, ...],
        gpu_num_blocks: int,
        block_size_factor: int,
    ):
        assert len(tensor_offsets) == len(page_bytes) == len(gpu_bases)
        self.data_base = data_base
        self.slot_stride = slot_stride
        self.num_slots = num_slots
        self.tp_rank = tp_rank
        self.tp_slice_bytes = tp_slice_bytes
        self.block_size_factor = block_size_factor
        self.gpu_num_blocks = gpu_num_blocks
        self.num_tensors = len(page_bytes)

        self.page_bytes = np.array(page_bytes, dtype=np.uint64)
        self.gpu_bases = np.array(gpu_bases, dtype=np.uint64)
        # host base of tensor t, for tp slice of this rank, at slot 0
        self.host_bases = np.array(
            [data_base + tp_rank * tp_slice_bytes + off for off in tensor_offsets],
            dtype=np.uint64,
        )
        self.bytes_per_slot_copy = int(sum(page_bytes)) * block_size_factor

    def build(
        self, gpu_block_ids: np.ndarray, slot_ids: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (gpu_addrs, host_addrs, sizes), one entry per tensor/sub-block.

        ``gpu_block_ids`` is at GPU-block granularity, ``slot_ids`` at offloaded
        (slot) granularity. When the two do not line up exactly, the leading
        ``(-len(gpu_block_ids)) % factor`` sub-blocks of the first slot are
        skipped -- matching ``cpu_gpu.SingleDirectionOffloadingHandler``.
        """
        factor = self.block_size_factor
        gpu_block_ids = np.ascontiguousarray(gpu_block_ids, dtype=np.uint64)
        slot_ids = np.ascontiguousarray(slot_ids, dtype=np.uint64)
        m = gpu_block_ids.size
        skip = (-m) % factor
        if slot_ids.size * factor != m + skip:
            raise ValueError(
                f"{m} GPU blocks (+{skip} skipped) do not fill {slot_ids.size} "
                f"slots of {factor} blocks each"
            )
        if m == 0:
            empty = np.empty(0, dtype=np.uint64)
            return empty, empty, empty

        if slot_ids.max() >= self.num_slots:
            raise ValueError(
                f"slot id {int(slot_ids.max())} out of range (num_slots="
                f"{self.num_slots})"
            )
        if gpu_block_ids.max() >= self.gpu_num_blocks:
            raise ValueError(
                f"GPU block id {int(gpu_block_ids.max())} out of range "
                f"(num_gpu_blocks={self.gpu_num_blocks})"
            )

        k = np.arange(skip, skip + m, dtype=np.uint64)
        slot_of = slot_ids[k // factor]
        sub_of = k % factor

        # (num_tensors, m)
        host = (
            self.host_bases[:, None]
            + slot_of[None, :] * np.uint64(self.slot_stride)
            + sub_of[None, :] * self.page_bytes[:, None]
        ).ravel()
        gpu = (
            self.gpu_bases[:, None] + gpu_block_ids[None, :] * self.page_bytes[:, None]
        ).ravel()
        sizes = np.repeat(self.page_bytes, m)
        return gpu, host, sizes

    def num_bytes(self, gpu_block_count: int) -> int:
        return int(self.page_bytes.sum()) * gpu_block_count
