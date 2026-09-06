# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Slot geometry for the RadixShmem shared KV cache.

Everything here is derivable from ``VllmConfig`` + ``KVCacheConfig``, i.e. before
any KV tensor exists. That is what lets the DP rank 0 scheduler create the shared
regions and every other process attach to them.

Slot layout (one slot == one offloaded block, shared by the whole TP group):

    slot_base = data_base + slot_id * slot_stride
    tp slice  = slot_base + tp_rank * tp_slice_bytes
    tensor t  = tp slice  + tensor_offset[t]
    sub-block = tensor t  + j * page_bytes[t]      j in [0, block_size_factor)

Padding (slot_stride - slot_bytes) only ever lands at the tail of a slot.
"""

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.kv_offload.base import CanonicalKVCaches

DEFAULT_INDEX_SHM_NAME = "/vllm_kv_index"
DEFAULT_DATA_SHM_NAME = "/vllm_kv_data"
DEFAULT_SLOT_ALIGN = 4096
DEFAULT_ATTACH_TIMEOUT_S = 300.0
DEFAULT_SENTINEL_DIR = "/dev/shm"


class GeometryMismatch(RuntimeError):
    """Raised when two processes disagree about the shared-region geometry."""


def _align_up(value: int, align: int) -> int:
    return (value + align - 1) // align * align


@dataclass(frozen=True)
class SlotGeometry:
    """Fully-resolved geometry of the shared index + SlotStore."""

    index_shm_name: str
    data_shm_name: str
    hugepage_path: str

    # tokens
    gpu_block_size: int
    offloaded_block_size: int
    block_size_factor: int

    # bytes
    per_rank_block_bytes: int  # all canonical tensors, one GPU block, one rank
    tp_slice_bytes: int  # per_rank_block_bytes * block_size_factor
    slot_bytes: int  # tp_size * tp_slice_bytes (unpadded)
    slot_align: int
    slot_stride: int  # slot_bytes rounded up to slot_align

    # counts
    tp_size: int
    num_slots: int

    @property
    def total_data_bytes(self) -> int:
        return self.num_slots * self.slot_stride

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SlotGeometry":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__})

    def check_same(self, other: "SlotGeometry", *, what: str) -> None:
        """Fail closed on any field mismatch, naming every offending field."""
        if self == other:
            return
        diffs = [
            f"{f}: local={getattr(self, f)!r} {what}={getattr(other, f)!r}"
            for f in self.__dataclass_fields__
            if getattr(self, f) != getattr(other, f)
        ]
        raise GeometryMismatch(
            "RadixShmem geometry mismatch between this process and "
            f"{what}; every process attaching the same shared region must "
            "derive an identical geometry. Differing fields:\n  " + "\n  ".join(diffs)
        )


def compute_geometry(
    vllm_config: "VllmConfig",
    kv_cache_config: "KVCacheConfig",
) -> SlotGeometry:
    """Derive the shared geometry from config alone (no tensors required)."""
    kv_transfer_config = vllm_config.kv_transfer_config
    assert kv_transfer_config is not None
    extra: dict[str, Any] = kv_transfer_config.kv_connector_extra_config or {}

    parallel_config = vllm_config.parallel_config
    if parallel_config.pipeline_parallel_size != 1:
        raise ValueError(
            "RadixShmemConnector requires pipeline_parallel_size == 1 "
            f"(got {parallel_config.pipeline_parallel_size})"
        )
    tp_size = parallel_config.tensor_parallel_size
    if tp_size != parallel_config.world_size:
        raise ValueError(
            "RadixShmemConnector expects world_size == tensor_parallel_size "
            f"(got world_size={parallel_config.world_size}, tp={tp_size})"
        )

    groups = kv_cache_config.kv_cache_groups
    if not groups:
        raise ValueError("RadixShmemConnector requires at least one KV cache group")

    gpu_block_sizes = {g.kv_cache_spec.block_size for g in groups}
    if len(gpu_block_sizes) != 1:
        raise ValueError(
            "RadixShmemConnector requires a single GPU block size, "
            f"got {gpu_block_sizes}"
        )
    gpu_block_size = gpu_block_sizes.pop()

    offloaded_block_size = int(extra.get("block_size") or gpu_block_size)
    if offloaded_block_size % gpu_block_size != 0:
        raise ValueError(
            f"offloaded block_size ({offloaded_block_size}) must be a multiple "
            f"of the GPU block size ({gpu_block_size})"
        )
    block_size_factor = offloaded_block_size // gpu_block_size

    # Total KV bytes one rank holds for one GPU block, summed over all layers.
    # Matches both canonicalization paths: per-layer tensors (n_layers x
    # page_size, FlashAttention additionally splitting each into K/V halves) and
    # the cross-layer single tensor (page_size x n_layers).
    per_rank_block_bytes = sum(
        g.kv_cache_spec.page_size_bytes * len(g.layer_names) for g in groups
    )
    if per_rank_block_bytes <= 0:
        raise ValueError("computed a non-positive KV block size")

    tp_slice_bytes = per_rank_block_bytes * block_size_factor
    slot_bytes = tp_slice_bytes * tp_size

    slot_align = int(extra.get("slot_align", DEFAULT_SLOT_ALIGN))
    if slot_align <= 0 or (slot_align & (slot_align - 1)) != 0:
        raise ValueError(f"slot_align must be a power of two, got {slot_align}")
    slot_stride = _align_up(slot_bytes, slot_align)

    # Node-wide budget shared by every DP rank: never divide by world_size.
    cpu_bytes_to_use = extra.get("cpu_bytes_to_use")
    if not cpu_bytes_to_use:
        raise ValueError(
            "cpu_bytes_to_use must be specified in kv_connector_extra_config"
        )
    num_slots = int(cpu_bytes_to_use) // slot_stride
    if num_slots <= 0:
        raise ValueError(
            f"cpu_bytes_to_use ({int(cpu_bytes_to_use)}) is smaller than one "
            f"slot ({slot_stride} bytes)"
        )

    return SlotGeometry(
        index_shm_name=str(extra.get("index_shm_name", DEFAULT_INDEX_SHM_NAME)),
        data_shm_name=str(extra.get("data_shm_name", DEFAULT_DATA_SHM_NAME)),
        hugepage_path=str(extra.get("hugepage_path", "")),
        gpu_block_size=gpu_block_size,
        offloaded_block_size=offloaded_block_size,
        block_size_factor=block_size_factor,
        per_rank_block_bytes=per_rank_block_bytes,
        tp_slice_bytes=tp_slice_bytes,
        slot_bytes=slot_bytes,
        slot_align=slot_align,
        slot_stride=slot_stride,
        tp_size=tp_size,
        num_slots=num_slots,
    )


@dataclass(frozen=True)
class TensorLayout:
    """Where each canonical tensor lives inside one TP slice."""

    page_bytes: tuple[int, ...]  # per canonical tensor, one GPU block
    offsets: tuple[int, ...]  # byte offset within the TP slice
    total_bytes: int


def tensor_layout(
    kv_caches: "CanonicalKVCaches", block_size_factor: int
) -> TensorLayout:
    """Lay canonical tensors out back-to-back inside a TP slice.

    Tensor t occupies ``page_bytes[t] * block_size_factor`` bytes, holding the
    ``block_size_factor`` GPU sub-blocks of one offloaded block contiguously.
    """
    page_bytes: list[int] = []
    offsets: list[int] = []
    cursor = 0
    for t in kv_caches.tensors:
        page_bytes.append(t.page_size_bytes)
        offsets.append(cursor)
        cursor += t.page_size_bytes * block_size_factor
    return TensorLayout(
        page_bytes=tuple(page_bytes), offsets=tuple(offsets), total_bytes=cursor
    )


def verify_against_canonical(
    geometry: SlotGeometry, kv_caches: "CanonicalKVCaches"
) -> TensorLayout:
    """Worker-side check that the real tensors match the published geometry."""
    layout = tensor_layout(kv_caches, geometry.block_size_factor)
    actual = sum(layout.page_bytes)
    if actual != geometry.per_rank_block_bytes:
        raise GeometryMismatch(
            "RadixShmem: the KV cache tensors on this worker do not match the "
            "geometry published by the DP rank 0 scheduler. Per-rank bytes per "
            f"GPU block: actual={actual}, published="
            f"{geometry.per_rank_block_bytes}. Canonical page sizes: "
            f"{layout.page_bytes}. This usually means an unsupported KV cache "
            "layout (e.g. layers sharing one physical tensor)."
        )
    if layout.total_bytes != geometry.tp_slice_bytes:
        raise GeometryMismatch(
            f"RadixShmem: TP slice size mismatch: laid out {layout.total_bytes} "
            f"bytes, geometry says {geometry.tp_slice_bytes}"
        )
    return layout
