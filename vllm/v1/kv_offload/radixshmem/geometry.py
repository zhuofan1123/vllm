# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Slot geometry for the RadixShmem shared KV cache.

Everything here is derivable from ``OffloadingConfig``, i.e. before any KV
tensor exists. That is what lets the owning scheduler create the shared regions
and every other process (other DP schedulers, all TP workers, other vLLM
instances on the node) attach to them and fail closed on any disagreement.

One slot holds one offloaded chunk (``blocks_per_chunk`` GPU blocks) of one KV
cache group, for the whole TP group::

    slot_base   = data_base + slot_id * slot_stride
    slice(w)    = slot_base + w * slice_bytes        w = tp_rank, or 0 if replicated
    tensor t    = slice(w) + tensor_offset[t]
    sub-block j = tensor t + j * page_bytes[t]      j in [0, blocks_per_chunk)

so from a single TP rank's point of view, canonical tensor ``t`` is a strided
``(num_slots, page_bytes[t] * blocks_per_chunk)`` int8 matrix with row stride
``slot_stride`` -- exactly the CPU-side shape ``SingleDirectionOffloadingHandler``
in ``kv_offload/cpu`` expects, so the transfer path is upstream's, unchanged.

Padding (``slot_stride - slot_bytes``) only ever lands at the tail of a slot.
"""

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vllm.v1.kv_offload.base import CanonicalKVCaches
    from vllm.v1.kv_offload.config import OffloadingConfig

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
    tokens_per_chunk: int  # of the first KV cache group; the index's block_size
    blocks_per_chunk: int

    # bytes
    worker_bytes_per_block: int  # one GPU block, one TP rank, all groups/layers
    slice_bytes: int  # worker_bytes_per_block * blocks_per_chunk
    num_slices: int  # tp_size, or 1 when the KV is replicated across TP ranks
    slot_bytes: int  # num_slices * slice_bytes (unpadded)
    slot_align: int
    slot_stride: int  # slot_bytes rounded up to slot_align

    # topology
    tp_size: int
    replicated: bool
    num_slots: int

    @property
    def total_data_bytes(self) -> int:
        return self.num_slots * self.slot_stride

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SlotGeometry":
        return cls(**d)

    def check_same(self, other: "SlotGeometry", *, what: str) -> None:
        """Fail closed if any field differs from what ``what`` published."""
        mine = self.to_dict()
        theirs = other.to_dict()
        diffs = [
            f"{k}: mine={mine[k]!r} theirs={theirs.get(k)!r}"
            for k in mine
            if mine[k] != theirs.get(k)
        ]
        if diffs:
            raise GeometryMismatch(
                f"RadixShmem geometry disagrees with {what}: " + "; ".join(diffs)
            )


def compute_geometry(config: "OffloadingConfig") -> SlotGeometry:
    """Derive the shared-region geometry from the normalized offloading config.

    Every process attaching the same region must compute the same geometry, so
    this reads nothing process-specific (no rank, no device).
    """
    extra: dict[str, Any] = dict(config.extra_config or {})
    parallel = config.parallel

    if parallel.pp_size != 1:
        raise ValueError(
            f"RadixShmem offloading requires pipeline_parallel_size == 1 "
            f"(got {parallel.pp_size})"
        )
    if parallel.world_size != parallel.tp_size:
        raise ValueError(
            "RadixShmem offloading expects world_size == tensor_parallel_size "
            f"(got world_size={parallel.world_size}, tp={parallel.tp_size}); "
            "context parallelism is not supported"
        )
    if not config.groups:
        raise ValueError("RadixShmem offloading requires at least one KV cache group")

    blocks_per_chunk = config.cache.blocks_per_chunk
    tokens_per_chunk = config.groups[0].tokens_per_block * blocks_per_chunk

    worker_bytes_per_block = config.worker_kv_bytes_per_block
    if worker_bytes_per_block <= 0:
        raise ValueError(
            "RadixShmem offloading needs worker_kv_bytes_per_block > 0 "
            "(the KV cache config has no tensors?)"
        )
    slice_bytes = worker_bytes_per_block * blocks_per_chunk

    replicated_raw = extra.get("replicated_kv")
    replicated = (
        bool(config.replicated_layout)
        if replicated_raw is None
        else _parse_bool(replicated_raw, "replicated_kv")
    )
    num_slices = 1 if replicated else parallel.tp_size
    slot_bytes = slice_bytes * num_slices

    slot_align = int(extra.get("slot_align", DEFAULT_SLOT_ALIGN))
    if slot_align <= 0 or (slot_align & (slot_align - 1)) != 0:
        raise ValueError(f"slot_align must be a power of two, got {slot_align}")
    slot_stride = _align_up(slot_bytes, slot_align)

    # Node-wide budget shared by every DP rank and every instance attaching
    # this region: never divide by world_size.
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
        tokens_per_chunk=tokens_per_chunk,
        blocks_per_chunk=blocks_per_chunk,
        worker_bytes_per_block=worker_bytes_per_block,
        slice_bytes=slice_bytes,
        num_slices=num_slices,
        slot_bytes=slot_bytes,
        slot_align=slot_align,
        slot_stride=slot_stride,
        tp_size=parallel.tp_size,
        replicated=replicated,
        num_slots=num_slots,
    )


def _parse_bool(value: Any, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("1", "true", "yes", "on"):
            return True
        if lowered in ("0", "false", "no", "off"):
            return False
    if isinstance(value, int):
        return bool(value)
    raise ValueError(f"{name} must be a boolean, got {value!r}")


@dataclass(frozen=True)
class TensorLayout:
    """Where each canonical tensor lives inside one TP slice."""

    page_bytes: tuple[int, ...]  # per canonical tensor, one GPU block
    offsets: tuple[int, ...]  # byte offset within the slice
    total_bytes: int


def tensor_layout(
    kv_caches: "CanonicalKVCaches", blocks_per_chunk: int
) -> TensorLayout:
    """Lay canonical tensors out back-to-back inside a slice.

    Tensor t occupies ``page_bytes[t] * blocks_per_chunk`` bytes, holding the
    ``blocks_per_chunk`` GPU sub-blocks of one chunk contiguously.
    """
    page_bytes: list[int] = []
    offsets: list[int] = []
    cursor = 0
    for t in kv_caches.tensors:
        page_bytes.append(t.page_size_bytes)
        offsets.append(cursor)
        cursor += t.page_size_bytes * blocks_per_chunk
    return TensorLayout(
        page_bytes=tuple(page_bytes), offsets=tuple(offsets), total_bytes=cursor
    )


def verify_against_canonical(
    geometry: SlotGeometry, kv_caches: "CanonicalKVCaches"
) -> TensorLayout:
    """Worker-side check that the real tensors fit the published geometry.

    The geometry was sized from ``worker_kv_bytes_per_block`` (the whole GPU
    KV allocation divided by the block count). Canonicalization may dedupe
    layers that alias the same bytes, so the real tensors can be smaller, but
    never larger.
    """
    layout = tensor_layout(kv_caches, geometry.blocks_per_chunk)
    if layout.total_bytes > geometry.slice_bytes:
        raise GeometryMismatch(
            "RadixShmem: the KV cache tensors on this worker do not fit the "
            "geometry published by the region owner. Bytes per chunk per rank: "
            f"actual={layout.total_bytes}, published={geometry.slice_bytes}. "
            f"Canonical page sizes: {layout.page_bytes}."
        )
    return layout
