# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Slot geometry for the RadixShmem shared KV cache.

Everything here is derivable from ``OffloadingConfig``, i.e. before any KV
tensor exists. That is what lets the owning scheduler create the shared regions
and every other process (other DP schedulers, all TP workers, other vLLM
instances on the node) attach to them and fail closed on any disagreement.

RadixShmem's SlotStore is one shared region split into up to three pools --
FULL, SWA, MAMBA -- each with its own slot size, matching the index's three
slot-id spaces. KV cache groups map onto pools by attention kind:

* full-attention groups share the FULL pool: one slot holds one chunk
  (``blocks_per_chunk`` GPU blocks) of every FULL group, for the whole TP group;
* windowed groups (sliding window, chunked local) share the SWA pool. Their
  chunks may be finer than FULL chunks (DeepSeek-V4: 64-token SWA blocks under
  128-token MLA blocks), so one SWA slot spans one FULL chunk of tokens, i.e.
  ``ratio`` consecutive SWA chunks, because the index hangs SWA slots on FULL
  path positions;
* recurrent groups (Mamba) share the MAMBA pool, one slot per chunk.

Inside a slot::

    slot_base   = pool_base + slot_id * slot_stride
    slice(w)    = slot_base + w * slice_bytes        w = tp_rank, or 0 if replicated
    tensor t    = slice(w) + tensor_offset[t]
    sub-block j = tensor t + j * page_bytes[t]      j in [0, sub_blocks)

so from one TP rank's point of view canonical tensor ``t`` of a pool is a strided
``(num_slots, page_bytes[t] * sub_blocks)`` int8 matrix with row stride
``slot_stride`` -- the CPU-side shape ``SingleDirectionOffloadingHandler`` in
``kv_offload/cpu`` expects, so the transfer path is upstream's, unchanged.
"""

import hashlib
from dataclasses import asdict, dataclass
from enum import IntEnum
from typing import TYPE_CHECKING, Any

from vllm.utils.math_utils import cdiv

if TYPE_CHECKING:
    from vllm.v1.kv_offload.config import OffloadingConfig

DEFAULT_INDEX_SHM_NAME = "/vllm_kv_index"
DEFAULT_DATA_SHM_NAME = "/vllm_kv_data"
DEFAULT_SLOT_ALIGN = 4096
DEFAULT_ATTACH_TIMEOUT_S = 300.0
DEFAULT_SENTINEL_DIR = "/dev/shm"


class PoolKind(IntEnum):
    """Slot pool / index component; values equal ``shmradix.ComponentType``."""

    FULL = 0
    SWA = 1
    MAMBA = 2

    @property
    def mask(self) -> int:
        return 1 << int(self)


class GeometryMismatch(RuntimeError):
    """Raised when two processes disagree about the shared-region geometry."""


def _align_up(value: int, align: int) -> int:
    return (value + align - 1) // align * align


@dataclass(frozen=True)
class GroupLayout:
    """How one KV cache group maps onto its pool."""

    group_idx: int
    kind: int  # PoolKind value
    tokens_per_chunk: int
    hashes_per_chunk: int
    # chunks of this group per FULL chunk (1 for FULL and MAMBA groups)
    ratio: int
    # KV bytes one rank holds per GPU block for this group's layers
    bytes_per_block: int
    # block-outermost layouts: (offset in the packed block, page bytes) per
    # layer; empty when the canonical tensors are already per layer
    layer_pages: tuple[tuple[int, int], ...] = ()


@dataclass(frozen=True)
class PoolSpec:
    kind: int  # PoolKind value
    num_slots: int
    # GPU blocks of one group spanned by one slot: blocks_per_chunk * ratio
    sub_blocks: int
    slice_bytes: int  # one writer's share of a slot
    num_slices: int  # tp_size, or 1 when the KV is replicated across TP ranks
    slot_bytes: int  # slice_bytes * num_slices (unpadded)
    slot_stride: int  # slot_bytes rounded up to slot_align


@dataclass(frozen=True)
class SlotGeometry:
    """Fully-resolved geometry of the shared index + SlotStore."""

    index_shm_name: str
    data_shm_name: str
    hugepage_path: str

    # tokens per FULL chunk; also the index's block_size
    tokens_per_chunk: int
    blocks_per_chunk: int
    # SWA window in FULL chunk positions, as the index counts it; 0 = no SWA
    swa_window_blocks: int

    tp_size: int
    replicated: bool
    slot_align: int

    groups: tuple[GroupLayout, ...]
    # present pools only, in PoolKind order
    pools: tuple[PoolSpec, ...]

    # model identity: a region holds one model's KV, and two models with the
    # same byte layout would otherwise attach to each other's cache
    model_fingerprint: str = ""

    # ----------------------------------------------------------- accessors

    def pool(self, kind: int) -> PoolSpec | None:
        for p in self.pools:
            if p.kind == int(kind):
                return p
        return None

    def require_pool(self, kind: int) -> PoolSpec:
        p = self.pool(kind)
        if p is None:
            raise GeometryMismatch(f"geometry has no {PoolKind(kind).name} pool")
        return p

    @property
    def pool_mask(self) -> int:
        return sum(1 << p.kind for p in self.pools)

    @property
    def num_slices(self) -> int:
        return 1 if self.replicated else self.tp_size

    @property
    def total_data_bytes(self) -> int:
        return sum(p.num_slots * p.slot_stride for p in self.pools)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SlotGeometry":
        d = dict(d)
        d["groups"] = tuple(
            GroupLayout(
                **{**g, "layer_pages": tuple(tuple(lp) for lp in g["layer_pages"])}
            )
            for g in d["groups"]
        )
        d["pools"] = tuple(PoolSpec(**p) for p in d["pools"])
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


_SLOT_OVERRIDE_KEYS = {
    PoolKind.FULL: "full_slots",
    PoolKind.SWA: "swa_slots",
    PoolKind.MAMBA: "mamba_slots",
}


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
    tokens_per_hash = config.cache.tokens_per_hash

    full_chunks = {
        g.tokens_per_block * blocks_per_chunk
        for g in config.groups
        if g.is_full_attention
    }
    if len(full_chunks) != 1:
        raise ValueError(
            "RadixShmem offloading needs exactly one full-attention chunk size to "
            f"anchor the radix paths on, got {sorted(full_chunks)}"
        )
    tokens_per_chunk = full_chunks.pop()

    replicated_raw = extra.get("replicated_kv")
    replicated = (
        bool(config.replicated_layout)
        if replicated_raw is None
        else _parse_bool(replicated_raw, "replicated_kv")
    )
    num_slices = 1 if replicated else parallel.tp_size

    slot_align = int(extra.get("slot_align", DEFAULT_SLOT_ALIGN))
    if slot_align <= 0 or (slot_align & (slot_align - 1)) != 0:
        raise ValueError(f"slot_align must be a power of two, got {slot_align}")

    groups: list[GroupLayout] = []
    swa_window_blocks = 0
    for idx, g in enumerate(config.groups):
        g_tokens_per_chunk = g.tokens_per_block * blocks_per_chunk
        if g.is_full_attention:
            kind, ratio = PoolKind.FULL, 1
        elif g.is_recurrent:
            kind, ratio = PoolKind.MAMBA, tokens_per_chunk // g_tokens_per_chunk
        else:
            kind, ratio = PoolKind.SWA, tokens_per_chunk // g_tokens_per_chunk
            if g.sliding_window_tokens is None:
                raise ValueError(f"windowed KV cache group {idx} has no window size")
            window_chunks = cdiv(g.sliding_window_tokens, g_tokens_per_chunk)
            swa_window_blocks = max(swa_window_blocks, cdiv(window_chunks, ratio))
        if ratio < 1 or ratio * g_tokens_per_chunk != tokens_per_chunk:
            raise ValueError(
                f"KV cache group {idx} chunk of {g_tokens_per_chunk} tokens does not "
                f"divide the full-attention chunk of {tokens_per_chunk} tokens"
            )
        if g.worker_kv_bytes_per_block <= 0:
            raise ValueError(
                f"KV cache group {idx} reports no KV bytes per block; the offloading "
                "config must carry worker_kv_bytes_per_block per group"
            )
        groups.append(
            GroupLayout(
                group_idx=idx,
                kind=int(kind),
                tokens_per_chunk=g_tokens_per_chunk,
                hashes_per_chunk=g_tokens_per_chunk // tokens_per_hash,
                ratio=ratio,
                bytes_per_block=g.worker_kv_bytes_per_block,
                layer_pages=tuple(tuple(lp) for lp in g.layer_pages),
            )
        )

    # slot sizes per pool
    pool_dims: dict[PoolKind, tuple[int, int]] = {}  # kind -> (sub_blocks, slice)
    for kind in PoolKind:
        members = [g for g in groups if g.kind == int(kind)]
        if not members:
            continue
        sub_blocks_set = {blocks_per_chunk * g.ratio for g in members}
        if len(sub_blocks_set) != 1:
            raise ValueError(
                f"{kind.name} groups have different chunk sizes "
                f"({sorted(sub_blocks_set)} GPU blocks per slot); not supported"
            )
        sub_blocks = sub_blocks_set.pop()
        pool_dims[kind] = (
            sub_blocks,
            sum(g.bytes_per_block for g in members) * sub_blocks,
        )

    # Node-wide budget shared by every DP rank and every instance attaching
    # this region: never divide by world_size. Pools without an explicit slot
    # count get as many slots as the FULL pool, which is cheap since their
    # slots are a small fraction of a FULL slot.
    cpu_bytes_to_use = extra.get("cpu_bytes_to_use")
    if not cpu_bytes_to_use:
        raise ValueError(
            "cpu_bytes_to_use must be specified in kv_connector_extra_config"
        )
    budget = int(cpu_bytes_to_use)
    strides = {
        kind: _align_up(slice_bytes * num_slices, slot_align)
        for kind, (_, slice_bytes) in pool_dims.items()
    }
    fixed: dict[PoolKind, int] = {}
    for kind in pool_dims:
        raw = extra.get(_SLOT_OVERRIDE_KEYS[kind])
        if raw is not None:
            fixed[kind] = int(raw)
            if fixed[kind] <= 0:
                raise ValueError(f"{_SLOT_OVERRIDE_KEYS[kind]} must be > 0")
    remaining = budget - sum(fixed[k] * strides[k] for k in fixed)
    per_full_slot = sum(strides[k] for k in pool_dims if k not in fixed)
    if PoolKind.FULL in fixed:
        num_full = fixed[PoolKind.FULL]
        derived = (
            (remaining - num_full * strides[PoolKind.FULL]) // per_full_slot
            if per_full_slot
            else num_full
        )
    else:
        num_full = remaining // per_full_slot if per_full_slot else 0
        derived = num_full
    if num_full <= 0 or derived <= 0:
        raise ValueError(
            f"cpu_bytes_to_use ({budget}) is too small for one slot of every pool "
            f"(strides {dict((k.name, v) for k, v in strides.items())})"
        )

    pools: list[PoolSpec] = []
    for kind in PoolKind:
        if kind not in pool_dims:
            continue
        sub_blocks, slice_bytes = pool_dims[kind]
        if kind in fixed:
            num_slots = fixed[kind]
        elif kind == PoolKind.FULL:
            num_slots = num_full
        else:
            num_slots = derived
        pools.append(
            PoolSpec(
                kind=int(kind),
                num_slots=num_slots,
                sub_blocks=sub_blocks,
                slice_bytes=slice_bytes,
                num_slices=num_slices,
                slot_bytes=slice_bytes * num_slices,
                slot_stride=strides[kind],
            )
        )

    fingerprint = hashlib.sha256(
        f"{config.model.name}|{config.model.dtype}|{config.kv_cache_layout}".encode()
    ).hexdigest()[:16]

    return SlotGeometry(
        index_shm_name=str(extra.get("index_shm_name", DEFAULT_INDEX_SHM_NAME)),
        data_shm_name=str(extra.get("data_shm_name", DEFAULT_DATA_SHM_NAME)),
        hugepage_path=str(extra.get("hugepage_path", "")),
        tokens_per_chunk=tokens_per_chunk,
        blocks_per_chunk=blocks_per_chunk,
        swa_window_blocks=swa_window_blocks,
        tp_size=parallel.tp_size,
        replicated=replicated,
        slot_align=slot_align,
        groups=tuple(groups),
        pools=tuple(pools),
        model_fingerprint=fingerprint,
    )


@dataclass(frozen=True)
class TensorLayout:
    """Where each canonical tensor of a pool lives inside one slice."""

    page_bytes: tuple[int, ...]  # per tensor, one GPU block
    offsets: tuple[int, ...]  # byte offset within the slice
    total_bytes: int


def tensor_layout(page_bytes: list[int], sub_blocks: int) -> TensorLayout:
    """Lay a pool's canonical tensors out back-to-back inside a slice.

    Tensor t occupies ``page_bytes[t] * sub_blocks`` bytes, holding the
    ``sub_blocks`` GPU sub-blocks of one slot contiguously.
    """
    offsets: list[int] = []
    cursor = 0
    for page in page_bytes:
        offsets.append(cursor)
        cursor += page * sub_blocks
    return TensorLayout(
        page_bytes=tuple(page_bytes), offsets=tuple(offsets), total_bytes=cursor
    )


def verify_pool_layout(pool: PoolSpec, layout: TensorLayout) -> None:
    """Worker-side check that the real tensors fit the published pool.

    The pool was sized from per-group ``worker_kv_bytes_per_block``.
    Canonicalization may dedupe layers that alias the same bytes, so the real
    tensors can be smaller, but never larger.
    """
    if layout.total_bytes > pool.slice_bytes:
        raise GeometryMismatch(
            f"RadixShmem: the {PoolKind(pool.kind).name} KV cache tensors on this "
            "worker do not fit the geometry published by the region owner. Bytes "
            f"per slot per rank: actual={layout.total_bytes}, "
            f"published={pool.slice_bytes}. Page sizes: {layout.page_bytes}."
        )
