# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scheduler side of RadixShmem offloading: the shared radix index as an
``OffloadingManager``.

Two layers:

* ``RadixIndex`` is a thin, testable ledger over one attached ``RadixClient``.
  It owns the rules every caller depends on: RadixShmem is the only slot
  allocator (and ``allocate`` is the only point where eviction happens), a hit
  is pinned by ``query(lock=True)`` and released exactly once through a
  ``LookupLease``, and ``insert`` is the only publish point -- strictly after
  the bytes landed, so the shared tree never has an unready entry.

* ``RadixShmemOffloadingManager`` adapts that to upstream's per-key
  ``OffloadingManager`` contract. The connector scheduler asks about one
  ``OffloadKey`` (block hash + KV group) at a time, while a radix tree only
  matches root-anchored prefixes, so the manager rebuilds each request's path
  from ``ReqContext.block_hashes`` and answers a whole step's worth of lookups
  from one pinned query.

The request's path is its sequence of full-attention chunk hashes; every KV
cache group maps onto positions of that path (see ``geometry.py``):

* FULL groups share one FULL slot per position (the slot holds all of their
  tensors), published as a prefix run with ``insert(component=FULL)``;
* SWA groups share one SWA slot per position, each holding the ``ratio`` SWA
  chunks that fall into that position's token span, published with
  ``insert(component=SWA)`` once every sub-chunk landed;
* MAMBA groups get one MAMBA slot per position, ``insert(component=MAMBA)``.

SWA and MAMBA publishes need the FULL path below them, so a completed store
publishes FULL first; anything the index rejects for a missing path is retried
on later completions.
"""

import time
from collections import defaultdict
from collections.abc import Collection, Iterable
from dataclasses import dataclass, field

import numpy as np

from vllm.distributed.kv_transfer.kv_connector.v1.offloading.metrics import (
    OffloadingConnectorStats,
)
from vllm.logger import init_logger
from vllm.v1.kv_offload.base import (
    LoadStoreSpec,
    LookupResult,
    Medium,
    OffloadingEvent,
    OffloadingManager,
    OffloadKey,
    PrepareStoreOutput,
    ReqContext,
    RequestOffloadingContext,
    ScheduleEndContext,
    get_offload_group_idx,
    make_offload_key,
)
from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec

from .geometry import GroupLayout, PoolKind, SlotGeometry

logger = init_logger(__name__)

# a FULL_PATH_MISSING publish is retried on this many later completions
_MAX_PUBLISH_RETRIES = 64


class RadixShmemMetrics:
    """Metric names emitted by ``RadixShmemOffloadingManager.get_stats``."""

    SLOTS_USED = "vllm:kv_offload_radixshmem_slots_used"
    SLOTS_TOTAL = "vllm:kv_offload_radixshmem_slots_total"
    OPEN_LEASES = "vllm:kv_offload_radixshmem_open_leases"
    HIT_BLOCKS = "vllm:kv_offload_radixshmem_hit_blocks"
    PUBLISHED_BLOCKS = "vllm:kv_offload_radixshmem_published_blocks"
    PUBLISH_REJECTED = "vllm:kv_offload_radixshmem_publish_rejected"
    ALLOC_FAILURES = "vllm:kv_offload_radixshmem_alloc_failures"
    INDEX_TIME = "vllm:kv_offload_radixshmem_index_time"


def to_u64(block_hashes: Iterable[bytes]) -> np.ndarray:
    """Reduce vLLM ``BlockHash`` bytes to the uint64 keys RadixShmem indexes on.

    vLLM's hash already folds in the parent hash, LoRA id, multimodal keys and
    cache salt, so truncation keeps the prefix semantics; it only shrinks the
    collision space. Every process attaching the same region must use exactly
    this function.
    """
    hashes = list(block_hashes)
    n = len(hashes)
    if n == 0:
        return np.empty(0, dtype=np.uint64)
    buf = bytearray(8 * n)
    for i, h in enumerate(hashes):
        buf[8 * i : 8 * i + 8] = h[:8]
    return np.ascontiguousarray(np.frombuffer(bytes(buf), dtype=np.uint64))


def _component(kind: int):
    import shmradix

    return shmradix.ComponentType(int(kind))


class LookupLease:
    """A pinned prefix hit. FULL and the auxiliary components (SWA / MAMBA) are
    queried separately -- a full-attention prefix stays a hit even where the
    windowed groups only kept a trailing window -- so ``hit_blocks`` is the FULL
    depth and ``swa_end`` may run deeper. Both queries' pins are released once.
    """

    __slots__ = (
        "hit_blocks",
        "slots",
        "swa_start",
        "swa_end",
        "swa_slots",
        "mamba_position",
        "mamba_slot",
        "_finalizes",
        "_owner",
        "_released",
    )

    def __init__(
        self,
        hit_blocks: int,
        slots: np.ndarray,
        swa_start: int,
        swa_end: int,
        swa_slots: np.ndarray,
        mamba_position: int,
        mamba_slot: int | None,
        finalizes: tuple,
        owner,
    ):
        self.hit_blocks = hit_blocks
        self.slots = slots
        self.swa_start = swa_start
        self.swa_end = swa_end
        self.swa_slots = swa_slots
        self.mamba_position = mamba_position
        self.mamba_slot = mamba_slot
        self._finalizes = finalizes
        self._owner = owner
        self._released = False

    @property
    def released(self) -> bool:
        return self._released

    def swa_slot(self, position: int) -> int | None:
        if self.swa_start <= position < self.swa_end and self.swa_slots.size:
            return int(self.swa_slots[position - self.swa_start])
        return None

    def release(self) -> None:
        """Drop the pins. Idempotent -- each ``finalize`` is one-shot underneath."""
        if self._released:
            return
        self._released = True
        try:
            for finalize in self._finalizes:
                finalize()
        finally:
            self._owner._forget(self)

    def __repr__(self) -> str:
        state = "released" if self._released else "held"
        return f"LookupLease({self.hit_blocks} FULL blocks, {state})"


class RadixIndex:
    """Ledger over one attached ``RadixClient`` (one per scheduler process)."""

    def __init__(self, client, *, mask: int = PoolKind.FULL.mask):
        self.client = client
        # FULL is queried on its own for the prefix depth; the windowed
        # components (SWA / MAMBA) are queried together, if present, so their
        # partial coverage never shortens the FULL hit
        self.full_mask = PoolKind.FULL.mask
        self.aux_mask = mask & ~PoolKind.FULL.mask
        self._leases: set[LookupLease] = set()
        # counters, for logging and the connector's stats
        self.num_lookups = 0
        self.num_hit_blocks = 0
        self.num_allocated = 0
        self.num_recycled = 0
        self.num_published = 0
        self.num_publish_rejected = 0
        self.num_alloc_failures = 0
        # wall time inside the shared index, split by op: with several
        # schedulers on one region this is where lock contention would show
        self.time_query_s = 0.0
        self.time_alloc_s = 0.0
        self.time_insert_s = 0.0
        self.num_queries = 0
        self.num_allocs = 0
        self.num_inserts = 0

    # ------------------------------------------------------------ lookup

    @staticmethod
    def _hit_blocks(result) -> int:
        """``common_hit`` of a query, or 0 when the request failed."""
        status = getattr(result, "status", None)
        if status is not None and int(status) != 0:
            return 0
        return int(result.common_hit)

    def lookup(
        self, hashes: np.ndarray, *, touch: bool = False, mask: int | None = None
    ) -> int:
        """Blocks of ``hashes`` (root-anchored) already in the shared cache.

        ``touch`` refreshes the LRU position of the hit; a plain probe does not.
        """
        if hashes.size == 0:
            return 0
        t0 = time.perf_counter()
        r = self.client.query(
            hashes,
            mask=self.full_mask if mask is None else mask,
            local_only=True,
            lock=False,
            update_meta=touch,
        )
        self.time_query_s += time.perf_counter() - t0
        self.num_queries += 1
        return self._hit_blocks(r)

    def lookup_and_pin(self, hashes: np.ndarray) -> LookupLease | None:
        """Match from the root and pin the hit in one shot.

        Returns ``None`` on a zero-block hit -- there is nothing to release in
        that case, and handing back a lease nobody has to free is a leak
        waiting to happen.
        """
        self.num_lookups += 1
        if hashes.size == 0:
            return None
        finalizes: list = []
        t0 = time.perf_counter()
        r = self.client.query(
            hashes, mask=self.full_mask, local_only=True, lock=True, update_meta=True
        )
        self.time_query_s += time.perf_counter() - t0
        self.num_queries += 1
        hit = self._hit_blocks(r)
        if hit == 0:
            # lock=True with a zero hit takes no ref, but finalize is still a
            # valid one-shot; call it so nothing is left dangling.
            r.finalize()
            return None
        finalizes.append(r.finalize)
        # Full slots come back as (source_rank, offset, slot_ids) runs tiling
        # [0, common_hit) in offset order; local_only gives at most one run.
        runs = [np.asarray(ids, dtype=np.int32) for _, _, ids in r.full_fragments]
        slots = np.concatenate(runs) if len(runs) > 1 else runs[0]
        if slots.size < hit:
            r.finalize()
            raise RuntimeError(
                f"RadixShmem: query hit {hit} blocks but returned "
                f"{slots.size} Full slots"
            )

        swa_start = swa_end = hit
        swa_slots = np.empty(0, dtype=np.int32)
        mamba_position, mamba_slot = -1, None
        if self.aux_mask:
            t0 = time.perf_counter()
            aux = self.client.query(
                hashes,
                mask=self.aux_mask,
                local_only=True,
                lock=True,
                update_meta=True,
            )
            self.time_query_s += time.perf_counter() - t0
            self.num_queries += 1
            aux_hit = self._hit_blocks(aux)
            got = np.asarray(getattr(aux, "swa_slots", ()), dtype=np.int32)
            if aux_hit > 0 and got.size:
                swa_slots = got
                swa_end = aux_hit
                swa_start = aux_hit - got.size
            mamba = getattr(aux, "mamba", None)
            if aux_hit > 0 and mamba is not None:
                mamba_position, mamba_slot = aux_hit - 1, int(mamba[1])
            finalizes.append(aux.finalize)

        lease = LookupLease(
            hit,
            np.ascontiguousarray(slots[:hit], dtype=np.int32),
            swa_start,
            swa_end,
            swa_slots,
            mamba_position,
            mamba_slot,
            tuple(finalizes),
            self,
        )
        self.num_hit_blocks += hit
        self._leases.add(lease)
        return lease

    def _forget(self, lease: LookupLease) -> None:
        self._leases.discard(lease)

    @property
    def num_open_leases(self) -> int:
        return len(self._leases)

    # -------------------------------------------------------- allocation

    def allocate(self, n: int, kind: int = PoolKind.FULL) -> np.ndarray | None:
        """Reserve ``n`` slots of a pool, evicting unpinned LRU entries as needed.

        Returns ``None`` when the pool cannot satisfy the request -- every
        candidate is either pinned or freshly published. The caller must skip
        the store, not retry in a loop.
        """
        if n <= 0:
            return np.empty(0, dtype=np.int32)
        t0 = time.perf_counter()
        slots = self.client.allocate_slots(n, component=_component(kind))
        self.time_alloc_s += time.perf_counter() - t0
        self.num_allocs += 1
        got = int(slots.size)
        if got < n:
            self.num_alloc_failures += 1
            if got:
                self.client.recycle_slots(slots, component=_component(kind))
            return None
        self.num_allocated += got
        return np.ascontiguousarray(slots, dtype=np.int32)

    def recycle(self, slots: np.ndarray, kind: int = PoolKind.FULL) -> None:
        if slots is None or slots.size == 0:
            return
        self.client.recycle_slots(
            np.ascontiguousarray(slots, dtype=np.int32), component=_component(kind)
        )
        self.num_recycled += int(slots.size)

    # ----------------------------------------------------------- publish

    def publish(
        self, hashes: np.ndarray, slots: np.ndarray, start: int
    ) -> tuple[int, np.ndarray]:
        """Register FULL ``slots`` for ``hashes[start:]`` after the data landed.

        ``hashes`` is the full root-anchored prefix; ``slots`` covers the tail
        beginning at block ``start``. ``insert`` has three outcomes and all of
        them are normal here, because another scheduler may have published the
        same prefix while this store was in flight:

        * ``matched < start`` -- the prefix this store sat on top of was
          evicted underneath us. Nothing is published; every slot comes back.
        * ``matched == start`` -- the expected case.
        * ``matched > start`` -- someone else got there first. The leading
          ``matched - start`` slots come back and the rest is published.

        Returns ``(blocks_published, unused_slots)``. ``auto_recycle`` is left
        on, so the unused slots are already back in the pool; they are returned
        only so the caller can account for them.
        """
        return self._insert(hashes, slots, start, PoolKind.FULL)

    def publish_component(
        self, hashes: np.ndarray, slots: np.ndarray, kind: int
    ) -> tuple[int, str | None]:
        """Publish SWA/MAMBA ``slots`` at the end of ``hashes``.

        SWA takes the trailing ``min(W, n)`` positions' slots, MAMBA exactly one.
        Returns ``(slots_published, error_name)``; ``error_name`` is set when the
        index rejected the whole publish (e.g. ``FULL_PATH_MISSING``).
        """
        published, unused, error = self._insert_raw(hashes, slots, 0, kind)
        if published <= 0 and error is not None and int(error) != 0:
            return 0, getattr(error, "name", str(error))
        return published, None

    def _insert(
        self, hashes: np.ndarray, slots: np.ndarray, start: int, kind: int
    ) -> tuple[int, np.ndarray]:
        published, unused, error = self._insert_raw(hashes, slots, start, kind)
        if published <= 0 and error is not None and int(error) != 0:
            logger.warning_once(
                "RadixShmem insert failed (%s); the shared index may be full. "
                "Raise max_nodes / data_pool_ratio if this persists.",
                error,
            )
        return max(published, 0), unused

    def _insert_raw(self, hashes, slots, start, kind):
        if slots.size == 0:
            return 0, np.empty(0, dtype=np.int32), None
        hashes = np.ascontiguousarray(hashes, dtype=np.uint64)
        slots = np.ascontiguousarray(slots, dtype=np.int32)
        t0 = time.perf_counter()
        res = self.client.insert(
            hashes, slots, start=start, auto_recycle=True, component=_component(kind)
        )
        self.time_insert_s += time.perf_counter() - t0
        self.num_inserts += 1
        unused = res.unused_slots
        published = int(slots.size) - int(unused.size)
        if published <= 0:
            self.num_publish_rejected += 1
        else:
            self.num_published += published
        return published, unused, res.error

    # ------------------------------------------------------------- misc

    def stats(self) -> dict[str, int]:
        return {
            "num_slots": int(self.client.mempool_total()),
            "used_slots": int(self.client.mempool_used()),
            "free_slots": int(self.client.mempool_free()),
            "open_leases": len(self._leases),
            "lookups": self.num_lookups,
            "hit_blocks": self.num_hit_blocks,
            "allocated": self.num_allocated,
            "recycled": self.num_recycled,
            "published": self.num_published,
            "publish_rejected": self.num_publish_rejected,
            "alloc_failures": self.num_alloc_failures,
            "queries": self.num_queries,
            "allocs": self.num_allocs,
            "inserts": self.num_inserts,
            "query_ms": round(self.time_query_s * 1e3, 1),
            "alloc_ms": round(self.time_alloc_s * 1e3, 1),
            "insert_ms": round(self.time_insert_s * 1e3, 1),
            "index_s": self.time_query_s + self.time_alloc_s + self.time_insert_s,
        }

    def close(self) -> None:
        """Release every outstanding pin, then drop the client.

        Leaking a ref here would pin those blocks in the shared tree for the
        lifetime of the *node*, not the process -- nothing else can ever
        decrement them.
        """
        if self._leases:
            logger.warning(
                "RadixShmem: releasing %d lookup lease(s) still open at shutdown",
                len(self._leases),
            )
        for lease in list(self._leases):
            try:
                lease.release()
            except Exception:
                logger.exception("RadixShmem: failed to release a lookup lease")
        self._leases.clear()
        self.client = None


# ------------------------------------------------------------------ manager


@dataclass
class _ReqState:
    """Per-request scratch, kept on the ReqContext and in ``_states``."""

    # root-anchored u64 path over the FULL chunks known so far (immutable
    # arrays; extending builds a new one, so older references stay valid)
    path: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.uint64))
    # per group: how many block hashes have been folded into `chunk_idx`
    indexed_hashes: dict[int, int] = field(default_factory=dict)
    # key -> chunk index within its group
    chunk_idx: dict[OffloadKey, int] = field(default_factory=dict)
    # this step's pinned lookup, None once handed to a load or when nothing hit
    lease: LookupLease | None = None
    looked_up: bool = False
    # pins that back an in-flight load; released in complete_load
    load_leases: list[LookupLease] = field(default_factory=list)

    def release_round(self) -> None:
        if self.lease is not None:
            self.lease.release()
        self.lease = None
        self.looked_up = False

    def release_all(self) -> None:
        self.release_round()
        for lease in self.load_leases:
            lease.release()
        self.load_leases.clear()


@dataclass
class _PendingSlot:
    """A slot reserved by prepare_store, published once every part landed."""

    kind: int
    position: int
    slot: int
    path: np.ndarray  # root-anchored path with len >= position + 1
    # (group, sub-chunk) parts still to be written into the slot
    # (group, sub-chunk) parts the connector offered for this slot and has not
    # yet confirmed stored. Populated from the actual store keys, not a fixed
    # theoretical set: a windowed group only offers its reachable window, so
    # requiring every sub-chunk would never complete (DeepSeek-V4).
    parts_left: set[tuple[int, int]] = field(default_factory=set)
    keys: list[OffloadKey] = field(default_factory=list)
    landed_any: bool = False
    retries: int = 0


class RadixShmemOffloadingManager(OffloadingManager):
    """``OffloadingManager`` over the node-shared RadixShmem index."""

    def __init__(
        self,
        regions,
        geometry: SlotGeometry,
        *,
        enable_events: bool = False,
    ):
        self.regions = regions
        self.geometry = geometry
        self.index = RadixIndex(regions.client, mask=geometry.pool_mask)
        self.groups: dict[int, GroupLayout] = {g.group_idx: g for g in geometry.groups}
        full = [g for g in geometry.groups if g.kind == PoolKind.FULL]
        self._full_groups = frozenset(g.group_idx for g in full)
        self._full_hashes_per_chunk = full[0].hashes_per_chunk
        self._states: dict[str, _ReqState] = {}
        # requests that pinned something this step and may need a sweep
        self._dirty: set[str] = set()
        # (request, kind, position) -> reserved slot awaiting publish
        self._pending: dict[tuple[str, int, int], _PendingSlot] = {}
        # complete SWA/MAMBA slots the index rejected for a missing FULL path
        self._retry: list[_PendingSlot] = []
        self._events: list[OffloadingEvent] | None = [] if enable_events else None
        self._stats_snapshot = self.index.stats()

    # --------------------------------------------------------- state

    def _state(self, req_context: ReqContext) -> _ReqState:
        state = req_context.get_state(_ReqState)
        if state is None:
            state = _ReqState()
            req_context.set_state(state)
            self._states[req_context.req_id] = state
        return state

    def _index_keys(self, state: _ReqState, req_context: ReqContext) -> None:
        """Fold any new block hashes of the request into the path and key map."""
        block_hashes = req_context.block_hashes
        if block_hashes is None:
            raise RuntimeError(
                "RadixShmem offloading needs ReqContext.block_hashes; the "
                "offloading connector scheduler did not provide them"
            )
        num_hashes = len(block_hashes)
        for group in self.groups.values():
            hpc = group.hashes_per_chunk
            done = state.indexed_hashes.get(group.group_idx, 0)
            num_chunks = num_hashes // hpc
            first_chunk = done // hpc
            if num_chunks <= first_chunk:
                continue
            for i in range(first_chunk, num_chunks):
                key = make_offload_key(block_hashes[(i + 1) * hpc - 1], group.group_idx)
                state.chunk_idx[key] = i
            state.indexed_hashes[group.group_idx] = num_chunks * hpc
        hpc = self._full_hashes_per_chunk
        num_positions = num_hashes // hpc
        if num_positions > state.path.size:
            new_hashes = [
                block_hashes[(p + 1) * hpc - 1]
                for p in range(state.path.size, num_positions)
            ]
            state.path = np.concatenate([state.path, to_u64(new_hashes)])

    def _locate(
        self, state: _ReqState, req_context: ReqContext, key: OffloadKey
    ) -> tuple[GroupLayout, int, int] | None:
        """(group, position on the path, sub-chunk) of ``key``, or None."""
        idx = state.chunk_idx.get(key)
        if idx is None:
            self._index_keys(state, req_context)
            idx = state.chunk_idx.get(key)
            if idx is None:
                return None
        group = self.groups[get_offload_group_idx(key)]
        return group, idx // group.ratio, idx % group.ratio

    # ------------------------------------------------------ OffloadingManager

    def on_new_request(self, req_context: ReqContext) -> RequestOffloadingContext:
        self._state(req_context)
        return RequestOffloadingContext()

    def _round_lease(self, state: _ReqState) -> LookupLease | None:
        if not state.looked_up:
            state.lease = self.index.lookup_and_pin(state.path)
            state.looked_up = True
        return state.lease

    def lookup(self, key: OffloadKey, req_context: ReqContext) -> LookupResult:
        state = self._state(req_context)
        located = self._locate(state, req_context, key)
        if located is None:
            # not derived from this request's hashes (e.g. a boundary key of a
            # partial recurrent tail) -- never offloaded by this manager
            return LookupResult.MISS
        group, position, _ = located
        self._dirty.add(req_context.req_id)
        lease = self._round_lease(state)
        if lease is None:
            return LookupResult.MISS
        if group.kind == PoolKind.FULL:
            return (
                LookupResult.HIT if position < lease.hit_blocks else LookupResult.MISS
            )
        if group.kind == PoolKind.SWA:
            # a windowed group hits on its own window, independent of the FULL
            # prefix depth (the connector loads only the window)
            return (
                LookupResult.HIT
                if lease.swa_slot(position) is not None
                else LookupResult.MISS
            )
        # MAMBA: only the checkpoint at its own hit boundary is usable
        return (
            LookupResult.HIT
            if lease.mamba_slot is not None and position == lease.mamba_position
            else LookupResult.MISS
        )

    def touch(self, keys: Collection[OffloadKey], req_context: ReqContext) -> None:
        """Refresh the LRU position of the request's path."""
        if not keys:
            return
        state = self._state(req_context)
        deepest = -1
        for key in keys:
            located = self._locate(state, req_context, key)
            if located is not None:
                deepest = max(deepest, located[1])
        if deepest < 0:
            return
        lease = state.lease if state.looked_up else None
        if lease is not None and lease.hit_blocks >= deepest + 1:
            return  # this step's pinned query already refreshed it
        self.index.lookup(state.path[: deepest + 1], touch=True)

    def prepare_load(
        self, keys: Collection[OffloadKey], req_context: ReqContext
    ) -> LoadStoreSpec:
        state = self._state(req_context)
        lease = state.lease if state.looked_up else None
        if lease is None:
            raise RuntimeError(
                "RadixShmem: load requested without a hit from this step's lookup"
            )
        slots: list[int] = []
        last: tuple[int, int] | None = None  # (group, position) just emitted
        for key in keys:
            located = self._locate(state, req_context, key)
            assert located is not None, f"load of a key foreign to the request: {key!r}"
            group, position, _ = located
            if group.kind == PoolKind.FULL:
                if position >= lease.hit_blocks:
                    raise RuntimeError(
                        f"RadixShmem: FULL load of position {position} beyond the "
                        f"hit ({lease.hit_blocks})"
                    )
                slot = int(lease.slots[position])
            elif group.kind == PoolKind.SWA:
                # one slot per position; the connector lists every sub-chunk key
                if last == (group.group_idx, position):
                    continue
                swa = lease.swa_slot(position)
                if swa is None:
                    raise RuntimeError(
                        f"RadixShmem: SWA position {position} was not a hit"
                    )
                slot = swa
            else:
                if lease.mamba_slot is None or position != lease.mamba_position:
                    raise RuntimeError("RadixShmem: MAMBA checkpoint was not a hit")
                slot = lease.mamba_slot
            slots.append(slot)
            last = (group.group_idx, position)
        # the pin moves to the load; the sweep must not release it
        state.load_leases.append(lease)
        state.lease = None
        return CPULoadStoreSpec(slots)

    def complete_load(
        self, keys: Collection[OffloadKey], req_context: ReqContext
    ) -> None:
        state = self._state(req_context)
        for lease in state.load_leases:
            lease.release()
        state.load_leases.clear()

    def prepare_store(
        self, keys: Collection[OffloadKey], req_context: ReqContext
    ) -> PrepareStoreOutput | None:
        state = self._state(req_context)
        req_id = req_context.req_id
        located_keys: list[tuple[OffloadKey, GroupLayout, int, int]] = []
        for key in keys:
            located = self._locate(state, req_context, key)
            if located is not None:
                located_keys.append((key, *located))
        if not located_keys:
            return PrepareStoreOutput([], CPULoadStoreSpec([]), [])

        # FULL positions already published by anyone are skipped; the index
        # only accepts FULL publishes as prefix extensions, so this is a probe
        # of the path up to the deepest FULL key
        full_positions = [p for _, g, p, _ in located_keys if g.kind == PoolKind.FULL]
        already = (
            self.index.lookup(
                state.path[: max(full_positions) + 1], mask=PoolKind.FULL.mask
            )
            if full_positions
            else 0
        )

        keys_to_store: list[OffloadKey] = []
        slots_out: list[int] = []
        allocation_failed = False
        last: tuple[int, int] | None = None
        for key, group, position, sub in located_keys:
            if group.kind == PoolKind.FULL and position < already:
                continue
            pending_key = (req_id, group.kind, position)
            pending = self._pending.get(pending_key)
            if pending is None:
                slots = self.index.allocate(1, group.kind)
                if slots is None:
                    allocation_failed = True
                    break
                pending = _PendingSlot(
                    kind=group.kind,
                    position=position,
                    slot=int(slots[0]),
                    path=state.path,
                )
                self._pending[pending_key] = pending
            # track exactly what the connector offers; a windowed group's slot
            # merges every windowed group's window at this position
            pending.parts_left.add((group.group_idx, sub))
            pending.keys.append(key)
            keys_to_store.append(key)
            # one CPU slot per (group, position): the connector's handler
            # expands it into the group's sub-blocks itself
            if last != (group.group_idx, position):
                slots_out.append(pending.slot)
            last = (group.group_idx, position)

        if allocation_failed and not keys_to_store:
            return None
        return PrepareStoreOutput(
            keys_to_store=keys_to_store,
            store_spec=CPULoadStoreSpec(slots_out),
            evicted_keys=[],
        )

    def complete_store(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
        success: bool = True,
    ) -> None:
        state = self._state(req_context)
        req_id = req_context.req_id
        completed: list[_PendingSlot] = []
        for key in keys:
            located = self._locate(state, req_context, key)
            if located is None:
                continue
            group, position, sub = located
            pending = self._pending.get((req_id, group.kind, position))
            if pending is None:
                continue
            pending.parts_left.discard((group.group_idx, sub))
            pending.landed_any = True
            if not pending.parts_left and pending not in completed:
                completed.append(pending)
                del self._pending[(req_id, group.kind, position)]
        if not completed:
            return
        if not success:
            for pending in completed:
                self.index.recycle(
                    np.array([pending.slot], dtype=np.int32), pending.kind
                )
            return

        stored: list[OffloadKey] = []
        # FULL first: the other components hang on the FULL path
        full = sorted(
            (p for p in completed if p.kind == PoolKind.FULL), key=lambda p: p.position
        )
        run: list[_PendingSlot] = []
        for item in full:
            if run and (
                item.position != run[-1].position + 1 or item.path is not run[-1].path
            ):
                stored.extend(self._publish_full_run(run))
                run = []
            run.append(item)
        if run:
            stored.extend(self._publish_full_run(run))

        others = [p for p in completed if p.kind != PoolKind.FULL] + self._retry
        self._retry = []
        for pending in others:
            stored.extend(self._publish_component(pending))

        if self._events is not None and stored:
            self._events.append(
                OffloadingEvent(keys=stored, medium=Medium.CPU, removed=False)
            )

    def _publish_full_run(self, run: list[_PendingSlot]) -> list[OffloadKey]:
        first, last = run[0], run[-1]
        slots = np.array([p.slot for p in run], dtype=np.int32)
        published, _ = self.index.publish(
            first.path[: last.position + 1], slots, first.position
        )
        if published < len(run):
            logger.debug(
                "RadixShmem: published %d of %d FULL chunks from position %d (the "
                "rest was already present, or the prefix below was evicted)",
                published,
                len(run),
                first.position,
            )
        # publish() keeps the tail: the leading (len - published) were redundant
        return [key for p in run[len(run) - published :] for key in p.keys]

    def _publish_component(self, pending: _PendingSlot) -> list[OffloadKey]:
        n = pending.position + 1
        if pending.kind == PoolKind.SWA and self.geometry.swa_window_blocks > 1:
            # the index publishes a whole trailing window at once; only the
            # single-position window (window <= one FULL chunk) is supported
            # per slot here, larger windows need every slot of the window
            window = self.geometry.swa_window_blocks
            logger.warning_once(
                "RadixShmem: SWA windows spanning %d FULL chunks are published one "
                "position at a time; positions below the request end will not be "
                "loadable from this store",
                window,
            )
        slots = np.array([pending.slot], dtype=np.int32)
        published, error = self.index.publish_component(
            pending.path[:n], slots, pending.kind
        )
        if published:
            return pending.keys
        if error == "FULL_PATH_MISSING" and pending.retries < _MAX_PUBLISH_RETRIES:
            # the FULL chunks below are still in flight in another job
            pending.retries += 1
            self._retry.append(pending)
            return []
        if error is not None:
            logger.debug(
                "RadixShmem: %s publish at position %d rejected (%s)",
                PoolKind(pending.kind).name,
                pending.position,
                error,
            )
        # rejected for good, or a peer's slot won: ours came back via auto_recycle
        return []

    def on_schedule_end(self, context: ScheduleEndContext) -> None:
        """Drop every pin taken this step that did not turn into a load."""
        for req_id in self._dirty:
            state = self._states.get(req_id)
            if state is not None:
                state.release_round()
        self._dirty.clear()

    def on_request_finished(self, req_context: ReqContext) -> None:
        state = self._states.pop(req_context.req_id, None)
        self._dirty.discard(req_context.req_id)
        if state is None:
            return
        state.release_round()
        # load pins stay until complete_load: a copy may still be in flight

    def take_events(self) -> Iterable[OffloadingEvent]:
        if not self._events:
            return ()
        events, self._events = self._events, []
        return events

    def has_pending_work(self) -> bool:
        return False

    def reset_cache(self) -> None:
        # The index is shared with other schedulers and instances; one engine's
        # reset must not evict what they rely on.
        logger.warning(
            "RadixShmem: reset_prefix_cache does not clear the node-shared "
            "index; restart the owning instance to drop it"
        )

    def get_stats(self) -> OffloadingConnectorStats | None:
        current = self.index.stats()
        previous, self._stats_snapshot = self._stats_snapshot, current
        stats = OffloadingConnectorStats()
        stats.set_gauge(RadixShmemMetrics.SLOTS_USED, current["used_slots"])
        stats.set_gauge(RadixShmemMetrics.SLOTS_TOTAL, current["num_slots"])
        stats.set_gauge(RadixShmemMetrics.OPEN_LEASES, current["open_leases"])
        # Prometheus rejects negative counter increments, so every delta is
        # clamped: the fields are monotonic, but float rounding is not.
        for name, field_name in (
            (RadixShmemMetrics.HIT_BLOCKS, "hit_blocks"),
            (RadixShmemMetrics.PUBLISHED_BLOCKS, "published"),
            (RadixShmemMetrics.PUBLISH_REJECTED, "publish_rejected"),
            (RadixShmemMetrics.ALLOC_FAILURES, "alloc_failures"),
            (RadixShmemMetrics.INDEX_TIME, "index_s"),
        ):
            delta = current[field_name] - previous[field_name]
            if delta > 0:
                stats.increase_counter(name, delta)
        return stats

    def release_all(self) -> None:
        """Give back every piece of shared state this process still holds.

        Pins would keep blocks unevictable and slots reserved for stores that
        never completed would sit allocated but unreachable, both for as long as
        the shared region lives -- which outlives this process.
        """
        for state in self._states.values():
            state.release_all()
        self._states.clear()
        self._dirty.clear()
        by_kind: dict[int, list[int]] = defaultdict(list)
        for pending in list(self._pending.values()) + self._retry:
            by_kind[pending.kind].append(pending.slot)
        self._pending.clear()
        self._retry.clear()
        for kind, slots in by_kind.items():
            self.index.recycle(np.array(slots, dtype=np.int32), kind)

    def shutdown(self) -> None:
        self.release_all()
        logger.info("RadixShmem index stats: %s", self.index.stats())
        self.index.close()
        self.regions.close()
