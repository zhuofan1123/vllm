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
  matches root-anchored prefixes, so the manager rebuilds each request's per
  group hash path from ``ReqContext.block_hashes`` and answers a whole step's
  worth of lookups from one pinned query.

Group handling (all groups share one slot pool; a slot is one chunk of one
group, see ``geometry.py``):

* full-attention groups are stored as radix *paths*: chunk ``i`` of group ``g``
  is node ``i`` on the path ``[h(0,g), h(1,g), ...]``, so a prefix hit is one
  query and LRU/pinning follow the tree;
* windowed groups (sliding window, chunked local, Mamba) are stored *flat*, one
  root child per chunk, because the connector only ever stores the last window
  of them and a path with a missing prefix cannot be inserted.

Group ``g`` is folded into the 64-bit key so the groups never collide in the
one tree; group 0 is left unmixed so single-group models key exactly as before.
"""

import time
from collections import defaultdict
from collections.abc import Collection, Iterable, Sequence
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
    get_offload_block_hash,
    get_offload_group_idx,
    make_offload_key,
)
from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec

logger = init_logger(__name__)

# splitmix64 increment; mixed into the key for every group but the first
_GROUP_MIX = np.uint64(0x9E3779B97F4A7C15)
_U64_MASK = (1 << 64) - 1


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


def to_u64(block_hashes: Iterable[bytes], group_idx: int = 0) -> np.ndarray:
    """Reduce vLLM ``BlockHash`` bytes to the uint64 keys RadixShmem indexes on.

    vLLM's hash already folds in the parent hash, LoRA id, multimodal keys and
    cache salt, so truncation keeps the prefix semantics; it only shrinks the
    collision space. The KV cache group is folded in so one tree can hold every
    group's chunks. Every process attaching the same region must use exactly
    this function.
    """
    hashes = list(block_hashes)
    n = len(hashes)
    if n == 0:
        return np.empty(0, dtype=np.uint64)
    buf = bytearray(8 * n)
    for i, h in enumerate(hashes):
        buf[8 * i : 8 * i + 8] = h[:8]
    arr = np.frombuffer(bytes(buf), dtype=np.uint64)
    if group_idx:
        arr = arr ^ np.uint64((group_idx * int(_GROUP_MIX)) & _U64_MASK)
    return np.ascontiguousarray(arr, dtype=np.uint64)


class LookupLease:
    """A pinned prefix hit: ``slots`` stay valid until ``release()``."""

    __slots__ = ("hit_blocks", "slots", "_finalize", "_owner", "_released")

    def __init__(self, hit_blocks: int, slots: np.ndarray, finalize, owner):
        self.hit_blocks = hit_blocks
        self.slots = slots
        self._finalize = finalize
        self._owner = owner
        self._released = False

    @property
    def released(self) -> bool:
        return self._released

    def release(self) -> None:
        """Drop the pin. Idempotent -- ``finalize`` is one-shot underneath."""
        if self._released:
            return
        self._released = True
        try:
            self._finalize()
        finally:
            self._owner._forget(self)

    def __repr__(self) -> str:
        state = "released" if self._released else "held"
        return f"LookupLease({self.hit_blocks} blocks, {state})"


class RadixIndex:
    """Ledger over one attached ``RadixClient`` (one per scheduler process)."""

    def __init__(self, client):
        self.client = client
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
        """``common_hit`` of a Full-only query, or 0 when the request failed."""
        status = getattr(result, "status", None)
        if status is not None and int(status) != 0:
            return 0
        return int(result.common_hit)

    def lookup(self, hashes: np.ndarray, *, touch: bool = False) -> int:
        """Blocks of ``hashes`` (root-anchored) already in the shared cache.

        ``touch`` refreshes the LRU position of the hit; a plain probe does not.
        """
        if hashes.size == 0:
            return 0
        t0 = time.perf_counter()
        r = self.client.query(hashes, local_only=True, lock=False, update_meta=touch)
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
        t0 = time.perf_counter()
        r = self.client.query(hashes, local_only=True, lock=True, update_meta=True)
        self.time_query_s += time.perf_counter() - t0
        self.num_queries += 1
        hit = self._hit_blocks(r)
        if hit == 0:
            # lock=True with a zero hit takes no ref, but finalize is still a
            # valid one-shot; call it so nothing is left dangling.
            r.finalize()
            return None
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
        slots = np.ascontiguousarray(slots[:hit], dtype=np.int32)
        self.num_hit_blocks += hit
        lease = LookupLease(hit, slots, r.finalize, self)
        self._leases.add(lease)
        return lease

    def _forget(self, lease: LookupLease) -> None:
        self._leases.discard(lease)

    @property
    def num_open_leases(self) -> int:
        return len(self._leases)

    # -------------------------------------------------------- allocation

    def allocate(self, n: int) -> np.ndarray | None:
        """Reserve ``n`` slots, evicting unpinned LRU entries as needed.

        Returns ``None`` when the pool cannot satisfy the request -- every
        candidate is either pinned or freshly published. The caller must skip
        the store, not retry in a loop.
        """
        if n <= 0:
            return np.empty(0, dtype=np.int32)
        t0 = time.perf_counter()
        slots = self.client.allocate_slots(n)
        self.time_alloc_s += time.perf_counter() - t0
        self.num_allocs += 1
        got = int(slots.size)
        if got < n:
            self.num_alloc_failures += 1
            if got:
                self.client.recycle_slots(slots)
            return None
        self.num_allocated += got
        return np.ascontiguousarray(slots, dtype=np.int32)

    def recycle(self, slots: np.ndarray) -> None:
        if slots is None or slots.size == 0:
            return
        self.client.recycle_slots(np.ascontiguousarray(slots, dtype=np.int32))
        self.num_recycled += int(slots.size)

    # ----------------------------------------------------------- publish

    def publish(
        self, hashes: np.ndarray, slots: np.ndarray, start: int
    ) -> tuple[int, np.ndarray]:
        """Register ``slots`` for ``hashes[start:]`` after the data landed.

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
        if slots.size == 0:
            return 0, np.empty(0, dtype=np.int32)
        hashes = np.ascontiguousarray(hashes, dtype=np.uint64)
        slots = np.ascontiguousarray(slots, dtype=np.int32)
        t0 = time.perf_counter()
        res = self.client.insert(hashes, slots, start=start, auto_recycle=True)
        self.time_insert_s += time.perf_counter() - t0
        self.num_inserts += 1
        unused = res.unused_slots
        published = int(slots.size) - int(unused.size)
        if int(res.error) != 0 and published == 0:
            logger.warning_once(
                "RadixShmem insert failed (%s); the shared index is full. "
                "Raise max_nodes / data_pool_ratio if this persists.",
                res.error,
            )
        if published <= 0:
            self.num_publish_rejected += 1
            return 0, unused
        self.num_published += published
        return published, unused

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


@dataclass(frozen=True)
class GroupInfo:
    """What the manager needs to know about one KV cache group."""

    group_idx: int
    # vLLM block hashes per offloaded chunk of this group
    hashes_per_chunk: int
    # sliding window / chunked local / Mamba: stored flat, not as a path
    windowed: bool


@dataclass
class _GroupHits:
    """One step's root-anchored lookup of a full-attention group."""

    hit_blocks: int
    # pinned hit; None once handed to a load (or when hit_blocks == 0)
    lease: LookupLease | None


@dataclass
class _ReqState:
    """Per-request scratch, kept on the ReqContext and in ``_states``."""

    # per group: mixed u64 path over the chunks known so far (immutable arrays;
    # extending builds a new one, so older references stay valid prefixes)
    paths: dict[int, np.ndarray] = field(default_factory=dict)
    # per group: how many block hashes have been folded into `paths`/`chunk_idx`
    indexed_hashes: dict[int, int] = field(default_factory=dict)
    # key -> chunk index within its group
    chunk_idx: dict[OffloadKey, int] = field(default_factory=dict)
    # current step's pinned lookups
    round_hits: dict[int, _GroupHits] = field(default_factory=dict)
    flat_leases: dict[OffloadKey, LookupLease] = field(default_factory=dict)
    # pins that back an in-flight load; released in complete_load
    load_leases: list[LookupLease] = field(default_factory=list)

    def release_round(self) -> None:
        for hits in self.round_hits.values():
            if hits.lease is not None:
                hits.lease.release()
        self.round_hits.clear()
        for lease in self.flat_leases.values():
            lease.release()
        self.flat_leases.clear()

    def release_all(self) -> None:
        self.release_round()
        for lease in self.load_leases:
            lease.release()
        self.load_leases.clear()


@dataclass(frozen=True)
class _PendingStore:
    """A slot reserved by prepare_store, published in complete_store."""

    group_idx: int
    chunk_idx: int
    slot: int
    # root-anchored path (len >= chunk_idx + 1) for path groups; the single
    # mixed key for flat groups
    hashes: np.ndarray


class RadixShmemOffloadingManager(OffloadingManager):
    """``OffloadingManager`` over the node-shared RadixShmem index."""

    def __init__(
        self,
        regions,
        groups: Sequence[GroupInfo],
        *,
        enable_events: bool = False,
    ):
        self.regions = regions
        self.index = RadixIndex(regions.client)
        self.groups: dict[int, GroupInfo] = {g.group_idx: g for g in groups}
        self._states: dict[str, _ReqState] = {}
        # requests that pinned something this step and may need a sweep
        self._dirty: set[str] = set()
        self._pending: dict[OffloadKey, _PendingStore] = {}
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
        """Fold any new block hashes of the request into paths and key map."""
        block_hashes = req_context.block_hashes
        if block_hashes is None:
            raise RuntimeError(
                "RadixShmem offloading needs ReqContext.block_hashes; the "
                "offloading connector scheduler did not provide them"
            )
        num_hashes = len(block_hashes)
        for group in self.groups.values():
            g = group.group_idx
            hpc = group.hashes_per_chunk
            done = state.indexed_hashes.get(g, 0)
            num_chunks = num_hashes // hpc
            first_chunk = done // hpc
            if num_chunks <= first_chunk:
                continue
            new_hashes = [
                block_hashes[(i + 1) * hpc - 1] for i in range(first_chunk, num_chunks)
            ]
            for i, h in enumerate(new_hashes, start=first_chunk):
                state.chunk_idx[make_offload_key(h, g)] = i
            new_u64 = to_u64(new_hashes, g)
            old = state.paths.get(g)
            state.paths[g] = new_u64 if old is None else np.concatenate([old, new_u64])
            state.indexed_hashes[g] = num_chunks * hpc

    def _locate(
        self, state: _ReqState, req_context: ReqContext, key: OffloadKey
    ) -> tuple[GroupInfo, int] | None:
        """(group, chunk index) of ``key`` within this request, or None."""
        idx = state.chunk_idx.get(key)
        if idx is None:
            self._index_keys(state, req_context)
            idx = state.chunk_idx.get(key)
            if idx is None:
                return None
        return self.groups[get_offload_group_idx(key)], idx

    @staticmethod
    def _flat_key(key: OffloadKey) -> np.ndarray:
        return to_u64([get_offload_block_hash(key)], get_offload_group_idx(key))

    # ------------------------------------------------------ OffloadingManager

    def on_new_request(self, req_context: ReqContext) -> RequestOffloadingContext:
        self._state(req_context)
        return RequestOffloadingContext()

    def lookup(self, key: OffloadKey, req_context: ReqContext) -> LookupResult:
        state = self._state(req_context)
        located = self._locate(state, req_context, key)
        if located is None:
            # not derived from this request's hashes (e.g. a boundary key of a
            # partial recurrent tail) -- never offloaded by this manager
            return LookupResult.MISS
        group, idx = located
        self._dirty.add(req_context.req_id)

        if group.windowed:
            if key in state.flat_leases:
                return LookupResult.HIT
            lease = self.index.lookup_and_pin(self._flat_key(key))
            if lease is None:
                return LookupResult.MISS
            state.flat_leases[key] = lease
            return LookupResult.HIT

        hits = state.round_hits.get(group.group_idx)
        if hits is None:
            lease = self.index.lookup_and_pin(state.paths[group.group_idx])
            hits = _GroupHits(
                hit_blocks=lease.hit_blocks if lease is not None else 0, lease=lease
            )
            state.round_hits[group.group_idx] = hits
        return LookupResult.HIT if idx < hits.hit_blocks else LookupResult.MISS

    def touch(self, keys: Collection[OffloadKey], req_context: ReqContext) -> None:
        """Refresh the LRU position of the request's path groups.

        Flat chunks are not touched: they are cheap to recompute and refreshing
        them one query at a time would cost more than it saves.
        """
        if not keys:
            return
        state = self._state(req_context)
        deepest: dict[int, int] = {}
        for key in keys:
            located = self._locate(state, req_context, key)
            if located is None or located[0].windowed:
                continue
            g = located[0].group_idx
            deepest[g] = max(deepest.get(g, -1), located[1])
        for g, idx in deepest.items():
            hits = state.round_hits.get(g)
            if hits is not None and hits.hit_blocks >= idx + 1:
                continue  # this step's pinned query already refreshed it
            self.index.lookup(state.paths[g][: idx + 1], touch=True)

    def prepare_load(
        self, keys: Collection[OffloadKey], req_context: ReqContext
    ) -> LoadStoreSpec:
        state = self._state(req_context)
        slots: list[int] = []
        for key in keys:
            located = self._locate(state, req_context, key)
            assert located is not None, f"load of a key foreign to the request: {key!r}"
            group, idx = located
            if group.windowed:
                lease = state.flat_leases.pop(key, None)
                if lease is None:
                    # lookup happened in an earlier step and was swept; re-pin
                    lease = self.index.lookup_and_pin(self._flat_key(key))
                    if lease is None:
                        raise RuntimeError(
                            "RadixShmem: chunk evicted between lookup and load"
                        )
                state.load_leases.append(lease)
                slots.append(int(lease.slots[0]))
                continue
            hits = state.round_hits.get(group.group_idx)
            if hits is None or hits.hit_blocks <= idx:
                raise RuntimeError(
                    f"RadixShmem: load of chunk {idx} (group {group.group_idx}) "
                    "that this step's lookup did not report as a hit"
                )
            lease = hits.lease
            if lease is not None:
                # the pin moves to the load; the sweep must not release it
                state.load_leases.append(lease)
                hits.lease = None
            slots.append(int(state.load_leases[-1].slots[idx]))
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
        by_group: dict[int, list[tuple[int, OffloadKey]]] = defaultdict(list)
        for key in keys:
            located = self._locate(state, req_context, key)
            if located is None:
                continue  # cannot place a key we cannot anchor
            by_group[located[0].group_idx].append((located[1], key))

        keys_to_store: list[OffloadKey] = []
        slots_out: list[int] = []
        allocation_failed = False
        for g, items in by_group.items():
            group = self.groups[g]
            items.sort()
            if group.windowed:
                for idx, key in items:
                    if key in self._pending:
                        continue
                    flat = self._flat_key(key)
                    if self.index.lookup(flat) > 0:
                        continue  # a peer already published it
                    slots = self.index.allocate(1)
                    if slots is None:
                        allocation_failed = True
                        break
                    self._pending[key] = _PendingStore(g, idx, int(slots[0]), flat)
                    keys_to_store.append(key)
                    slots_out.append(int(slots[0]))
                continue

            path = state.paths[g]
            end = items[-1][0] + 1
            assert end <= path.size
            # somebody else may have published part of this prefix already;
            # only move the bytes that are genuinely missing
            already = self.index.lookup(path[:end])
            wanted = [
                (idx, key)
                for idx, key in items
                if idx >= already and key not in self._pending
            ]
            if not wanted:
                continue
            slots = self.index.allocate(len(wanted))
            if slots is None:
                allocation_failed = True
                continue
            for (idx, key), slot in zip(wanted, slots):
                self._pending[key] = _PendingStore(g, idx, int(slot), path)
                keys_to_store.append(key)
                slots_out.append(int(slot))

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
        pending: list[tuple[OffloadKey, _PendingStore]] = []
        for key in keys:
            entry = self._pending.pop(key, None)
            if entry is not None:
                pending.append((key, entry))
        if not pending:
            return
        if not success:
            self.index.recycle(np.array([p.slot for _, p in pending], dtype=np.int32))
            return

        by_group: dict[int, list[tuple[OffloadKey, _PendingStore]]] = defaultdict(list)
        for key, entry in pending:
            by_group[entry.group_idx].append((key, entry))

        stored: list[OffloadKey] = []
        for g, entries in by_group.items():
            entries.sort(key=lambda e: e[1].chunk_idx)
            if self.groups[g].windowed:
                for key, entry in entries:
                    published, _ = self.index.publish(
                        entry.hashes, np.array([entry.slot], dtype=np.int32), 0
                    )
                    if published:
                        stored.append(key)
                continue
            # publish maximal runs of consecutive chunks in one insert each
            run: list[tuple[OffloadKey, _PendingStore]] = []
            for item in entries:
                if run and (
                    item[1].chunk_idx != run[-1][1].chunk_idx + 1
                    or item[1].hashes is not run[-1][1].hashes
                ):
                    stored.extend(self._publish_run(run))
                    run = []
                run.append(item)
            if run:
                stored.extend(self._publish_run(run))

        if self._events is not None and stored:
            self._events.append(
                OffloadingEvent(keys=stored, medium=Medium.CPU, removed=False)
            )

    def _publish_run(
        self, run: list[tuple[OffloadKey, _PendingStore]]
    ) -> list[OffloadKey]:
        first = run[0][1]
        last = run[-1][1]
        slots = np.array([p.slot for _, p in run], dtype=np.int32)
        published, _ = self.index.publish(
            first.hashes[: last.chunk_idx + 1], slots, first.chunk_idx
        )
        if published < len(run):
            logger.debug(
                "RadixShmem: published %d of %d chunks starting at %d (the rest "
                "was already present, or the prefix below was evicted)",
                published,
                len(run),
                first.chunk_idx,
            )
        # publish() keeps the tail: the leading (len - published) were redundant
        return [key for key, _ in run[len(run) - published :]]

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
        if self._pending:
            self.index.recycle(
                np.array([p.slot for p in self._pending.values()], dtype=np.int32)
            )
            self._pending.clear()

    def shutdown(self) -> None:
        self.release_all()
        logger.info("RadixShmem index stats: %s", self.index.stats())
        self.index.close()
        self.regions.close()
