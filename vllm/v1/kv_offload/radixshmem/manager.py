# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The RadixShmem index ledger, as seen by one DP scheduler.

This is deliberately *not* an ``OffloadingManager``. That interface is
per-request-suffix (``lookup`` gets hashes sliced from ``start_block_idx``),
while a radix tree only knows how to match from the root -- so the scheduler
here hands whole prefixes down and this class stays a thin, testable wrapper
over ``RadixClient``.

Ownership rules, all of which the scheduler depends on:

* RadixShmem is the only slot allocator. ``allocate`` is also the only point
  where eviction happens (ref-protected LRU), so it can fail while the tree is
  pinned solid.
* A hit is pinned by ``query(lock=True)``, which matches and takes the ref in
  one lock. The returned lease MUST be released exactly once; ``LookupLease``
  makes double release a no-op and the manager tracks live leases so shutdown
  can drain them.
* ``insert`` is the only publish point, and it happens strictly after the data
  is in the SlotStore -- there is no unready state in the shared tree.
"""

import time

import numpy as np

from vllm.logger import init_logger
from vllm.v1.core.kv_cache_utils import BlockHash

logger = init_logger(__name__)


def to_u64(block_hashes: list[BlockHash]) -> np.ndarray:
    """Reduce vLLM ``BlockHash`` bytes to the uint64 RadixShmem indexes on.

    vLLM's hash already folds in the parent hash, LoRA id, multimodal keys and
    cache salt, so truncation keeps the prefix semantics; it only shrinks the
    collision space. Every process attaching the same region must use exactly
    this function.
    """
    n = len(block_hashes)
    if n == 0:
        return np.empty(0, dtype=np.uint64)
    buf = bytearray(8 * n)
    for i, h in enumerate(block_hashes):
        buf[8 * i : 8 * i + 8] = h[:8]
    return np.frombuffer(bytes(buf), dtype=np.uint64)


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


class RadixShmemManager:
    """Scheduler-side ledger over one attached ``RadixClient``."""

    def __init__(self, client, *, offloaded_block_size: int):
        self.client = client
        self.offloaded_block_size = offloaded_block_size
        self._leases: set[LookupLease] = set()
        # counters, for logging and the connector's stats
        self.num_lookups = 0
        self.num_hit_blocks = 0
        self.num_allocated = 0
        self.num_recycled = 0
        self.num_published = 0
        self.num_insert_rejected = 0
        self.num_alloc_failures = 0
        # wall time inside the shared index, split by op: with several DP
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

    def lookup(self, hashes: np.ndarray) -> int:
        """Blocks of ``hashes`` (root-anchored) already in the shared cache."""
        if hashes.size == 0:
            return 0
        t0 = time.perf_counter()
        r = self.client.query(hashes, local_only=True, lock=False, update_meta=False)
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
        them are normal here, because another DP rank may have published the
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
            self.num_insert_rejected += 1
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
            "insert_rejected": self.num_insert_rejected,
            "alloc_failures": self.num_alloc_failures,
            "queries": self.num_queries,
            "allocs": self.num_allocs,
            "inserts": self.num_inserts,
            "query_ms": round(self.time_query_s * 1e3, 1),
            "alloc_ms": round(self.time_alloc_s * 1e3, 1),
            "insert_ms": round(self.time_insert_s * 1e3, 1),
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
