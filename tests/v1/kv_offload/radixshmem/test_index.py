# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The ``RadixIndex`` ledger over a real shared index (CPU only)."""

import contextlib
import os

import numpy as np
import pytest

from vllm.v1.kv_offload.radixshmem.manager import RadixIndex, to_u64

shmradix = pytest.importorskip("shmradix")

BLOCK_SIZE = 16
NUM_SLOTS = 64


@pytest.fixture
def manager(request):
    name = f"/rs_mgr_{os.getpid()}_{abs(hash(request.node.name)) % 10**6}"
    with contextlib.suppress(OSError):
        os.unlink(f"/dev/shm{name}")
    # the bare index (no server process, no store): RadixIndex needs nothing else
    cfg = shmradix._core.ShmConfig()
    cfg.max_nodes = 4 * NUM_SLOTS + 1024
    cfg.max_blocks = NUM_SLOTS
    cfg.block_size = BLOCK_SIZE
    server = shmradix._core.RadixServer(name, cfg)
    client = shmradix._core.RadixClient(server)
    m = RadixIndex(client)
    yield m
    m.close()
    del client, server
    with contextlib.suppress(OSError):
        os.unlink(f"/dev/shm{name}")


def hashes(*ids: int) -> np.ndarray:
    return np.array(ids, dtype=np.uint64)


def store(m: RadixIndex, h: np.ndarray, start: int = 0):
    """Allocate + publish the tail of ``h`` starting at block ``start``."""
    slots = m.allocate(int(h.size) - start)
    assert slots is not None
    return m.publish(h, slots, start)


# ------------------------------------------------------------ dimension reduction


def test_to_u64_is_little_endian_prefix():
    from vllm.v1.core.kv_cache_utils import BlockHash

    a = BlockHash(b"\x01" + b"\x00" * 31)
    b = BlockHash(bytes(range(32)))
    out = to_u64([a, b])
    assert out.dtype == np.uint64
    assert out[0] == 1
    assert out[1] == int.from_bytes(bytes(range(8)), "little")


def test_to_u64_empty():
    assert to_u64([]).shape == (0,)


# ------------------------------------------------------------------- basic flow


def test_publish_then_lookup(manager):
    h = hashes(1, 2, 3, 4)
    published, unused = store(manager, h)
    assert published == 4
    assert unused.size == 0
    assert manager.lookup(h) == 4
    # a longer prefix sharing the first 4 blocks still hits exactly 4
    assert manager.lookup(hashes(1, 2, 3, 4, 5, 6)) == 4
    # a divergent prefix hits only the common part
    assert manager.lookup(hashes(1, 2, 9)) == 2
    assert manager.lookup(hashes(9, 9)) == 0


def test_lookup_empty_prefix(manager):
    assert manager.lookup(np.empty(0, dtype=np.uint64)) == 0
    assert manager.lookup_and_pin(np.empty(0, dtype=np.uint64)) is None


def test_miss_returns_no_lease(manager):
    assert manager.lookup_and_pin(hashes(7, 8)) is None
    assert manager.num_open_leases == 0


def test_lease_slots_match_publish(manager):
    h = hashes(1, 2, 3)
    slots = manager.allocate(3)
    manager.publish(h, slots, 0)
    lease = manager.lookup_and_pin(h)
    assert lease is not None
    assert lease.hit_blocks == 3
    assert list(lease.slots) == list(slots)
    lease.release()


# ------------------------------------------------------------------- pin safety


def test_pinned_blocks_survive_allocation_pressure(manager):
    """A leased prefix must not be evicted, even under a full-pool allocate."""
    h = hashes(*range(1, 9))
    store(manager, h)
    lease = manager.lookup_and_pin(h)
    assert lease is not None and lease.hit_blocks == 8

    # churn far more than the pool holds; only unpinned entries may be evicted
    for gen in range(1, 12):
        filler = hashes(*(1000 * gen + i for i in range(16)))
        s = manager.allocate(16)
        if s is None:
            break
        manager.publish(filler, s, 0)

    assert manager.lookup(h) == 8, "pinned prefix was evicted"
    assert list(manager.lookup_and_pin(h).slots) == list(lease.slots)
    lease.release()


def test_allocate_fails_when_everything_is_pinned(manager):
    h = hashes(*range(1, NUM_SLOTS + 1))
    slots = manager.allocate(NUM_SLOTS)
    assert slots is not None
    manager.publish(h, slots, 0)
    lease = manager.lookup_and_pin(h)
    assert lease is not None and lease.hit_blocks == NUM_SLOTS

    assert manager.allocate(1) is None
    assert manager.num_alloc_failures == 1
    # nothing was half-taken on the failure path
    assert manager.client.mempool_free() == 0

    lease.release()
    again = manager.allocate(1)
    assert again is not None
    manager.recycle(again)


def test_release_is_idempotent(manager):
    h = hashes(1, 2)
    store(manager, h)
    lease = manager.lookup_and_pin(h)
    lease.release()
    lease.release()
    lease.release()
    assert manager.num_open_leases == 0
    # the refs really are back to zero: everything is evictable again
    for gen in range(1, 12):
        s = manager.allocate(16)
        if s is None:
            break
        manager.publish(hashes(*(1000 * gen + i for i in range(16))), s, 0)
    assert manager.lookup(h) < 2


def test_close_drains_open_leases(manager):
    h = hashes(5, 6, 7)
    store(manager, h)
    lease = manager.lookup_and_pin(h)
    assert lease is not None
    client = manager.client
    manager.close()
    assert manager.client is None
    # the pin is gone, so the whole pool is allocatable again
    got = client.allocate_slots(NUM_SLOTS)
    assert got.size == NUM_SLOTS
    client.recycle_slots(got)


# ---------------------------------------------------------------- insert states


def test_publish_with_start_skips_matched_prefix(manager):
    base = hashes(1, 2, 3)
    store(manager, base)
    ext = hashes(1, 2, 3, 4, 5)
    slots = manager.allocate(2)
    published, unused = manager.publish(ext, slots, start=3)
    assert published == 2 and unused.size == 0
    assert manager.lookup(ext) == 5


def test_publish_races_a_peer_that_got_there_first(manager):
    """matched > start: the leading slots come back, the tail is published."""
    ext = hashes(1, 2, 3, 4, 5)
    # peer published the first 4 blocks while our store was in flight
    peer_slots = manager.allocate(4)
    manager.publish(ext[:4], peer_slots, 0)

    used_before = manager.client.mempool_used()  # the peer's 4
    mine = manager.allocate(5)
    published, unused = manager.publish(ext, mine, start=0)
    assert published == 1
    assert unused.size == 4
    assert list(unused) == list(mine[:4])
    assert manager.lookup(ext) == 5
    # the 4 duplicates went straight back to the pool
    assert manager.client.mempool_used() == used_before + 1


def test_publish_rejected_when_prefix_vanished(manager):
    """matched < start: nothing is published and every slot is returned."""
    ext = hashes(1, 2, 3, 4, 5)
    used_before = manager.client.mempool_used()
    slots = manager.allocate(2)
    published, unused = manager.publish(ext, slots, start=3)  # blocks 0..2 absent
    assert published == 0
    assert unused.size == 2
    assert manager.num_publish_rejected == 1
    assert manager.client.mempool_used() == used_before
    assert manager.lookup(ext) == 0


def test_publish_empty_is_a_noop(manager):
    published, unused = manager.publish(hashes(1, 2), np.empty(0, dtype=np.int32), 2)
    assert published == 0 and unused.size == 0


# ----------------------------------------------------------------- bookkeeping


def test_slots_are_conserved_across_a_full_cycle(manager):
    baseline = manager.client.mempool_used()
    for i in range(20):
        h = hashes(*(100 * i + j for j in range(4)))
        slots = manager.allocate(4)
        assert slots is not None
        lease = None
        published, _ = manager.publish(h, slots, 0)
        assert published == 4
        lease = manager.lookup_and_pin(h)
        assert lease is not None
        lease.release()
    assert manager.num_open_leases == 0
    # everything published is still owned by the tree, nothing leaked loose
    manager.client.reset()
    assert manager.client.mempool_used() == baseline


def test_recycle_returns_slots(manager):
    used = manager.client.mempool_used()
    slots = manager.allocate(5)
    assert manager.client.mempool_used() == used + 5
    manager.recycle(slots)
    assert manager.client.mempool_used() == used
    assert manager.num_recycled == 5


def test_stats_shape(manager):
    s = manager.stats()
    assert s["num_slots"] == NUM_SLOTS
    assert set(s) >= {"used_slots", "free_slots", "open_leases", "published"}
