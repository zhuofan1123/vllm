# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""``RadixShmemOffloadingManager`` against a real shared index (CPU only).

Drives the ``OffloadingManager`` contract the way the upstream offloading
connector scheduler does: per-key lookups, ``prepare_load`` / ``complete_load``
around a load, ``prepare_store`` / ``complete_store`` around a store, and the
``on_schedule_end`` sweep between steps. The hybrid fixture is DeepSeek-V4
shaped (see ``hybrid_groups``): one SWA slot spans two SWA chunks.
"""

import numpy as np
import pytest

from vllm.v1.kv_offload.base import (
    LookupResult,
    ScheduleEndContext,
    make_offload_key,
)
from vllm.v1.kv_offload.radixshmem.bootstrap import attach_regions, create_regions
from vllm.v1.kv_offload.radixshmem.geometry import PoolKind
from vllm.v1.kv_offload.radixshmem.manager import RadixShmemOffloadingManager

from .utils import (
    block_hashes,
    geometry_for,
    hybrid_groups,
    keys_for,
    make_offloading_config,
    req_context,
    unique_tag,
)

STEP = ScheduleEndContext(new_req_ids=(), preempted_req_ids=())


@pytest.fixture
def regions(request):
    param = getattr(request, "param", None) or {}
    config = make_offloading_config(
        tag=unique_tag(request),
        tp_size=2,
        groups=param.get("groups"),
        extra=param.get("extra"),
    )
    r = create_regions(geometry_for(config), dict(config.extra_config))
    yield r
    r.close()


@pytest.fixture
def manager(regions):
    m = RadixShmemOffloadingManager(regions, regions.geometry)
    yield m
    m.release_all()
    m.index.close()


def store(manager, ctx, keys, *, success=True):
    """prepare_store -> (workers copy) -> complete_store; returns the output."""
    out = manager.prepare_store(keys, ctx)
    assert out is not None
    manager.complete_store(out.keys_to_store, ctx, success=success)
    return out


def lookups(manager, ctx, keys):
    return [manager.lookup(k, ctx) for k in keys]


# --------------------------------------------------------------------- store/hit


def test_store_then_hit_from_another_request(manager):
    hashes = block_hashes(4)
    a = req_context("a", hashes)
    manager.on_new_request(a)
    keys = keys_for(hashes)
    assert lookups(manager, a, keys) == [LookupResult.MISS] * 4
    manager.on_schedule_end(STEP)

    out = store(manager, a, keys)
    assert out.keys_to_store == keys
    assert len(out.store_spec.block_ids) == 4

    b = req_context("b", hashes)
    manager.on_new_request(b)
    assert lookups(manager, b, keys) == [LookupResult.HIT] * 4
    # one root-anchored query per step answers every key of the request:
    # a's miss lookup, the store's already-published probe, b's hit lookup
    assert manager.index.num_queries == 3
    manager.on_schedule_end(STEP)


def test_prefix_hit_stops_at_divergence(manager):
    hashes = block_hashes(4)
    a = req_context("a", hashes)
    store(manager, a, keys_for(hashes))

    other = hashes[:2] + block_hashes(2, salt=9)
    b = req_context("b", other)
    assert lookups(manager, b, keys_for(other)) == [
        LookupResult.HIT,
        LookupResult.HIT,
        LookupResult.MISS,
        LookupResult.MISS,
    ]
    manager.on_schedule_end(STEP)


def test_second_store_skips_what_a_peer_published(manager):
    hashes = block_hashes(4)
    store(manager, req_context("a", hashes), keys_for(hashes))
    used = manager.index.client.mempool_used()
    out = manager.prepare_store(keys_for(hashes), req_context("b", hashes))
    assert out is not None and out.keys_to_store == []
    assert manager.index.client.mempool_used() == used


def test_partial_overlap_stores_only_the_tail(manager):
    hashes = block_hashes(4)
    store(manager, req_context("a", hashes[:2]), keys_for(hashes[:2]))
    out = store(manager, req_context("b", hashes), keys_for(hashes))
    assert out.keys_to_store == keys_for(hashes)[2:]
    c = req_context("c", hashes)
    assert lookups(manager, c, keys_for(hashes)) == [LookupResult.HIT] * 4
    manager.on_schedule_end(STEP)


def test_failed_store_returns_its_slots(manager):
    hashes = block_hashes(3)
    ctx = req_context("a", hashes)
    used = manager.index.client.mempool_used()
    store(manager, ctx, keys_for(hashes), success=False)
    assert manager.index.client.mempool_used() == used
    b = req_context("b", hashes)
    assert manager.lookup(keys_for(hashes)[0], b) is LookupResult.MISS


def test_store_is_not_visible_before_complete_store(manager):
    hashes = block_hashes(2)
    ctx = req_context("a", hashes)
    out = manager.prepare_store(keys_for(hashes), ctx)
    b = req_context("b", hashes)
    assert manager.lookup(keys_for(hashes)[0], b) is LookupResult.MISS
    manager.on_schedule_end(STEP)
    manager.complete_store(out.keys_to_store, ctx)
    assert manager.lookup(keys_for(hashes)[0], b) is LookupResult.HIT
    manager.on_schedule_end(STEP)


def test_keys_foreign_to_the_request_are_misses(manager):
    hashes = block_hashes(2)
    ctx = req_context("a", hashes)
    stranger = make_offload_key(block_hashes(1, salt=5)[0], 0)
    assert manager.lookup(stranger, ctx) is LookupResult.MISS
    out = manager.prepare_store([stranger], ctx)
    assert out is not None and out.keys_to_store == []


# ------------------------------------------------------------------- load pins


def test_load_pins_until_complete_load(manager):
    hashes = block_hashes(4)
    store(manager, req_context("a", hashes), keys_for(hashes))

    b = req_context("b", hashes)
    keys = keys_for(hashes)
    assert lookups(manager, b, keys) == [LookupResult.HIT] * 4
    # GPU already has the first two chunks: load only the tail
    spec = manager.prepare_load(keys[2:], b)
    assert len(spec.block_ids) == 2
    manager.on_schedule_end(STEP)  # sweep must not drop the load's pin
    assert manager.index.num_open_leases == 1

    # a full-pool churn cannot evict what the load still reads
    for gen in range(1, 40):
        filler = block_hashes(16, salt=100 + gen)
        out = manager.prepare_store(keys_for(filler), req_context(f"f{gen}", filler))
        if out is None:
            break
        manager.complete_store(out.keys_to_store, req_context(f"f{gen}", filler))
    c = req_context("c", hashes)
    assert lookups(manager, c, keys) == [LookupResult.HIT] * 4
    manager.on_schedule_end(STEP)

    manager.complete_load(keys[2:], b)
    assert manager.index.num_open_leases == 0


def test_load_slots_match_the_stored_slots(manager):
    hashes = block_hashes(4)
    out = store(manager, req_context("a", hashes), keys_for(hashes))
    b = req_context("b", hashes)
    keys = keys_for(hashes)
    lookups(manager, b, keys)
    spec = manager.prepare_load(keys[1:], b)
    assert list(spec.block_ids) == list(out.store_spec.block_ids[1:])
    manager.complete_load(keys[1:], b)
    manager.on_schedule_end(STEP)


def test_unused_lookup_pins_are_swept_at_step_end(manager):
    hashes = block_hashes(4)
    store(manager, req_context("a", hashes), keys_for(hashes))
    b = req_context("b", hashes)
    lookups(manager, b, keys_for(hashes))
    assert manager.index.num_open_leases == 1
    manager.on_schedule_end(STEP)
    assert manager.index.num_open_leases == 0
    # the next step re-queries from scratch
    assert lookups(manager, b, keys_for(hashes)) == [LookupResult.HIT] * 4
    assert manager.index.num_open_leases == 1
    manager.on_request_finished(b)
    assert manager.index.num_open_leases == 0


def test_hashes_growing_between_steps_extend_the_path(manager):
    hashes = block_hashes(6)
    live = list(hashes[:3])
    ctx = req_context("a", live)
    keys = keys_for(hashes)
    assert lookups(manager, ctx, keys[:3]) == [LookupResult.MISS] * 3
    manager.on_schedule_end(STEP)
    live.extend(hashes[3:])  # the request decoded three more blocks
    out = store(manager, ctx, keys)
    assert out.keys_to_store == keys
    b = req_context("b", hashes)
    assert lookups(manager, b, keys) == [LookupResult.HIT] * 6
    manager.on_schedule_end(STEP)


# --------------------------------------------------------------- hybrid (SWA)

HYBRID = {"groups": hybrid_groups()}


def hybrid_keys(hashes):
    """(full keys, swa keys) for a hybrid request; 2 hashes per FULL chunk."""
    return keys_for(hashes, 0, hashes_per_chunk=2), keys_for(hashes, 1)


@pytest.mark.parametrize("regions", [HYBRID], indirect=True)
def test_swa_window_is_stored_in_its_own_pool_and_hits_per_position(manager):
    hashes = block_hashes(8)  # 4 FULL positions, 8 SWA chunks
    a = req_context("a", hashes)
    full_keys, swa_keys = hybrid_keys(hashes)
    swa_before = manager.regions.client.swa_mempool_total()
    # the connector stores the whole FULL group and the SWA tail window
    # (chunks 6, 7 == position 3)
    out = store(manager, a, full_keys + swa_keys[6:])
    assert out.keys_to_store == full_keys + swa_keys[6:]
    # one CPU slot per (group, position): 4 FULL + 1 SWA
    assert len(out.store_spec.block_ids) == 5
    assert manager.regions.client.mempool_used() == 4  # FULL pool
    assert manager.regions.client.swa_mempool_total() == swa_before

    b = req_context("b", hashes)
    assert lookups(manager, b, full_keys) == [LookupResult.HIT] * 4
    assert (
        lookups(manager, b, swa_keys)
        == [LookupResult.MISS] * 6 + [LookupResult.HIT] * 2
    )
    spec = manager.prepare_load(full_keys[2:] + swa_keys[6:], b)
    # 2 FULL positions + 1 SWA slot, the very slot the store used
    assert list(spec.block_ids) == list(out.store_spec.block_ids[2:4]) + [
        out.store_spec.block_ids[4]
    ]
    manager.complete_load(full_keys[2:] + swa_keys[6:], b)
    manager.on_schedule_end(STEP)
    assert manager.index.num_open_leases == 0


@pytest.mark.parametrize("regions", [HYBRID], indirect=True)
def test_swa_publish_waits_for_the_full_path(manager):
    """SWA slots hang on FULL positions: if the FULL chunks land later, the SWA
    publish is retried when they do."""
    hashes = block_hashes(4)
    ctx = req_context("a", hashes)
    full_keys, swa_keys = hybrid_keys(hashes)
    swa_out = manager.prepare_store(swa_keys[2:], ctx)
    full_out = manager.prepare_store(full_keys, ctx)
    manager.complete_store(swa_out.keys_to_store, ctx)  # FULL not there yet
    b = req_context("b", hashes)
    assert lookups(manager, b, swa_keys[2:]) == [LookupResult.MISS] * 2
    manager.on_schedule_end(STEP)
    manager.complete_store(full_out.keys_to_store, ctx)  # retries the SWA publish
    assert lookups(manager, b, full_keys + swa_keys[2:]) == [LookupResult.HIT] * 4
    manager.on_schedule_end(STEP)


@pytest.mark.parametrize("regions", [HYBRID], indirect=True)
def test_swa_slot_publishes_on_the_offered_window_parts(manager):
    """A windowed group offers only its reachable window; the merged SWA slot
    publishes once those offered sub-chunks land (not every theoretical one),
    and the whole position then hits."""
    hashes = block_hashes(4)
    ctx = req_context("a", hashes)
    full_keys, swa_keys = hybrid_keys(hashes)
    store(manager, ctx, full_keys)
    # the connector stores position 1's window: both its sub-chunks (2, 3)
    out = manager.prepare_store(swa_keys[2:4], ctx)
    probe = req_context("probe", hashes)
    assert manager.lookup(swa_keys[2], probe) is LookupResult.MISS  # not landed
    manager.on_schedule_end(STEP)
    manager.complete_store(out.keys_to_store, ctx)
    b = req_context("b", hashes)
    assert lookups(manager, b, swa_keys[2:4]) == [LookupResult.HIT] * 2
    manager.on_schedule_end(STEP)


@pytest.mark.parametrize("regions", [HYBRID], indirect=True)
def test_swa_slot_does_not_stall_on_unoffered_sub_chunks(manager):
    """The connector offering only part of a position's span must still publish
    -- requiring every theoretical sub-chunk would never complete for a
    windowed group that only keeps a trailing window (DeepSeek-V4)."""
    hashes = block_hashes(4)
    ctx = req_context("a", hashes)
    full_keys, swa_keys = hybrid_keys(hashes)
    store(manager, ctx, full_keys)
    # only the tail sub-chunk of position 1 is offered
    store(manager, ctx, [swa_keys[3]])
    assert not manager._pending  # published, nothing stuck
    b = req_context("b", hashes)
    assert manager.lookup(swa_keys[3], b) is LookupResult.HIT
    manager.on_schedule_end(STEP)


@pytest.mark.parametrize("regions", [HYBRID], indirect=True)
def test_full_prefix_hits_independently_of_the_swa_window(manager):
    """FULL and SWA are queried separately: a deep FULL prefix hits even where
    the windowed group kept only its trailing window (upstream loads only the
    window)."""
    hashes = block_hashes(8)
    ctx = req_context("a", hashes)
    full_keys, swa_keys = hybrid_keys(hashes)
    store(manager, ctx, full_keys + swa_keys[2:4])  # SWA window at position 1 only
    b = req_context("b", hashes)
    # every FULL position is a hit, regardless of where SWA was kept
    assert lookups(manager, b, full_keys) == [LookupResult.HIT] * 4
    # SWA hits only where its window was stored (chunks 2, 3 == position 1)
    assert (
        lookups(manager, b, swa_keys)
        == [LookupResult.MISS] * 2 + [LookupResult.HIT] * 2 + [LookupResult.MISS] * 4
    )
    spec = manager.prepare_load(full_keys + swa_keys[2:4], b)
    assert len(spec.block_ids) == 5  # 4 FULL slots + 1 SWA slot
    manager.complete_load(full_keys + swa_keys[2:4], b)
    manager.on_schedule_end(STEP)


# ------------------------------------------------------------------ resources


@pytest.mark.parametrize(
    "regions", [{"extra": {"background_evict_ratio": 0}}], indirect=True
)
def test_allocation_failure_when_everything_is_pinned(regions):
    """With the background evictor off, a fully pinned pool refuses to allocate."""
    m = RadixShmemOffloadingManager(regions, regions.geometry)
    n = regions.geometry.require_pool(PoolKind.FULL).num_slots
    hashes = block_hashes(n)
    store(m, req_context("a", hashes), keys_for(hashes))
    b = req_context("b", hashes)
    assert lookups(m, b, keys_for(hashes)) == [LookupResult.HIT] * n  # pins all
    more = block_hashes(1, salt=7)
    assert m.prepare_store(keys_for(more), req_context("c", more)) is None
    m.on_schedule_end(STEP)
    assert m.prepare_store(keys_for(more), req_context("c", more)) is not None
    m.release_all()
    m.index.close()


def test_release_all_returns_reserved_slots(regions):
    m = RadixShmemOffloadingManager(regions, regions.geometry)
    baseline = regions.client.mempool_used()
    hashes = block_hashes(4)
    ctx = req_context("a", hashes)
    out = m.prepare_store(keys_for(hashes), ctx)
    assert out is not None and regions.client.mempool_used() == baseline + 4
    m.release_all()  # a crash here would leak those slots for the region's life
    assert regions.client.mempool_used() == baseline
    m.index.close()


def test_stats_report_shared_pool_usage(manager):
    hashes = block_hashes(3)
    store(manager, req_context("a", hashes), keys_for(hashes))
    stats = manager.get_stats()
    reduced = stats.reduce()
    assert reduced["vllm:kv_offload_radixshmem_slots_used"] >= 3
    assert reduced["vllm:kv_offload_radixshmem_published_blocks"] == 3
    assert reduced["vllm:kv_offload_radixshmem_index_time"] > 0
    # deltas are counters: never negative, even with no activity in between
    for _ in range(50):
        for key, value in manager.get_stats().reduce().items():
            if not key.endswith(("slots_used", "slots_total", "open_leases")):
                assert value >= 0, (key, value)


def test_two_schedulers_share_one_region(regions):
    """What one scheduler process publishes is a hit for another, no RPC."""
    peer_regions = attach_regions(regions.geometry, timeout_s=30, role="peer")
    owner = RadixShmemOffloadingManager(regions, regions.geometry)
    peer = RadixShmemOffloadingManager(peer_regions, peer_regions.geometry)
    try:
        hashes = block_hashes(4)
        store(owner, req_context("a", hashes), keys_for(hashes))
        ctx = req_context("b", hashes)
        assert lookups(peer, ctx, keys_for(hashes)) == [LookupResult.HIT] * 4
        spec = peer.prepare_load(keys_for(hashes), ctx)
        assert np.array_equal(spec.block_ids, np.asarray(spec.block_ids))
        peer.complete_load(keys_for(hashes), ctx)
        peer.on_schedule_end(STEP)
    finally:
        peer.release_all()
        peer.index.close()
        owner.release_all()
        owner.index.close()
        peer_regions.close()
