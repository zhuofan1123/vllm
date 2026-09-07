# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Geometry derivation and the owner / attacher handshake (CPU only)."""

import json
import multiprocessing as mp
import os
import time

import pytest

from vllm.v1.kv_offload.radixshmem.bootstrap import (
    SENTINEL_VERSION,
    attach_regions,
    check_none_hash,
    create_regions,
    open_regions,
    sentinel_path,
)
from vllm.v1.kv_offload.radixshmem.geometry import (
    GeometryMismatch,
    PoolKind,
    SlotGeometry,
    compute_geometry,
)

from .utils import (
    NUM_LAYERS,
    PAGE_BYTES,
    group,
    hybrid_groups,
    make_offloading_config,
    unique_tag,
)

# ------------------------------------------------------------------ geometry


def test_geometry_math():
    g = compute_geometry(make_offloading_config(tag="geo", tp_size=2))
    assert g.blocks_per_chunk == 1 and g.tokens_per_chunk == 16
    [full] = g.pools
    assert full.kind == PoolKind.FULL
    assert g.groups[0].sub_blocks == 1
    assert full.slice_bytes == PAGE_BYTES * NUM_LAYERS
    assert full.num_slices == 2
    assert full.slot_bytes == PAGE_BYTES * NUM_LAYERS * 2
    assert full.slot_stride == full.slot_bytes  # already 4K-aligned
    assert full.num_slots == (8 << 20) // full.slot_stride
    assert not g.replicated and g.swa_window_blocks == 0
    assert g.pool_mask == PoolKind.FULL.mask


def test_geometry_blocks_per_chunk():
    g = compute_geometry(
        make_offloading_config(tag="geo", tp_size=4, blocks_per_chunk=4)
    )
    full = g.require_pool(PoolKind.FULL)
    assert g.blocks_per_chunk == 4 and g.tokens_per_chunk == 64
    assert g.groups[0].sub_blocks == 4
    assert full.slice_bytes == PAGE_BYTES * NUM_LAYERS * 4
    assert full.slot_bytes == full.slice_bytes * 4


def test_geometry_hybrid_pools():
    """SWA chunks finer than FULL chunks share one SWA slot per FULL position."""
    g = compute_geometry(
        make_offloading_config(tag="geo", tp_size=1, groups=hybrid_groups())
    )
    assert g.tokens_per_chunk == 32
    full, swa = g.pools
    assert (full.kind, swa.kind) == (PoolKind.FULL, PoolKind.SWA)
    assert g.groups[1].ratio == 2 and g.groups[1].hashes_per_chunk == 1
    assert g.groups[0].ratio == 1 and g.groups[0].hashes_per_chunk == 2
    assert g.groups[1].sub_blocks == 2  # two 16-token SWA blocks per position
    assert swa.slice_bytes == PAGE_BYTES * 2
    assert full.slice_bytes == 2 * PAGE_BYTES
    assert g.swa_window_blocks == 1  # 32-token window == one FULL position
    assert g.pool_mask == PoolKind.FULL.mask | PoolKind.SWA.mask
    # both pools sized from the same budget by their default shares (0.70 / 0.25)
    assert full.num_slots > swa.num_slots > 0
    assert g.total_data_bytes <= 8 << 20


def test_geometry_pool_slot_overrides():
    g = compute_geometry(
        make_offloading_config(
            tag="geo", tp_size=1, groups=hybrid_groups(), extra={"swa_slots": 8}
        )
    )
    assert g.require_pool(PoolKind.SWA).num_slots == 8
    full = g.require_pool(PoolKind.FULL)
    assert (
        full.num_slots * full.slot_stride + 8 * g.require_pool(PoolKind.SWA).slot_stride
        <= 8 << 20
    )


def test_geometry_replicated_stores_one_slice():
    g = compute_geometry(
        make_offloading_config(tag="geo", tp_size=4, extra={"replicated_kv": True})
    )
    full = g.require_pool(PoolKind.FULL)
    assert g.replicated and full.num_slices == 1
    assert full.slot_bytes == full.slice_bytes
    # vLLM's own detection is the default
    g2 = compute_geometry(
        make_offloading_config(tag="geo", tp_size=4, replicated_layout=True)
    )
    assert g2.replicated
    g3 = compute_geometry(
        make_offloading_config(
            tag="geo", tp_size=4, replicated_layout=True, extra={"replicated_kv": "0"}
        )
    )
    assert not g3.replicated and g3.require_pool(PoolKind.FULL).num_slices == 4


def test_geometry_pads_to_slot_align():
    g = compute_geometry(
        make_offloading_config(
            tag="geo", tp_size=1, groups=[group(bytes_per_block=1000)]
        )
    )
    full = g.require_pool(PoolKind.FULL)
    assert full.slot_bytes == 1000
    assert full.slot_stride == 4096


def test_geometry_rejects_unsupported_parallelism():
    from dataclasses import replace

    config = make_offloading_config(tag="geo")
    with pytest.raises(ValueError, match="pipeline_parallel_size"):
        compute_geometry(replace(config, parallel=replace(config.parallel, pp_size=2)))
    with pytest.raises(ValueError, match="world_size"):
        compute_geometry(
            replace(config, parallel=replace(config.parallel, world_size=4))
        )


def test_geometry_rejects_misaligned_windowed_chunks():
    groups = [group(32, kind="full"), group(24, kind="swa", window=24)]
    with pytest.raises(ValueError, match="does not divide"):
        compute_geometry(make_offloading_config(tag="geo", groups=groups))


def test_geometry_requires_cpu_bytes():
    from dataclasses import replace

    config = make_offloading_config(tag="geo")
    extra = dict(config.extra_config)
    del extra["cpu_bytes_to_use"]
    with pytest.raises(ValueError, match="cpu_bytes_to_use"):
        compute_geometry(replace(config, extra_config=extra))


def test_check_same_reports_every_differing_field():
    a = compute_geometry(make_offloading_config(tag="geo", tp_size=2))
    b = SlotGeometry.from_dict({**a.to_dict(), "blocks_per_chunk": 4, "tp_size": 8})
    with pytest.raises(GeometryMismatch) as exc:
        a.check_same(b, what="peer")
    msg = str(exc.value)
    assert "blocks_per_chunk" in msg and "tp_size" in msg


def test_geometry_round_trips_through_the_sentinel_dict():
    a = compute_geometry(
        make_offloading_config(tag="geo", tp_size=1, groups=hybrid_groups())
    )
    assert SlotGeometry.from_dict(a.to_dict()) == a


# ---------------------------------------------------------------- handshake


def _owner_proc(tag, ready_evt, done_evt, err_q):
    """Create the regions, stamp a pattern into slice 0 of slot 3, wait, exit."""
    try:
        config = make_offloading_config(tag=tag, tp_size=2)
        g = compute_geometry(config)
        regions = create_regions(g, dict(config.extra_config))
        slice_bytes = g.require_pool(PoolKind.FULL).slice_bytes
        regions.store.write_slot(3, bytes([0xAA]) * slice_bytes)
        err_q.put(("owner_pid", os.getpid()))
        ready_evt.set()
        assert done_evt.wait(60)
        regions.close()
    except Exception as e:  # surface into the parent
        err_q.put(("error", f"{type(e).__name__}: {e}"))
        ready_evt.set()


def _spawn_owner(tag):
    ctx = mp.get_context("spawn")
    ready, done = ctx.Event(), ctx.Event()
    q = ctx.Queue()
    p = ctx.Process(target=_owner_proc, args=(tag, ready, done, q))
    p.start()
    assert ready.wait(60), "owner never became ready"
    kind, payload = q.get(timeout=5)
    assert kind == "owner_pid", payload
    return p, done, payload


def test_owner_publishes_and_attacher_reads(request):
    tag = unique_tag(request)
    p, done, owner_pid = _spawn_owner(tag)
    try:
        config = make_offloading_config(tag=tag, tp_size=2)
        g = compute_geometry(config)
        regions = attach_regions(g, timeout_s=30, role="dp1")
        assert not regions.is_owner
        assert regions.owner_pid == owner_pid
        full = g.require_pool(PoolKind.FULL)
        assert regions.store.num_slots == full.num_slots
        assert regions.store.slot_bytes == full.slot_stride

        slot = regions.store.read_slot(3)
        assert slot[: full.slice_bytes] == b"\xaa" * full.slice_bytes
        # the other TP slice was never written
        assert slot[full.slice_bytes :] == b"\x00" * (
            full.slot_stride - full.slice_bytes
        )

        # attacher writes the TP1 slice; owner-side memory is the same memory
        mv = regions.store.slot_view(3)
        mv[full.slice_bytes : full.slot_bytes] = b"\xbb" * full.slice_bytes
        del mv
        assert regions.store.read_slot(3)[full.slice_bytes : full.slot_bytes] == (
            b"\xbb" * full.slice_bytes
        )
        regions.close()
    finally:
        done.set()
        p.join(30)
    assert p.exitcode == 0
    # owner removed the sentinel on the way out
    path = sentinel_path(g, "/dev/shm")
    deadline = time.time() + 5
    while os.path.exists(path) and time.time() < deadline:
        time.sleep(0.05)
    assert not os.path.exists(path)


def test_auto_role_attaches_to_a_live_owner(request):
    """A second instance's dp0 scheduler must join, not steal, the live region."""
    tag = unique_tag(request)
    p, done, owner_pid = _spawn_owner(tag)
    try:
        config = make_offloading_config(tag=tag, tp_size=2)
        g = compute_geometry(config)
        regions = open_regions(
            g, dict(config.extra_config), prefer_owner=True, role="dp0", timeout_s=30
        )
        assert not regions.is_owner and regions.owner_pid == owner_pid
        # the owner's data is intact: nothing was unlinked and re-created
        slice_bytes = g.require_pool(PoolKind.FULL).slice_bytes
        assert regions.store.read_slot(3)[:slice_bytes] == b"\xaa" * slice_bytes
        regions.close()

        with pytest.raises(RuntimeError, match="live pid"):
            open_regions(
                g,
                dict(config.extra_config),
                shm_role="owner",
                prefer_owner=True,
                role="dp0",
            )
    finally:
        done.set()
        p.join(30)


def test_auto_role_creates_when_nobody_owns(request):
    tag = unique_tag(request)
    config = make_offloading_config(tag=tag, tp_size=2)
    g = compute_geometry(config)
    regions = open_regions(
        g, dict(config.extra_config), prefer_owner=True, role="dp0", timeout_s=5
    )
    try:
        assert regions.is_owner and regions.owner_pid == os.getpid()
        assert os.path.exists(sentinel_path(g, "/dev/shm"))
    finally:
        regions.close()


def test_hybrid_pools_are_created_and_attached(request):
    """SWA pool exists in store and index, and the attacher sees the same."""
    tag = unique_tag(request)
    config = make_offloading_config(tag=tag, tp_size=1, groups=hybrid_groups())
    g = compute_geometry(config)
    owner = create_regions(g, dict(config.extra_config))
    try:
        assert owner.store.pool_mask == g.pool_mask
        assert (
            owner.client.swa_mempool_total() == g.require_pool(PoolKind.SWA).num_slots
        )
        assert owner.client.component_config.swa_window_blocks == 1
        peer = attach_regions(g, timeout_s=10, role="peer")
        try:
            assert (
                peer.store.pool(PoolKind.SWA).num_slots
                == owner.store.pool(PoolKind.SWA).num_slots
            )
            # the SWA pool is a separate range: writing a SWA slot leaves FULL alone
            swa_bytes = g.require_pool(PoolKind.SWA).slot_stride
            peer.store.write_slot(0, bytes([0x5A]) * swa_bytes, PoolKind.SWA)
            assert owner.store.read_slot(0, PoolKind.SWA)[:8] == b"\x5a" * 8
            assert owner.store.read_slot(0)[:8] == b"\x00" * 8
        finally:
            peer.close()
    finally:
        owner.close()


def test_attach_geometry_mismatch_fails_closed(request):
    tag = unique_tag(request)
    p, done, _ = _spawn_owner(tag)
    try:
        # same names, different chunking -> must refuse to attach
        config = make_offloading_config(tag=tag, tp_size=2, blocks_per_chunk=4)
        g = compute_geometry(config)
        with pytest.raises(GeometryMismatch):
            attach_regions(g, timeout_s=10, role="dp1")
    finally:
        done.set()
        p.join(30)


def test_attach_times_out_without_owner(request):
    g = compute_geometry(make_offloading_config(tag=unique_tag(request)))
    t0 = time.monotonic()
    with pytest.raises(TimeoutError):
        attach_regions(g, timeout_s=1.0, role="dp1")
    assert time.monotonic() - t0 >= 1.0


def test_stale_sentinel_from_dead_owner_waits_then_times_out(request, tmp_path):
    g = compute_geometry(make_offloading_config(tag=unique_tag(request)))
    path = sentinel_path(g, str(tmp_path))
    with open(path, "w") as f:
        json.dump(
            {"version": SENTINEL_VERSION, "pid": 2**22, "geometry": g.to_dict()}, f
        )
    # a restart legitimately sees the old owner's sentinel first, so this must
    # wait for a replacement and only fail at the deadline
    with pytest.raises(TimeoutError, match="dead owner"):
        attach_regions(g, sentinel_dir=str(tmp_path), timeout_s=1, role="dp1")


def test_reused_pid_does_not_pass_for_a_live_owner(request, tmp_path):
    """A sentinel naming a live pid with another start time is a leftover."""
    from vllm.v1.kv_offload.radixshmem.bootstrap import _owner_alive

    g = compute_geometry(make_offloading_config(tag=unique_tag(request)))
    payload = {"version": SENTINEL_VERSION, "pid": os.getpid(), "pid_start": 1}
    assert not _owner_alive(payload)
    payload["pid_start"] = None
    assert _owner_alive(payload)  # legacy sentinel without start time
    config = make_offloading_config(tag=unique_tag(request), tp_size=2)
    path = sentinel_path(g, str(tmp_path))
    with open(path, "w") as f:
        json.dump({**payload, "pid_start": 1, "geometry": g.to_dict()}, f)
    regions = open_regions(
        g,
        dict(config.extra_config),
        prefer_owner=True,
        role="dp0",
        sentinel_dir=str(tmp_path),
        timeout_s=5,
    )
    try:
        assert regions.is_owner  # created fresh instead of attaching to junk
    finally:
        regions.close()


def test_geometry_carries_the_model_identity():
    a = compute_geometry(make_offloading_config(tag="geo"))
    from dataclasses import replace

    from vllm.v1.kv_offload.config import OffloadingModelConfig

    other = replace(
        make_offloading_config(tag="geo"),
        model=OffloadingModelConfig(name="another-model", dtype="float16"),
    )
    b = compute_geometry(other)
    assert a.model_fingerprint != b.model_fingerprint
    with pytest.raises(GeometryMismatch, match="model_fingerprint"):
        a.check_same(b, what="peer")


def test_diverging_none_hash_is_rejected(monkeypatch):
    """A scheduler whose NONE_HASH differs can never share a prefix -- fail loud."""
    from vllm.v1.core import kv_cache_utils

    monkeypatch.setattr(kv_cache_utils, "NONE_HASH", b"\x01" * 32, raising=False)
    check_none_hash(("01" * 32), what="the owner")  # agreement is silent
    with pytest.raises(GeometryMismatch, match="PYTHONHASHSEED"):
        check_none_hash("02" * 32, what="the owner")


def test_none_hash_check_skipped_when_unavailable(monkeypatch):
    """Nothing published (older owner) or nothing local -- no false alarm."""
    from vllm.v1.core import kv_cache_utils

    monkeypatch.setattr(kv_cache_utils, "NONE_HASH", b"\x01" * 32, raising=False)
    check_none_hash(None, what="the owner")
    monkeypatch.delattr(kv_cache_utils, "NONE_HASH", raising=False)
    check_none_hash("02" * 32, what="the owner")
