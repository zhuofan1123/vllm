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
    SlotGeometry,
    compute_geometry,
)

from .utils import NUM_LAYERS, PAGE_BYTES, make_offloading_config, unique_tag

# ------------------------------------------------------------------ geometry


def test_geometry_math():
    g = compute_geometry(make_offloading_config(tag="geo", tp_size=2))
    assert g.blocks_per_chunk == 1
    assert g.worker_bytes_per_block == PAGE_BYTES * NUM_LAYERS
    assert g.slice_bytes == PAGE_BYTES * NUM_LAYERS
    assert g.num_slices == 2
    assert g.slot_bytes == PAGE_BYTES * NUM_LAYERS * 2
    assert g.slot_stride == g.slot_bytes  # already 4K-aligned
    assert g.num_slots == (8 << 20) // g.slot_stride
    assert not g.replicated


def test_geometry_blocks_per_chunk():
    g = compute_geometry(
        make_offloading_config(tag="geo", tp_size=4, blocks_per_chunk=4)
    )
    assert g.blocks_per_chunk == 4
    assert g.tokens_per_chunk == 64
    assert g.slice_bytes == PAGE_BYTES * NUM_LAYERS * 4
    assert g.slot_bytes == g.slice_bytes * 4


def test_geometry_replicated_stores_one_slice():
    g = compute_geometry(
        make_offloading_config(tag="geo", tp_size=4, extra={"replicated_kv": True})
    )
    assert g.replicated and g.num_slices == 1
    assert g.slot_bytes == g.slice_bytes
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
    assert not g3.replicated and g3.num_slices == 4


def test_geometry_pads_to_slot_align():
    g = compute_geometry(
        make_offloading_config(tag="geo", tp_size=1, worker_bytes_per_block=1000)
    )
    assert g.slot_bytes == 1000
    assert g.slot_stride == 4096


def test_geometry_rejects_unsupported_parallelism():
    config = make_offloading_config(tag="geo")
    bad_pp = OffloadingParallelConfigPatch(config, pp_size=2)
    with pytest.raises(ValueError, match="pipeline_parallel_size"):
        compute_geometry(bad_pp)
    bad_world = OffloadingParallelConfigPatch(config, world_size=4)
    with pytest.raises(ValueError, match="world_size"):
        compute_geometry(bad_world)


def OffloadingParallelConfigPatch(config, **fields):
    from dataclasses import replace

    return replace(config, parallel=replace(config.parallel, **fields))


def test_geometry_requires_cpu_bytes():
    config = make_offloading_config(tag="geo")
    extra = dict(config.extra_config)
    del extra["cpu_bytes_to_use"]
    from dataclasses import replace

    with pytest.raises(ValueError, match="cpu_bytes_to_use"):
        compute_geometry(replace(config, extra_config=extra))


def test_check_same_reports_every_differing_field():
    a = compute_geometry(make_offloading_config(tag="geo", tp_size=2))
    b = SlotGeometry.from_dict({**a.to_dict(), "blocks_per_chunk": 4, "tp_size": 8})
    with pytest.raises(GeometryMismatch) as exc:
        a.check_same(b, what="peer")
    msg = str(exc.value)
    assert "blocks_per_chunk" in msg and "tp_size" in msg


# ---------------------------------------------------------------- handshake


def _owner_proc(tag, ready_evt, done_evt, err_q):
    """Create the regions, stamp a pattern into slice 0 of slot 3, wait, exit."""
    try:
        config = make_offloading_config(tag=tag, tp_size=2)
        g = compute_geometry(config)
        regions = create_regions(g, dict(config.extra_config))
        regions.store.write_slot(3, bytes([0xAA]) * g.slice_bytes)
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
        assert regions.store.num_slots == g.num_slots
        assert regions.store.slot_bytes == g.slot_stride

        slot = regions.store.read_slot(3)
        assert slot[: g.slice_bytes] == b"\xaa" * g.slice_bytes
        # the other TP slice was never written
        assert slot[g.slice_bytes :] == b"\x00" * (g.slot_stride - g.slice_bytes)

        # attacher writes the TP1 slice; owner-side memory is the same memory
        mv = regions.store.slot_view(3)
        mv[g.slice_bytes : g.slot_bytes] = b"\xbb" * g.slice_bytes
        del mv
        assert regions.store.read_slot(3)[g.slice_bytes : g.slot_bytes] == (
            b"\xbb" * g.slice_bytes
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
        assert regions.store.read_slot(3)[: g.slice_bytes] == b"\xaa" * g.slice_bytes
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


def test_stale_sentinel_from_dead_owner_is_rejected(request, tmp_path):
    g = compute_geometry(make_offloading_config(tag=unique_tag(request)))
    path = sentinel_path(g, str(tmp_path))
    with open(path, "w") as f:
        json.dump(
            {"version": SENTINEL_VERSION, "pid": 2**22, "geometry": g.to_dict()}, f
        )
    with pytest.raises(RuntimeError, match="not running"):
        attach_regions(g, sentinel_dir=str(tmp_path), timeout_s=5, role="dp1")


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
