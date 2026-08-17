# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Geometry derivation and the DP0-owner / attacher handshake.

CPU only: no CUDA, no model, no vLLM engine. Run inside the radixshmem_vllm
container with PYTHONPATH=/raid/zfl/vllm:/raid/zfl/RadixShmem/python.
"""

import multiprocessing as mp
import os
import time
from types import SimpleNamespace

import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.radixshmem import (
    GeometryMismatch,
    SlotGeometry,
    attach_regions,
    compute_geometry,
    create_regions,
    sentinel_path,
)
from vllm.distributed.kv_transfer.kv_connector.v1.radixshmem.bootstrap import (
    SENTINEL_VERSION,
    check_none_hash,
)

shmradix = pytest.importorskip("shmradix")
pytest.importorskip("shmradix._data")

PAGE_BYTES = 4096
NUM_LAYERS = 4


def make_configs(
    *,
    tp_size: int = 2,
    gpu_block_size: int = 16,
    offloaded_block_size: int | None = None,
    cpu_bytes: int = 64 << 20,
    index_name: str = "/rs_test_index",
    data_name: str = "/rs_test_data",
    slot_align: int = 4096,
    num_layers: int = NUM_LAYERS,
    page_bytes: int = PAGE_BYTES,
):
    extra = {
        "index_shm_name": index_name,
        "data_shm_name": data_name,
        "cpu_bytes_to_use": cpu_bytes,
        "slot_align": slot_align,
    }
    if offloaded_block_size is not None:
        extra["block_size"] = offloaded_block_size
    vllm_config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(kv_connector_extra_config=extra),
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=1,
            tensor_parallel_size=tp_size,
            world_size=tp_size,
            data_parallel_index=0,
        ),
    )
    spec = SimpleNamespace(block_size=gpu_block_size, page_size_bytes=page_bytes)
    kv_cache_config = SimpleNamespace(
        kv_cache_groups=[
            SimpleNamespace(
                kv_cache_spec=spec, layer_names=[f"l{i}" for i in range(num_layers)]
            )
        ]
    )
    return vllm_config, kv_cache_config, extra


def test_geometry_math():
    vllm_config, kv_cache_config, _ = make_configs(tp_size=2)
    g = compute_geometry(vllm_config, kv_cache_config)

    assert g.block_size_factor == 1
    assert g.per_rank_block_bytes == PAGE_BYTES * NUM_LAYERS
    assert g.tp_slice_bytes == PAGE_BYTES * NUM_LAYERS
    assert g.slot_bytes == PAGE_BYTES * NUM_LAYERS * 2
    assert g.slot_stride == g.slot_bytes  # already 4K-aligned
    assert g.num_slots == (64 << 20) // g.slot_stride


def test_geometry_block_size_factor():
    vllm_config, kv_cache_config, _ = make_configs(
        tp_size=4, gpu_block_size=16, offloaded_block_size=64
    )
    g = compute_geometry(vllm_config, kv_cache_config)
    assert g.block_size_factor == 4
    assert g.tp_slice_bytes == PAGE_BYTES * NUM_LAYERS * 4
    assert g.slot_bytes == g.tp_slice_bytes * 4


def test_geometry_pads_to_slot_align():
    vllm_config, kv_cache_config, _ = make_configs(
        tp_size=1, page_bytes=1000, num_layers=1, slot_align=4096
    )
    g = compute_geometry(vllm_config, kv_cache_config)
    assert g.slot_bytes == 1000
    assert g.slot_stride == 4096


def test_geometry_rejects_pp():
    vllm_config, kv_cache_config, _ = make_configs()
    vllm_config.parallel_config.pipeline_parallel_size = 2
    with pytest.raises(ValueError, match="pipeline_parallel_size"):
        compute_geometry(vllm_config, kv_cache_config)


def test_geometry_requires_cpu_bytes():
    vllm_config, kv_cache_config, extra = make_configs()
    del extra["cpu_bytes_to_use"]
    with pytest.raises(ValueError, match="cpu_bytes_to_use"):
        compute_geometry(vllm_config, kv_cache_config)


def test_check_same_reports_every_differing_field():
    vllm_config, kv_cache_config, _ = make_configs(tp_size=2)
    a = compute_geometry(vllm_config, kv_cache_config)
    b = SlotGeometry.from_dict({**a.to_dict(), "block_size_factor": 4, "tp_size": 8})
    with pytest.raises(GeometryMismatch) as exc:
        a.check_same(b, what="peer")
    msg = str(exc.value)
    assert "block_size_factor" in msg and "tp_size" in msg


# ---------------------------------------------------------------- handshake


def _owner_proc(names, ready_evt, done_evt, err_q):
    """DP0: create the regions, stamp a pattern, wait, then exit."""
    try:
        index_name, data_name = names
        vllm_config, kv_cache_config, extra = make_configs(
            tp_size=2, cpu_bytes=8 << 20, index_name=index_name, data_name=data_name
        )
        g = compute_geometry(vllm_config, kv_cache_config)
        regions = create_regions(g, extra)
        # TP rank 0 half of slot 3 gets 0xAA, the index gets one entry.
        buf = bytearray(g.tp_slice_bytes)
        for i in range(len(buf)):
            buf[i] = 0xAA
        regions.store.write_slot(3, buf)
        err_q.put(("owner_pid", os.getpid()))
        ready_evt.set()
        assert done_evt.wait(60)
        regions.close()
    except Exception as e:  # surface into the parent
        err_q.put(("error", f"{type(e).__name__}: {e}"))
        ready_evt.set()


def test_owner_publishes_and_attacher_reads(tmp_path):
    ctx = mp.get_context("spawn")
    index_name = f"/rs_bs_index_{os.getpid()}"
    data_name = f"/rs_bs_data_{os.getpid()}"
    ready, done = ctx.Event(), ctx.Event()
    q = ctx.Queue()
    p = ctx.Process(target=_owner_proc, args=((index_name, data_name), ready, done, q))
    p.start()
    try:
        assert ready.wait(60), "owner never became ready"
        kind, payload = q.get(timeout=5)
        assert kind == "owner_pid", payload
        owner_pid = payload

        vllm_config, kv_cache_config, _ = make_configs(
            tp_size=2, cpu_bytes=8 << 20, index_name=index_name, data_name=data_name
        )
        g = compute_geometry(vllm_config, kv_cache_config)
        regions = attach_regions(g, timeout_s=30, role="dp1")
        assert not regions.is_owner
        assert regions.owner_pid == owner_pid
        assert regions.store.num_slots == g.num_slots
        assert regions.store.slot_bytes == g.slot_stride

        slot = regions.store.read_slot(3)
        assert slot[: g.tp_slice_bytes] == b"\xaa" * g.tp_slice_bytes
        # the other TP slice was never written
        assert slot[g.tp_slice_bytes :] == b"\x00" * (
            g.slot_stride - g.tp_slice_bytes
        )

        # attacher writes the TP1 slice; owner-side memory is the same memory
        mv = regions.store.slot_view(3)
        mv[g.tp_slice_bytes : g.slot_bytes] = b"\xbb" * g.tp_slice_bytes
        del mv
        assert regions.store.read_slot(3)[g.tp_slice_bytes : g.slot_bytes] == (
            b"\xbb" * g.tp_slice_bytes
        )

        regions.close()
    finally:
        done.set()
        p.join(30)
    assert p.exitcode == 0
    # owner removed the sentinel on the way out
    vllm_config, kv_cache_config, _ = make_configs(
        index_name=index_name, data_name=data_name
    )
    g = compute_geometry(vllm_config, kv_cache_config)
    deadline = time.time() + 5
    while os.path.exists(sentinel_path(g, "/dev/shm")) and time.time() < deadline:
        time.sleep(0.05)
    assert not os.path.exists(sentinel_path(g, "/dev/shm"))


def test_attach_geometry_mismatch_fails_closed():
    ctx = mp.get_context("spawn")
    index_name = f"/rs_mm_index_{os.getpid()}"
    data_name = f"/rs_mm_data_{os.getpid()}"
    ready, done = ctx.Event(), ctx.Event()
    q = ctx.Queue()
    p = ctx.Process(target=_owner_proc, args=((index_name, data_name), ready, done, q))
    p.start()
    try:
        assert ready.wait(60)
        kind, payload = q.get(timeout=5)
        assert kind == "owner_pid", payload

        # same names, different block_size_factor -> must refuse to attach
        vllm_config, kv_cache_config, _ = make_configs(
            tp_size=2,
            cpu_bytes=8 << 20,
            index_name=index_name,
            data_name=data_name,
            offloaded_block_size=64,
        )
        g = compute_geometry(vllm_config, kv_cache_config)
        with pytest.raises(GeometryMismatch):
            attach_regions(g, timeout_s=10, role="dp1")
    finally:
        done.set()
        p.join(30)


def test_attach_times_out_without_owner():
    vllm_config, kv_cache_config, _ = make_configs(
        index_name=f"/rs_absent_index_{os.getpid()}",
        data_name=f"/rs_absent_data_{os.getpid()}",
    )
    g = compute_geometry(vllm_config, kv_cache_config)
    t0 = time.monotonic()
    with pytest.raises(TimeoutError):
        attach_regions(g, timeout_s=1.0, role="dp1")
    assert time.monotonic() - t0 >= 1.0


def test_stale_sentinel_from_dead_owner_is_rejected(tmp_path):
    vllm_config, kv_cache_config, _ = make_configs(
        index_name=f"/rs_stale_index_{os.getpid()}",
        data_name=f"/rs_stale_data_{os.getpid()}",
    )
    g = compute_geometry(vllm_config, kv_cache_config)
    path = sentinel_path(g, str(tmp_path))
    import json

    with open(path, "w") as f:
        json.dump(
            {"version": SENTINEL_VERSION, "pid": 2**22, "geometry": g.to_dict()}, f
        )
    with pytest.raises(RuntimeError, match="not running"):
        attach_regions(g, sentinel_dir=str(tmp_path), timeout_s=5, role="dp1")


def test_diverging_none_hash_is_rejected(monkeypatch):
    """A DP rank whose NONE_HASH differs can never share a prefix -- fail loud."""
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
