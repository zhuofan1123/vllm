# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Two DP ranks, one shared index: the point of the whole connector.

Each process runs its own ``RadixShmemConnectorScheduler`` over the same shared
regions, exactly as two DP engine cores would. CPU only -- no CUDA, no model.
Run inside the radixshmem_vllm container with
PYTHONPATH=/raid/zfl/vllm:/raid/zfl/RadixShmem/python.
"""

import multiprocessing as mp
import os
import signal
import time

import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.radixshmem import (
    attach_regions,
    compute_geometry,
    create_regions,
    sentinel_path,
)
from vllm.distributed.kv_transfer.kv_connector.v1.radixshmem.geometry import (
    DEFAULT_SENTINEL_DIR,
)
from vllm.distributed.kv_transfer.kv_connector.v1.radixshmem.scheduler import (
    RadixShmemConnectorScheduler,
)

from .test_scheduler import (
    GPU_BLOCK_SIZE,
    fake_output,
    fake_request,
    make_configs,
    store_and_ack,
)

pytest.importorskip("shmradix")
pytest.importorskip("shmradix._data")

TIMEOUT_S = 60


def _build(tag: str, *, owner: bool, dp_rank: int):
    vllm_config, kv_cache_config, extra = make_configs(tag=tag)
    vllm_config.parallel_config.data_parallel_index = dp_rank
    geometry = compute_geometry(vllm_config, kv_cache_config)
    if owner:
        regions = create_regions(geometry, extra)
    else:
        regions = attach_regions(geometry, timeout_s=TIMEOUT_S, role=f"dp{dp_rank}")
    sched = RadixShmemConnectorScheduler(
        geometry=geometry, regions=regions, vllm_config=vllm_config
    )
    return sched, extra


def _peer_proc(tag, stage, err_q):
    """DP1: attach, store a prefix, then read back what DP0 stored."""
    try:
        sched, _ = _build(tag, owner=False, dp_rank=1)
        try:
            # 1. publish a prefix of our own
            store_and_ack(
                sched, fake_request("p1", 4, salt=1), [0, 1, 2, 3], 4 * GPU_BLOCK_SIZE
            )
            stage.put("peer_stored")
            assert stage.get(timeout=TIMEOUT_S) == "owner_stored"

            # 2. the prefix DP0 published must be a hit here
            n, async_load = sched.get_num_new_matched_tokens(
                fake_request("p2", 4, salt=0), 0
            )
            sched.build_connector_meta(fake_output([]))
            err_q.put(("hit", n, async_load))
        finally:
            sched.release_all()
            sched.manager.close()
            sched.regions.close()
    except BaseException as exc:  # noqa: BLE001
        err_q.put(("error", repr(exc), None))


@pytest.mark.timeout(180)
def test_two_ranks_share_one_index():
    tag = f"share_{os.getpid()}"
    ctx = mp.get_context("spawn")
    stage, err_q = ctx.Queue(), ctx.Queue()

    owner, _ = _build(tag, owner=True, dp_rank=0)
    baseline = owner.manager.client.mempool_used()
    p = ctx.Process(target=_peer_proc, args=(tag, stage, err_q))
    p.start()
    try:
        assert stage.get(timeout=TIMEOUT_S) == "peer_stored"

        # what DP1 published is visible here without any coordination
        peer_prefix = owner._prefix_u64(fake_request("x", 4, salt=1), 4)
        assert owner.manager.lookup(peer_prefix) == 4

        store_and_ack(
            owner, fake_request("o1", 4, salt=0), [8, 9, 10, 11], 4 * GPU_BLOCK_SIZE
        )
        stage.put("owner_stored")

        kind, n, async_load = err_q.get(timeout=TIMEOUT_S)
        assert kind == "hit", n
        assert (n, async_load) == (4 * GPU_BLOCK_SIZE, True)

        p.join(TIMEOUT_S)
        assert p.exitcode == 0
        # 8 published slots, and every pin the peer took has been given back
        assert owner.manager.client.mempool_used() == baseline + 8
        assert owner.manager.allocate(owner.geometry.num_slots) is not None
    finally:
        if p.is_alive():
            p.kill()
            p.join(5)
        owner.close()


def _crash_proc(tag, ready, err_q):
    """Create the regions, signal ready, then hang until killed."""
    try:
        sched, _ = _build(tag, owner=True, dp_rank=0)
        store_and_ack(sched, fake_request("c", 2, salt=7), [0, 1], 2 * GPU_BLOCK_SIZE)
        ready.put(sched.regions.owner_pid)
        time.sleep(600)
    except BaseException as exc:  # noqa: BLE001
        err_q.put(repr(exc))


@pytest.mark.timeout(180)
def test_new_owner_reclaims_after_the_old_one_is_killed():
    tag = f"crash_{os.getpid()}"
    ctx = mp.get_context("spawn")
    ready, err_q = ctx.Queue(), ctx.Queue()
    p = ctx.Process(target=_crash_proc, args=(tag, ready, err_q))
    p.start()
    try:
        owner_pid = ready.get(timeout=TIMEOUT_S)
        assert owner_pid == p.pid
        os.kill(p.pid, signal.SIGKILL)
        p.join(TIMEOUT_S)

        # the sentinel is still on disk, pointing at a pid that no longer exists
        stale = compute_geometry(*make_configs(tag=tag)[:2])
        assert os.path.exists(sentinel_path(stale, DEFAULT_SENTINEL_DIR))

        sched, _ = _build(tag, owner=True, dp_rank=0)
        try:
            # reclaimed, not reused: the killed owner's entries are gone
            assert (
                sched.manager.lookup(sched._prefix_u64(fake_request("c", 2, salt=7), 2))
                == 0
            )
            assert sched.manager.client.mempool_used() == 0
        finally:
            sched.close()
    finally:
        if p.is_alive():
            p.kill()
            p.join(5)
    assert err_q.empty(), err_q.get()


def _live_owner_proc(tag, ready, stop, err_q):
    try:
        sched, _ = _build(tag, owner=True, dp_rank=0)
        ready.put(sched.regions.owner_pid)
        stop.get(timeout=600)
        sched.close()
        err_q.put("ok")
    except BaseException as exc:  # noqa: BLE001
        err_q.put(repr(exc))


@pytest.mark.timeout(180)
def test_second_owner_refuses_a_live_region():
    tag = f"live_{os.getpid()}"
    ctx = mp.get_context("spawn")
    ready, stop, err_q = ctx.Queue(), ctx.Queue(), ctx.Queue()
    p = ctx.Process(target=_live_owner_proc, args=(tag, ready, stop, err_q))
    p.start()
    try:
        ready.get(timeout=TIMEOUT_S)
        _, _, extra = make_configs(tag=tag)
        geometry = compute_geometry(*make_configs(tag=tag)[:2])
        # blowing away a region another engine core is actively using would
        # corrupt it, so this has to fail loudly
        with pytest.raises(RuntimeError, match="already"):
            create_regions(geometry, extra)
    finally:
        stop.put("stop")
        p.join(TIMEOUT_S)
        if p.is_alive():
            p.kill()
            p.join(5)
    assert err_q.get(timeout=TIMEOUT_S) == "ok"
