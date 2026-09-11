# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The manager's defer / promote logic around ``RadixClient.pull_async``.

A real index and store in this process stand in for the local tree; the
remote pull itself is a fake ``PullJob`` whose completion publishes the peer's
blocks into that index, exactly what the client's completer does after the
server's RDMA read lands. The real cross-node path is exercised end to end
against radixshmem's own multi-node tests.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from vllm.v1.kv_offload.base import LookupResult, ScheduleEndContext
from vllm.v1.kv_offload.radixshmem.geometry import PoolKind
from vllm.v1.kv_offload.radixshmem.manager import (
    RadixShmemOffloadingManager,
    RemoteFetchPolicy,
)

from .utils import (
    block_hashes,
    geometry_for,
    keys_for,
    make_offloading_config,
    open_test_client,
    req_context,
    unique_tag,
)

STEP = ScheduleEndContext(new_req_ids=(), preempted_req_ids=())


class FakeJob:
    """What ``pull_async`` hands back; ``land()`` plays the completer."""

    def __init__(self, manager, path, local_hit, remote):
        self._manager = manager
        self._path = path
        self.local_hit = local_hit
        self.planned_hit = local_hit + remote
        self._done = remote == 0
        self.cancelled = False
        self.remote_blocks = 0

    def land(self):
        """The peer's blocks arrive: publish them as the client would."""
        n = self.planned_hit - self.local_hit
        slots = self._manager.index.allocate(n, PoolKind.FULL)
        self._manager.index.publish(
            self._path[: self.planned_hit], slots, self.local_hit
        )
        self.remote_blocks = n
        self._done = True

    def done(self):
        return self._done

    def wait(self, timeout=None):
        assert self._done
        return SimpleNamespace(
            common_hit=self.planned_hit if self.remote_blocks else self.local_hit,
            remote_blocks=self.remote_blocks,
            finalize=lambda: None,
        )

    def cancel(self):
        self.cancelled = True


@pytest.fixture
def rig(request):
    config = make_offloading_config(tag=unique_tag(request), tp_size=2)
    client, server = open_test_client(config)
    m = RadixShmemOffloadingManager(client, geometry_for(config).adopt(client))
    # a standalone server has no peers, so the manager disabled pulls; turn
    # them on and route submissions to a fake job the test controls
    m._remote = RemoteFetchPolicy(min_blocks=2, timeout_ms=1000)
    jobs: list[FakeJob] = []
    remote = {"blocks": 4}

    def submit(path):
        local = m.index.lookup(path)
        job = FakeJob(m, path, local, remote["blocks"])
        jobs.append(job)
        return job

    m._submit_remote = submit
    yield SimpleNamespace(manager=m, jobs=jobs, remote=remote)
    m.release_all()
    m.index.close()
    client.close()
    server.close()


def _store(m, ctx, keys):
    out = m.prepare_store(keys, ctx)
    m.complete_store(out.keys_to_store, ctx)
    m.on_schedule_end(STEP)


def test_short_local_hit_defers_then_hits_once_promoted(rig):
    m = rig.manager
    hashes = block_hashes(6)
    _store(m, req_context("a", hashes[:2]), keys_for(hashes[:2]))
    ctx = req_context("b", hashes)
    keys = keys_for(hashes)
    m.on_new_request(ctx)
    assert [m.lookup(k, ctx) for k in keys[:3]] == [
        LookupResult.HIT,
        LookupResult.HIT,
        LookupResult.RETRY,
    ]
    assert m.lookup(keys[3], ctx) == LookupResult.RETRY  # one pull per request
    assert len(rig.jobs) == 1 and m.has_pending_work()
    m.on_schedule_end(STEP)

    rig.jobs[0].land()
    assert [m.lookup(k, ctx) for k in keys] == [LookupResult.HIT] * 6
    assert not m.has_pending_work() and m.num_remote_blocks == 4
    spec = m.prepare_load(keys, ctx)
    assert len(spec.block_ids) == 6
    m.complete_load(keys, ctx)
    m.on_schedule_end(STEP)


def test_gap_below_the_minimum_is_a_plain_miss(rig):
    m = rig.manager
    hashes = block_hashes(3)
    _store(m, req_context("a", hashes[:2]), keys_for(hashes[:2]))
    ctx = req_context("b", hashes)
    m.on_new_request(ctx)
    assert m.lookup(keys_for(hashes)[2], ctx) == LookupResult.MISS
    assert rig.jobs == []


def test_no_peer_has_more_completes_at_once(rig):
    m = rig.manager
    rig.remote["blocks"] = 0
    hashes = block_hashes(5)
    ctx = req_context("b", hashes)
    m.on_new_request(ctx)
    keys = keys_for(hashes)
    assert m.lookup(keys[0], ctx) == LookupResult.MISS
    assert len(rig.jobs) == 1 and not m.has_pending_work()
    m.on_schedule_end(STEP)
    # asked once per admission: the next step does not submit again
    assert m.lookup(keys[0], ctx) == LookupResult.MISS
    assert len(rig.jobs) == 1


def test_finished_request_cancels_its_pull(rig):
    m = rig.manager
    hashes = block_hashes(5)
    ctx = req_context("b", hashes)
    m.on_new_request(ctx)
    assert m.lookup(keys_for(hashes)[0], ctx) == LookupResult.RETRY
    m.on_schedule_end(STEP)
    m.on_request_finished(ctx)
    assert rig.jobs[0].cancelled and not m.has_pending_work()


def test_pulls_are_off_without_peers(request):
    config = make_offloading_config(tag=unique_tag(request))
    client, server = open_test_client(config)
    try:
        m = RadixShmemOffloadingManager(
            client, geometry_for(config).adopt(client), remote=RemoteFetchPolicy()
        )
        hashes = block_hashes(5)
        ctx = req_context("b", hashes)
        m.on_new_request(ctx)
        assert m.lookup(keys_for(hashes)[0], ctx) == LookupResult.MISS
        m.release_all()
        m.index.close()
    finally:
        client.close()
        server.close()


def test_path_is_uint64(rig):
    hashes = block_hashes(2)
    ctx = req_context("a", hashes)
    rig.manager.on_new_request(ctx)
    rig.manager.lookup(keys_for(hashes)[0], ctx)
    assert rig.manager._states["a"].path.dtype == np.uint64
