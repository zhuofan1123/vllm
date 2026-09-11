# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Who starts the node's RadixServer, who attaches, and what they must agree on.

CPU only, standalone servers (no etcd, no RDMA): the first ``open_client`` per
name starts a server in this process, every later one attaches to it.
"""

import threading

import pytest

from vllm.v1.kv_offload.radixshmem.bootstrap import open_client, server_config
from vllm.v1.kv_offload.radixshmem.geometry import (
    GeometryMismatch,
    PoolKind,
    compute_geometry,
)

from .utils import (
    PAGE_BYTES,
    TOKENS_PER_BLOCK,
    geometry_for,
    group,
    hybrid_groups,
    make_offloading_config,
    open_test_client,
    unique_tag,
)


def test_first_starts_the_server_and_the_rest_attach(request):
    config = make_offloading_config(tag=unique_tag(request))
    first, server = open_test_client(config)
    assert server is not None
    try:
        second, none = open_test_client(config)
        worker, none2 = open_test_client(config, may_start=False, read_only=True)
        assert none is None and none2 is None
        g1, g2, g3 = (geometry_for(config).adopt(c) for c in (first, second, worker))
        assert g1 == g2 == g3
        full = g1.require_pool(PoolKind.FULL)
        assert (
            full.num_slots == geometry_for(config).require_pool(PoolKind.FULL).num_slots
        )
        # one store: bytes written through one client are read by the others
        first.slot_view(3)[:4] = b"vllm"
        assert bytes(worker.slot_view(3)[:4]) == b"vllm"
        second.close()
        worker.close()
    finally:
        first.close()
        server.close()


def test_worker_waits_for_a_server(request):
    config = make_offloading_config(tag=unique_tag(request))
    got: list = []

    def wait_then_attach():
        got.append(open_test_client(config, may_start=False, read_only=True))

    waiter = threading.Thread(target=wait_then_attach)
    waiter.start()
    waiter.join(1.0)
    assert waiter.is_alive()  # nothing to attach to yet
    scheduler, server = open_test_client(config)
    try:
        waiter.join(20.0)
        assert got, "the worker did not attach once the server came up"
        client, none = got[0]
        assert none is None and client.local_read_only
        client.close()
    finally:
        scheduler.close()
        server.close()


def test_worker_times_out_without_a_server(request):
    config = make_offloading_config(tag=unique_tag(request))
    with pytest.raises(TimeoutError):
        open_client(
            compute_geometry(config),
            dict(config.extra_config),
            role="tp0",
            may_start=False,
            timeout_s=1.0,
        )


def test_another_model_is_refused_by_the_geometry(request):
    tag = unique_tag(request)
    owner_config = make_offloading_config(tag=tag)
    scheduler, server = open_test_client(owner_config)
    try:
        # same name, different chunking: the server's index counts other blocks
        other = make_offloading_config(tag=tag, tokens_per_block=2 * TOKENS_PER_BLOCK)
        client, none = open_test_client(other)
        assert none is None
        with pytest.raises(GeometryMismatch, match="tokens per chunk"):
            geometry_for(other).adopt(client)
        client.close()
        # same chunking, wider slots: the server's slots are too narrow
        bigger = make_offloading_config(
            tag=tag, groups=[group(bytes_per_block=8 * PAGE_BYTES)]
        )
        client, _ = open_test_client(bigger)
        with pytest.raises(GeometryMismatch, match="slot"):
            geometry_for(bigger).adopt(client)
        client.close()
    finally:
        scheduler.close()
        server.close()


def test_server_config_takes_the_geometry_and_named_fields(request):
    config = make_offloading_config(
        tag=unique_tag(request),
        groups=hybrid_groups(),
        extra={
            "expected_min_nodes": 3,
            "registry": "etcd://h:2379",
            "transfer_devices": ["mlx5_0"],
            "background_evict_ratio": 0.0,
        },
    )
    g = compute_geometry(config)
    cfg = server_config(
        g, dict(config.extra_config), name="/x", endpoint="unix:///tmp/x"
    )
    assert cfg.index.tokens_per_block == g.tokens_per_chunk
    assert cfg.index.full_slots == g.require_pool(PoolKind.FULL).num_slots
    assert cfg.index.swa_slots == g.require_pool(PoolKind.SWA).num_slots
    assert cfg.index.swa_window_blocks == g.swa_window_blocks
    assert cfg.index.background_evict_ratio == 0.0
    assert cfg.data.full_slot_bytes == g.require_pool(PoolKind.FULL).slot_stride
    assert cfg.data.data_bytes == g.total_data_bytes
    assert cfg.data.prefault is False
    assert cfg.data.transfer_devices == ["mlx5_0"]
    assert cfg.cluster.expected_min_nodes == 3 and cfg.distributed
    assert cfg.cluster.registry == "etcd://h:2379"
    # the server resolves exactly the counts the geometry asked for
    planned = cfg.resolved_geometry()["pools"]
    assert planned["full"]["num_slots"] == cfg.index.full_slots
    assert planned["swa"]["num_slots"] == cfg.index.swa_slots
