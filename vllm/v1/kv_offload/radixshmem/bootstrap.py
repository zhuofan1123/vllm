# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bring-up of the node's ``RadixServer`` and this process's ``RadixClient``.

One RadixServer per node owns the shared index, the SlotStore and, in a
cluster, the transfer engine that pulls KV from peer nodes; every vLLM process
is a RadixClient of it. The server is started by whichever scheduler gets there
first: schedulers take a file lock, probe the server's gRPC socket and, finding
no live server, start one in-process (sized from their geometry) before letting
go of the lock. Everyone else -- later DP ranks, other instances on the node,
every TP worker -- probes, finds it, and attaches. Workers never start a server:
they are built before any scheduler exists and simply wait for the socket.
"""

import atexit
import contextlib
import fcntl
import os
import time
from collections.abc import Iterator
from dataclasses import fields
from typing import Any

from vllm.logger import init_logger

from .geometry import DEFAULT_ATTACH_TIMEOUT_S, PoolKind, SlotGeometry

logger = init_logger(__name__)

DEFAULT_NAME = "/vllm_kv"
_PROBE_TIMEOUT_S = 3.0


def _shmradix() -> tuple[Any, Any]:
    try:
        import shmradix
        from shmradix import rpc
    except ImportError as e:
        raise ImportError(
            "RadixShmem offloading needs the `shmradix` package (RadixShmem with "
            "its `_data` extension and grpcio)"
        ) from e
    return shmradix, rpc


def _probe(rpc: Any, endpoint: str) -> bool:
    """Whether a RadixServer (starting or ready) answers at ``endpoint``."""
    if endpoint.startswith("unix://") and not os.path.exists(endpoint[7:]):
        return False
    stub = rpc.Stub(endpoint)
    try:
        stub.call("Describe", timeout=_PROBE_TIMEOUT_S)
        return True
    except Exception:
        return False
    finally:
        stub.close()


def _pick(cls: Any, extra: dict[str, Any]) -> dict[str, Any]:
    names = {f.name for f in fields(cls)}
    return {k: v for k, v in extra.items() if k in names}


def server_config(
    geometry: SlotGeometry, extra: dict[str, Any], *, name: str, endpoint: str
) -> Any:
    """``RadixServerConfig`` for a server sized from ``geometry``.

    The slot shape and counts come from the geometry. Any other field of
    shmradix's ``IndexConfig`` / ``DataPlaneConfig`` / ``ClusterConfig`` may be
    set under its own name in ``kv_connector_extra_config`` (``prefault``,
    ``background_evict_ratio``, ``expected_min_nodes``, ``registry``,
    ``rpc_address``, ``transfer_devices``, ...).
    """
    shmradix, _ = _shmradix()
    full = geometry.require_pool(PoolKind.FULL)
    swa = geometry.pool(PoolKind.SWA)
    mamba = geometry.pool(PoolKind.MAMBA)
    index = shmradix.IndexConfig(
        **{
            **_pick(shmradix.IndexConfig, extra),
            "name": name,
            "tokens_per_block": geometry.tokens_per_chunk,
            "full_slots": full.num_slots,
            "swa_slots": swa.num_slots if swa else 0,
            "swa_window_blocks": geometry.swa_window_blocks if swa else 0,
            "mamba_slots": mamba.num_slots if mamba else 0,
        }
    )
    data = shmradix.DataPlaneConfig(
        **{
            **_pick(shmradix.DataPlaneConfig, extra),
            "data_bytes": geometry.total_data_bytes,
            "full_slot_bytes": full.slot_stride,
            "swa_slot_bytes": swa.slot_stride if swa else 0,
            "mamba_slot_bytes": mamba.slot_stride if mamba else 0,
            "slot_align": geometry.slot_align,
        }
    )
    return shmradix.RadixServerConfig(
        index=index,
        data=data,
        cluster=shmradix.ClusterConfig(**_pick(shmradix.ClusterConfig, extra)),
        hugepage_path=str(extra.get("hugepage_path", "")),
        endpoint=endpoint,
    )


def describe_pools(published: dict[str, Any]) -> str:
    return ", ".join(
        f"{kind.upper()} {p['num_slots']} x {p['slot_bytes']} B"
        for kind, p in published["pools"].items()
    )


@contextlib.contextmanager
def _start_lock(name: str) -> Iterator[None]:
    path = f"/dev/shm/{name.strip('/').replace('/', '_')}.vllm.lock"
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)


def start_server(
    geometry: SlotGeometry, extra: dict[str, Any], *, name: str, endpoint: str
) -> Any:
    """Start the node's RadixServer in this process; ready on return."""
    shmradix, _ = _shmradix()
    cfg = server_config(geometry, extra, name=name, endpoint=endpoint)
    # a crashed owner leaves its index region behind; the server recreates the
    # store itself but would map an existing index as is
    with contextlib.suppress(OSError):
        os.unlink(os.path.join(cfg.hugepage_path or "/dev/shm", name.lstrip("/")))
    t0 = time.monotonic()
    server = shmradix.RadixServer(cfg).start()
    atexit.register(server.close)
    logger.info(
        "RadixShmem: started the node's RadixServer %s in %.1f s (%s%s)",
        name,
        time.monotonic() - t0,
        describe_pools(server.geometry),
        f"; cluster rank {server.index.rank()}/{server.index.world_size()}"
        if server.index.is_distributed()
        else "",
    )
    return server


def open_client(
    geometry: SlotGeometry,
    extra: dict[str, Any],
    *,
    role: str,
    may_start: bool,
    read_only: bool = False,
    timeout_s: float = DEFAULT_ATTACH_TIMEOUT_S,
) -> tuple[Any, Any]:
    """Connect this process to the node's RadixServer, starting one if allowed.

    Returns ``(client, server)``: ``server`` is the in-process RadixServer when
    this call started it (the cache lives exactly as long as it does), else
    ``None``. With ``may_start=False`` the call waits up to ``timeout_s`` for
    a server to appear. ``read_only`` attaches the light form of the client
    (index and SlotStore only, no cluster plane), which is all a worker needs.
    """
    shmradix, rpc = _shmradix()
    name = str(extra.get("name", DEFAULT_NAME))
    endpoint = str(extra.get("endpoint") or rpc.endpoint_for(name))
    deadline = time.monotonic() + timeout_s
    server = None
    if may_start:
        with _start_lock(name):
            if not _probe(rpc, endpoint):
                server = start_server(geometry, extra, name=name, endpoint=endpoint)
    else:
        next_log = time.monotonic() + 30.0
        while not _probe(rpc, endpoint):
            now = time.monotonic()
            if now >= deadline:
                raise TimeoutError(
                    f"RadixShmem: {role} found no RadixServer at {endpoint} within "
                    f"{timeout_s:.0f} s"
                )
            if now >= next_log:
                logger.info(
                    "RadixShmem: %s waiting for a RadixServer at %s", role, endpoint
                )
                next_log = now + 30.0
            time.sleep(0.5)
    try:
        client = shmradix.RadixClient(
            name,
            endpoint=endpoint,
            timeout_s=max(1.0, deadline - time.monotonic()),
            max_outstanding=int(extra.get("max_inflight_fetches", 8)),
            local_read_only=read_only,
        )
    except BaseException:
        if server is not None:
            server.close()
        raise
    logger.info(
        "RadixShmem: %s attached %s (%s)", role, name, describe_pools(client.geometry)
    )
    return client, server
