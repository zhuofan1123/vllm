# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bootstrap of the node-shared RadixShmem regions.

The DP rank 0 scheduler process owns both regions: it derives the geometry from
config (see ``geometry.py``), creates the radix index and the SlotStore, and then
publishes a sentinel file. Every other process -- the remaining DP schedulers and
all TP workers -- waits for that sentinel, attaches, and fails closed if its own
geometry disagrees with the published one.

The sentinel is what makes the handshake safe:

* it is written *last*, so its presence means both regions are ready;
* it carries the owner pid, so a leftover file from a crashed run is detected
  instead of being attached to (``SlotStore::create`` unlinks the stale region,
  so an old mapping stays alive but points at different memory);
* it carries the geometry, so attachers compare against what the owner actually
  built rather than re-deriving and hoping.
"""

import atexit
import json
import os
import time
from dataclasses import dataclass
from typing import Any

from vllm.logger import init_logger

from .geometry import (
    DEFAULT_ATTACH_TIMEOUT_S,
    DEFAULT_SENTINEL_DIR,
    GeometryMismatch,
    SlotGeometry,
)

logger = init_logger(__name__)

SENTINEL_VERSION = 2


def _import_shmradix() -> tuple[Any, Any]:
    try:
        import shmradix
        from shmradix import _data as shmradix_data
    except ImportError as e:
        raise RuntimeError(
            "RadixShmemConnector requires the `shmradix` package (with its "
            "`_data` extension) on PYTHONPATH. Build RadixShmem and add "
            "<RadixShmem>/python to PYTHONPATH."
        ) from e
    return shmradix, shmradix_data


def sentinel_path(geometry: SlotGeometry, sentinel_dir: str) -> str:
    base = geometry.data_shm_name.lstrip("/").replace("/", "_")
    return os.path.join(sentinel_dir, f"{base}.ready")


def _pid_alive(pid: int) -> bool:
    return pid > 0 and os.path.exists(f"/proc/{pid}")


def _region_file(name: str, hugepage_path: str) -> str:
    base = name.lstrip("/")
    return os.path.join(hugepage_path or "/dev/shm", base)


def _unlink_quiet(path: str) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass


@dataclass
class SharedRegions:
    """Handle on the shared index + SlotStore, owned or attached."""

    geometry: SlotGeometry
    client: Any  # shmradix.RadixClient
    store: Any  # shmradix._data.SlotStore
    is_owner: bool
    owner_pid: int
    server: Any = None  # shmradix.RadixServer, owner only
    _sentinel: str | None = None
    _closed: bool = False

    def close(self) -> None:
        """Tear down in the only safe order: sentinel, store, index."""
        if self._closed:
            return
        self._closed = True
        if self.is_owner and self._sentinel:
            _unlink_quiet(self._sentinel)
        self.store = None
        self.client = None
        self.server = None

    @property
    def data_ptr(self) -> int:
        """Address of slot 0 in this process (the registerable range)."""
        import numpy as np

        return np.frombuffer(self.store.data_view(), dtype=np.uint8).ctypes.data


def _make_shm_config(geometry: SlotGeometry, extra: dict[str, Any]) -> Any:
    shmradix, _ = _import_shmradix()
    # The node pool is not grown on demand and overflowing it is a segfault, not
    # an exception, so size it explicitly rather than relying on the default.
    max_nodes = int(extra.get("max_nodes") or (2 * geometry.num_slots + 1024))
    if max_nodes >= 2**32:
        raise ValueError(f"max_nodes ({max_nodes}) exceeds the uint32 node id space")
    cfg = shmradix.ShmConfig()
    cfg.max_nodes = max_nodes
    cfg.max_blocks = geometry.num_slots
    cfg.block_size = geometry.offloaded_block_size
    cfg.hugepage_path = geometry.hugepage_path
    if "data_pool_ratio" in extra:
        cfg.data_pool_ratio = float(extra["data_pool_ratio"])
    return cfg


def create_regions(
    geometry: SlotGeometry,
    extra: dict[str, Any],
    *,
    sentinel_dir: str = DEFAULT_SENTINEL_DIR,
    force_reclaim: bool = False,
) -> SharedRegions:
    """Owner path: create both regions, then publish the sentinel."""
    shmradix, shmradix_data = _import_shmradix()

    sentinel = sentinel_path(geometry, sentinel_dir)
    existing = _read_sentinel(sentinel)
    if existing is not None and _pid_alive(existing["pid"]) and not force_reclaim:
        raise RuntimeError(
            f"RadixShmem: {sentinel} already claims regions "
            f"{geometry.index_shm_name} / {geometry.data_shm_name}, owned by "
            f"live pid {existing['pid']}. Another vLLM instance is using these "
            "names; pick different index_shm_name/data_shm_name or stop it."
        )

    # Stale leftovers: unlink before create so nothing maps the old region.
    _unlink_quiet(sentinel)
    _unlink_quiet(_region_file(geometry.index_shm_name, geometry.hugepage_path))
    shmradix_data.SlotStore.destroy(geometry.data_shm_name, geometry.hugepage_path)

    shm_cfg = _make_shm_config(geometry, extra)
    server = shmradix.RadixServer(geometry.index_shm_name, shm_cfg)
    client = shmradix.RadixClient(server)

    store_cfg = shmradix_data.SlotStoreConfig()
    store_cfg.name = geometry.data_shm_name
    store_cfg.num_slots = geometry.num_slots
    store_cfg.slot_bytes = geometry.slot_bytes
    store_cfg.slot_align = geometry.slot_align
    store_cfg.hugepage_path = geometry.hugepage_path
    store_cfg.prefault = bool(extra.get("prefault", True))
    t0 = time.monotonic()
    store = shmradix_data.SlotStore.create(store_cfg)
    create_s = time.monotonic() - t0

    _check_attached(geometry, client, store)

    regions = SharedRegions(
        geometry=geometry,
        client=client,
        store=store,
        is_owner=True,
        owner_pid=os.getpid(),
        server=server,
        _sentinel=sentinel,
    )
    _write_sentinel(sentinel, geometry)
    atexit.register(regions.close)

    if os.getenv("PYTHONHASHSEED") is None:
        logger.warning(
            "RadixShmem: PYTHONHASHSEED is not set, so vLLM seeds NONE_HASH from "
            "os.urandom and every DP scheduler hashes the same tokens "
            "differently -- nothing can be shared between ranks. Set it to the "
            "same fixed value in every process."
        )

    logger.info(
        "RadixShmem owner ready: index=%s (max_blocks=%d, max_nodes=%d, "
        "block_size=%d) data=%s (%d slots x %d B stride = %.2f GiB, "
        "created in %.1fs) sentinel=%s",
        geometry.index_shm_name,
        geometry.num_slots,
        shm_cfg.max_nodes,
        geometry.offloaded_block_size,
        geometry.data_shm_name,
        geometry.num_slots,
        geometry.slot_stride,
        geometry.total_data_bytes / 2**30,
        create_s,
        sentinel,
    )
    return regions


def attach_regions(
    geometry: SlotGeometry,
    *,
    sentinel_dir: str = DEFAULT_SENTINEL_DIR,
    timeout_s: float = DEFAULT_ATTACH_TIMEOUT_S,
    role: str = "attacher",
    check_block_hashes: bool = False,
) -> SharedRegions:
    """Non-owner path: wait for the sentinel, attach, verify.

    ``check_block_hashes`` is for schedulers: only they compute BlockHashes, so
    only they can disagree with the owner about what a prefix hashes to.
    """
    shmradix, shmradix_data = _import_shmradix()

    sentinel = sentinel_path(geometry, sentinel_dir)
    deadline = time.monotonic() + timeout_s

    published, owner_pid, owner_none_hash = _wait_for_sentinel(sentinel, deadline, role)
    geometry.check_same(published, what="the DP rank 0 owner")
    if check_block_hashes:
        check_none_hash(owner_none_hash, what="the DP rank 0 owner")

    client = _retry_until(
        lambda: shmradix.RadixClient(geometry.index_shm_name),
        deadline,
        what=f"attach index {geometry.index_shm_name}",
    )
    store = _retry_until(
        lambda: shmradix_data.SlotStore.attach(
            geometry.data_shm_name, geometry.hugepage_path, 5000
        ),
        deadline,
        what=f"attach SlotStore {geometry.data_shm_name}",
    )

    _check_attached(geometry, client, store)
    if not _pid_alive(owner_pid):
        raise RuntimeError(
            f"RadixShmem: owner pid {owner_pid} died while {role} was attaching; "
            "the regions it published may already have been unlinked"
        )

    logger.info(
        "RadixShmem %s attached: index=%s data=%s (%d slots x %d B, owner pid %d)",
        role,
        geometry.index_shm_name,
        geometry.data_shm_name,
        store.num_slots,
        store.slot_bytes,
        owner_pid,
    )
    return SharedRegions(
        geometry=geometry,
        client=client,
        store=store,
        is_owner=False,
        owner_pid=owner_pid,
    )


def _check_attached(geometry: SlotGeometry, client: Any, store: Any) -> None:
    """Cross-check the two regions against each other and the geometry."""
    if store.num_slots != geometry.num_slots:
        raise GeometryMismatch(
            f"SlotStore has {store.num_slots} slots, geometry says "
            f"{geometry.num_slots}"
        )
    if store.slot_bytes != geometry.slot_stride:
        raise GeometryMismatch(
            f"SlotStore stride is {store.slot_bytes} B, geometry says "
            f"{geometry.slot_stride} B (slot_align={geometry.slot_align})"
        )
    if store.slot_bytes < geometry.slot_bytes:
        raise GeometryMismatch(
            f"SlotStore stride {store.slot_bytes} B cannot hold one slot "
            f"({geometry.slot_bytes} B)"
        )
    mempool_total = client.mempool_total()
    if mempool_total != geometry.num_slots:
        raise GeometryMismatch(
            f"index mempool has {mempool_total} slots, SlotStore has "
            f"{geometry.num_slots}; slot ids would run off the data region"
        )
    if client.block_size() != geometry.offloaded_block_size:
        raise GeometryMismatch(
            f"index block_size is {client.block_size()} tokens, geometry says "
            f"{geometry.offloaded_block_size}"
        )
    if geometry.hugepage_path and not store.is_hugepage:
        raise GeometryMismatch(
            f"hugepage_path={geometry.hugepage_path!r} was requested but the "
            "SlotStore fell back to plain shared memory"
        )


def none_hash_fingerprint() -> str | None:
    """vLLM's ``NONE_HASH``, the root every BlockHash chain hangs off.

    ``init_none_hash`` seeds it from ``os.urandom`` when ``PYTHONHASHSEED`` is
    unset, so two DP schedulers then hash identical tokens to different
    BlockHashes -- the shared tree silently degenerates into one private subtree
    per rank. Returns None before the engine has initialized it.
    """
    from vllm.v1.core import kv_cache_utils

    none_hash = getattr(kv_cache_utils, "NONE_HASH", None)
    return bytes(none_hash).hex() if none_hash is not None else None


def check_none_hash(published: str | None, *, what: str) -> None:
    """Fail closed when this process would hash blocks differently."""
    local = none_hash_fingerprint()
    if local is None or published is None or local == published:
        return
    raise GeometryMismatch(
        "RadixShmem: this process derives a different vLLM NONE_HASH than "
        f"{what} ({local[:16]}... vs {published[:16]}...), so identical tokens "
        "would hash to different BlockHashes and no prefix could ever be shared "
        "between DP ranks. NONE_HASH is randomized per process unless "
        "PYTHONHASHSEED is set -- set it to the same fixed value in every "
        "process that attaches this region."
    )


def _write_sentinel(path: str, geometry: SlotGeometry) -> None:
    payload = {
        "version": SENTINEL_VERSION,
        "pid": os.getpid(),
        "geometry": geometry.to_dict(),
        "none_hash": none_hash_fingerprint(),
    }
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _read_sentinel(path: str) -> dict[str, Any] | None:
    try:
        with open(path) as f:
            payload = json.load(f)
    except (OSError, ValueError):
        return None
    if payload.get("version") != SENTINEL_VERSION:
        return None
    return payload


def _wait_for_sentinel(
    path: str, deadline: float, role: str
) -> tuple[SlotGeometry, int, str | None]:
    next_log = time.monotonic() + 30.0
    while True:
        payload = _read_sentinel(path)
        if payload is not None:
            pid = int(payload["pid"])
            if not _pid_alive(pid):
                raise RuntimeError(
                    f"RadixShmem: sentinel {path} names owner pid {pid}, which "
                    "is not running. This is a leftover from a crashed run -- "
                    "delete the sentinel and the shm regions, then restart."
                )
            return (
                SlotGeometry.from_dict(payload["geometry"]),
                pid,
                payload.get("none_hash"),
            )
        now = time.monotonic()
        if now > deadline:
            raise TimeoutError(
                f"RadixShmem: {role} timed out waiting for {path}. The DP rank 0 "
                "scheduler is responsible for creating it; check its logs."
            )
        if now > next_log:
            next_log = now + 30.0
            logger.info("RadixShmem %s still waiting for %s", role, path)
        time.sleep(0.05)


def _retry_until(fn: Any, deadline: float, *, what: str) -> Any:
    last: Exception | None = None
    while True:
        try:
            return fn()
        except Exception as e:  # region may not be visible yet
            last = e
        if time.monotonic() > deadline:
            raise TimeoutError(f"RadixShmem: timed out trying to {what}") from last
        time.sleep(0.05)
