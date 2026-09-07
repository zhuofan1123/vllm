# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bootstrap of the node-shared RadixShmem regions.

Exactly one process on the node owns both regions: it derives the geometry from
config (see ``geometry.py``), creates the radix index and the SlotStore, and then
publishes a sentinel file. Every other process -- the remaining DP schedulers,
all TP workers, and the schedulers/workers of *other vLLM instances* that share
the cache -- waits for that sentinel, attaches, and fails closed if its own
geometry disagrees with the published one.

The sentinel is what makes the handshake safe:

* it is written *last*, so its presence means both regions are ready;
* it carries the owner pid, so a leftover file from a crashed run is detected
  instead of being attached to (``SlotStore::create`` unlinks the stale region,
  so an old mapping stays alive but points at different memory);
* it carries the geometry, so attachers compare against what the owner actually
  built rather than re-deriving and hoping.

Ownership is decided under a file lock (``<sentinel>.lock``) so two instances
starting at the same time cannot both create: the second one blocks, then sees
the first one's live sentinel and attaches instead.
"""

import atexit
import contextlib
import fcntl
import json
import os
import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from vllm.logger import init_logger

from .geometry import (
    DEFAULT_ATTACH_TIMEOUT_S,
    DEFAULT_SENTINEL_DIR,
    GeometryMismatch,
    PoolKind,
    SlotGeometry,
)

logger = init_logger(__name__)

SENTINEL_VERSION = 4

SHM_ROLES = ("auto", "owner", "attach")


def _import_shmradix() -> tuple[Any, Any]:
    try:
        import shmradix
        from shmradix import _data as shmradix_data
    except ImportError as e:
        raise RuntimeError(
            "RadixShmem offloading requires the `shmradix` package (with its "
            "`_data` extension) on PYTHONPATH. Build RadixShmem and add "
            "<RadixShmem>/python to PYTHONPATH."
        ) from e
    return shmradix, shmradix_data


def sentinel_path(geometry: SlotGeometry, sentinel_dir: str) -> str:
    base = geometry.data_shm_name.lstrip("/").replace("/", "_")
    return os.path.join(sentinel_dir, f"{base}.ready")


def _pid_alive(pid: int) -> bool:
    return pid > 0 and os.path.exists(f"/proc/{pid}")


def _proc_start_time(pid: int) -> int | None:
    """Kernel start time (clock ticks) of ``pid``; None if it is gone."""
    try:
        with open(f"/proc/{pid}/stat") as f:
            stat = f.read()
    except OSError:
        return None
    # field 22, counted after the ")" that ends the (possibly spaced) comm
    fields = stat[stat.rindex(")") + 2 :].split()
    return int(fields[19])


def _owner_alive(payload: dict[str, Any]) -> bool:
    """Whether the process that wrote a sentinel is still the one running.

    Pids get reused: after a hard kill, a fresh process may sit on the old
    owner's pid within minutes, so liveness is pid *and* start time. Sentinels
    written before start times were recorded fall back to the pid alone.
    """
    pid = int(payload["pid"])
    start = _proc_start_time(pid)
    if start is None:
        return False
    expected = payload.get("pid_start")
    return expected is None or int(expected) == start


def _region_file(name: str, hugepage_path: str) -> str:
    base = name.lstrip("/")
    return os.path.join(hugepage_path or "/dev/shm", base)


def _unlink_quiet(path: str) -> None:
    with contextlib.suppress(OSError):
        os.unlink(path)


@contextlib.contextmanager
def _ownership_lock(sentinel: str) -> Iterator[None]:
    """Serialize the create-or-attach decision across processes on the node."""
    fd = os.open(f"{sentinel}.lock", os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        with contextlib.suppress(OSError):
            fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


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

    def data_view(self) -> memoryview:
        """Writable view over the whole registerable data range (all pools)."""
        return self.store.data_view()

    def pool_view(self, kind: int) -> memoryview:
        """Writable view over one pool (num_slots x slot_stride)."""
        return self.store.pool_view(int(kind))

    @property
    def data_ptr(self) -> int:
        """Address of slot 0 in this process (the registerable range)."""
        import numpy as np

        return np.frombuffer(self.data_view(), dtype=np.uint8).ctypes.data


def _make_shm_config(geometry: SlotGeometry, extra: dict[str, Any]) -> Any:
    shmradix, _ = _import_shmradix()
    # The node pool is not grown on demand and overflowing it is a segfault, not
    # an exception, so size it explicitly rather than relying on the default.
    full = geometry.require_pool(PoolKind.FULL)
    max_nodes = int(extra.get("max_nodes") or (2 * full.num_slots + 1024))
    if max_nodes >= 2**32:
        raise ValueError(f"max_nodes ({max_nodes}) exceeds the uint32 node id space")
    cfg = shmradix.ShmConfig()
    cfg.max_nodes = max_nodes
    cfg.max_blocks = full.num_slots
    cfg.block_size = geometry.tokens_per_chunk
    cfg.hugepage_path = geometry.hugepage_path
    # the index's three slot-id spaces mirror the store's three pools
    cfg.component_mask = geometry.pool_mask
    swa = geometry.pool(PoolKind.SWA)
    if swa is not None:
        cfg.swa_max_blocks = swa.num_slots
        cfg.swa_window_blocks = geometry.swa_window_blocks
    mamba = geometry.pool(PoolKind.MAMBA)
    if mamba is not None:
        cfg.mamba_max_slots = mamba.num_slots
        cfg.mamba_state_bytes = mamba.slot_stride
    if "data_pool_ratio" in extra:
        cfg.data_pool_ratio = float(extra["data_pool_ratio"])
    if "background_evict_ratio" in extra:
        # free fraction the index's background evictor keeps; 0 disables it
        cfg.background_evict_ratio = float(extra["background_evict_ratio"])
    return cfg


def open_regions(
    geometry: SlotGeometry,
    extra: dict[str, Any],
    *,
    shm_role: str = "auto",
    prefer_owner: bool,
    role: str,
    sentinel_dir: str = DEFAULT_SENTINEL_DIR,
    timeout_s: float = DEFAULT_ATTACH_TIMEOUT_S,
    check_block_hashes: bool = False,
) -> SharedRegions:
    """Create or attach, according to ``shm_role``.

    ``auto`` (default): attach if a live owner already published this region,
    otherwise create when ``prefer_owner`` (the DP rank 0 scheduler of an
    instance) or wait for a creator when not (other DP ranks). ``owner`` /
    ``attach`` force one path; ``owner`` refuses to steal a live region unless
    ``force_reclaim`` is set in ``extra``.
    """
    if shm_role not in SHM_ROLES:
        raise ValueError(f"shm_role must be one of {SHM_ROLES}, got {shm_role!r}")
    force_reclaim = bool(extra.get("force_reclaim", False))

    if shm_role == "owner":
        return create_regions(
            geometry, extra, sentinel_dir=sentinel_dir, force_reclaim=force_reclaim
        )
    if shm_role == "attach" or not prefer_owner:
        return attach_regions(
            geometry,
            sentinel_dir=sentinel_dir,
            timeout_s=timeout_s,
            role=role,
            check_block_hashes=check_block_hashes,
        )

    sentinel = sentinel_path(geometry, sentinel_dir)
    with _ownership_lock(sentinel):
        existing = _read_sentinel(sentinel)
        live_owner = existing is not None and _owner_alive(existing)
        if not live_owner:
            return create_regions(
                geometry, extra, sentinel_dir=sentinel_dir, force_reclaim=False
            )
    logger.info(
        "RadixShmem %s: regions %s already owned by pid %s, attaching",
        role,
        geometry.data_shm_name,
        existing["pid"] if existing else "?",
    )
    return attach_regions(
        geometry,
        sentinel_dir=sentinel_dir,
        timeout_s=timeout_s,
        role=role,
        check_block_hashes=check_block_hashes,
    )


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
    if existing is not None and _owner_alive(existing) and not force_reclaim:
        raise RuntimeError(
            f"RadixShmem: {sentinel} already claims regions "
            f"{geometry.index_shm_name} / {geometry.data_shm_name}, owned by "
            f"live pid {existing['pid']}. Another vLLM instance is using these "
            "names; pick different index_shm_name/data_shm_name, use "
            "shm_role=auto/attach to share them, or stop it."
        )

    # Stale leftovers: unlink before create so nothing maps the old region.
    _unlink_quiet(sentinel)
    _unlink_quiet(_region_file(geometry.index_shm_name, geometry.hugepage_path))
    shmradix_data.SlotStore.destroy(geometry.data_shm_name, geometry.hugepage_path)

    shm_cfg = _make_shm_config(geometry, extra)
    server = shmradix.RadixServer(geometry.index_shm_name, shm_cfg)
    client = shmradix.RadixClient(server)

    def pool_cfg(kind: PoolKind):
        pool = geometry.pool(kind)
        if pool is None:
            return shmradix_data.SlotPoolConfig()
        # the stride is what the store rounds to anyway; passing it keeps the
        # index's mamba_state_bytes and the store's slot_bytes identical
        return shmradix_data.SlotPoolConfig(pool.num_slots, pool.slot_stride)

    store_cfg = shmradix_data.SlotStoreConfig(
        name=geometry.data_shm_name,
        full=pool_cfg(PoolKind.FULL),
        swa=pool_cfg(PoolKind.SWA),
        mamba=pool_cfg(PoolKind.MAMBA),
        slot_align=geometry.slot_align,
        hugepage_path=geometry.hugepage_path,
        prefault=bool(extra.get("prefault", True)),
    )
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

    logger.info(
        "RadixShmem owner ready: index=%s (max_nodes=%d, tokens_per_chunk=%d, "
        "swa_window_blocks=%d) data=%s (%s; %d slice(s); %.2f GiB, created in "
        "%.1fs) sentinel=%s",
        geometry.index_shm_name,
        shm_cfg.max_nodes,
        geometry.tokens_per_chunk,
        geometry.swa_window_blocks,
        geometry.data_shm_name,
        describe_pools(geometry),
        geometry.num_slices,
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

    published, owner_pid, owner_start, owner_none_hash = _wait_for_sentinel(
        sentinel, deadline, role
    )
    geometry.check_same(published, what="the region owner")
    if check_block_hashes:
        check_none_hash(owner_none_hash, what="the region owner")

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
    if not _owner_alive({"pid": owner_pid, "pid_start": owner_start}):
        raise RuntimeError(
            f"RadixShmem: owner pid {owner_pid} died while {role} was attaching; "
            "the regions it published may already have been unlinked"
        )

    logger.info(
        "RadixShmem %s attached: index=%s data=%s (%s, owner pid %d)",
        role,
        geometry.index_shm_name,
        geometry.data_shm_name,
        describe_pools(geometry),
        owner_pid,
    )
    return SharedRegions(
        geometry=geometry,
        client=client,
        store=store,
        is_owner=False,
        owner_pid=owner_pid,
    )


def describe_pools(geometry: SlotGeometry) -> str:
    return ", ".join(
        f"{PoolKind(p.kind).name} {p.num_slots} x {p.slot_stride} B"
        for p in geometry.pools
    )


def _check_attached(geometry: SlotGeometry, client: Any, store: Any) -> None:
    """Cross-check the two regions against each other and the geometry."""
    shmradix, _ = _import_shmradix()
    if int(store.pool_mask) != geometry.pool_mask:
        raise GeometryMismatch(
            f"SlotStore has pools {int(store.pool_mask):#x}, geometry says "
            f"{geometry.pool_mask:#x}"
        )
    for pool in geometry.pools:
        kind = PoolKind(pool.kind)
        actual = store.pool(int(kind))
        if actual.num_slots != pool.num_slots:
            raise GeometryMismatch(
                f"SlotStore {kind.name} pool has {actual.num_slots} slots, geometry "
                f"says {pool.num_slots}"
            )
        if actual.slot_bytes != pool.slot_stride:
            raise GeometryMismatch(
                f"SlotStore {kind.name} stride is {actual.slot_bytes} B, geometry "
                f"says {pool.slot_stride} B (slot_align={geometry.slot_align})"
            )
    try:
        # index slot-id spaces vs store pools, pool by pool
        shmradix.check_pools_aligned(client, store)
    except ValueError as e:
        raise GeometryMismatch(f"RadixShmem: {e}") from e
    if client.block_size() != geometry.tokens_per_chunk:
        raise GeometryMismatch(
            f"index block_size is {client.block_size()} tokens, geometry says "
            f"{geometry.tokens_per_chunk}"
        )
    if geometry.hugepage_path and not store.is_hugepage:
        raise GeometryMismatch(
            f"hugepage_path={geometry.hugepage_path!r} was requested but the "
            "SlotStore fell back to plain shared memory"
        )


def none_hash_fingerprint() -> str | None:
    """vLLM's ``NONE_HASH``, the root every BlockHash chain hangs off.

    ``init_none_hash`` seeds it from ``os.urandom`` for xxhash when
    ``PYTHONHASHSEED`` is unset, so two schedulers then hash identical tokens
    to different BlockHashes -- the shared tree silently degenerates into one
    private subtree per process. Returns None before the engine has
    initialized it.
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
        "would hash to different BlockHashes and no prefix could ever be shared. "
        "With xxhash NONE_HASH is randomized per process unless PYTHONHASHSEED "
        "is set -- set it to the same fixed value in every process that attaches "
        "this region, or use --prefix-caching-hash-algo sha256."
    )


def _write_sentinel(path: str, geometry: SlotGeometry) -> None:
    payload = {
        "version": SENTINEL_VERSION,
        "pid": os.getpid(),
        "pid_start": _proc_start_time(os.getpid()),
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
) -> tuple[SlotGeometry, int, int | None, str | None]:
    next_log = time.monotonic() + 30.0
    stale_pid: int | None = None
    while True:
        payload = _read_sentinel(path)
        if payload is not None:
            pid = int(payload["pid"])
            if _owner_alive(payload):
                return (
                    SlotGeometry.from_dict(payload["geometry"]),
                    pid,
                    payload.get("pid_start"),
                    payload.get("none_hash"),
                )
            # A dead owner's sentinel is what a restart finds first: workers
            # start attaching before the new scheduler has replaced it, so keep
            # waiting rather than failing; only a timeout makes it an error.
            if stale_pid != pid:
                stale_pid = pid
                logger.info(
                    "RadixShmem %s: %s names dead owner pid %d; waiting for a "
                    "new owner to replace it",
                    role,
                    path,
                    pid,
                )
        now = time.monotonic()
        if now > deadline:
            if stale_pid is not None:
                raise TimeoutError(
                    f"RadixShmem: {role} timed out waiting for {path}, which "
                    f"still names dead owner pid {stale_pid}. This is a leftover "
                    "from a crashed run -- delete the sentinel and the shm "
                    "regions, or start the owning scheduler."
                )
            raise TimeoutError(
                f"RadixShmem: {role} timed out waiting for {path}. The owning "
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
