# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""``OffloadingSpec`` for a node-shared CPU KV cache backed by RadixShmem.

Every DP rank -- and every vLLM instance on the node that points at the same
server name -- shares one radix index *and* one slot store, so a prefix
offloaded by any of them is a real cache hit for all of them. This is the
difference from ``CPUOffloadingSpec``, where each engine keeps a private pool.
With the server configured as a cluster member, a prefix held by another node
is pulled over RDMA and becomes a local hit too.

Bootstrap: the first scheduler to get there starts the node's ``RadixServer``
in-process, sized from the geometry it derives from config; every other
scheduler and every TP worker connects to it as a ``RadixClient``, adopts the
slot counts it published, and refuses to run if its own slot shape does not fit.

The SlotStore has one pool per attention kind (FULL / SWA / MAMBA), sized from
the KV cache groups; SWA and Mamba chunks go through the index's own components
rather than sharing the full-attention slots (see ``geometry.py``).

Selected with ``--kv-offloading-backend radixshmem``; ``--kv-offloading-size``
is then the budget for the *whole node*, not per rank or per instance. Optional
knobs in ``kv_connector_extra_config``:

* ``name`` / ``endpoint``: the server's name (its shm and socket names derive
  from it; change it to run two independent caches on one host) and, for TCP,
  its gRPC endpoint;
* ``replicated_kv``: store one TP slice instead of ``tp_size`` when every rank
  holds identical KV bytes (MLA / MQA models); defaults to vLLM's detection;
* ``full_slots`` / ``swa_slots`` / ``mamba_slots``: slot counts per pool;
  unset pools split ``cpu_bytes_to_use`` by a default share;
* ``slot_align``, ``hugepage_path``, ``attach_timeout_s``, and any field of
  shmradix's ``IndexConfig`` / ``DataPlaneConfig`` / ``ClusterConfig`` under
  its own name (``prefault``, ``background_evict_ratio``,
  ``expected_min_nodes``, ``registry``, ``rpc_address``, ``transfer_devices``,
  ...) for the server this process may start;
* ``remote_lookup_min_blocks`` / ``p2p_fetch_timeout_s`` /
  ``max_inflight_fetches``: when and how a short local hit pulls from a peer.

Limits: ``PP=1``, no context parallelism, and ``reset_prefix_cache`` does not
clear the shared index.
"""

import os
from typing import Any

from vllm.logger import init_logger
from vllm.v1.kv_offload.base import (
    CanonicalKVCaches,
    OffloadingCounterMetadata,
    OffloadingGaugeMetadata,
    OffloadingManager,
    OffloadingMetricMetadata,
    OffloadingSpec,
    OffloadingWorker,
)
from vllm.v1.kv_offload.config import OffloadingConfig

from .bootstrap import open_client
from .geometry import DEFAULT_ATTACH_TIMEOUT_S, SlotGeometry, compute_geometry
from .manager import (
    RadixShmemMetrics,
    RadixShmemOffloadingManager,
    RemoteFetchPolicy,
)
from .worker import RadixShmemOffloadingWorker

logger = init_logger(__name__)


def _warn_if_block_hashes_are_per_process(config: OffloadingConfig) -> None:
    """vLLM seeds NONE_HASH from os.urandom for xxhash unless PYTHONHASHSEED is set.

    Checked from config because the spec is built before ``init_none_hash``
    runs in the engine core. Every process sharing the region must hash a
    prefix identically or nothing can ever be shared.
    """
    hash_algo = str(config.cache.prefix_caching_hash_algo)
    if hash_algo.startswith("xxhash") and os.getenv("PYTHONHASHSEED") is None:
        logger.warning(
            "RadixShmem: --prefix-caching-hash-algo %s is not collision resistant, "
            "so with PYTHONHASHSEED unset vLLM seeds NONE_HASH from os.urandom and "
            "every scheduler hashes the same tokens differently -- nothing can be "
            "shared. Set PYTHONHASHSEED to the same fixed value in every process, "
            "or use --prefix-caching-hash-algo sha256.",
            hash_algo,
        )


class RadixShmemOffloadingSpec(OffloadingSpec):
    @classmethod
    def build_metric_definitions(
        cls, extra_config: dict[str, Any]
    ) -> dict[str, OffloadingMetricMetadata]:
        return {
            RadixShmemMetrics.SLOTS_USED: OffloadingGaugeMetadata(
                documentation="Slots of the node-shared RadixShmem pool in use "
                "(published or reserved), across every attached engine.",
            ),
            RadixShmemMetrics.SLOTS_TOTAL: OffloadingGaugeMetadata(
                documentation="Total slots of the node-shared RadixShmem pool.",
            ),
            RadixShmemMetrics.OPEN_LEASES: OffloadingGaugeMetadata(
                documentation="Prefix hits this scheduler currently pins in the "
                "shared index (lookups awaiting a load, loads in flight).",
            ),
            RadixShmemMetrics.HIT_BLOCKS: OffloadingCounterMetadata(
                documentation="Offloaded chunks matched by this scheduler's "
                "prefix lookups in the shared index.",
            ),
            RadixShmemMetrics.PUBLISHED_BLOCKS: OffloadingCounterMetadata(
                documentation="Chunks this scheduler published into the shared "
                "index after their data landed.",
            ),
            RadixShmemMetrics.PUBLISH_REJECTED: OffloadingCounterMetadata(
                documentation="Store completions whose chunks could not be "
                "published (prefix evicted underneath, or index full).",
            ),
            RadixShmemMetrics.ALLOC_FAILURES: OffloadingCounterMetadata(
                documentation="Slot allocations refused because the pool was "
                "fully pinned.",
            ),
            RadixShmemMetrics.INDEX_TIME: OffloadingCounterMetadata(
                documentation="Wall time spent inside the shared index "
                "(query/allocate/insert), in seconds.",
            ),
        }

    def __init__(self, config: OffloadingConfig):
        super().__init__(config)
        extra = dict(config.extra_config or {})
        self.geometry: SlotGeometry = compute_geometry(config)
        # tells the connector worker that only TP rank 0 needs to store
        self.replicated_layout = self.geometry.replicated
        self._attach_timeout_s = float(
            extra.get("attach_timeout_s", DEFAULT_ATTACH_TIMEOUT_S)
        )
        self._extra = extra

    # ---------------------------------------------------------- scheduler

    def get_manager(self) -> OffloadingManager:
        _warn_if_block_hashes_are_per_process(self.config)
        dp_index = self.config.parallel.data_parallel_index
        client, server = open_client(
            self.geometry,
            self._extra,
            role=f"dp{dp_index} scheduler ({self.config.engine_id[:8]})",
            may_start=True,
            timeout_s=self._attach_timeout_s,
        )
        return RadixShmemOffloadingManager(
            client,
            self.geometry.adopt(client),
            enable_events=self.kv_events_config.enable_kv_cache_events,
            server=server,
            remote=RemoteFetchPolicy.from_extra(self._extra),
        )

    # ------------------------------------------------------------- worker

    def get_worker(self, kv_caches: CanonicalKVCaches) -> OffloadingWorker:
        rank = self.config.parallel.rank

        def attach() -> tuple[Any, SlotGeometry, int]:
            client, _ = open_client(
                self.geometry,
                self._extra,
                role=f"tp{rank} worker ({self.config.engine_id[:8]})",
                may_start=False,
                read_only=True,
                timeout_s=self._attach_timeout_s,
            )
            return client, self.geometry.adopt(client), rank

        return RadixShmemOffloadingWorker(attach=attach, kv_caches=kv_caches)
