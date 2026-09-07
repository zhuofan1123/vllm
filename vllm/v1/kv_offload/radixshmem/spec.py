# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""``OffloadingSpec`` for a node-shared CPU KV cache backed by RadixShmem.

Every DP rank -- and every vLLM instance on the node that points at the same
shared-memory names -- shares one radix index *and* one slot store, so a prefix
offloaded by any of them is a real cache hit for all of them. This is the
difference from ``CPUOffloadingSpec``, where each engine keeps a private pool.

Bootstrap: the first scheduler to start (DP rank 0 of the first instance)
derives the geometry from config, creates both shared regions, and publishes a
sentinel; every other scheduler and every TP worker waits for that sentinel and
attaches, refusing to run if its own geometry disagrees.

The SlotStore has one pool per attention kind (FULL / SWA / MAMBA), sized from
the KV cache groups; SWA and Mamba chunks go through the index's own components
rather than sharing the full-attention slots (see ``geometry.py``).

Selected with ``--kv-offloading-backend radixshmem``; ``--kv-offloading-size``
is then the budget for the *whole node*, not per rank or per instance. Optional
knobs in ``kv_connector_extra_config``:

* ``index_shm_name`` / ``data_shm_name``: change both to run two independent
  caches on one host;
* ``shm_role``: ``auto`` (default: attach to a live owner, else create),
  ``owner`` or ``attach``;
* ``replicated_kv``: store one TP slice instead of ``tp_size`` when every rank
  holds identical KV bytes (MLA / MQA models); defaults to vLLM's detection;
* ``full_slots`` / ``swa_slots`` / ``mamba_slots``: slot counts per pool;
  unset pools get as many slots as the FULL pool within ``cpu_bytes_to_use``;
* ``block_size`` / ``blocks_per_chunk``, ``slot_align``, ``hugepage_path``,
  ``max_nodes``, ``data_pool_ratio``, ``background_evict_ratio``,
  ``sentinel_dir``, ``attach_timeout_s``, ``force_reclaim``, ``prefault``.

Limits: ``PP=1``, no context parallelism, one node, and ``reset_prefix_cache``
does not clear the shared index.
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

from .bootstrap import SharedRegions, attach_regions, open_regions
from .geometry import (
    DEFAULT_ATTACH_TIMEOUT_S,
    DEFAULT_SENTINEL_DIR,
    SlotGeometry,
    compute_geometry,
)
from .manager import RadixShmemMetrics, RadixShmemOffloadingManager
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
        self._shm_role = str(extra.get("shm_role", "auto"))
        self._sentinel_dir = str(extra.get("sentinel_dir", DEFAULT_SENTINEL_DIR))
        self._attach_timeout_s = float(
            extra.get("attach_timeout_s", DEFAULT_ATTACH_TIMEOUT_S)
        )
        self._extra = extra

    # ---------------------------------------------------------- scheduler

    def get_manager(self) -> OffloadingManager:
        _warn_if_block_hashes_are_per_process(self.config)
        parallel = self.config.parallel
        dp_index = parallel.data_parallel_index
        regions = open_regions(
            self.geometry,
            self._extra,
            shm_role=self._shm_role,
            # dp rank 0 of an instance creates unless a live owner exists;
            # the other dp ranks always wait for one
            prefer_owner=(dp_index == 0),
            role=f"dp{dp_index} scheduler ({self.config.engine_id[:8]})",
            sentinel_dir=self._sentinel_dir,
            timeout_s=self._attach_timeout_s,
            check_block_hashes=True,
        )
        return RadixShmemOffloadingManager(
            regions,
            self.geometry,
            enable_events=self.kv_events_config.enable_kv_cache_events,
        )

    # ------------------------------------------------------------- worker

    def get_worker(self, kv_caches: CanonicalKVCaches) -> OffloadingWorker:
        rank = self.config.parallel.rank

        def attach() -> tuple[SharedRegions, int]:
            regions = attach_regions(
                self.geometry,
                sentinel_dir=self._sentinel_dir,
                timeout_s=self._attach_timeout_s,
                role=f"tp{rank} worker ({self.config.engine_id[:8]})",
                adopt_published=True,
            )
            return regions, rank

        return RadixShmemOffloadingWorker(attach=attach, kv_caches=kv_caches)
