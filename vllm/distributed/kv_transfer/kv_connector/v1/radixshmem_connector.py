# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A node-shared CPU KV cache backed by RadixShmem.

Every DP rank on the node shares one radix index *and* one slot store, so a
prefix offloaded by any rank is a real cache hit for all of them -- unlike the
stock ``OffloadingConnector``, where each DP rank keeps a private CPU pool.

Bootstrap: the DP rank 0 scheduler derives the geometry from config, creates
both shared regions, and publishes a sentinel. Every other process waits for
that sentinel and attaches, refusing to run if its own geometry disagrees.

Workers attach lazily, on their first transfer. They are constructed *before*
any scheduler exists (``initialize_from_config`` runs while the engine core is
still sizing the KV cache), so attaching eagerly would deadlock against the
owner that has not been created yet.

Every DP rank must derive the same ``NONE_HASH`` -- the root of every
``BlockHash`` chain. With ``--prefix-caching-hash-algo sha256`` vLLM uses a fixed
seed; with xxhash it seeds from ``os.urandom(32)`` unless ``PYTHONHASHSEED`` is
set, so each scheduler process would hash identical tokens differently and no
prefix could ever be shared. The owner publishes its fingerprint in the sentinel
and the other schedulers refuse to attach on a mismatch.

Selected with ``--kv-offloading-backend radixshmem``; ``--kv-offloading-size``
is then the budget for the *whole node*, not per rank. Optional knobs go in
``kv_connector_extra_config``: ``index_shm_name`` / ``data_shm_name`` (change
both to run two instances on one host), ``block_size`` (offloaded block size,
a multiple of the GPU block size), ``slot_align``, ``hugepage_path``,
``max_nodes``, ``sentinel_dir``, ``attach_timeout_s``, ``force_reclaim``.

Limits: ``PP=1``, one node, ``reset_prefix_cache`` does not clear the shared
index, and the instance has to be restarted if the owner process dies.
"""

import os
from collections.abc import Iterable
from typing import Any

import torch

from vllm.config import VllmConfig
from vllm.distributed.kv_events import KVCacheEvent
from vllm.distributed.kv_transfer.kv_connector.v1 import (
    KVConnectorBase_V1,
    KVConnectorRole,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorMetadata,
    KVConnectorWorkerMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import (
    KVConnectorPromMetrics,
    KVConnectorStats,
    PromMetric,
    PromMetricT,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.metrics import (
    OffloadingConnectorStats,
    OffloadPromMetrics,
)
from vllm.distributed.kv_transfer.kv_connector.v1.radixshmem import (
    attach_regions,
    compute_geometry,
    create_regions,
)
from vllm.distributed.kv_transfer.kv_connector.v1.radixshmem.geometry import (
    DEFAULT_ATTACH_TIMEOUT_S,
    DEFAULT_SENTINEL_DIR,
)
from vllm.distributed.kv_transfer.kv_connector.v1.radixshmem.metadata import (
    RadixShmemMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.radixshmem.scheduler import (
    RadixShmemConnectorScheduler,
)
from vllm.distributed.kv_transfer.kv_connector.v1.radixshmem.worker import (
    RadixShmemConnectorWorker,
)
from vllm.forward_context import ForwardContext
from vllm.logger import init_logger
from vllm.v1.attention.backend import AttentionMetadata
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import Request

logger = init_logger(__name__)


def _warn_if_block_hashes_are_per_process(vllm_config: VllmConfig) -> None:
    """vLLM seeds NONE_HASH from os.urandom for xxhash unless PYTHONHASHSEED is set.

    Checked from config rather than from ``kv_cache_utils.NONE_HASH`` because the
    connector is built before ``init_none_hash`` runs in the engine core.
    """
    hash_algo = vllm_config.cache_config.prefix_caching_hash_algo
    if hash_algo.startswith("xxhash") and os.getenv("PYTHONHASHSEED") is None:
        logger.warning(
            "RadixShmem: --prefix-caching-hash-algo %s is not collision resistant, "
            "so with PYTHONHASHSEED unset vLLM seeds NONE_HASH from os.urandom and "
            "every DP scheduler hashes the same tokens differently -- nothing can "
            "be shared between ranks. Set PYTHONHASHSEED to the same fixed value "
            "in every process, or use --prefix-caching-hash-algo sha256.",
            hash_algo,
        )


class RadixShmemConnector(KVConnectorBase_V1):
    @property
    def requires_kv_delivery(self) -> bool:
        # a best-effort cache: a dropped store is only a future miss
        return False

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: KVCacheConfig,
    ):
        super().__init__(vllm_config, role, kv_cache_config)
        assert vllm_config.kv_transfer_config is not None
        extra = vllm_config.kv_transfer_config.kv_connector_extra_config

        self.geometry = compute_geometry(vllm_config, kv_cache_config)
        self._sentinel_dir = str(extra.get("sentinel_dir", DEFAULT_SENTINEL_DIR))
        self._attach_timeout_s = float(
            extra.get("attach_timeout_s", DEFAULT_ATTACH_TIMEOUT_S)
        )
        self.connector_scheduler: RadixShmemConnectorScheduler | None = None
        self.connector_worker: RadixShmemConnectorWorker | None = None

        if role == KVConnectorRole.SCHEDULER:
            # data_parallel_index, not data_parallel_rank: for dense models
            # vLLM makes every DP engine look like DP=1 (rank forced to 0) and
            # only the index still says which engine this is. Using the rank
            # would make every engine try to own the region.
            dp_rank = vllm_config.parallel_config.data_parallel_index
            if dp_rank == 0:
                _warn_if_block_hashes_are_per_process(vllm_config)
                regions = create_regions(
                    self.geometry,
                    extra,
                    sentinel_dir=self._sentinel_dir,
                    force_reclaim=bool(extra.get("force_reclaim", False)),
                )
            else:
                regions = attach_regions(
                    self.geometry,
                    sentinel_dir=self._sentinel_dir,
                    timeout_s=self._attach_timeout_s,
                    role=f"dp{dp_rank} scheduler",
                    check_block_hashes=True,
                )
            self.connector_scheduler = RadixShmemConnectorScheduler(
                geometry=self.geometry, regions=regions, vllm_config=vllm_config
            )
        else:
            self.connector_worker = RadixShmemConnectorWorker(
                attach=self._attach_worker_regions,
                vllm_config=vllm_config,
                kv_cache_config=kv_cache_config,
            )

    def _attach_worker_regions(self):
        from vllm.distributed.parallel_state import get_tensor_model_parallel_rank

        tp_rank = get_tensor_model_parallel_rank()
        regions = attach_regions(
            self.geometry,
            sentinel_dir=self._sentinel_dir,
            timeout_s=self._attach_timeout_s,
            role=f"tp{tp_rank} worker",
        )
        return regions, tp_rank

    # ---------------------------------------------------------------- worker

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        assert self.connector_worker is not None
        self.connector_worker.register_kv_caches(kv_caches)

    def handle_preemptions(self, kv_connector_metadata: KVConnectorMetadata):
        assert self.connector_worker is not None
        assert isinstance(kv_connector_metadata, RadixShmemMetadata)
        self.connector_worker.handle_preemptions(kv_connector_metadata)

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        assert self.connector_worker is not None
        assert isinstance(self._connector_metadata, RadixShmemMetadata)
        self.connector_worker.start_kv_transfers(self._connector_metadata)

    def wait_for_layer_load(self, layer_name: str) -> None:
        pass

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: "AttentionMetadata",
        **kwargs,
    ) -> None:
        pass

    def wait_for_save(self):
        assert self.connector_worker is not None
        assert isinstance(self._connector_metadata, RadixShmemMetadata)
        self.connector_worker.prepare_store_kv(self._connector_metadata)

    def get_finished(self, finished_req_ids: set[str]) -> tuple[set[str], set[str]]:
        assert self.connector_worker is not None
        return self.connector_worker.get_finished(finished_req_ids)

    def build_connector_worker_meta(self) -> KVConnectorWorkerMetadata | None:
        assert self.connector_worker is not None
        return self.connector_worker.build_connector_worker_meta()

    # ------------------------------------------------------------- scheduler

    def get_num_new_matched_tokens(
        self, request: "Request", num_computed_tokens: int
    ) -> tuple[int | None, bool]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.get_num_new_matched_tokens(
            request, num_computed_tokens
        )

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        assert self.connector_scheduler is not None
        return self.connector_scheduler.update_state_after_alloc(
            request, blocks, num_external_tokens
        )

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.build_connector_meta(scheduler_output)

    def update_connector_output(self, connector_output: KVConnectorOutput):
        assert self.connector_scheduler is not None
        self.connector_scheduler.update_connector_output(connector_output)

    def request_finished(
        self, request: "Request", block_ids: list[int]
    ) -> tuple[bool, dict[str, Any] | None]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.request_finished(request, block_ids)

    def take_events(self) -> Iterable[KVCacheEvent]:
        assert self.connector_scheduler is not None
        return self.connector_scheduler.take_events()

    # ----------------------------------------------------------------- misc

    def get_kv_connector_stats(self) -> KVConnectorStats | None:
        if self.connector_worker is None:
            return None
        return self.connector_worker.get_kv_connector_stats()

    @classmethod
    def build_kv_connector_stats(
        cls, data: dict[str, Any] | None = None
    ) -> KVConnectorStats | None:
        return (
            OffloadingConnectorStats(data=data)
            if data is not None
            else OffloadingConnectorStats()
        )

    @classmethod
    def build_prom_metrics(
        cls,
        vllm_config: VllmConfig,
        metric_types: dict[type[PromMetric], type[PromMetricT]],
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[object]],
    ) -> KVConnectorPromMetrics:
        return OffloadPromMetrics(
            vllm_config, metric_types, labelnames, per_engine_labelvalues
        )

    def shutdown(self):
        if self.connector_scheduler is not None:
            self.connector_scheduler.close()
            self.connector_scheduler = None
        if self.connector_worker is not None:
            self.connector_worker.close()
            self.connector_worker = None
