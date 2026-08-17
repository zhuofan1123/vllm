# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker side of the RadixShmem connector.

Same shape as ``OffloadingConnectorWorker`` -- it reuses that class's KV cache
canonicalization wholesale -- with two changes:

* A finished store is reported to the scheduler by store id (via
  ``build_connector_worker_meta``) rather than folded into ``finished_sending``.
  ``finished_sending`` is per request and only fires once the request stops
  generating, which is far too late to publish a slot that other DP ranks could
  be using already.
* The shared regions are attached from a background thread started when the KV
  caches are registered, and joined by the first transfer. Workers are built
  during ``initialize_from_config``, before any scheduler exists, so attaching
  eagerly there would wait forever on the DP rank 0 scheduler that creates the
  regions; doing it inline on the first transfer instead costs ~1 s per 10 GiB
  (cudaHostRegister) with the engine step loop stopped. The thread overlaps
  that with the rest of startup.
"""

import threading
import time
from collections import defaultdict
from collections.abc import Callable

import torch

from vllm.distributed.kv_transfer.kv_connector.v1.metrics import KVConnectorStats
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.metrics import (
    OffloadingConnectorStats,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.worker import (
    OffloadingConnectorWorker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.radixshmem.metadata import (
    RadixShmemMetadata,
    RadixShmemWorkerMetadata,
    ReqId,
)
from vllm.logger import init_logger
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.kv_offload.mediums import CPULoadStoreSpec, GPULoadStoreSpec
from vllm.v1.kv_offload.spec import CanonicalKVCaches
from vllm.v1.kv_offload.worker.radixshmem import RadixShmemOffloadingHandlers
from vllm.v1.kv_offload.worker.worker import OffloadingWorker, TransferSpec

logger = init_logger(__name__)


class RadixShmemConnectorWorker:
    def __init__(self, *, attach: Callable[[], tuple], vllm_config, kv_cache_config):
        self._attach = attach
        self.worker = OffloadingWorker()
        self.handlers: RadixShmemOffloadingHandlers | None = None
        self._kv_caches: CanonicalKVCaches | None = None
        self._attach_lock = threading.Lock()
        self._attach_thread: threading.Thread | None = None
        self._attach_error: BaseException | None = None
        self.kv_connector_stats = OffloadingConnectorStats()

        self._job_counter = 0
        # job_id -> (req_id, store_id or None)
        self._jobs: dict[int, tuple[ReqId, int | None]] = {}
        self._load_job: dict[ReqId, int] = {}
        self._store_jobs = defaultdict[ReqId, set[int]](set)
        self._unsubmitted_store_jobs: list[tuple[int, TransferSpec]] = []
        self._finished_reqs_waiting_for_store: set[ReqId] = set()
        self._completed_stores: dict[int, int] = {}

        # the canonicalization in OffloadingConnectorWorker is generic; borrow it
        # rather than duplicating several hundred lines of backend layout logic
        self._canonicalizer = _Canonicalizer(vllm_config, kv_cache_config, self)

    # -------------------------------------------------- KV cache registration

    def register_kv_caches(
        self, kv_caches: dict[str, torch.Tensor | list[torch.Tensor]]
    ):
        self._canonicalizer.register_kv_caches(kv_caches)

    def register_cross_layers_kv_cache(
        self, kv_cache: torch.Tensor, attn_backend: type[AttentionBackend]
    ):
        self._canonicalizer.register_cross_layers_kv_cache(kv_cache, attn_backend)

    def _register_handlers(self, kv_caches: CanonicalKVCaches):
        assert self._kv_caches is None, "KV caches registered twice"
        self._kv_caches = kv_caches
        self._start_attach()

    def _start_attach(self) -> None:
        """Attach in the background, as early as the caches allow.

        The attach itself waits on the DP rank 0 scheduler's sentinel and then
        pins the whole data region (~1 s per 10 GiB of cudaHostRegister). Doing
        that on the first store puts the entire cost on live requests -- it
        stalls the engine step loop, so every request in flight on this rank
        eats it. Here it overlaps the rest of startup instead.
        """
        if self._attach_thread is not None:
            return
        device = torch.cuda.current_device() if torch.cuda.is_available() else None

        def run() -> None:
            if device is not None:
                torch.cuda.set_device(device)
            try:
                self._attach_now()
            except BaseException as e:  # re-raised on the first transfer
                self._attach_error = e

        self._attach_thread = threading.Thread(
            target=run, name="radixshmem-attach", daemon=True
        )
        self._attach_thread.start()

    def _attach_now(self) -> None:
        with self._attach_lock:
            if self.handlers is not None:
                return
            assert self._kv_caches is not None, "transfer before register_kv_caches"
            regions, tp_rank = self._attach()
            handlers = RadixShmemOffloadingHandlers(
                regions=regions, kv_caches=self._kv_caches, tp_rank=tp_rank
            )
            self.worker.register_handler(
                GPULoadStoreSpec, CPULoadStoreSpec, handlers.gpu_to_cpu_handler
            )
            self.worker.register_handler(
                CPULoadStoreSpec, GPULoadStoreSpec, handlers.cpu_to_gpu_handler
            )
            # published last: handlers is what _ensure_attached checks
            self.handlers = handlers

    def _ensure_attached(self) -> None:
        """Join the background attach; do it here if it never started."""
        if self.handlers is not None:
            return
        t0 = time.perf_counter()
        if self._attach_thread is not None:
            self._attach_thread.join()
            if self._attach_error is not None:
                raise self._attach_error
        self._attach_now()
        waited = time.perf_counter() - t0
        if waited > 0.05:
            logger.info(
                "RadixShmem worker attach blocked the first transfer for %.3f s",
                waited,
            )

    # ------------------------------------------------------------- transfers

    def _generate_job_id(self) -> int:
        job_id = self._job_counter
        self._job_counter = job_id + 1
        return job_id

    def _submit_deferred_stores(self) -> None:
        if not self._unsubmitted_store_jobs:
            return
        self._ensure_attached()
        for job_id, spec in self._unsubmitted_store_jobs:
            assert self.worker.transfer_async(job_id, spec)
        self._unsubmitted_store_jobs.clear()

    def handle_preemptions(self, metadata: RadixShmemMetadata):
        self._submit_deferred_stores()
        for req_id in metadata.reqs_to_flush or ():
            job_ids = self._store_jobs.get(req_id)
            if job_ids:
                # the GPU blocks are about to be reused, so the reads must land
                self.worker.wait(job_ids)

    def start_kv_transfers(self, metadata: RadixShmemMetadata):
        self._submit_deferred_stores()
        if metadata.reqs_to_load:
            self._ensure_attached()
        for req_id, spec in metadata.reqs_to_load.items():
            job_id = self._generate_job_id()
            self._jobs[job_id] = (req_id, None)
            assert req_id not in self._load_job
            self._load_job[req_id] = job_id
            assert self.worker.transfer_async(job_id, spec)

    def prepare_store_kv(self, metadata: RadixShmemMetadata):
        for store in metadata.stores:
            job_id = self._generate_job_id()
            self._jobs[job_id] = (store.req_id, store.store_id)
            self._store_jobs[store.req_id].add(job_id)
            # deferred to the next step so offloading never delays sampling
            self._unsubmitted_store_jobs.append((job_id, store.spec))

    def get_finished(self, finished_req_ids: set[str]) -> tuple[set[str], set[str]]:
        finished_sending: set[str] = set()
        finished_recving: set[str] = set()
        for result in self.worker.get_finished():
            assert result.success
            req_id, store_id = self._jobs.pop(result.job_id)
            if (
                result.transfer_time
                and result.transfer_size is not None
                and result.transfer_type is not None
            ):
                self.kv_connector_stats.record_transfer(
                    num_bytes=result.transfer_size,
                    time=result.transfer_time,
                    transfer_type=result.transfer_type,
                )
            if store_id is not None:
                # tell the scheduler this rank's slice of those slots is written
                self._completed_stores[store_id] = (
                    self._completed_stores.get(store_id, 0) + 1
                )
                req_jobs = self._store_jobs[req_id]
                req_jobs.discard(result.job_id)
                if req_jobs:
                    continue
                if req_id in self._finished_reqs_waiting_for_store:
                    self._finished_reqs_waiting_for_store.remove(req_id)
                    finished_sending.add(req_id)
                    del self._store_jobs[req_id]
            else:
                assert self._load_job.pop(req_id) == result.job_id
                finished_recving.add(req_id)

        for req_id in finished_req_ids:
            pending = self._store_jobs.get(req_id)
            if pending:
                self._finished_reqs_waiting_for_store.add(req_id)
            elif pending is not None:
                finished_sending.add(req_id)
                del self._store_jobs[req_id]

        return finished_sending, finished_recving

    def build_connector_worker_meta(self) -> RadixShmemWorkerMetadata | None:
        if not self._completed_stores:
            return None
        meta = RadixShmemWorkerMetadata(completed_stores=self._completed_stores)
        self._completed_stores = {}
        return meta

    def get_kv_connector_stats(self) -> KVConnectorStats | None:
        if self.kv_connector_stats.is_empty():
            return None
        stats, self.kv_connector_stats = (
            self.kv_connector_stats,
            OffloadingConnectorStats(),
        )
        return stats

    def close(self) -> None:
        # let a background attach finish first, or it would pin the region
        # again right after this tore it down; bounded, since a thread still
        # waiting on the owner's sentinel would hold shutdown for its timeout
        if self._attach_thread is not None:
            self._attach_thread.join(timeout=30.0)
            if self._attach_thread.is_alive():
                logger.warning("RadixShmem: attach still running at close; leaving it")
                return
            self._attach_thread = None
        if self.handlers is not None:
            self.handlers.close()
            self.handlers = None
        self._kv_caches = None


class _Canonicalizer(OffloadingConnectorWorker):
    """Reuses ``OffloadingConnectorWorker``'s KV cache layout handling only.

    That class derives the canonical (num_blocks, page_bytes) views from every
    supported attention backend layout -- a lot of backend-specific reasoning
    with no RadixShmem content. Subclassing and overriding the one hook it
    exposes is cheaper and safer than copying it.
    """

    def __init__(self, vllm_config, kv_cache_config, owner: RadixShmemConnectorWorker):
        # deliberately not calling super().__init__: only the two register_*
        # methods are used, and they need just these two attributes
        self.spec = _SpecShim(vllm_config, kv_cache_config)
        self._owner = owner

    def _register_handlers(self, kv_caches: CanonicalKVCaches):
        self._owner._register_handlers(kv_caches)


class _SpecShim:
    """The subset of ``OffloadingSpec`` the canonicalization actually reads."""

    def __init__(self, vllm_config, kv_cache_config):
        self.vllm_config = vllm_config
        self.kv_cache_config = kv_cache_config
