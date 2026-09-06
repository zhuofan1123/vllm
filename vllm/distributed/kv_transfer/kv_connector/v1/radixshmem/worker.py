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
    _TransferMetricName,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.worker import (
    OffloadingConnectorWorker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.radixshmem.metadata import (
    RadixShmemMetadata,
    RadixShmemWorkerMetadata,
    ReqId,
    TransferSpec,
)
from vllm.logger import init_logger
from vllm.v1.kv_offload.base import CanonicalKVCaches, GPULoadStoreSpec
from vllm.v1.kv_offload.radixshmem.worker import RadixShmemOffloadingHandlers

logger = init_logger(__name__)


class RadixShmemConnectorWorker:
    def __init__(self, *, attach: Callable[[], tuple], vllm_config, kv_cache_config):
        self._attach = attach
        self.worker: RadixShmemOffloadingHandlers | None = None
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

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        self._canonicalizer.register_kv_caches(kv_caches)

    def _on_canonical_kv_caches(self, kv_caches: CanonicalKVCaches):
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
            if self.worker is not None:
                return
            assert self._kv_caches is not None, "transfer before register_kv_caches"
            regions, tp_rank = self._attach()
            # published last: worker is what _ensure_attached checks
            self.worker = RadixShmemOffloadingHandlers(
                regions=regions, kv_caches=self._kv_caches, tp_rank=tp_rank
            )

    def _ensure_attached(self) -> RadixShmemOffloadingHandlers:
        """Join the background attach; do it here if it never started."""
        if self.worker is not None:
            return self.worker
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
        assert self.worker is not None
        return self.worker

    # ------------------------------------------------------------- transfers

    def _generate_job_id(self) -> int:
        job_id = self._job_counter
        self._job_counter = job_id + 1
        return job_id

    def _submit(self, job_id: int, spec: TransferSpec) -> None:
        worker = self._ensure_attached()
        src_spec, dst_spec = spec
        if isinstance(src_spec, GPULoadStoreSpec):
            ok = worker.submit_store(job_id, src_spec, dst_spec)
        else:
            assert isinstance(dst_spec, GPULoadStoreSpec)
            ok = worker.submit_load(job_id, src_spec, dst_spec)
        assert ok, f"RadixShmem: transfer job {job_id} was rejected"

    def _submit_deferred_stores(self) -> None:
        if not self._unsubmitted_store_jobs:
            return
        for job_id, spec in self._unsubmitted_store_jobs:
            self._submit(job_id, spec)
        self._unsubmitted_store_jobs.clear()

    def handle_preemptions(self, metadata: RadixShmemMetadata):
        self._submit_deferred_stores()
        for req_id in metadata.reqs_to_flush or ():
            job_ids = self._store_jobs.get(req_id)
            if job_ids:
                # the GPU blocks are about to be reused, so the reads must land
                self._ensure_attached().wait(job_ids)

    def start_kv_transfers(self, metadata: RadixShmemMetadata):
        self._submit_deferred_stores()
        for req_id, spec in metadata.reqs_to_load.items():
            job_id = self._generate_job_id()
            self._jobs[job_id] = (req_id, None)
            assert req_id not in self._load_job
            self._load_job[req_id] = job_id
            self._submit(job_id, spec)

    def prepare_store_kv(self, metadata: RadixShmemMetadata):
        for store in metadata.stores:
            job_id = self._generate_job_id()
            self._jobs[job_id] = (store.req_id, store.store_id)
            self._store_jobs[store.req_id].add(job_id)
            # deferred to the next step so offloading never delays sampling
            self._unsubmitted_store_jobs.append((job_id, store.spec))

    def _record_transfer(self, *, store: bool, num_bytes: int, seconds: float) -> None:
        if store:
            bytes_name = _TransferMetricName.STORE_BYTES
            time_name = _TransferMetricName.STORE_TIME
            size_name = _TransferMetricName.STORE_SIZE
        else:
            bytes_name = _TransferMetricName.LOAD_BYTES
            time_name = _TransferMetricName.LOAD_TIME
            size_name = _TransferMetricName.LOAD_SIZE
        stats = self.kv_connector_stats
        stats.increase_counter(bytes_name, num_bytes)
        stats.increase_counter(time_name, seconds)
        stats.observe_histogram(size_name, num_bytes)

    def get_finished(self, finished_req_ids: set[str]) -> tuple[set[str], set[str]]:
        finished_sending: set[str] = set()
        finished_recving: set[str] = set()
        if self.worker is not None:
            for result in self.worker.get_finished():
                assert result.success
                req_id, store_id = self._jobs.pop(result.job_id)
                if (
                    result.transfer_time is not None
                    and result.transfer_size is not None
                ):
                    self._record_transfer(
                        store=store_id is not None,
                        num_bytes=result.transfer_size,
                        seconds=result.transfer_time,
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
        if self.worker is not None:
            self.worker.close()
            self.worker = None
        self._kv_caches = None


class _Canonicalizer(OffloadingConnectorWorker):
    """Reuses ``OffloadingConnectorWorker``'s KV cache layout handling only.

    That class derives the canonical (num_blocks, page_bytes) views from every
    supported attention backend layout -- a lot of backend-specific reasoning
    with no RadixShmem content. Subclassing and overriding the one hook it
    calls at the end (``_init_worker``) is cheaper and safer than copying it.
    """

    def __init__(self, vllm_config, kv_cache_config, owner: RadixShmemConnectorWorker):
        # deliberately not calling super().__init__: only register_kv_caches is
        # used, and it reads just these two attributes before _init_worker
        self.vllm_config = vllm_config
        self.kv_cache_config = kv_cache_config
        self._owner = owner

    def _init_worker(self, kv_caches: CanonicalKVCaches) -> None:
        self._owner._on_canonical_kv_caches(kv_caches)
