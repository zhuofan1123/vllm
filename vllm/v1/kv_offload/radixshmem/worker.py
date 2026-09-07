# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker side of RadixShmem offloading.

Nothing is allocated here: the CPU side is one node-wide shared SlotStore that
the owning scheduler created, and this worker only attaches to it, pins the
mapping for DMA, and exposes its own slice of every slot as strided int8
tensors. The transfers themselves are upstream's
``SingleDirectionOffloadingHandler`` -- batch copies over the same address
arithmetic the private CPU pool uses, so there is no RadixShmem copy path.

Attaching is deferred to a background thread started at construction, joined by
the first transfer. Workers are built during ``initialize_from_config``, before
any scheduler exists, so attaching eagerly would wait forever on the scheduler
that creates the regions; doing it inline on the first transfer instead costs
~1 s per 10 GiB (cudaHostRegister) with the engine step loop stopped.
"""

import threading
import time
from collections.abc import Callable

import torch

from vllm.logger import init_logger
from vllm.v1.kv_offload.base import (
    CanonicalKVCaches,
    GPULoadStoreSpec,
    LoadStoreSpec,
    OffloadingWorker,
    TransferResult,
)
from vllm.v1.kv_offload.cpu.gpu_worker import SingleDirectionOffloadingHandler

from .bootstrap import SharedRegions
from .geometry import verify_against_canonical

logger = init_logger(__name__)


def host_register(ptr: int, nbytes: int) -> None:
    """cudaHostRegister a raw range (the whole SlotStore mapping)."""
    err = torch.cuda.cudart().cudaHostRegister(ptr, nbytes, 0)
    if err.value != 0:
        raise RuntimeError(
            f"cudaHostRegister({ptr:#x}, {nbytes}) failed: {err}. Registering a "
            "multi-GB shared region can fail on a low RLIMIT_MEMLOCK."
        )


def host_unregister(ptr: int) -> None:
    err = torch.cuda.cudart().cudaHostUnregister(ptr)
    if err.value != 0:
        logger.warning("cudaHostUnregister(%#x) failed: %s", ptr, err)


def slot_views(
    regions: SharedRegions, kv_caches: CanonicalKVCaches, writer_idx: int
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Strided per-tensor views of this writer's slice of every slot.

    Returns the flat int8 tensor over the whole mapping (keep it alive: the
    views borrow its storage) and, per canonical tensor ``t``, an int8 view of
    shape ``(num_slots, page_bytes[t] * blocks_per_chunk)`` with row stride
    ``slot_stride`` -- the CPU-side shape ``SingleDirectionOffloadingHandler``
    expects, addressed by slot id.
    """
    geometry = regions.geometry
    layout = verify_against_canonical(geometry, kv_caches)
    if not 0 <= writer_idx < geometry.num_slices:
        raise ValueError(
            f"writer index {writer_idx} out of range for {geometry.num_slices} "
            "slice(s) per slot"
        )
    base = torch.frombuffer(regions.data_view(), dtype=torch.int8)
    if base.numel() < geometry.total_data_bytes:
        raise RuntimeError(
            f"SlotStore mapping is {base.numel()} B, geometry needs "
            f"{geometry.total_data_bytes} B"
        )
    slice_base = writer_idx * geometry.slice_bytes
    views = [
        torch.as_strided(
            base,
            (geometry.num_slots, page * geometry.blocks_per_chunk),
            (geometry.slot_stride, 1),
            storage_offset=slice_base + offset,
        )
        for page, offset in zip(layout.page_bytes, layout.offsets)
    ]
    return base, views


class RadixShmemOffloadingWorker(OffloadingWorker):
    """``OffloadingWorker`` over the node-shared SlotStore."""

    def __init__(
        self,
        *,
        attach: Callable[[], tuple[SharedRegions, int]],
        kv_caches: CanonicalKVCaches,
        blocks_per_chunk: int,
    ):
        self._attach = attach
        self._kv_caches = kv_caches
        self._blocks_per_chunk = blocks_per_chunk

        self.regions: SharedRegions | None = None
        self.tp_rank: int | None = None
        self._base: torch.Tensor | None = None
        self._registered_ptr: int | None = None
        self._store_handler: SingleDirectionOffloadingHandler | None = None
        self._load_handler: SingleDirectionOffloadingHandler | None = None

        self._attach_lock = threading.Lock()
        self._attach_error: BaseException | None = None
        device = torch.cuda.current_device() if torch.cuda.is_available() else None

        def run() -> None:
            if device is not None:
                torch.cuda.set_device(device)
            try:
                self._attach_now()
            except BaseException as e:  # re-raised on the first transfer
                self._attach_error = e

        self._attach_thread: threading.Thread | None = threading.Thread(
            target=run, name="radixshmem-attach", daemon=True
        )
        self._attach_thread.start()

    # ------------------------------------------------------------- attach

    def _attach_now(self) -> None:
        with self._attach_lock:
            if self._store_handler is not None:
                return
            regions, tp_rank = self._attach()
            geometry = regions.geometry
            writer_idx = 0 if geometry.replicated else tp_rank
            base, cpu_tensors = slot_views(regions, self._kv_caches, writer_idx)

            t0 = time.perf_counter()
            host_register(base.data_ptr(), geometry.total_data_bytes)
            register_s = time.perf_counter() - t0
            self._registered_ptr = base.data_ptr()

            gpu_tensors = [
                t.tensor.view(torch.int8).view((-1, t.page_size_bytes))
                for t in self._kv_caches.tensors
            ]
            refs = self._kv_caches.group_data_refs
            store_handler = SingleDirectionOffloadingHandler(
                gpu_tensors=gpu_tensors,
                cpu_tensors=cpu_tensors,
                blocks_per_chunk=self._blocks_per_chunk,
                layer_refs_per_group=refs,
                gpu_to_cpu=True,
            )
            load_handler = SingleDirectionOffloadingHandler(
                gpu_tensors=gpu_tensors,
                cpu_tensors=cpu_tensors,
                blocks_per_chunk=self._blocks_per_chunk,
                layer_refs_per_group=refs,
                gpu_to_cpu=False,
            )
            self.regions = regions
            self.tp_rank = tp_rank
            self._base = base
            self._load_handler = load_handler
            # published last: this is what _handlers() checks
            self._store_handler = store_handler

            logger.info(
                "RadixShmem worker tp_rank=%d attached: %d slots x %d B stride, "
                "writing slice %d of %d, %d canonical tensor(s), "
                "cudaHostRegister(%.2f GiB) took %.3f s",
                tp_rank,
                geometry.num_slots,
                geometry.slot_stride,
                writer_idx,
                geometry.num_slices,
                len(cpu_tensors),
                geometry.total_data_bytes / 2**30,
                register_s,
            )

    def _handlers(
        self,
    ) -> tuple[SingleDirectionOffloadingHandler, SingleDirectionOffloadingHandler]:
        """Join the background attach; do it here if it never finished."""
        if self._store_handler is None:
            t0 = time.perf_counter()
            if self._attach_thread is not None:
                self._attach_thread.join()
                self._attach_thread = None
                if self._attach_error is not None:
                    raise self._attach_error
            self._attach_now()
            waited = time.perf_counter() - t0
            if waited > 0.05:
                logger.info(
                    "RadixShmem worker attach blocked the first transfer for %.3f s",
                    waited,
                )
        assert self._store_handler is not None and self._load_handler is not None
        return self._store_handler, self._load_handler

    # -------------------------------------------------- OffloadingWorker

    def submit_store(
        self, job_id: int, src_spec: GPULoadStoreSpec, dst_spec: LoadStoreSpec
    ) -> bool:
        store, _ = self._handlers()
        return store.transfer_async(job_id, src_spec, dst_spec)

    def submit_load(
        self, job_id: int, src_spec: LoadStoreSpec, dst_spec: GPULoadStoreSpec
    ) -> bool:
        _, load = self._handlers()
        return load.transfer_async(job_id, src_spec, dst_spec)

    def get_finished(self) -> list[TransferResult]:
        if self._store_handler is None or self._load_handler is None:
            return []
        return self._store_handler.get_finished() + self._load_handler.get_finished()

    def wait(self, job_ids: set[int]) -> None:
        if self._store_handler is None or self._load_handler is None:
            return
        self._store_handler.wait(job_ids)
        self._load_handler.wait(job_ids)

    def shutdown(self) -> None:
        """Drain, unpin, detach -- in that order.

        Unregistering while a copy is in flight, or dropping the mapping before
        unregistering, is a use-after-free inside the driver.
        """
        # let a background attach finish first, or it would pin the region
        # again right after this tore it down; bounded, since a thread still
        # waiting on the owner's sentinel would hold shutdown for its timeout
        if self._attach_thread is not None:
            self._attach_thread.join(timeout=30.0)
            if self._attach_thread.is_alive():
                logger.warning(
                    "RadixShmem: attach still running at shutdown; leaving it"
                )
                return
            self._attach_thread = None
        for handler in (self._store_handler, self._load_handler):
            if handler is None:
                continue
            try:
                handler.shutdown()
            except Exception:
                logger.exception("RadixShmem: error draining transfers at shutdown")
        self._store_handler = None
        self._load_handler = None
        if self._registered_ptr is not None:
            host_unregister(self._registered_ptr)
            self._registered_ptr = None
        self._base = None
        if self.regions is not None:
            self.regions.close()
            self.regions = None
