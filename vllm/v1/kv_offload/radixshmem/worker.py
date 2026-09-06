# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker-side GPU <-> SlotStore transfers for the RadixShmem connector.

Unlike ``CPUOffloadingWorker``, nothing is allocated here: the CPU side is one
node-wide shared region created by the DP rank 0 scheduler, and this worker only
attaches to it, pins its TP slice of every slot, and DMAs in and out.

The slot layout is packed, so a slot is not a contiguous run of pages the way a
``swap_blocks`` destination is::

    slot s : [ rank 0 | rank 1 | ... | rank tp-1 ][ pad to slot_align ]
    rank r : [ tensor 0 sub-blocks ][ tensor 1 sub-blocks ] ...

Each rank writes only its own slice, so the ranks never collide and no lock is
needed on the data plane -- the index (which decides *which* slot) is where
coordination happens.
"""

import time
from collections import deque
from dataclasses import dataclass

import torch

from vllm.logger import init_logger
from vllm.v1.kv_offload.base import (
    BlockIDsLoadStoreSpec,
    CanonicalKVCaches,
    GPULoadStoreSpec,
    LoadStoreSpec,
    OffloadingWorker,
    TransferResult,
)
from vllm.v1.kv_offload.radixshmem.mem import (
    PackedSlotAddresser,
    copy_addrs,
    host_register,
    host_unregister,
)

logger = init_logger(__name__)

# (src, dst) pair handed to a handler; one of the two is always the GPU side.
TransferSpec = tuple[LoadStoreSpec, LoadStoreSpec]


@dataclass
class _Transfer:
    job_id: int
    stream: torch.cuda.Stream
    start_event: torch.Event
    end_event: torch.Event
    num_bytes: int


class SlotTransferHandler:
    """One direction of GPU <-> SlotStore traffic, submitted in order.

    Mirrors ``SingleDirectionOffloadingHandler`` in ``kv_offload/cpu``: a stream
    per in-flight job, each chained to the previous job's end event so
    completion order matches submission order.
    """

    def __init__(self, addresser: PackedSlotAddresser, *, gpu_to_cpu: bool):
        self.addresser = addresser
        self.gpu_to_cpu = gpu_to_cpu

        self._transfer_events: dict[int, torch.Event] = {}
        self._transfers: deque[_Transfer] = deque()
        self._stream_pool: list[torch.cuda.Stream] = []
        self._event_pool: list[torch.Event] = []

    def transfer_async(self, job_id: int, transfer_spec: TransferSpec) -> bool:
        src_spec, dst_spec = transfer_spec
        assert isinstance(src_spec, BlockIDsLoadStoreSpec)
        assert isinstance(dst_spec, BlockIDsLoadStoreSpec)
        gpu_spec, slot_spec = (
            (src_spec, dst_spec) if self.gpu_to_cpu else (dst_spec, src_spec)
        )

        gpu_addrs, host_addrs, sizes = self.addresser.build(
            gpu_spec.block_ids, slot_spec.block_ids
        )
        if gpu_addrs.size == 0:
            return False

        stream = self._stream_pool.pop() if self._stream_pool else torch.cuda.Stream()
        start_event = self._take_event()
        end_event = self._take_event()

        if self.gpu_to_cpu:
            # the KV being offloaded is only valid after the model step
            stream.wait_stream(torch.cuda.current_stream())
        if self._transfers:
            stream.wait_event(self._transfers[-1].end_event)

        src, dst = (
            (gpu_addrs, host_addrs) if self.gpu_to_cpu else (host_addrs, gpu_addrs)
        )
        with torch.cuda.stream(stream):
            start_event.record(stream)
            copy_addrs(src, dst, sizes, stream)
            end_event.record(stream)

        self._transfer_events[job_id] = end_event
        self._transfers.append(
            _Transfer(
                job_id=job_id,
                stream=stream,
                start_event=start_event,
                end_event=end_event,
                num_bytes=self.addresser.num_bytes(int(gpu_spec.block_ids.size)),
            )
        )
        return True

    def _take_event(self) -> torch.Event:
        if self._event_pool:
            return self._event_pool.pop()
        return torch.Event(enable_timing=True)

    def get_finished(self) -> list[TransferResult]:
        results: list[TransferResult] = []
        while self._transfers and self._transfers[0].end_event.query():
            t = self._transfers.popleft()
            results.append(
                TransferResult(
                    job_id=t.job_id,
                    success=True,
                    transfer_size=t.num_bytes,
                    transfer_time=t.start_event.elapsed_time(t.end_event) * 1e-3,
                )
            )
            self._stream_pool.append(t.stream)
            self._event_pool.append(t.end_event)
            self._event_pool.append(t.start_event)
            del self._transfer_events[t.job_id]
        return results

    def wait(self, job_ids: set[int]) -> None:
        for job_id in job_ids:
            event = self._transfer_events.get(job_id)
            if event is not None:
                event.synchronize()

    def drain(self) -> None:
        """Block until nothing is in flight. Required before unregistering."""
        for t in list(self._transfers):
            t.end_event.synchronize()
        self.get_finished()


class RadixShmemOffloadingHandlers(OffloadingWorker):
    """Attaches the shared SlotStore and drives both transfer directions.

    This is the ``OffloadingWorker`` for the RadixShmem medium: ``submit_store``
    is GPU -> slot, ``submit_load`` is slot -> GPU. The two directions are
    independent handlers so a burst of stores never queues behind a load.
    """

    def __init__(
        self,
        *,
        regions,  # SharedRegions
        kv_caches: CanonicalKVCaches,
        tp_rank: int,
    ):
        from vllm.distributed.kv_transfer.kv_connector.v1.radixshmem import (
            tensor_layout,
            verify_against_canonical,
        )

        geometry = regions.geometry
        # fails closed if the tensors this worker actually got do not add up to
        # the byte count the owner already sized the region from
        verify_against_canonical(geometry, kv_caches)
        layout = tensor_layout(kv_caches, geometry.block_size_factor)

        self.regions = regions
        self.geometry = geometry
        self.tp_rank = tp_rank
        self._registered_ptr: int | None = None

        gpu_views = [
            t.tensor.view(torch.int8).view((-1, t.page_size_bytes))
            for t in kv_caches.tensors
        ]
        # keep the views alive: data_ptr() below must stay valid
        self._gpu_views = gpu_views
        num_gpu_blocks = gpu_views[0].shape[0]
        for v in gpu_views:
            assert v.is_cuda and v.shape[0] == num_gpu_blocks

        data_base = regions.data_ptr
        t0 = time.perf_counter()
        host_register(data_base, geometry.total_data_bytes)
        self.host_register_s = time.perf_counter() - t0
        self._registered_ptr = data_base

        addresser = PackedSlotAddresser(
            data_base=data_base,
            slot_stride=geometry.slot_stride,
            num_slots=geometry.num_slots,
            tp_rank=tp_rank,
            tp_slice_bytes=geometry.tp_slice_bytes,
            tensor_offsets=layout.offsets,
            page_bytes=layout.page_bytes,
            gpu_bases=tuple(v.data_ptr() for v in gpu_views),
            gpu_num_blocks=num_gpu_blocks,
            block_size_factor=geometry.block_size_factor,
        )
        self.addresser = addresser
        self.gpu_to_cpu_handler = SlotTransferHandler(addresser, gpu_to_cpu=True)
        self.cpu_to_gpu_handler = SlotTransferHandler(addresser, gpu_to_cpu=False)

        logger.info(
            "RadixShmem worker tp_rank=%d attached: %d slots x %d B stride, "
            "slice [%d, %d) of each slot, %d GPU blocks, %d tensors, "
            "cudaHostRegister(%.2f GiB) took %.3f s",
            tp_rank,
            geometry.num_slots,
            geometry.slot_stride,
            tp_rank * geometry.tp_slice_bytes,
            (tp_rank + 1) * geometry.tp_slice_bytes,
            num_gpu_blocks,
            len(gpu_views),
            geometry.total_data_bytes / 2**30,
            self.host_register_s,
        )

    # ------------------------------------------------------ OffloadingWorker

    def submit_store(
        self, job_id: int, src_spec: GPULoadStoreSpec, dst_spec: LoadStoreSpec
    ) -> bool:
        return self.gpu_to_cpu_handler.transfer_async(job_id, (src_spec, dst_spec))

    def submit_load(
        self, job_id: int, src_spec: LoadStoreSpec, dst_spec: GPULoadStoreSpec
    ) -> bool:
        return self.cpu_to_gpu_handler.transfer_async(job_id, (src_spec, dst_spec))

    def get_finished(self) -> list[TransferResult]:
        return (
            self.gpu_to_cpu_handler.get_finished()
            + self.cpu_to_gpu_handler.get_finished()
        )

    def wait(self, job_ids: set[int]) -> None:
        self.gpu_to_cpu_handler.wait(job_ids)
        self.cpu_to_gpu_handler.wait(job_ids)

    def shutdown(self) -> None:
        self.close()

    # ------------------------------------------------------------- teardown

    def close(self) -> None:
        """Drain, unpin, detach -- in that order.

        Unregistering while a copy is in flight, or dropping the mapping before
        unregistering, is a use-after-free inside the driver.
        """
        try:
            self.gpu_to_cpu_handler.drain()
            self.cpu_to_gpu_handler.drain()
        except Exception:
            logger.exception("RadixShmem: error draining transfers at shutdown")
        if self._registered_ptr is not None:
            host_unregister(self._registered_ptr)
            self._registered_ptr = None
        self._gpu_views = []
        if self.regions is not None:
            self.regions.close()
            self.regions = None
