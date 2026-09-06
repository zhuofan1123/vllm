# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scheduler side of the RadixShmem connector.

Structurally this follows ``OffloadingConnectorScheduler``, because the hard
parts it solves -- async scheduling, preemption, tracking which GPU block holds
which offloaded block -- are unchanged. Three things differ, and they are the
reason this is a separate class rather than an ``OffloadingManager``:

* **Lookups are root-anchored.** A radix tree matches from the root, so the
  scheduler hands down the whole prefix and slices the result, instead of
  handing down a suffix starting at ``start_block_idx``.
* **A hit is pinned when it is found**, not when it is used. ``lookup_and_pin``
  matches and takes the ref under one lock; without that, another DP rank's
  ``allocate`` could evict the blocks between the lookup and the load. Leases
  that never turn into a load are swept at the end of the step.
* **Publishing waits for every TP rank.** A slot is only complete once all TP
  ranks have written their slice, so workers report store completions back and
  the insert happens when the count reaches ``world_size``.
"""

from collections import defaultdict
from collections.abc import Iterable
from itertools import islice
from typing import Any

import numpy as np

from vllm.distributed.kv_events import BlockStored, KVCacheEvent
from vllm.distributed.kv_transfer.kv_connector.utils import yield_req_data
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata
from vllm.distributed.kv_transfer.kv_connector.v1.radixshmem.metadata import (
    RadixShmemMetadata,
    RadixShmemWorkerMetadata,
    ReqId,
    StoreJob,
    TransferSpec,
)
from vllm.logger import init_logger
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_offload.base import GPULoadStoreSpec
from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec
from vllm.v1.kv_offload.radixshmem.manager import LookupLease, RadixShmemManager, to_u64
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import Request

from .bootstrap import SharedRegions
from .geometry import SlotGeometry

logger = init_logger(__name__)


class PendingStore:
    """An offload whose DMA is still in flight on at least one TP rank."""

    __slots__ = ("hashes", "block_hashes", "slots", "start", "req_id", "acks")

    def __init__(self, hashes, block_hashes, slots, start: int, req_id: ReqId):
        self.hashes = hashes
        self.block_hashes = block_hashes
        self.slots = slots
        self.start = start
        self.req_id = req_id
        self.acks = 0


class RadixShmemConnectorScheduler:
    def __init__(
        self,
        *,
        geometry: SlotGeometry,
        regions: SharedRegions,
        vllm_config,
    ):
        self.geometry = geometry
        self.regions = regions
        self.gpu_block_size = geometry.gpu_block_size
        self.offloaded_block_size = geometry.offloaded_block_size
        self.block_size_factor = geometry.block_size_factor
        self.manager = RadixShmemManager(
            regions.client, offloaded_block_size=self.offloaded_block_size
        )
        # a slot is published only after every TP rank wrote its own slice
        self.world_size = vllm_config.parallel_config.world_size
        kv_events = vllm_config.kv_events_config
        self._emit_events = bool(
            kv_events is not None and kv_events.enable_kv_cache_events
        )

        self._requests: dict[ReqId, Request] = {}
        self._request_block_ids: dict[ReqId, list[int]] = {}
        self._reqs_to_load: dict[ReqId, TransferSpec] = {}
        self._next_stored_block_idx: dict[ReqId, int] = {}

        # leases taken during get_num_new_matched_tokens, before we know whether
        # the request will actually be allocated this step
        self._pending_leases: dict[ReqId, LookupLease] = {}
        # leases held for the duration of a load
        self._load_leases: dict[ReqId, LookupLease] = {}

        self._next_store_id = 0
        self._pending_stores: dict[int, PendingStore] = {}
        # requests with at least one store in flight -- their GPU blocks must
        # not be freed yet
        self._store_ids_by_req = defaultdict[ReqId, set[int]](set)

        self._events: list[KVCacheEvent] = []

    # ---------------------------------------------------------------- helpers

    def _block_hashes(
        self, req: Request, start_idx: int = 0, end_idx: int | None = None
    ) -> Iterable[BlockHash]:
        """vLLM hashes per GPU block; take one per offloaded block."""
        return islice(
            req.block_hashes,
            self.block_size_factor * start_idx + self.block_size_factor - 1,
            self.block_size_factor * end_idx if end_idx else None,
            self.block_size_factor,
        )

    def _prefix_u64(self, req: Request, num_blocks: int) -> np.ndarray:
        return to_u64(list(self._block_hashes(req, end_idx=num_blocks)))

    # ----------------------------------------------------------------- lookup

    def get_num_new_matched_tokens(
        self, request: Request, num_computed_tokens: int
    ) -> tuple[int | None, bool]:
        num_blocks = request.num_tokens // self.offloaded_block_size
        if num_blocks == 0:
            return 0, False

        full_block_tokens = self.offloaded_block_size * num_blocks
        if full_block_tokens - num_computed_tokens < self.offloaded_block_size:
            # less than a whole offloaded block left to gain
            return 0, False

        lease = self.manager.lookup_and_pin(self._prefix_u64(request, num_blocks))
        if lease is None:
            return 0, False

        num_hit_tokens = self.offloaded_block_size * lease.hit_blocks
        if num_hit_tokens - num_computed_tokens < self.offloaded_block_size:
            # the shared cache knows nothing the GPU does not already have
            lease.release()
            return 0, False

        # the pin is held until update_state_after_alloc decides what to do with
        # it; anything left over is swept in build_connector_meta
        old = self._pending_leases.pop(request.request_id, None)
        if old is not None:
            old.release()
        self._pending_leases[request.request_id] = lease

        logger.debug(
            "Request %s hit %d shared tokens after %d GPU tokens",
            request.request_id,
            num_hit_tokens - num_computed_tokens,
            num_computed_tokens,
        )
        return num_hit_tokens - num_computed_tokens, True

    def update_state_after_alloc(
        self, request: Request, blocks: KVCacheBlocks, num_external_tokens: int
    ):
        req_id = request.request_id
        self._requests[req_id] = request
        # filled in by _get_reqs_to_store
        self._request_block_ids[req_id] = []

        lease = self._pending_leases.pop(req_id, None)
        if num_external_tokens == 0:
            if lease is not None:
                lease.release()
            return
        assert lease is not None, "external tokens promised without a pinned hit"

        block_ids = blocks.get_block_ids()[0]
        num_computed_gpu_blocks = sum(
            block.block_hash is not None for block in blocks.blocks[0]
        )
        num_computed_tokens = num_computed_gpu_blocks * self.gpu_block_size
        full_block_tokens = num_computed_tokens + num_external_tokens
        assert full_block_tokens % self.offloaded_block_size == 0

        num_pending_gpu_blocks = len(block_ids) - num_computed_gpu_blocks
        assert num_external_tokens == num_pending_gpu_blocks * self.gpu_block_size

        start_block_idx = num_computed_tokens // self.offloaded_block_size
        num_blocks = full_block_tokens // self.offloaded_block_size
        assert lease.hit_blocks >= num_blocks

        # the lease is root-anchored, so the slots for this load are a plain slice
        src_spec = CPULoadStoreSpec(lease.slots[start_block_idx:num_blocks].tolist())
        dst_spec = GPULoadStoreSpec(
            block_ids[num_computed_gpu_blocks:],
            group_sizes=(num_pending_gpu_blocks,),
            block_indices=(num_computed_gpu_blocks,),
        )

        self._reqs_to_load[req_id] = (src_spec, dst_spec)
        # hold the pin until the data has actually been read out of the slots
        self._load_leases[req_id] = lease
        self._next_stored_block_idx[req_id] = num_blocks

    # ------------------------------------------------------------------ store

    def _get_reqs_to_store(self, scheduler_output: SchedulerOutput) -> list[StoreJob]:
        stores: list[StoreJob] = []
        for req_id, new_block_id_groups, preempted in yield_req_data(scheduler_output):
            if preempted:
                self._request_block_ids[req_id] = []
            if new_block_id_groups:
                self._request_block_ids[req_id] += new_block_id_groups[0]

            block_ids = self._request_block_ids[req_id]
            req = self._requests[req_id]
            new_tokens = scheduler_output.num_scheduled_tokens[req_id]
            # with async scheduling some tokens may not be there yet
            total_tokens = min(req.num_computed_tokens + new_tokens, req.num_tokens)
            num_blocks = total_tokens // self.offloaded_block_size
            start_block_idx = self._next_stored_block_idx.get(req_id, 0)
            if num_blocks - start_block_idx <= 0:
                continue
            assert len(req.block_hashes) >= num_blocks * self.block_size_factor

            hashes = self._prefix_u64(req, num_blocks)
            # somebody else may have published part of this prefix already; only
            # move the bytes that are genuinely missing
            already = self.manager.lookup(hashes)
            store_start = max(start_block_idx, min(already, num_blocks))
            num_new = num_blocks - store_start
            if num_new <= 0:
                self._next_stored_block_idx[req_id] = num_blocks
                continue

            slots = self.manager.allocate(num_new)
            if slots is None:
                logger.warning_once(
                    "RadixShmem: no free slots for %d block(s); the shared cache "
                    "is fully pinned. Offloading is skipped this step.",
                    num_new,
                )
                continue

            src_block_ids: list[int] = []
            for offloaded_idx in range(store_start, num_blocks):
                base = offloaded_idx * self.block_size_factor
                src_block_ids.extend(block_ids[base : base + self.block_size_factor])
            assert len(src_block_ids) == num_new * self.block_size_factor

            dst_spec = CPULoadStoreSpec(slots.tolist())
            src_spec = GPULoadStoreSpec(
                src_block_ids,
                group_sizes=(len(src_block_ids),),
                block_indices=(store_start * self.block_size_factor,),
            )

            store_id = self._next_store_id
            self._next_store_id += 1
            block_hashes = (
                list(self._block_hashes(req, store_start, num_blocks))
                if self._emit_events
                else None
            )
            self._pending_stores[store_id] = PendingStore(
                hashes=hashes,
                block_hashes=block_hashes,
                slots=slots,
                start=store_start,
                req_id=req_id,
            )
            self._store_ids_by_req[req_id].add(store_id)
            stores.append(
                StoreJob(store_id=store_id, req_id=req_id, spec=(src_spec, dst_spec))
            )
            self._next_stored_block_idx[req_id] = num_blocks

            logger.debug(
                "Request %s offloading blocks [%d, %d) into %d slots",
                req_id,
                store_start,
                num_blocks,
                num_new,
            )
        return stores

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        meta = RadixShmemMetadata(
            reqs_to_load=self._reqs_to_load,
            stores=self._get_reqs_to_store(scheduler_output),
            reqs_to_flush=scheduler_output.preempted_req_ids,
        )
        self._reqs_to_load = {}

        # Any hit we pinned this step that did not become a load belongs to a
        # request the scheduler could not allocate. Dropping the pin here is the
        # only chance we get -- nothing will call back about those requests.
        if self._pending_leases:
            for lease in self._pending_leases.values():
                lease.release()
            self._pending_leases.clear()

        return meta

    # ------------------------------------------------------------- completion

    def update_connector_output(self, connector_output: KVConnectorOutput):
        worker_meta = connector_output.kv_connector_worker_meta
        if isinstance(worker_meta, RadixShmemWorkerMetadata):
            for store_id, count in worker_meta.completed_stores.items():
                self._ack_store(store_id, count)

        for req_id in connector_output.finished_recving or []:
            lease = self._load_leases.pop(req_id, None)
            if lease is not None:
                lease.release()

        for req_id in connector_output.finished_sending or []:
            self._store_ids_by_req.pop(req_id, None)

    def _ack_store(self, store_id: int, count: int) -> None:
        pending = self._pending_stores.get(store_id)
        if pending is None:
            return
        pending.acks += count
        if pending.acks < self.world_size:
            return
        del self._pending_stores[store_id]
        self._store_ids_by_req[pending.req_id].discard(store_id)

        published, _ = self.manager.publish(
            pending.hashes, pending.slots, pending.start
        )
        if published and self._emit_events and pending.block_hashes:
            # publish() may have dropped a leading duplicate run
            tail = pending.block_hashes[len(pending.block_hashes) - published :]
            self._events.append(
                BlockStored(
                    block_hashes=tail,
                    parent_block_hash=None,
                    token_ids=[],
                    lora_id=None,
                    block_size=self.offloaded_block_size,
                    medium="CPU",
                    lora_name=None,
                )
            )

    # ----------------------------------------------------------------- misc

    def request_finished(
        self, request: Request, block_ids: list[int]
    ) -> tuple[bool, dict[str, Any] | None]:
        req_id = request.request_id
        self._requests.pop(req_id, None)
        self._request_block_ids.pop(req_id, None)
        self._next_stored_block_idx.pop(req_id, None)
        # Hold the GPU blocks until the worker says so. Membership, not
        # emptiness: the worker reports finished_sending for every request it
        # ever stored for, and the scheduler asserts that such a request is
        # still alive. Acking the last store is not the same as being told.
        return req_id in self._store_ids_by_req, None

    def take_events(self) -> Iterable[KVCacheEvent]:
        events, self._events = self._events, []
        return events

    def release_all(self) -> None:
        """Give back every piece of shared state this process still holds.

        Pins would keep blocks unevictable and slots reserved for stores that
        never completed would sit allocated but unreachable, both for as long as
        the shared region lives -- which outlives this process.
        """
        for lease in list(self._pending_leases.values()):
            lease.release()
        self._pending_leases.clear()
        for lease in list(self._load_leases.values()):
            lease.release()
        self._load_leases.clear()
        for pending in self._pending_stores.values():
            self.manager.recycle(pending.slots)
        self._pending_stores.clear()

    def close(self) -> None:
        self.release_all()
        logger.info("RadixShmem scheduler stats: %s", self.manager.stats())
        self.manager.close()
        self.regions.close()
