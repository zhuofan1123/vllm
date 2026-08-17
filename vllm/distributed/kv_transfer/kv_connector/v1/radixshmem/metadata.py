# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scheduler <-> worker messages for the RadixShmem connector."""

from dataclasses import dataclass, field

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorMetadata,
    KVConnectorWorkerMetadata,
)
from vllm.v1.kv_offload.worker.worker import TransferSpec

ReqId = str


@dataclass
class StoreJob:
    """One GPU->slot offload, tracked by an id the scheduler owns.

    Store completion cannot be keyed by request: a long request offloads many
    times, and each batch has to be published into the shared index on its own.
    """

    store_id: int
    req_id: ReqId
    spec: TransferSpec


@dataclass
class RadixShmemMetadata(KVConnectorMetadata):
    reqs_to_load: dict[ReqId, TransferSpec] = field(default_factory=dict)
    stores: list[StoreJob] = field(default_factory=list)
    reqs_to_flush: set[ReqId] | None = None


@dataclass
class RadixShmemWorkerMetadata(KVConnectorWorkerMetadata):
    """Worker -> scheduler: store ids whose DMA landed on this worker.

    A slot is only safe to publish once *every* TP rank has written its slice,
    so the scheduler counts these up to ``world_size`` before inserting into
    the shared index. ``aggregate`` sums the per-worker reports for one step.
    """

    completed_stores: dict[int, int]

    def aggregate(
        self, other: "KVConnectorWorkerMetadata"
    ) -> "KVConnectorWorkerMetadata":
        assert isinstance(other, RadixShmemWorkerMetadata)
        merged = dict(self.completed_stores)
        for store_id, count in other.completed_stores.items():
            merged[store_id] = merged.get(store_id, 0) + count
        return RadixShmemWorkerMetadata(completed_stores=merged)
