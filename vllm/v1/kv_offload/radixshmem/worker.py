# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker side of RadixShmem offloading.

Nothing is allocated here: the CPU side is one node-wide shared SlotStore that
the owning scheduler created, and this worker only attaches to it, pins the
mapping for DMA, and exposes its own slice of every slot as strided int8
tensors. The transfers themselves are upstream's
``SingleDirectionOffloadingHandler`` -- batch copies over the same address
arithmetic the private CPU pool uses, so there is no RadixShmem copy path.

The store has one pool per attention kind (see ``geometry.py``), so there is
one pair of handlers per pool and a connector job is split by KV cache group
before submission; the parent job completes when every part has.

Attaching is deferred to a background thread started at construction, joined by
the first transfer. Workers are built during ``initialize_from_config``, before
any scheduler exists, so attaching eagerly would wait forever on the scheduler
that creates the regions; doing it inline on the first transfer instead costs
~1 s per 10 GiB (cudaHostRegister) with the engine step loop stopped.
"""

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import torch

from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv
from vllm.v1.kv_offload.base import (
    CanonicalKVCacheRef,
    CanonicalKVCaches,
    GPULoadStoreSpec,
    LoadStoreSpec,
    OffloadingWorker,
    TransferResult,
)
from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec
from vllm.v1.kv_offload.cpu.gpu_worker import SingleDirectionOffloadingHandler

from .bootstrap import SharedRegions
from .geometry import (
    PoolKind,
    PoolSpec,
    SlotGeometry,
    tensor_layout,
    verify_pool_layout,
)

logger = init_logger(__name__)

# a parent job id j fans out to child ids j * _CHILD_STRIDE + kind
_CHILD_STRIDE = 8


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


@dataclass
class PoolPlan:
    """GPU tensors and per-group refs that go through one pool's handlers."""

    pool: PoolSpec
    gpu_tensors: list[torch.Tensor] = field(default_factory=list)
    # per KV cache group (all groups; empty for groups of other pools)
    refs_per_group: list[list[CanonicalKVCacheRef]] = field(default_factory=list)
    group_idxs: list[int] = field(default_factory=list)

    @property
    def page_bytes(self) -> list[int]:
        return [t.shape[1] for t in self.gpu_tensors]


def plan_pools(geometry: SlotGeometry, kv_caches: CanonicalKVCaches) -> list[PoolPlan]:
    """Split the canonical KV caches by pool.

    Layer-outermost layouts already come as one canonical tensor per layer (or
    per set of aliased layers); each group's refs pick its tensors. Packed
    block-outermost layouts (e.g. DeepSeek-V4) come as a single tensor covering
    the whole block for every group, so per-layer strided views are carved out
    of it using the layer offsets the geometry carries.
    """
    num_groups = len(geometry.groups)
    packed = (
        len(kv_caches.tensors) == 1
        and any(g.layer_pages for g in geometry.groups)
        and all(
            len(refs) == 1 and refs[0].tensor_idx == 0
            for refs in kv_caches.group_data_refs
        )
    )
    plans: dict[int, PoolPlan] = {
        p.kind: PoolPlan(pool=p, refs_per_group=[[] for _ in range(num_groups)])
        for p in geometry.pools
    }
    if packed:
        packed_tensor = kv_caches.tensors[0].tensor
        block_stride = kv_caches.tensors[0].page_size_bytes
        base = packed_tensor.view(torch.int8).view((-1, block_stride))
        num_blocks = base.shape[0]
        for group in geometry.groups:
            plan = plans[group.kind]
            for offset, page in group.layer_pages:
                if offset + page > block_stride:
                    raise RuntimeError(
                        f"layer page [{offset}, {offset + page}) exceeds the packed "
                        f"block of {block_stride} bytes"
                    )
                view = torch.as_strided(
                    base, (num_blocks, page), (block_stride, 1), storage_offset=offset
                )
                plan.gpu_tensors.append(view)
                plan.refs_per_group[group.group_idx].append(
                    CanonicalKVCacheRef(len(plan.gpu_tensors) - 1, page)
                )
            plan.group_idxs.append(group.group_idx)
        return [plans[p.kind] for p in geometry.pools]

    for group in geometry.groups:
        plan = plans[group.kind]
        local_idx: dict[int, int] = {}
        for ref in kv_caches.group_data_refs[group.group_idx]:
            if ref.tensor_idx not in local_idx:
                t = kv_caches.tensors[ref.tensor_idx]
                plan.gpu_tensors.append(
                    t.tensor.view(torch.int8).view((-1, t.page_size_bytes))
                )
                local_idx[ref.tensor_idx] = len(plan.gpu_tensors) - 1
            plan.refs_per_group[group.group_idx].append(
                CanonicalKVCacheRef(local_idx[ref.tensor_idx], ref.page_size_bytes)
            )
        plan.group_idxs.append(group.group_idx)
    return [plans[p.kind] for p in geometry.pools]


def pool_views(
    regions: SharedRegions, plan: PoolPlan, writer_idx: int
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Strided per-tensor views of this writer's slice of every slot of a pool.

    Returns the flat int8 tensor over the pool (keep it alive: the views borrow
    its storage) and, per GPU tensor ``t`` of the plan, an int8 view of shape
    ``(num_slots, page_bytes[t] * sub_blocks)`` with row stride ``slot_stride``
    -- the CPU-side shape ``SingleDirectionOffloadingHandler`` expects,
    addressed by slot id.
    """
    pool = plan.pool
    layout = tensor_layout(plan.page_bytes, pool.sub_blocks)
    verify_pool_layout(pool, layout)
    if not 0 <= writer_idx < pool.num_slices:
        raise ValueError(
            f"writer index {writer_idx} out of range for {pool.num_slices} "
            "slice(s) per slot"
        )
    base = torch.frombuffer(regions.pool_view(pool.kind), dtype=torch.int8)
    if base.numel() < pool.num_slots * pool.slot_stride:
        raise RuntimeError(
            f"{PoolKind(pool.kind).name} pool mapping is {base.numel()} B, geometry "
            f"needs {pool.num_slots * pool.slot_stride} B"
        )
    slice_base = writer_idx * pool.slice_bytes
    views = [
        torch.as_strided(
            base,
            (pool.num_slots, page * pool.sub_blocks),
            (pool.slot_stride, 1),
            storage_offset=slice_base + offset,
        )
        for page, offset in zip(layout.page_bytes, layout.offsets)
    ]
    return base, views


@dataclass
class _PoolHandlers:
    plan: PoolPlan
    base: torch.Tensor
    store: SingleDirectionOffloadingHandler
    load: SingleDirectionOffloadingHandler


class RadixShmemOffloadingWorker(OffloadingWorker):
    """``OffloadingWorker`` over the node-shared, per-pool SlotStore."""

    def __init__(
        self,
        *,
        attach: Callable[[], tuple[SharedRegions, int]],
        kv_caches: CanonicalKVCaches,
    ):
        self._attach = attach
        self._kv_caches = kv_caches

        self.regions: SharedRegions | None = None
        self.tp_rank: int | None = None
        self._data_base: torch.Tensor | None = None
        self._registered_ptr: int | None = None
        self._pools: dict[int, _PoolHandlers] | None = None
        # parent job -> outstanding child job ids, and their finished results
        self._children: dict[int, set[int]] = {}
        self._child_results: dict[int, list[TransferResult]] = {}

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
            if self._pools is not None:
                return
            regions, tp_rank = self._attach()
            geometry = regions.geometry
            writer_idx = 0 if geometry.replicated else tp_rank

            data_base = torch.frombuffer(regions.data_view(), dtype=torch.int8)
            t0 = time.perf_counter()
            host_register(data_base.data_ptr(), data_base.numel())
            register_s = time.perf_counter() - t0
            self._registered_ptr = data_base.data_ptr()

            pools: dict[int, _PoolHandlers] = {}
            for plan in plan_pools(geometry, self._kv_caches):
                if not plan.gpu_tensors:
                    continue
                base, cpu_tensors = pool_views(regions, plan, writer_idx)
                common = dict(
                    gpu_tensors=plan.gpu_tensors,
                    cpu_tensors=cpu_tensors,
                    blocks_per_chunk=plan.pool.sub_blocks,
                    layer_refs_per_group=plan.refs_per_group,
                )
                pools[plan.pool.kind] = _PoolHandlers(
                    plan=plan,
                    base=base,
                    store=SingleDirectionOffloadingHandler(gpu_to_cpu=True, **common),
                    load=SingleDirectionOffloadingHandler(gpu_to_cpu=False, **common),
                )
            self.regions = regions
            self.tp_rank = tp_rank
            self._data_base = data_base
            # published last: this is what _ensure_attached checks
            self._pools = pools

            logger.info(
                "RadixShmem worker tp_rank=%d attached: pools %s, writing slice %d "
                "of %d, cudaHostRegister(%.2f GiB) took %.3f s",
                tp_rank,
                ", ".join(
                    f"{PoolKind(k).name}({len(h.plan.gpu_tensors)} tensors, "
                    f"{h.plan.pool.sub_blocks} sub-blocks)"
                    for k, h in pools.items()
                ),
                writer_idx,
                geometry.num_slices,
                data_base.numel() / 2**30,
                register_s,
            )

    def _ensure_attached(self) -> dict[int, _PoolHandlers]:
        """Join the background attach; do it here if it never finished."""
        if self._pools is None:
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
        assert self._pools is not None
        return self._pools

    # ------------------------------------------------------------- jobs

    def _split(
        self, job_id: int, gpu_spec: GPULoadStoreSpec, cpu_spec: LoadStoreSpec
    ) -> list[tuple[int, int, GPULoadStoreSpec, CPULoadStoreSpec]]:
        """Split one connector job into (child_id, pool, gpu, cpu) per pool.

        The CPU ids are consumed group by group exactly as the handler does:
        ``cdiv(group_size + block_index % sub_blocks, sub_blocks)`` per group.
        """
        assert isinstance(cpu_spec, CPULoadStoreSpec) or hasattr(cpu_spec, "block_ids")
        pools = self._ensure_attached()
        assert self.regions is not None
        geometry = self.regions.geometry
        groups = geometry.groups
        num_groups = len(groups)
        gpu_ids = gpu_spec.block_ids
        cpu_ids = cpu_spec.block_ids  # type: ignore[attr-defined]
        per_pool: dict[int, tuple[list[int], list[int], list[int], list[int]]] = {}
        gpu_off = 0
        cpu_off = 0
        for g_idx, (size, first_block) in enumerate(
            zip(gpu_spec.group_sizes, gpu_spec.block_indices)
        ):
            if size == 0:
                continue
            kind = groups[g_idx].kind
            sub_blocks = geometry.require_pool(kind).sub_blocks
            n_cpu = cdiv(size + first_block % sub_blocks, sub_blocks)
            entry = per_pool.setdefault(
                kind, ([], [0] * num_groups, [0] * num_groups, [])
            )
            entry[0].extend(int(b) for b in gpu_ids[gpu_off : gpu_off + size])
            entry[1][g_idx] = int(size)
            entry[2][g_idx] = int(first_block)
            entry[3].extend(int(c) for c in cpu_ids[cpu_off : cpu_off + n_cpu])
            gpu_off += size
            cpu_off += n_cpu
        assert gpu_off == len(gpu_ids), (gpu_off, len(gpu_ids))
        assert cpu_off == len(cpu_ids), (cpu_off, len(cpu_ids))
        parts = []
        for kind, (blocks, sizes, indices, slots) in per_pool.items():
            if kind not in pools:
                raise RuntimeError(f"no handlers for pool {PoolKind(kind).name}")
            parts.append(
                (
                    job_id * _CHILD_STRIDE + kind,
                    kind,
                    GPULoadStoreSpec(blocks, group_sizes=sizes, block_indices=indices),
                    CPULoadStoreSpec(slots),
                )
            )
        return parts

    def _submit(self, job_id: int, gpu_spec, cpu_spec, *, store: bool) -> bool:
        parts = self._split(job_id, gpu_spec, cpu_spec)
        if not parts:
            return False
        pools = self._ensure_attached()
        children: set[int] = set()
        for child_id, kind, gpu, cpu in parts:
            handler = pools[kind].store if store else pools[kind].load
            ok = (
                handler.transfer_async(child_id, gpu, cpu)
                if store
                else handler.transfer_async(child_id, cpu, gpu)
            )
            if not ok:
                return False
            children.add(child_id)
        self._children[job_id] = children
        self._child_results[job_id] = []
        return True

    # -------------------------------------------------- OffloadingWorker

    def submit_store(
        self, job_id: int, src_spec: GPULoadStoreSpec, dst_spec: LoadStoreSpec
    ) -> bool:
        return self._submit(job_id, src_spec, dst_spec, store=True)

    def submit_load(
        self, job_id: int, src_spec: LoadStoreSpec, dst_spec: GPULoadStoreSpec
    ) -> bool:
        return self._submit(job_id, dst_spec, src_spec, store=False)

    def get_finished(self) -> list[TransferResult]:
        if not self._pools:
            return []
        results: list[TransferResult] = []
        for handlers in self._pools.values():
            for child in handlers.store.get_finished() + handlers.load.get_finished():
                parent = child.job_id // _CHILD_STRIDE
                outstanding = self._children.get(parent)
                if outstanding is None:
                    continue
                outstanding.discard(child.job_id)
                self._child_results[parent].append(child)
                if outstanding:
                    continue
                parts = self._child_results.pop(parent)
                del self._children[parent]
                results.append(
                    TransferResult(
                        job_id=parent,
                        success=all(p.success for p in parts),
                        transfer_size=sum(p.transfer_size or 0 for p in parts),
                        transfer_time=max(p.transfer_time or 0.0 for p in parts),
                    )
                )
        return results

    def wait(self, job_ids: set[int]) -> None:
        if not self._pools:
            return
        children = {c for j in job_ids for c in self._children.get(j, ())}
        for handlers in self._pools.values():
            handlers.store.wait(children)
            handlers.load.wait(children)

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
        for handlers in (self._pools or {}).values():
            for handler in (handlers.store, handlers.load):
                try:
                    handler.shutdown()
                except Exception:
                    logger.exception("RadixShmem: error draining transfers at shutdown")
        self._pools = None
        if self._registered_ptr is not None:
            host_unregister(self._registered_ptr)
            self._registered_ptr = None
        self._data_base = None
        if self.regions is not None:
            self.regions.close()
            self.regions = None
