# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker side of RadixShmem offloading.

Nothing is allocated here: the CPU side is one node-wide shared SlotStore that
the owning scheduler created, and this worker only attaches to it, pins the
mapping for DMA, and exposes its own slice of every slot as strided int8
tensors. The transfers themselves are upstream's
``SingleDirectionOffloadingHandler`` -- batch copies over the same address
arithmetic the private CPU pool uses, so there is no RadixShmem copy path.

The store has one pool per attention kind (see ``geometry.py``), and inside a
pool every KV cache group owns its own byte region of each slot -- a group's
region holds its ``sub_blocks`` GPU blocks of one full-attention position, and
groups of one pool may differ in ``sub_blocks`` (DeepSeek-V4's windowed groups
span 4 / 32 / 64 blocks). So there is one pair of handlers *per group*, and a
connector job is split by group before submission; the parent job completes
when every part has.

Attaching is deferred to a background thread started at construction, joined by
the first transfer. Workers are built during ``initialize_from_config``, before
any scheduler exists, so attaching eagerly would wait forever on the scheduler
that creates the regions; doing it inline on the first transfer instead costs
~1 s per 10 GiB (cudaHostRegister) with the engine step loop stopped.
"""

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

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
    GroupLayout,
    PoolKind,
    SlotGeometry,
    group_tensor_layout,
    verify_group_layout,
)

logger = init_logger(__name__)

# a parent job id j fans out to child ids j * _CHILD_STRIDE + group_idx
_CHILD_STRIDE = 64


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
class _GroupTensors:
    """GPU tensors and canonical refs for one KV cache group."""

    group: GroupLayout
    gpu_tensors: list[torch.Tensor]
    refs: list[CanonicalKVCacheRef]  # into gpu_tensors, in layer order

    @property
    def page_bytes(self) -> list[int]:
        return [t.shape[1] for t in self.gpu_tensors]


def plan_groups(
    geometry: SlotGeometry, kv_caches: CanonicalKVCaches
) -> list[_GroupTensors]:
    """Split the canonical KV caches into per-group GPU tensors.

    Layer-outermost layouts already come as one canonical tensor per layer (or
    per set of aliased layers); each group's refs pick its tensors. Packed
    block-outermost layouts (e.g. DeepSeek-V4) come as a single tensor covering
    the whole block for every group, so per-layer strided views are carved out
    of it using the layer offsets the geometry carries.
    """
    packed = (
        len(kv_caches.tensors) == 1
        and any(g.layer_pages for g in geometry.groups)
        and all(
            len(refs) == 1 and refs[0].tensor_idx == 0
            for refs in kv_caches.group_data_refs
        )
    )
    plans: list[_GroupTensors] = []
    if packed:
        packed_tensor = kv_caches.tensors[0].tensor
        block_stride = kv_caches.tensors[0].page_size_bytes
        base = packed_tensor.view(torch.int8).view((-1, block_stride))
        num_blocks = base.shape[0]
        for group in geometry.groups:
            tensors: list[torch.Tensor] = []
            refs: list[CanonicalKVCacheRef] = []
            for offset, page in group.layer_pages:
                if offset + page > block_stride:
                    raise RuntimeError(
                        f"layer page [{offset}, {offset + page}) exceeds the packed "
                        f"block of {block_stride} bytes"
                    )
                tensors.append(
                    torch.as_strided(
                        base,
                        (num_blocks, page),
                        (block_stride, 1),
                        storage_offset=offset,
                    )
                )
                refs.append(CanonicalKVCacheRef(len(tensors) - 1, page))
            plans.append(_GroupTensors(group, tensors, refs))
        return plans

    for group in geometry.groups:
        tensors = []
        refs = []
        local_idx: dict[int, int] = {}
        for ref in kv_caches.group_data_refs[group.group_idx]:
            if ref.tensor_idx not in local_idx:
                t = kv_caches.tensors[ref.tensor_idx]
                tensors.append(t.tensor.view(torch.int8).view((-1, t.page_size_bytes)))
                local_idx[ref.tensor_idx] = len(tensors) - 1
            refs.append(
                CanonicalKVCacheRef(local_idx[ref.tensor_idx], ref.page_size_bytes)
            )
        plans.append(_GroupTensors(group, tensors, refs))
    return plans


def group_cpu_views(
    regions: SharedRegions, plan: _GroupTensors, writer_idx: int
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Strided per-tensor views of this writer's region of a group's pool.

    Returns the flat int8 tensor over the pool (keep it alive: the views borrow
    its storage) and, per GPU tensor ``t`` of the group, an int8 view of shape
    ``(num_slots, page_bytes[t] * sub_blocks)`` with row stride ``slot_stride``
    -- the CPU-side shape ``SingleDirectionOffloadingHandler`` expects,
    addressed by slot id.
    """
    group = plan.group
    pool = regions.geometry.require_pool(PoolKind(group.kind))
    layout = group_tensor_layout(group, plan.page_bytes)
    verify_group_layout(pool, group, layout)
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
            (pool.num_slots, page * group.sub_blocks),
            (pool.slot_stride, 1),
            storage_offset=slice_base + offset,
        )
        for page, offset in zip(layout.page_bytes, layout.offsets)
    ]
    return base, views


@dataclass
class _GroupHandlers:
    group: GroupLayout
    cpu_base: torch.Tensor
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
        self._groups: dict[int, _GroupHandlers] | None = None
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
            if self._groups is not None:
                return
            regions, tp_rank = self._attach()
            geometry = regions.geometry
            writer_idx = 0 if geometry.replicated else tp_rank

            data_base = torch.frombuffer(regions.data_view(), dtype=torch.int8)
            t0 = time.perf_counter()
            host_register(data_base.data_ptr(), data_base.numel())
            register_s = time.perf_counter() - t0
            self._registered_ptr = data_base.data_ptr()

            groups: dict[int, _GroupHandlers] = {}
            for plan in plan_groups(geometry, self._kv_caches):
                if not plan.gpu_tensors:
                    continue
                cpu_base, cpu_tensors = group_cpu_views(regions, plan, writer_idx)
                common = dict(
                    gpu_tensors=plan.gpu_tensors,
                    cpu_tensors=cpu_tensors,
                    blocks_per_chunk=plan.group.sub_blocks,
                    layer_refs_per_group=[plan.refs],
                )
                groups[plan.group.group_idx] = _GroupHandlers(
                    group=plan.group,
                    cpu_base=cpu_base,
                    store=SingleDirectionOffloadingHandler(gpu_to_cpu=True, **common),
                    load=SingleDirectionOffloadingHandler(gpu_to_cpu=False, **common),
                )
            self.regions = regions
            self.tp_rank = tp_rank
            self._data_base = data_base
            # published last: this is what _ensure_attached checks
            self._groups = groups

            logger.info(
                "RadixShmem worker tp_rank=%d attached: %d groups over pools %s, "
                "writing slice %d of %d, cudaHostRegister(%.2f GiB) took %.3f s",
                tp_rank,
                len(groups),
                ", ".join(
                    f"{PoolKind(p.kind).name}({p.num_slots}x{p.slot_stride}B)"
                    for p in geometry.pools
                ),
                writer_idx,
                geometry.num_slices,
                data_base.numel() / 2**30,
                register_s,
            )

    def _ensure_attached(self) -> dict[int, _GroupHandlers]:
        """Join the background attach; do it here if it never finished."""
        if self._groups is None:
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
        assert self._groups is not None
        return self._groups

    # ------------------------------------------------------------- jobs

    def _split(
        self, job_id: int, gpu_spec: GPULoadStoreSpec, cpu_spec: LoadStoreSpec
    ) -> list[tuple[int, int, GPULoadStoreSpec, CPULoadStoreSpec]]:
        """Split one connector job into (child_id, group, gpu, cpu) per group.

        The connector orders GPU blocks and CPU slots by group; each group of
        ``group_size`` blocks consumes ``cdiv(group_size + block_index %
        sub_blocks, sub_blocks)`` CPU slots -- exactly what the handler consumes.
        """
        groups = self._ensure_attached()
        gpu_ids = gpu_spec.block_ids
        cpu_ids = cpu_spec.block_ids  # type: ignore[attr-defined]
        parts: list[tuple[int, int, GPULoadStoreSpec, CPULoadStoreSpec]] = []
        gpu_off = 0
        cpu_off = 0
        for g_idx, (size, first_block) in enumerate(
            zip(gpu_spec.group_sizes, gpu_spec.block_indices)
        ):
            if size == 0:
                continue
            handlers = groups.get(g_idx)
            if handlers is None:
                raise RuntimeError(f"no handlers for KV cache group {g_idx}")
            sub_blocks = handlers.group.sub_blocks
            n_cpu = cdiv(size + first_block % sub_blocks, sub_blocks)
            parts.append(
                (
                    job_id * _CHILD_STRIDE + g_idx,
                    g_idx,
                    GPULoadStoreSpec(
                        [int(b) for b in gpu_ids[gpu_off : gpu_off + size]],
                        group_sizes=(size,),
                        block_indices=(first_block,),
                    ),
                    CPULoadStoreSpec(
                        [int(c) for c in cpu_ids[cpu_off : cpu_off + n_cpu]]
                    ),
                )
            )
            gpu_off += size
            cpu_off += n_cpu
        assert gpu_off == len(gpu_ids), (gpu_off, len(gpu_ids))
        assert cpu_off == len(cpu_ids), (cpu_off, len(cpu_ids))
        return parts

    def _submit(self, job_id: int, gpu_spec, cpu_spec, *, store: bool) -> bool:
        parts = self._split(job_id, gpu_spec, cpu_spec)
        if not parts:
            return False
        groups = self._ensure_attached()
        children: set[int] = set()
        for child_id, g_idx, gpu, cpu in parts:
            handler = groups[g_idx].store if store else groups[g_idx].load
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
        if not self._groups:
            return []
        results: list[TransferResult] = []
        for handlers in self._groups.values():
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
        if not self._groups:
            return
        children = {c for j in job_ids for c in self._children.get(j, ())}
        for handlers in self._groups.values():
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
        for handlers in (self._groups or {}).values():
            for handler in (handlers.store, handlers.load):
                try:
                    handler.shutdown()
                except Exception:
                    logger.exception("RadixShmem: error draining transfers at shutdown")
        self._groups = None
        if self._registered_ptr is not None:
            host_unregister(self._registered_ptr)
            self._registered_ptr = None
        self._data_base = None
        if self.regions is not None:
            self.regions.close()
            self.regions = None
