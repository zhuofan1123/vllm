# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared helpers for the RadixShmem offloading tests.

All tests here need a real ``shmradix`` build (index + ``_data`` extension); the
shared regions are created under unique names per test and torn down after.
"""

import os
from collections.abc import Sequence
from typing import Any

import pytest

from vllm.v1.kv_offload.base import ReqContext, make_offload_key
from vllm.v1.kv_offload.config import (
    OffloadingCacheConfig,
    OffloadingConfig,
    OffloadingGroupConfig,
    OffloadingModelConfig,
    OffloadingParallelConfig,
)
from vllm.v1.kv_offload.radixshmem.geometry import SlotGeometry, compute_geometry

pytest.importorskip("shmradix")
pytest.importorskip("shmradix._data")

TOKENS_PER_BLOCK = 16
PAGE_BYTES = 4096
NUM_LAYERS = 2


def make_offloading_config(
    *,
    tag: str,
    tp_size: int = 2,
    tokens_per_block: int = TOKENS_PER_BLOCK,
    blocks_per_chunk: int = 1,
    worker_bytes_per_block: int = PAGE_BYTES * NUM_LAYERS,
    cpu_bytes: int = 8 << 20,
    groups: Sequence[tuple[int, bool]] | None = None,
    rank: int = 0,
    data_parallel_index: int = 0,
    replicated_layout: bool = False,
    extra: dict[str, Any] | None = None,
) -> OffloadingConfig:
    """An ``OffloadingConfig`` the way ``build_offloading_config`` would shape it.

    ``groups`` is a list of (tokens_per_block, is_full_attention); the default
    is one full-attention group.
    """
    extra_config: dict[str, Any] = {
        "spec_name": "RadixShmemOffloadingSpec",
        "index_shm_name": f"/rs_t_idx_{tag}",
        "data_shm_name": f"/rs_t_dat_{tag}",
        "cpu_bytes_to_use": cpu_bytes,
        "slot_align": 4096,
        "attach_timeout_s": 60,
    }
    if extra:
        extra_config.update(extra)
    if groups is None:
        groups = [(tokens_per_block, True)]
    return OffloadingConfig(
        groups=tuple(
            OffloadingGroupConfig(
                tokens_per_block=tpb,
                layer_names=(f"g{i}_l{j}" for j in range(NUM_LAYERS)),
                is_full_attention=full,
            )
            for i, (tpb, full) in enumerate(groups)
        ),
        worker_kv_bytes_per_block=worker_bytes_per_block,
        enable_kv_cache_events=False,
        extra_config=extra_config,
        engine_id=f"engine-{tag}-dp{data_parallel_index}",
        model=OffloadingModelConfig(name="test-model", dtype="float16"),
        cache=OffloadingCacheConfig(
            tokens_per_hash=tokens_per_block, blocks_per_chunk=blocks_per_chunk
        ),
        parallel=OffloadingParallelConfig(
            rank=rank,
            world_size=tp_size,
            tp_size=tp_size,
            pp_size=1,
            pcp_size=1,
            dcp_size=1,
            data_parallel_index=data_parallel_index,
            data_parallel_size=1,
            data_parallel_rank_local=None,
            is_parallelism_agnostic=False,
        ),
        replicated_layout=replicated_layout,
    )


def geometry_for(config: OffloadingConfig) -> SlotGeometry:
    return compute_geometry(config)


def unique_tag(request) -> str:
    return f"{os.getpid()}_{abs(hash(request.node.name)) % 10**6}"


def block_hashes(num_hashes: int, *, salt: int = 0) -> list[bytes]:
    """Deterministic 32-byte block hashes, distinct per (salt, index)."""
    return [
        (salt * 100000 + i).to_bytes(8, "little") + b"\x00" * 24
        for i in range(num_hashes)
    ]


def req_context(req_id: str, hashes: list[bytes]) -> ReqContext:
    return ReqContext(req_id=req_id, block_hashes=hashes)


def keys_for(hashes: list[bytes], group_idx: int = 0, hashes_per_chunk: int = 1):
    """Offload keys of every complete chunk, in chunk order."""
    n = len(hashes) // hashes_per_chunk
    return [
        make_offload_key(hashes[(i + 1) * hashes_per_chunk - 1], group_idx)
        for i in range(n)
    ]
