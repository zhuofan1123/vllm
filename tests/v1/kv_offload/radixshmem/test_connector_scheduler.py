# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The upstream offloading connector scheduler over the RadixShmem spec.

Two connector schedulers (two "instances") in one process share one region:
what the first stores is an external prefix hit for the second, end to end
through ``get_num_new_matched_tokens`` / ``update_state_after_alloc`` /
``build_connector_meta`` / ``update_connector_output``. CPU only; the worker
side is simulated by acknowledging every job.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from tests.v1.kv_connector.unit.offloading_connector.test_config import (
    _make_kv_cache_config,
    _make_vllm_config,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
    OffloadingWorkerMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.config import (
    build_offloading_config,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
    OffloadingConnectorScheduler,
)
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.kv_cache_utils import BlockHash, KVCacheBlock
from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec
from vllm.v1.kv_offload.factory import OffloadingSpecFactory
from vllm.v1.kv_offload.radixshmem.manager import RadixShmemOffloadingManager
from vllm.v1.kv_offload.radixshmem.spec import RadixShmemOffloadingSpec
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import RequestStatus

from .utils import block_hashes, unique_tag

BLOCK = 16


def make_scheduler(tag: str) -> OffloadingConnectorScheduler:
    vllm_config = _make_vllm_config(
        extra_config={
            "spec_name": "RadixShmemOffloadingSpec",
            "index_shm_name": f"/rs_cs_idx_{tag}",
            "data_shm_name": f"/rs_cs_dat_{tag}",
            "cpu_bytes_to_use": 4 << 20,
            "attach_timeout_s": 30,
        }
    )
    vllm_config.speculative_config = None
    # part of the model fingerprint; a MagicMock would differ per scheduler
    vllm_config.cache_config.kv_cache_layout = "BLHNC"
    kv_cache_config = _make_kv_cache_config()
    spec = OffloadingSpecFactory.create_spec(
        build_offloading_config(vllm_config, kv_cache_config)
    )
    assert isinstance(spec, RadixShmemOffloadingSpec)
    return OffloadingConnectorScheduler(spec, vllm_config, kv_cache_config)


def make_request(req_id: str, hashes: list[bytes]) -> MagicMock:
    request = MagicMock()
    request.request_id = req_id
    request.kv_transfer_params = None
    request.num_prompt_tokens = len(hashes) * BLOCK
    request.num_tokens = len(hashes) * BLOCK
    request.num_computed_tokens = 0
    request.block_hashes = [BlockHash(h) for h in hashes]
    request.all_token_ids = list(range(len(hashes) * BLOCK))
    request.lora_request = None
    request.skip_reading_prefix_cache = False
    request.status = RequestStatus.RUNNING
    request.is_finished.return_value = False
    return request


def fake_output(entries, finished=()):
    """entries: (req_id, new_block_ids, num_scheduled_tokens) per request."""
    return SimpleNamespace(
        scheduled_new_reqs=[
            SimpleNamespace(req_id=r, block_ids=([list(b)] if b else ()))
            for r, b, _ in entries
        ],
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=[], new_block_ids=[], resumed_req_ids=set()
        ),
        num_scheduled_tokens={r: t for r, _, t in entries},
        finished_req_ids=set(finished),
        preempted_req_ids=None,
        kv_connector_block_state=None,
        kv_cache_block_copies=None,
    )


def ack(scheduler, job_ids):
    scheduler.update_connector_output(
        KVConnectorOutput(
            kv_connector_worker_meta=OffloadingWorkerMetadata(
                completed_jobs={j: 1 for j in job_ids}
            )
        )
    )


@pytest.fixture
def schedulers(request):
    tag = unique_tag(request)
    first = make_scheduler(tag)
    second = make_scheduler(tag)  # same names -> attaches to the live owner
    yield first, second
    second.shutdown()
    first.shutdown()


def test_prefix_stored_by_one_scheduler_loads_on_another(schedulers):
    first, second = schedulers
    assert isinstance(first.manager, RadixShmemOffloadingManager)
    assert first.manager.regions.is_owner and not second.manager.regions.is_owner

    hashes = block_hashes(4)
    req = make_request("a", hashes)
    first.on_new_request(req)
    assert first.get_num_new_matched_tokens(req, 0) == (0, False)
    first.update_state_after_alloc(req, KVCacheBlocks(([],)), 0)

    meta = first.build_connector_meta(fake_output([("a", [1, 2, 3, 4], 4 * BLOCK)]))
    [job_id] = meta.store_jobs
    store = meta.store_jobs[job_id]
    assert store.src_spec.block_ids.tolist() == [1, 2, 3, 4]
    assert isinstance(store.dst_spec, CPULoadStoreSpec)
    slots = store.dst_spec.block_ids.tolist()
    assert len(slots) == 4

    # not visible until every worker acked the copy
    probe = make_request("p", hashes)
    second.on_new_request(probe)
    assert second.get_num_new_matched_tokens(probe, 0) == (0, False)
    second.build_connector_meta(fake_output([]))
    ack(first, [job_id])

    req2 = make_request("b", hashes)
    second.on_new_request(req2)
    num_hit, is_async = second.get_num_new_matched_tokens(req2, BLOCK)
    assert (num_hit, is_async) == (3 * BLOCK, True)

    blocks = [KVCacheBlock(10), KVCacheBlock(11), KVCacheBlock(12), KVCacheBlock(13)]
    blocks[0]._block_hash = "cached"  # the GPU already had the first block
    second.update_state_after_alloc(req2, KVCacheBlocks((blocks,)), 3 * BLOCK)
    meta = second.build_connector_meta(fake_output([("b", [10, 11, 12, 13], 0)]))
    [load_id] = meta.load_jobs
    load = meta.load_jobs[load_id]
    assert load.dst_spec.block_ids.tolist() == [11, 12, 13]
    assert load.dst_spec.block_indices == [1]
    assert load.src_spec.block_ids.tolist() == slots[1:]  # the very slots stored
    assert second.manager.index.num_open_leases == 1
    ack(second, [load_id])
    assert second.manager.index.num_open_leases == 0


def test_unallocated_hit_is_unpinned_at_step_end(schedulers):
    first, second = schedulers
    hashes = block_hashes(2)
    req = make_request("a", hashes)
    first.on_new_request(req)
    first.get_num_new_matched_tokens(req, 0)
    first.update_state_after_alloc(req, KVCacheBlocks(([],)), 0)
    meta = first.build_connector_meta(fake_output([("a", [1, 2], 2 * BLOCK)]))
    ack(first, list(meta.store_jobs))

    req2 = make_request("b", hashes)
    second.on_new_request(req2)
    assert second.get_num_new_matched_tokens(req2, 0) == (2 * BLOCK, True)
    assert second.manager.index.num_open_leases == 1
    # the core scheduler could not allocate: no update_state_after_alloc
    second.build_connector_meta(fake_output([]))
    assert second.manager.index.num_open_leases == 0
