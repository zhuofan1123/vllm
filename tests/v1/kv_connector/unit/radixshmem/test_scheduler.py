# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scheduler-side bookkeeping against a real radix index.

CPU only: no CUDA, no model, no engine. The vLLM objects the scheduler reads
(Request, KVCacheBlocks, SchedulerOutput) are faked down to the handful of
attributes it actually touches. Run inside the radixshmem_vllm container with
PYTHONPATH=/raid/zfl/vllm:/raid/zfl/RadixShmem/python.
"""

import os
from types import SimpleNamespace

import pytest

from vllm.distributed.kv_transfer.kv_connector.v1.radixshmem import (
    compute_geometry,
    create_regions,
)
from vllm.distributed.kv_transfer.kv_connector.v1.radixshmem.metadata import (
    RadixShmemWorkerMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.radixshmem.scheduler import (
    RadixShmemConnectorScheduler,
)

pytest.importorskip("shmradix")
pytest.importorskip("shmradix._data")

GPU_BLOCK_SIZE = 16
PAGE_BYTES = 4096
NUM_LAYERS = 2
TP_SIZE = 2


def make_configs(*, tag: str, factor: int = 1, cpu_bytes: int = 8 << 20):
    extra = {
        "index_shm_name": f"/rs_sched_idx_{tag}",
        "data_shm_name": f"/rs_sched_dat_{tag}",
        "cpu_bytes_to_use": cpu_bytes,
        "slot_align": 4096,
    }
    if factor != 1:
        extra["block_size"] = GPU_BLOCK_SIZE * factor
    vllm_config = SimpleNamespace(
        kv_transfer_config=SimpleNamespace(kv_connector_extra_config=extra),
        parallel_config=SimpleNamespace(
            pipeline_parallel_size=1,
            tensor_parallel_size=TP_SIZE,
            world_size=TP_SIZE,
            data_parallel_index=0,
        ),
        kv_events_config=None,
    )
    kv_cache_config = SimpleNamespace(
        kv_cache_groups=[
            SimpleNamespace(
                kv_cache_spec=SimpleNamespace(
                    block_size=GPU_BLOCK_SIZE, page_size_bytes=PAGE_BYTES
                ),
                layer_names=[f"l{i}" for i in range(NUM_LAYERS)],
            )
        ]
    )
    return vllm_config, kv_cache_config, extra


@pytest.fixture
def sched(request):
    tag = f"{os.getpid()}_{request.node.name[:24]}"
    factor = getattr(request, "param", 1)
    vllm_config, kv_cache_config, extra = make_configs(tag=tag, factor=factor)
    geometry = compute_geometry(vllm_config, kv_cache_config)
    regions = create_regions(geometry, extra)
    s = RadixShmemConnectorScheduler(
        geometry=geometry, regions=regions, vllm_config=vllm_config
    )
    try:
        yield s
    finally:
        s.close()


# ------------------------------------------------------------------- fakes


def fake_request(req_id: str, num_blocks: int, factor: int = 1, *, salt: int = 0):
    """A request whose per-GPU-block hashes are deterministic in (salt, index)."""
    n_gpu_blocks = num_blocks * factor
    block_hashes = [
        (salt * 100000 + i).to_bytes(8, "little") + b"\x00" * 24
        for i in range(n_gpu_blocks)
    ]
    return SimpleNamespace(
        request_id=req_id,
        num_tokens=n_gpu_blocks * GPU_BLOCK_SIZE,
        num_computed_tokens=0,
        block_hashes=block_hashes,
    )


def fake_blocks(block_ids: list[int], num_computed: int):
    """KVCacheBlocks stand-in: cached blocks carry a hash, new ones do not."""
    blocks = [
        SimpleNamespace(block_hash=(b"h" if i < num_computed else None))
        for i in range(len(block_ids))
    ]
    return SimpleNamespace(blocks=[blocks], get_block_ids=lambda: [list(block_ids)])


def fake_output(entries, preempted=None):
    """entries: list of (req_id, new_block_ids, num_scheduled_tokens)."""
    return SimpleNamespace(
        scheduled_new_reqs=[
            SimpleNamespace(req_id=r, block_ids=([list(b)] if b else ()))
            for r, b, _ in entries
        ],
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=[], new_block_ids=[], resumed_req_ids=set()
        ),
        num_scheduled_tokens={r: t for r, _, t in entries},
        preempted_req_ids=None,
    )


def store_and_ack(sched, req, block_ids, num_tokens):
    """Run one full store round-trip and return the StoreJobs it produced."""
    sched.update_state_after_alloc(req, fake_blocks(block_ids, 0), 0)
    meta = sched.build_connector_meta(
        fake_output([(req.request_id, block_ids, num_tokens)])
    )
    for job in meta.stores:
        sched.update_connector_output(
            SimpleNamespace(
                kv_connector_worker_meta=RadixShmemWorkerMetadata(
                    completed_stores={job.store_id: sched.world_size}
                ),
                finished_recving=None,
                finished_sending=None,
            )
        )
    return meta.stores


# ------------------------------------------------------------------- tests


def test_store_then_hit(sched):
    req = fake_request("a", 4)
    block_ids = list(range(4))
    stores = store_and_ack(sched, req, block_ids, 4 * GPU_BLOCK_SIZE)
    assert len(stores) == 1
    assert len(stores[0].spec[1].block_ids) == 4  # one slot per offloaded block

    # a second request with the same prefix must hit all four blocks
    req2 = fake_request("b", 4)
    num_tokens, async_load = sched.get_num_new_matched_tokens(req2, 0)
    assert async_load is True
    assert num_tokens == 4 * GPU_BLOCK_SIZE
    sched.build_connector_meta(fake_output([]))  # sweeps the unused pin


def test_no_hit_before_any_store(sched):
    req = fake_request("a", 4)
    assert sched.get_num_new_matched_tokens(req, 0) == (0, False)


def test_hit_ignored_when_gpu_already_has_it(sched):
    req = fake_request("a", 4)
    store_and_ack(sched, req, list(range(4)), 4 * GPU_BLOCK_SIZE)

    req2 = fake_request("b", 4)
    # GPU prefix cache already covers all four blocks -> nothing to gain
    assert sched.get_num_new_matched_tokens(req2, 4 * GPU_BLOCK_SIZE) == (0, False)
    assert not sched._pending_leases


def test_load_spec_slices_the_root_anchored_lease(sched):
    req = fake_request("a", 4)
    stores = store_and_ack(sched, req, list(range(4)), 4 * GPU_BLOCK_SIZE)
    stored_slots = list(stores[0].spec[1].block_ids)

    req2 = fake_request("b", 4)
    num_tokens, _ = sched.get_num_new_matched_tokens(req2, 2 * GPU_BLOCK_SIZE)
    assert num_tokens == 2 * GPU_BLOCK_SIZE

    block_ids = [10, 11, 12, 13]
    sched.update_state_after_alloc(req2, fake_blocks(block_ids, 2), num_tokens)
    src, dst = sched._reqs_to_load["b"]
    # only the two blocks the GPU is missing, and they are the *tail* slots
    assert list(src.block_ids) == stored_slots[2:]
    assert list(dst.block_ids) == [12, 13]
    assert "b" in sched._load_leases


def test_load_lease_released_on_finished_recving(sched):
    req = fake_request("a", 2)
    store_and_ack(sched, req, list(range(2)), 2 * GPU_BLOCK_SIZE)

    req2 = fake_request("b", 2)
    n, _ = sched.get_num_new_matched_tokens(req2, 0)
    sched.update_state_after_alloc(req2, fake_blocks([9, 8], 0), n)
    assert "b" in sched._load_leases
    sched.update_connector_output(
        SimpleNamespace(
            kv_connector_worker_meta=None,
            finished_recving=["b"],
            finished_sending=None,
        )
    )
    assert "b" not in sched._load_leases


def test_unallocated_pin_is_swept(sched):
    req = fake_request("a", 4)
    store_and_ack(sched, req, list(range(4)), 4 * GPU_BLOCK_SIZE)
    used_after_store = sched.manager.client.mempool_used()

    req2 = fake_request("b", 4)
    sched.get_num_new_matched_tokens(req2, 0)
    assert sched._pending_leases  # pinned, but never allocated
    sched.build_connector_meta(fake_output([]))
    assert not sched._pending_leases
    # the pin is gone, so those slots are evictable again
    assert sched.manager.client.mempool_used() == used_after_store
    assert sched.manager.allocate(sched.geometry.num_slots) is not None


def test_publish_waits_for_every_tp_rank(sched):
    req = fake_request("a", 2)
    sched.update_state_after_alloc(req, fake_blocks([0, 1], 0), 0)
    meta = sched.build_connector_meta(fake_output([("a", [0, 1], 2 * GPU_BLOCK_SIZE)]))
    store_id = meta.stores[0].store_id
    assert sched.world_size == 2

    ack = lambda n: sched.update_connector_output(  # noqa: E731
        SimpleNamespace(
            kv_connector_worker_meta=RadixShmemWorkerMetadata(
                completed_stores={store_id: n}
            ),
            finished_recving=None,
            finished_sending=None,
        )
    )

    ack(1)
    assert store_id in sched._pending_stores
    assert sched.manager.lookup(sched._prefix_u64(req, 2)) == 0  # not visible yet
    ack(1)
    assert store_id not in sched._pending_stores
    assert sched.manager.lookup(sched._prefix_u64(req, 2)) == 2


def test_request_finished_holds_blocks_while_store_in_flight(sched):
    req = fake_request("a", 2)
    sched.update_state_after_alloc(req, fake_blocks([0, 1], 0), 0)
    sched.build_connector_meta(fake_output([("a", [0, 1], 2 * GPU_BLOCK_SIZE)]))
    assert sched.request_finished(req, [0, 1]) == (True, None)


def test_request_finished_waits_for_the_worker_not_for_the_acks(sched):
    """A fully acked store still owes a finished_sending, so hold the blocks.

    The worker reports finished_sending for every request it ever stored for,
    and the scheduler asserts such a request is still in flight; freeing on the
    last ack instead would kill the engine when that report arrives.
    """
    req = fake_request("a", 2)
    store_and_ack(sched, req, list(range(2)), 2 * GPU_BLOCK_SIZE)
    assert sched._store_ids_by_req.get("a") == set()  # every store acked
    assert sched.request_finished(req, [0, 1]) == (True, None)


def test_request_finished_releases_when_nothing_pending(sched):
    req = fake_request("a", 2)
    store_and_ack(sched, req, list(range(2)), 2 * GPU_BLOCK_SIZE)
    sched.update_connector_output(
        SimpleNamespace(
            kv_connector_worker_meta=None,
            finished_recving=None,
            finished_sending=["a"],
        )
    )
    assert sched.request_finished(req, [0, 1]) == (False, None)


def test_second_store_skips_what_a_peer_published(sched):
    req = fake_request("a", 4)
    store_and_ack(sched, req, list(range(4)), 4 * GPU_BLOCK_SIZE)

    # a different request over the same prefix, offloaded from scratch
    req2 = fake_request("b", 4)
    stores = store_and_ack(sched, req2, [20, 21, 22, 23], 4 * GPU_BLOCK_SIZE)
    assert stores == []  # everything was already in the shared index


def test_partial_overlap_stores_only_the_tail(sched):
    store_and_ack(sched, fake_request("a", 2), [0, 1], 2 * GPU_BLOCK_SIZE)
    # same first two blocks, two more after them
    req2 = fake_request("b", 4)
    stores = store_and_ack(sched, req2, [4, 5, 6, 7], 4 * GPU_BLOCK_SIZE)
    assert len(stores) == 1
    assert len(stores[0].spec[1].block_ids) == 2
    assert list(stores[0].spec[0].block_ids) == [6, 7]
    assert sched.manager.lookup(sched._prefix_u64(req2, 4)) == 4


@pytest.mark.parametrize("sched", [4], indirect=True)
def test_block_size_factor_groups_gpu_blocks(sched):
    assert sched.block_size_factor == 4
    req = fake_request("a", 2, factor=4)
    stores = store_and_ack(sched, req, list(range(8)), 8 * GPU_BLOCK_SIZE)
    assert len(stores) == 1
    src, dst = stores[0].spec
    assert len(dst.block_ids) == 2  # two offloaded blocks
    assert list(src.block_ids) == list(range(8))  # eight GPU blocks

    req2 = fake_request("b", 2, factor=4)
    num_tokens, _ = sched.get_num_new_matched_tokens(req2, 0)
    assert num_tokens == 2 * 4 * GPU_BLOCK_SIZE
    sched.build_connector_meta(fake_output([]))


def test_partial_trailing_block_is_not_offloaded(sched):
    req = fake_request("a", 3)
    req.num_tokens = 3 * GPU_BLOCK_SIZE + 5  # ragged tail
    stores = store_and_ack(sched, req, [0, 1, 2], 3 * GPU_BLOCK_SIZE + 5)
    assert len(stores[0].spec[1].block_ids) == 3


def test_release_returns_slots_of_never_acked_stores(sched):
    baseline = sched.manager.client.mempool_used()
    req = fake_request("a", 4)
    sched.update_state_after_alloc(req, fake_blocks(list(range(4)), 0), 0)
    meta = sched.build_connector_meta(
        fake_output([("a", list(range(4)), 4 * GPU_BLOCK_SIZE)])
    )
    assert meta.stores
    assert sched.manager.client.mempool_used() == baseline + 4

    # a crash here would leak those four slots for the life of the region
    sched.release_all()
    assert sched.manager.client.mempool_used() == baseline
