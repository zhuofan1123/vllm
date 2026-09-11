# RadixShmem KV offloading — integration state

Node-shared (and optionally cross-node) CPU KV cache for vLLM v1, backed by the
external **RadixShmem** library (`shmradix`: a shared-memory radix prefix index,
a SlotStore, and a per-node `RadixServer` that also runs cross-node transfers).
Unlike the built-in `CPUOffloadingSpec` (private per-engine pool), one index and
one SlotStore are shared by every DP rank, every TP worker, and every other vLLM
instance on the node that uses the same server name.

Branch `radixshmem-kv-offload-main` (upstream/main + these commits). Re-based on
the 2026-09 RadixShmem redesign (`RadixServer` / `RadixClient` over gRPC,
`pull_async`, `local_read_only`): vLLM is a **client only**, plus the one
scheduler that starts the node's server in-process.

## Wiring

RadixShmem is an `OffloadingSpec` plugin of upstream's native `OffloadingConnector`:
`--kv-offloading-backend radixshmem` → `vllm/config/vllm.py` sets
`kv_connector="OffloadingConnector"` and `spec_name="RadixShmemOffloadingSpec"`;
registered in `vllm/v1/kv_offload/factory.py`. The connector scheduler drives
our `OffloadingManager` through the per-key contract in `vllm/v1/kv_offload/base.py`
(`lookup()` → `HIT / MISS / RETRY`, `prepare_*/complete_*`). Cross-node deferral
relies on `RETRY` only; no scheduler changes.

## Files

- `geometry.py` — `SlotGeometry`/`GroupLayout`/`PoolSpec` from `OffloadingConfig`
  (FULL / SWA / MAMBA pools, group regions inside a slot). `compute_geometry`
  sizes the server; `adopt(client)` takes the server's slot counts and real
  strides and fails closed on chunking / pool / window / slot-width mismatch.
- `bootstrap.py` — `open_client(geometry, extra, role, may_start, read_only)`:
  under a flock on `/dev/shm/<name>.vllm.lock`, probe the gRPC socket; no live
  server → `start_server` (an in-process `shmradix.RadixServer` built by
  `server_config`, which maps `IndexConfig`/`DataPlaneConfig`/`ClusterConfig`
  fields by name from `kv_connector_extra_config`). Then `RadixClient(name)`.
  Workers use `may_start=False, read_only=True` (`local_read_only` client).
- `spec.py` — `RadixShmemOffloadingSpec`: `get_manager` (open + adopt + manager),
  `get_worker` (deferred read-only attach), metrics.
- `manager.py` — `RadixIndex` (ledger over the client: query/pin/allocate/
  publish) and `RadixShmemOffloadingManager`. Remote pulls: a beyond-hit lookup
  with a gap ≥ `remote_lookup_min_blocks` calls `client.pull_async(path,
  mask=pool_mask, lock=False, block=False)` once per admission and returns
  `RETRY`; `_collect_remote` finishes the `PullJob` at the next lookup and drops
  the round lease so the re-pin sees the promoted prefix.
- `worker.py` — attaches the SlotStore through the client, `cudaHostRegister`s
  it, GPU↔slot copies via upstream's `SingleDirectionOffloadingHandler` (one
  handler pair per group). Unchanged by the redesign apart from the attach.

## Decisions

- Server is started by vLLM (first scheduler through the lock), not required
  externally; the cache lives as long as that process. Multi-instance sharing
  works because later instances find the live server.
- Both scheduler and worker recompute the geometry from their own config and
  only adopt counts/strides from the server; no geometry is passed between
  processes (the old sentinel is gone). Worker byte layout must fit the
  server's stride or the worker fails closed.
- One combined-mask `pull_async` per request (FULL+SWA+MAMBA): for hybrids a
  FULL prefix the windowed groups cannot cover is useless to the request anyway.
  Local lookups still query FULL and aux independently.
- Dropped safety nets vs. the sentinel design: NONE_HASH fingerprint (mismatch
  only means no sharing; warning kept) and model fingerprint (rely on `name`;
  a `tag` field in radixshmem would restore it).

## Testing

- Container `zfl-dev`; venv `/workspace/vllm/.venv`; `shmradix` editable from
  `/workspace/radixshmem` (branch dev). Rebuild `.so` after C++ changes.
- `PYTHONHASHSEED=0 .venv/bin/python -m pytest tests/v1/kv_offload/radixshmem -q --noconftest`
  (59 tests, CPU + GPU worker round trips; each test starts its own in-process
  RadixServer under a unique name).
- `vllm serve` e2e (2026-09-09, DeepSeek-V4-Flash-0731, scripts in
  `/workspace/e2e-radixshmem/`): single instance TP2 (`serve_solo.sh` +
  `client_clean.py`, warm hit 21/22 chunks, output equal to two cold computes
  on a warmed server, logprob top-1 agrees) and DP2xTP2 (`serve_dp2.sh` +
  `client_dp.py`: DP0 started the server, DP1 + 4 workers attached, a prefix
  stored by DP1 loaded on DP0). Greedy text is not a reliable oracle for this
  model: consecutive cold computes on one engine differ. Graceful stop needs
  `--shutdown-timeout N`; with the default 0 the engine core is SIGKILLed
  before `RadixServer.close()` and regions stay in `/dev/shm`. DP2 needs
  `VLLM_ENGINE_READY_TIMEOUT_S` above 600 (two ranks load 160 GB at once).
  The DSv4 FP8 checkpoint currently fails in DeepGEMM `o_proj` before any
  offloading code runs (model-side issue on this branch).
- Not yet done: cluster (multi-node) `vllm serve` e2e.
