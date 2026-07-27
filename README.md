<p align="center">
  <img src="logo.svg" width="140" alt="Crayon"/>
</p>

<h1 align="center">Crayon</h1>

<p align="center">
A distributed task runtime for RL workloads, in Rust.<br/>
One coordinator, disposable workers, a plasma-style shared-memory object store —
built to beat Ray where RL actually hurts: dispatch latency and data movement.
</p>

## Why

RL rollouts run on ephemeral per-task workers: durability is worthless,
dispatch latency and payload movement are everything. Crayon optimizes exactly
that path and measures itself against Ray on every claim below (V100 host,
8-core Xeon, Linux, release builds; Ray 2.56).

**The win is co-location.** The large-payload lever is the shared-memory arena,
which only fires when client, coordinator, and worker share a host (proof =
the client can mmap the arena file). Across hosts, transfers fall back to the
8 MiB RPC frame and the arena advantage is gone. Crayon is a same-host RL
accelerator first; it runs distributed as a correctness fallback, not as its
fast path.

**Storage** (`storage-benchmark` vs `ray_storage_benchmark.py`, p50):

| Payload | put Crayon / Ray | get Crayon / Ray | Round trip |
|---|---|---|---|
| 1 KB | **80 µs** / 304 µs | **39 µs** / 70 µs | 3.1× |
| 1 MB | **159 µs** / 528 µs | **110 µs** / 151 µs | 2.5× |
| 16 MB | **1.1 ms** / 1.6 ms | **1.0 ms** / 3.3 ms | 2.4× |
| 1 GB | **50 ms** / 66 ms | **132 ms** / 620 ms | 3.8× |

Read amplification is 0.0 at the measured sizes (counting allocator): a get is
a slice out of the mmap, no copy. The shared-memory arena carries same-host
payloads of any size — the 8 MiB RPC frame cap applies only across hosts.

**Scheduling** (`rl-benchmark` vs `ray_rl_benchmark.py`, identical bandit
workload, 6 workers × 16 parallel rollouts × 300 iterations):
**11 799 episodes/s vs 1 552 — 7.6× faster.**

**Real LLM RL** (Qwen3.5-0.8B on one V100, 2 rollout actors + judge + REINFORCE
learner): the 2 GB policy broadcasts through the arena in **381 ms**; 30
training iterations lift arithmetic accuracy 0.24 → 0.55. On the full GSM8K
test split (1 319 problems, greedy), Crayon and Ray produce **bit-identical
scores** (305 correct) at identical GPU-bound throughput — the pipeline adds
no data loss and no overhead where the model dominates.

## Design

- One authoritative coordinator; workers register versioned operations
  (`namespace.name.vN`) — closures are never shipped.
- Plasma-style arena: a single mmap'd file per coordinator; same-host clients
  reserve/commit and read objects as slices. Puts < 1 MiB are
  content-addressed (blake3, dedup); larger puts skip hashing for speed.
- Fenced at-least-once execution: stale sessions and stale attempts cannot
  publish results; mutations are idempotent by request id.
- Event-driven control plane: blocking gets, long-poll workers, batched
  submit — no polling loops; pooled client connections.

## Install

```bash
cargo install crayon-rs --version 0.6.2   # library `crayon`, binary `crayon-cluster`
```

Python client (pyo3, arena-aware — gigabyte puts from Python):

```bash
cd crayon-py && maturin build --release
pip install target/wheels/crayon-*.whl
```

```python
import crayon
c = crayon.local_cluster(workers=4)           # spawns coordinator + workers here
oid = c.put(b"\x00" * (1 << 30))              # 1 GB, via shared memory
task, out = c.submit("llm", "rollout", 1, [b'{"seeds":[1],"max_new_tokens":48}'])
print(c.get(out, timeout_ms=120_000))
c.release(oid)                                # cluster is killed when `c` is dropped
```

Connect to an already-running cluster instead with `crayon.Client("host:port")`.
`local_cluster` needs the `crayon-cluster` binary on `PATH` or at
`$CRAYON_CLUSTER_BIN`.

## Run

One command brings up a coordinator + workers on this host and blocks until
Ctrl-C — the `ray.init()` analogue:

```bash
cargo build --release --bin crayon-cluster

# terminal 1: coordinator + 2 workers, prints the coordinator address
target/release/crayon-cluster local 2

# terminal 2: submit against the printed address
target/release/crayon-cluster submit 127.0.0.1:<port> 20 22
# 42
```

Or wire the processes by hand (what `local` does for you):

```bash
target/release/crayon-cluster coordinator 127.0.0.1:7000   # terminal 1
target/release/crayon-cluster worker 127.0.0.1:7000 127.0.0.1:7001   # terminal 2
target/release/crayon-cluster submit 127.0.0.1:7000 20 22   # terminal 3
```

End-to-end LLM RL / GSM8K (real model, three roles, Python-driven):

```bash
python3 benchmarks/crayon_llm_rl.py --model <hf-model-dir> --iterations 30
python3 benchmarks/crayon_gsm8k.py  --model <hf-model-dir>            # full test split
```

## Verify

```bash
cargo fmt --all -- --check
cargo clippy --all-targets -- -D warnings
cargo test
```

Benchmark methodology: [docs/benchmark.md](docs/benchmark.md). Architecture:
[docs/architecture.md](docs/architecture.md).

## Non-goals

Coordinator HA, actors, arbitrary code shipping, distributed reference
counting, lineage replay, hard preemption, autoscaling, exactly-once external
side effects.
