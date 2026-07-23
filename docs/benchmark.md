# Coordinator benchmark

`cluster-benchmark` measures the current coordinator/worker runtime with real
`crayon-cluster` processes on loopback. It does not use the removed 0.1.x
Python/V100 RL harness; those numbers belong to a different architecture.

## Scenarios

1. **Task** — submit a registered `builtin:add@1` operation, wait for the typed
   terminal result, and verify the output (`20 + 22 = 42`). This exercises
   submit, coordinator scheduling, worker polling, execution, completion
   publication, status polling, object location, and result retrieval.
2. **DAG/object transfer** — submit `builtin:copy@1` with a deterministic
   inline payload, wait for it to complete, then submit a second `copy` that
   depends on the first task's worker-local output object. The final payload is
   verified by BLAKE3 digest. This exercises dependency scheduling, direct
   worker-local object fetch, and checksum verification.

## Run

```bash
cargo build --release --bin crayon-cluster --bin cluster-benchmark

./target/release/cluster-benchmark \
  --workers 1 \
  --concurrency 1 \
  --warmups 3 \
  --samples 30 \
  --payload-bytes 4096 \
  --artifact-dir benchmark_artifacts/v100-w1-4k
```

Arguments:

- `--workers`: number of real worker processes.
- `--concurrency`: number of outstanding benchmark tasks.
- `--warmups`: samples excluded from `summary.json` but kept in `samples.csv`.
- `--samples`: measured samples per scenario.
- `--payload-bytes`: deterministic DAG payload size (capped by the `copy`
  inline limit of 64 KiB).
- `--artifact-dir`: output directory.

## Artifacts

Each run emits:

- `manifest.json`: command, Git SHA/dirty state, Rust version, OS, CPU, logical
  CPU count, topology, and parameters.
- `samples.csv`: raw warmup and measured timings.
- `summary.json`: count, median, mean, standard deviation, p25/p75/p90/p95,
  minimum, maximum, and operations/second for each scenario.
- `process.log`: merged coordinator/worker stdout/stderr.

## Measurement rules

- Release binaries are used.
- The benchmark starts its own coordinator and workers; startup is excluded.
- Every output is verified before a sample is recorded; a mismatch fails the run.
- Warmup and measured rows are separate.
- The topology is loopback; results are not multi-host network numbers.

## Interpretation limits

These numbers measure the control-plane and worker-local object path of the
0.2 coordinator runtime. They are not a universal framework-overhead comparison
and are not comparable to the deleted 0.1.x Python/RL benchmark. Payload size
has little effect at 4–64 KiB because control-plane latency dominates.

# RL rollout benchmark

`rl_benchmark` is a long-running distributed RL training simulation that
exercises multi-stage task graphs, object dependencies, retries, and
cancellation on a real coordinator/worker cluster. It simulates the
rollout-collection phase: the coordinator hands out seeded environment
rollouts to workers, each worker runs a fixed number of steps with the current
policy parameters, and the coordinator aggregates episode returns and nudges
the policy each iteration.

This is a verification workload, not a performance claim. It exists to keep the
runtime honest under sustained load and to provide an apples-to-apples
comparison point with `benchmarks/ray_rl_benchmark.py`.

## Run

```bash
cargo build --release --bin crayon-cluster --bin rl_benchmark

./target/release/rl_benchmark \
  --workers 2 \
  --parallelism 8 \
  --iterations 100 \
  --steps 100 \
  --policy-dim 16 \
  --artifact-dir benchmark_artifacts/rl
```

The benchmark starts its own coordinator and workers on loopback; startup is
excluded from timing. Every iteration submits `--parallelism` rollout tasks,
waits for all of them, and applies a policy update. A task failure is logged
but does not abort the run.

Arguments:

- `--workers`: number of real worker processes.
- `--parallelism`: outstanding rollout tasks per iteration.
- `--iterations`: number of policy-update iterations.
- `--steps`: environment steps per rollout.
- `--policy-dim`: linear policy parameter dimension.
- `--max-attempts`: task retry limit.
- `--seed`: base seed for environment generation.
- `--artifact-dir`: output directory.

## Artifacts

Each run emits:

- `manifest.json`: topology, parameters, total episodes, total time, and
  episodes/second.
- `samples.csv`: per-iteration episode count, mean return, and elapsed
  milliseconds.
- `process-*.log`: merged coordinator/worker stdout/stderr.

## Ray comparison

`benchmarks/ray_rl_benchmark.py` runs the same seeded bandit environment,
linear policy, and per-iteration update under Ray so the two systems can be
compared on identical per-task compute. It emits the same `manifest.json` and
`samples.csv` schema. Install Ray (`pip install ray`) and run:

```bash
python benchmarks/ray_rl_benchmark.py \
  --workers 2 \
  --parallelism 8 \
  --iterations 100 \
  --steps 100
```

