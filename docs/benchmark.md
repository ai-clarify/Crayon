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
