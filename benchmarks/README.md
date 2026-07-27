# Crayon vs Ray — RL rollout benchmark

Apples-to-apples comparison of Crayon's distributed task runtime against Ray on
an **identical** workload: the only difference is the executor.

## Workload

The rollout-collection phase of RL, run by both systems with the same parameters:

- a coordinator hands out seeded environment rollouts to worker processes,
- each worker runs a fixed-step LCG bandit rollout under the current linear
  policy (`theta`),
- the coordinator aggregates episode returns and nudges `theta` each iteration.

The per-task compute is a byte-for-byte port of the same LCG bandit
(`src/bin/crayon_cluster.rs::run_rollout` ↔ `benchmarks/ray_rl_benchmark.py::rollout`),
so the two runs learn the same curve and the benchmark measures **system
overhead, not algorithm speed**. Identical `mean_return` at the final iteration
(2524 ≈ 2.5e+03) confirms the workloads match exactly.

## Result

Config for every run:
`--workers 4 --parallelism 16 --iterations 1000 --steps 500 --policy-dim 32 --seed 42`.

On a server-class host (Intel Xeon Platinum 8260, 8 logical CPUs):

| System         | episodes/s | total time | per iteration |
|----------------|-----------:|-----------:|--------------:|
| **Crayon**     |   **9924** |     1.61 s |        1.6 ms |
| Ray 2.56.1     |       1764 |     9.07 s |        9.1 ms |
| **Speed-up**   |   **5.6×** |            |               |

Crayon's per-iteration cost is flat to completion — no degradation as tasks
accumulate — so the run length is unbounded. On the Xeon host this workload was
6.1× *slower* than Ray before the 0.3.0 control-plane rewrite; it is now 5.6×
faster — a ~34× swing on the same hardware.

## Scaling with worker count

Control-plane-bound sweep (`--parallelism 32 --iterations 150 --steps 1
--policy-dim 8`), one worker process per logical node, on the Xeon host. This
isolates scheduler overhead: the task itself is nearly free, so throughput
tracks how well each system's control plane scales with node count.

| workers | Crayon (ep/s) | Ray 2.56 (ep/s) | Crayon advantage |
|--------:|--------------:|----------------:|-----------------:|
|       1 |          5468 |             955 |            5.7×  |
|       2 |          7682 |               — |               —  |
|       4 |         10655 |            2071 |            5.1×  |
|       8 |         13860 |            1865 |            7.4×  |
|      16 |         15432 |            1163 |           13.3×  |

Crayon rises with worker count past the host's core count. **Ray peaks at 4
workers and then regresses** — adding workers makes it slower, as central (GCS +
Python) scheduling contention and plasma pressure outweigh the added
parallelism. The advantage therefore widens with scale: 5.7× at one worker,
13.3× at sixteen, with the trend pointing higher on a real multi-node cluster
where per-task central-scheduler cost dominates.

## Scaling with per-task compute

Fixed config (`--workers 4 --parallelism 16 --iterations 1000 --policy-dim 32`),
varying `--steps` to make each rollout heavier, on the Xeon host:

| steps/task | Crayon total | Ray total | speed-up | saved / 1000 iters |
|-----------:|-------------:|----------:|---------:|-------------------:|
|          1 |       1.56 s |    8.04 s |    5.2×  |             6.5 s  |
|        100 |       1.61 s |    8.24 s |    5.1×  |             6.6 s  |
|        500 |       1.57 s |    9.11 s |    5.8×  |             7.5 s  |
|       2000 |       1.63 s |   11.99 s |    7.4×  |            10.4 s  |

Crayon's wall-clock is **flat** as the task gets heavier (1.56 → 1.63 s) because
its four workers run rollouts in true parallel. Ray gets **linearly slower**
(8.04 → 11.99 s): its Python scheduling layer and the GIL serialize even the
compute, so heavier tasks are not fully parallelized. The consequence is
counter-intuitive but measured — **the heavier the per-task compute, the larger
Crayon's end-to-end advantage** (7.4× at steps=2000), the opposite of a
framework whose only edge is scheduling overhead.

## Why Crayon is faster here

The workload is control-plane-bound: tasks are tiny (~0.5 ms) and numerous, so
whoever spends less per task on scheduling and result transfer wins.

- **Event-driven, not polled.** Workers long-poll for work and clients block on
  results; the coordinator parks each request on a `Notify` and wakes it the
  instant state changes. No fixed poll interval sits on the critical path.
- **Batched round-trips.** One RPC submits a whole iteration's tasks and one
  blocking RPC awaits them all (`submit_batch` / `results`, mirroring
  `ray.get([refs])`), instead of one submit + one wait per task.
- **Inline small results.** Task completions ship outputs ≤64 KiB inline, so the
  coordinator answers a result fetch in one hop instead of redirecting the
  client to the producing worker.
- **O(1) scheduling.** The coordinator keeps a runnable queue and a waiting
  index, so assignment and dependency reconciliation never scan the full task
  table.

Ray's overhead here is its Python scheduling layer plus plasma object-store
serialization on every small result — fixed costs that dominate when the task
itself is nearly free.

## When the gap closes

If per-task compute is large (raise `--steps`), both systems become CPU-bound
and converge to the same hardware ceiling — the framework stops mattering. The
measured speed-up reflects control-plane efficiency, which is exactly what shows
up in fan-out workloads of many small tasks.

## Reproduce

```sh
# Crayon (release)
cargo build --release --bin rl-benchmark --bin crayon-cluster
./target/release/rl-benchmark \
  --workers 4 --parallelism 16 --iterations 1000 --steps 500 \
  --policy-dim 32 --seed 42 --artifact-dir /tmp/crayon-rl

# Ray (same config)
pip install 'ray>=2.5'
python3 benchmarks/ray_rl_benchmark.py \
  --workers 4 --parallelism 16 --iterations 1000 --steps 500 \
  --policy-dim 32 --seed 42 --artifact-dir /tmp/ray-rl
```

Each writes `manifest.json` (summary) and `samples.csv` (per-iteration) to its
artifact directory.
