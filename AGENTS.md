# Crayon — Agent Contract

Project-specific rules only; generic Rust/git knowledge is intentionally
absent. `CLAUDE.md` is a symlink to this file — edit here.

## What this is

A Rust distributed task runtime (coordinator + workers + client) competing
with Ray on **RL rollout workloads**: ephemeral per-task containers, so
durability is low-value and **dispatch latency + same-host data movement are
the levers**. Every performance claim is measured against Ray on the `v100`
host, never asserted.

## Architecture map

| Path | Role |
|---|---|
| `src/coordinator.rs` | Cluster state machine: tasks, objects, leases, scheduling (locality look-ahead), retry classes |
| `src/cluster.rs` | RPC server + dispatch, replay cache, client connection pool, blake3 checksum |
| `src/arena.rs` | Plasma-style shared-memory arena: single mmap, reserve/commit, parallel memcpy |
| `src/client.rs` | `ClusterClient`: arena-aware put/get, submit, blocking batch fetch |
| `src/data_plane.rs` | Worker-local object store |
| `src/protocol.rs` | Wire types; `MAX_FRAME_BYTES` = 8 MiB caps every TCP frame |
| `src/bin/crayon_cluster.rs` | Coordinator/worker binary; worker op registry incl. `llm-actor`/`llm-judge` roles |
| `crayon-py/` | pyo3 Python client (`import crayon`); build with maturin |
| `benchmarks/` | Ray baselines (`ray_*.py`) and the Python-driven LLM RL e2e (`crayon_llm_rl.py` + `llm_sidecar.py`) |

## Invariants (violating these is a bug, not a choice)

- **Same-host proof is arena mappability.** A client that can mmap the arena
  file is co-located; everything else falls back to TCP and is bounded by the
  8 MiB frame. Only the arena path may exceed one frame.
- **Puts < 1 MiB are content-addressed** (id = blake3, dedup); **≥ 1 MiB skip
  hashing** — random id, all-zero checksum means "unhashed, size-only check".
  Both sides of every checksum consumer must honor the all-zero sentinel.
- **Worker task outputs must stay ≤ ~8 MiB** (inline ≤ 64 KiB, else served
  worker-local over one frame). Large data flows client→arena→worker, never
  worker→client.
- **Mutations are idempotent by request id** (replay cache); a pooled
  connection retry must dial fresh, and `is_mutation` must list every mutating
  `ClientRequest` variant.
- **Wire shapes duplicated across languages stay in sync by hand**: the
  seed→prompt derivation and JSON shapes live in `src/bin/crayon_cluster.rs`,
  `benchmarks/llm_sidecar.py`, and `benchmarks/crayon_llm_rl.py`.

## Benchmarks are the ground truth

- Perf work starts by measuring, ends by re-measuring; wall-clock numbers from
  the Linux target, not macOS (no /dev/shm there — results mislead).
- v100 host: `/usr/bin/ssh v100` (plain `ssh` breaks in non-interactive
  shells), code in `~/crayon-bench/`, sync with
  `rsync -az -e /usr/bin/ssh --exclude target --exclude .git . v100:~/crayon-bench/`.
- Storage: `storage-benchmark` vs `benchmarks/ray_storage_benchmark.py`.
  RL: `rl-benchmark` vs `benchmarks/ray_rl_benchmark.py`. Real-LLM e2e:
  `benchmarks/crayon_llm_rl.py --model ~/models/Qwen3.5-0.8B`.

## Workflow

- **MANDATORY: extreme brevity in code and comments.** As few as possible —
  fewer lines, fewer words, fewer comments. Delete any comment that restates
  the code; keep only what explains *why* or a non-obvious invariant. Every
  line must earn its place. Not a preference — a hard rule.
- **Simplify before every commit.** Before staging changes, do a simplification pass:
  delete dead code, replace custom logic with stdlib/native equivalents, collapse
  duplication, drop unused abstractions. A commit that adds functionality must
  also remove anything it makes obsolete. "It works" is not sufficient — it must
  also be minimal. This is mandatory, not optional.
- `cargo test` green before every commit; `cargo build --release` for anything
  benchmarked (debug numbers are noise).
- Python client: `cd crayon-py && maturin build --release`, then
  `pip install target/wheels/*.whl --force-reinstall`.
- Smallest working diff; delete before adding; mark deliberate shortcuts with
  a `ponytail:` comment naming the ceiling and upgrade path.
- Repo artifacts (code, docs, commits) in English.
