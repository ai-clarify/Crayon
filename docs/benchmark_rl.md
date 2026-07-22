# RL benchmark methodology

`benchmark_rl.py` compares four explicitly different execution modes:

- `crayon-shared`: in-process Crayon tasks share the live model object.
- `crayon-serialized`: Crayon stores pickled payloads through its raw-byte API.
- `ray-actor`: persistent Ray actors hold models; this is the primary practical Ray baseline.
- `ray-stateless`: each task rebuilds a model; this is a diagnostic, not the primary comparison.

No performance claim is published in the repository. Generate results on the target machine and retain the emitted manifest and raw samples.

## Reproduction

```bash
python benchmark_rl.py \
  --backend crayon-shared \
  --workers 2 \
  --batch 8 \
  --warmup-steps 2 \
  --measure-steps 10 \
  --repetitions 3 \
  --seed 42 \
  --artifact-dir artifacts/crayon-shared

python benchmark_rl.py \
  --backend ray-actor \
  --workers 2 \
  --batch 8 \
  --warmup-steps 2 \
  --measure-steps 10 \
  --repetitions 3 \
  --seed 42 \
  --artifact-dir artifacts/ray-actor
```

Use `--small-model` for CPU smoke tests.

## Correctness and timing

Each chunk receives a deterministic seed derived from the base seed, repetition, step, and chunk index. Before accepting a timed row, the harness compares completions and log probabilities with a local reference. CUDA is synchronized at timing boundaries.

Initialization is excluded from steady-state rows. Warmup and measured rows remain separate in `samples.csv`. `summary.json` reports median, mean, standard deviation, p25, p75, p90, p95, minimum, and maximum.

## Artifacts

Each run emits:

- `manifest.json`: command, git state, software/hardware information, and script hash.
- `semantic_checks.json`: per-step equivalence results.
- `samples.csv`: raw warmup and measured timings.
- `gpu_memory.json`: process-tree GPU memory observations, or an explicit unsupported state.
- `summary.json`: aggregate timing and throughput statistics.

## Interpretation limits

Crayon currently executes workers as in-process blocking threads. Python callables still use the GIL. Ray actors use separate processes. These modes test different architectures; neither result is a universal framework-overhead number.

Networking uses deadline-bound chunked messages but assembles complete payloads in receiver memory. Crayon does not provide distributed task scheduling, process isolation, hard cancellation of running code, durable GCS metadata, TLS, or authentication.
