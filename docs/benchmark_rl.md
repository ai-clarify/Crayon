# Crayon vs Ray: 0.6B LLM RL Benchmark

## Test Environment

| Item | Value |
|------|-------|
| GPU | Tesla V100-SXM2-32GB |
| Model | Synthetic 0.6B Transformer (12 layers, hidden=2048, vocab=1000) |
| Training backend | PyTorch (Adam optimizer, pure Python) |
| Algorithm | GRPO (slime-style: lr=1e-6, clip=0.2, temperature=1.0) |
| Batch | 8 samples (2 prompts × 4 group size) |
| Sequence | prompt_len=16, gen_len=16 |

## What's Identical

- Model architecture and parameters (same `Transformer` class)
- Training logic (`train_step`, GRPO loss computation)
- All hyperparameters (lr, clip, temperature, etc.)
- Rollout forward+sampling logic (same `rollout_task` function body)
- GPU device (same V100)

## What Differs

Only the distribution framework and its model transport mechanism:

| | Crayon | Ray |
|---|---|---|
| Worker architecture | In-process tokio threads | Separate OS processes |
| Model transport (Mode A) | Python object reference (zero copy) | `pickle.dumps` → plasma → `pickle.loads` per step |
| Model copies | 1 (shared across workers) | N (one per worker process) |
| Parallelism | GIL-serialized Python, concurrent GPU | True process parallelism |

## Mode A: Each Backend's Optimal Path

Each backend uses its best available transport. Crayon passes the model by
reference (in-process); Ray must serialize it (separate processes).

### Steady-state step time (after warmup)

| Workers | Crayon | Ray | Speedup |
|---------|--------|-----|---------|
| 1 | **1.58s** | 21.1s | Crayon 13.4x |
| 2 | **1.67s** | 21.2s | Crayon 12.7x |
| 4 | **1.78s** | OOM* | — |

*Ray 4 workers OOMs: 4 × 5GB model + 12GB trainer = 32GB > V100 capacity.

### Key Observations

1. **Crayon's 13x speedup comes entirely from zero-copy model transport.**
   Crayon passes the live `model` Python object to workers; Ray must pickle
   2.4GB of weights every step and reload them in each worker.

2. **Crayon workers do NOT speed up with more workers** (1.58s → 1.78s).
   The GIL serializes Python rollout code; only GPU forward passes overlap.
   Adding workers adds scheduling overhead without throughput gain at this
   batch size.

3. **Ray workers also do NOT speed up** (21.1s → 21.2s). The bottleneck is
   model serialization/transfer, not forward compute. More workers just mean
   more copies of the same model to load.

4. **Crayon uses 12.4GB GPU memory regardless of worker count** (one shared
   model). Ray uses 12.4GB at 1-2 workers but OOMs at 4.

## Mode B: Both Serialize (Framework Overhead Isolation)

Force Crayon to pickle the model through its object store, identical data
flow to Ray. This isolates pure framework overhead (object store, scheduling,
serialization) from the zero-copy architecture advantage.

| Workers | Crayon (serialized) | Ray | Winner |
|---------|---------------------|-----|--------|
| 2 | 50.0s | 21.2s | **Ray 2.4x** |

### Why Crayon is slower in Mode B

1. **bincode vs plasma**: Crayon's object store uses bincode serialization
   + in-memory `Vec<u8>`. Ray uses pickle + plasma shared memory, which is
   faster for large objects.

2. **No model caching**: `rollout_task_serialized` rebuilds the model
   (`Transformer()` + `load_state_dict` + `.to(device)`) on every call.
   Ray workers are separate processes that exit after the task, freeing
   memory. Crayon workers are persistent threads, so models accumulate
   (11GB → 18.6GB GPU memory leak).

3. **Rust-Python boundary**: Every `spawn`/`get` call crosses the PyO3
   boundary, adding overhead per task.

## Adversarial Review: Is the Comparison Fair?

### Mode A

**Q: Is Crayon's 13x advantage artificial, since Ray "could" also avoid
serialization?**

A: No. Ray's workers are separate OS processes — they physically cannot
access the trainer's Python objects. Serialization is mandatory for Ray.
Crayon's in-process design is a deliberate architectural choice that
enables zero-copy. This is a real advantage, not a benchmark artifact.

**Q: Does Crayon's GIL-serialized execution make the "4 workers" claim
misleading?**

A: Yes, partially. Crayon's workers share the GIL, so Python rollout code
runs serially. At batch=8, the GPU is the bottleneck and 1 worker already
saturates it. For larger batches or CPU-bound tasks, more workers would
help. The benchmark uses 1-2 workers for Crayon to match Ray's effective
parallelism at this scale.

**Q: Could Ray use an actor to hold the model and avoid per-step
serialization?**

A: Yes, but that changes the programming model (actor-based vs task-based).
This benchmark compares the task-based rollout pattern, which is the
dominant RL paradigm (e.g., vLLM, SGLang rollout workers). In that pattern,
Ray must serialize the model every step.

### Mode B

**Q: Is Crayon's 2.4x deficit in Mode B a real framework weakness?**

A: Yes, but it's an implementation gap, not an architectural one. The bincode
object store is slower than plasma for large objects, and the lack of model
caching causes memory leaks. These are fixable:
- Replace bincode with shared memory for large objects
- Cache deserialized models in workers instead of rebuilding each step
- Reduce PyO3 boundary crossings

## Conclusion

| Dimension | Winner | Margin |
|-----------|--------|--------|
| Real-world throughput (Mode A) | **Crayon** | 13x (zero-copy advantage) |
| Pure framework overhead (Mode B) | Ray | 2.4x (plasma + caching) |
| Memory efficiency | **Crayon** | 1 model copy vs N |
| Scalability to many workers | Ray | True process parallelism |

**Bottom line**: Crayon's in-process architecture gives it a 13x advantage
for single-node RL workloads where the model fits in one GPU. Ray's
multi-process design scales better across nodes and avoids the GIL, but
pays a serialization tax on every step.
