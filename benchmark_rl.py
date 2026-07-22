"""Reproducible RL distribution benchmark for Crayon and Ray.

Backends:
  crayon-shared       In-process Crayon tasks sharing the live model object.
  crayon-serialized   Crayon raw-byte object/task path with one pickle layer.
  ray-actor           Persistent Ray actor holding one model (primary baseline).
  ray-stateless       Rebuild model per task (diagnostic only).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path


def derive_seed(base: int, repetition: int, step: int, chunk: int) -> int:
    payload = f"{base}:{repetition}:{step}:{chunk}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**31)


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("cannot summarize an empty sample")
    ordered = sorted(values)

    def percentile(p: float) -> float:
        index = (len(ordered) - 1) * p
        low = int(index)
        high = min(low + 1, len(ordered) - 1)
        return ordered[low] + (ordered[high] - ordered[low]) * (index - low)

    return {
        "count": len(values),
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "p25": percentile(0.25),
        "p75": percentile(0.75),
        "p90": percentile(0.90),
        "p95": percentile(0.95),
        "min": min(values),
        "max": max(values),
    }


def process_tree_pids(root: int) -> set[int]:
    try:
        output = subprocess.check_output(["ps", "-axo", "pid=,ppid="], text=True)
    except (OSError, subprocess.CalledProcessError):
        return {root}
    children: dict[int, list[int]] = {}
    for line in output.splitlines():
        try:
            pid, parent = map(int, line.split())
        except ValueError:
            continue
        children.setdefault(parent, []).append(pid)
    found, stack = {root}, [root]
    while stack:
        for child in children.get(stack.pop(), []):
            if child not in found:
                found.add(child)
                stack.append(child)
    return found


def parse_nvidia_smi(text: str, pids: set[int]) -> dict[str, object]:
    rows = []
    for line in text.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 3:
            continue
        try:
            pid, used = int(parts[0]), float(parts[2].split()[0])
        except ValueError:
            continue
        if pid in pids:
            rows.append({"pid": pid, "gpu": parts[1], "memory_mib": used})
    return {"supported": True, "total_mib": sum(row["memory_mib"] for row in rows), "rows": rows}


def gpu_memory_tree() -> dict[str, object]:
    command = [
        "nvidia-smi",
        "--query-compute-apps=pid,gpu_uuid,used_memory",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError):
        return {"supported": False, "total_mib": None, "rows": []}
    return parse_nvidia_smi(output, process_tree_pids(os.getpid()))


def git_metadata() -> dict[str, object]:
    def run(*args: str) -> str | None:
        try:
            return subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    return {
        "commit": run("git", "rev-parse", "HEAD"),
        "dirty": bool(run("git", "status", "--porcelain")),
    }


def load_torch():
    try:
        import numpy as np
        import torch
        import torch.nn as nn
    except ImportError as error:
        raise SystemExit("benchmark requires numpy and torch") from error
    return np, torch, nn


def build_components(small_model: bool):
    np, torch, nn = load_torch()
    hidden, layers, heads, intermediate = ((64, 2, 4, 128) if small_model else (2048, 12, 16, 8192))
    vocab = 1000

    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = nn.MultiheadAttention(hidden, heads, batch_first=True)
            self.ff1 = nn.Linear(hidden, intermediate, bias=False)
            self.ff2 = nn.Linear(intermediate, hidden, bias=False)

        def forward(self, value):
            attended, _ = self.attn(value, value, value, need_weights=False)
            return value + attended + self.ff2(torch.relu(self.ff1(value)))

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(vocab, hidden)
            self.layers = nn.ModuleList([Layer() for _ in range(layers)])
            self.output = nn.Linear(hidden, vocab)

        def forward(self, tokens):
            value = self.embed(tokens)
            for layer in self.layers:
                value = layer(value)
            return self.output(value[:, -1])

    return np, torch, Model, vocab


def rollout(model, prompts, seed: int, generation_length: int):
    np, torch, _ = load_torch()
    device = next(model.parameters()).device
    tokens = torch.as_tensor(prompts, dtype=torch.long, device=device)
    generator = torch.Generator(device=device).manual_seed(seed)
    completions, log_probs = [], []
    model.eval()
    with torch.no_grad():
        for _ in range(generation_length):
            log_prob = torch.log_softmax(model(tokens), dim=-1)
            next_token = torch.multinomial(log_prob.exp(), 1, generator=generator).squeeze(1)
            completions.append(next_token)
            log_probs.append(log_prob.gather(1, next_token[:, None]).squeeze(1))
            tokens = torch.cat([tokens, next_token[:, None]], dim=1)
    return (
        torch.stack(completions, 1).cpu().numpy().astype(np.int64),
        torch.stack(log_probs, 1).cpu().numpy().astype(np.float32),
    )


def make_backend(name, workers, Model, device, generation_length):
    import pickle

    if name.startswith("crayon"):
        try:
            import crayon
        except ImportError as error:
            raise SystemExit("Crayon backend requires the built Python extension") from error
        runtime = crayon.Ray(workers)

        class CrayonBackend:
            def run(self, model, chunks, seeds):
                if name == "crayon-shared":
                    refs = [runtime.spawn(rollout, model, chunk, seed, generation_length) for chunk, seed in zip(chunks, seeds)]
                else:
                    payload = pickle.dumps(model.state_dict(), protocol=pickle.HIGHEST_PROTOCOL)
                    state = runtime.put(payload)

                    def serialized(data, prompts, seed, length):
                        instance = Model().to(device)
                        instance.load_state_dict(pickle.loads(data))
                        return rollout(instance, prompts, seed, length)

                    refs = [runtime.spawn(serialized, state, chunk, seed, generation_length) for chunk, seed in zip(chunks, seeds)]
                return runtime.get_batch(refs)

            def close(self):
                return None

        return CrayonBackend()

    try:
        import ray
    except ImportError as error:
        raise SystemExit("Ray backend requires `pip install ray`") from error
    ray.init(num_cpus=workers, include_dashboard=False, ignore_reinit_error=True)

    if name == "ray-actor":
        @ray.remote
        class RolloutActor:
            def __init__(self):
                self.model = Model().to(device)

            def set_state(self, state):
                self.model.load_state_dict(state)

            def run(self, prompts, seed, length):
                return rollout(self.model, prompts, seed, length)

        actors = [RolloutActor.remote() for _ in range(workers)]

        class RayActorBackend:
            def run(self, model, chunks, seeds):
                ray.get([actor.set_state.remote(model.state_dict()) for actor in actors])
                return ray.get([actors[i % len(actors)].run.remote(chunk, seed, generation_length) for i, (chunk, seed) in enumerate(zip(chunks, seeds))])

            def close(self):
                ray.shutdown()

        return RayActorBackend()

    @ray.remote
    def stateless(state, prompts, seed, length):
        model = Model().to(device)
        model.load_state_dict(state)
        return rollout(model, prompts, seed, length)

    class RayStatelessBackend:
        def run(self, model, chunks, seeds):
            state = ray.put(model.state_dict())
            return ray.get([stateless.remote(state, chunk, seed, generation_length) for chunk, seed in zip(chunks, seeds)])

        def close(self):
            ray.shutdown()

    return RayStatelessBackend()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["crayon-shared", "crayon-serialized", "ray-actor", "ray-stateless"], required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--prompt-len", type=int, default=16)
    parser.add_argument("--gen-len", type=int, default=16)
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--measure-steps", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--small-model", action="store_true")
    args = parser.parse_args()
    if min(args.workers, args.batch, args.measure_steps, args.repetitions) <= 0:
        parser.error("workers, batch, measure-steps, and repetitions must be positive")

    np, torch, Model, vocab = build_components(args.small_model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    canonical = Model().state_dict()
    backend = make_backend(args.backend, args.workers, Model, device, args.gen_len)
    rows, checks, gpu_rows = [], [], []
    artifact = args.artifact_dir
    artifact.mkdir(parents=True, exist_ok=True)

    try:
        for repetition in range(args.repetitions):
            model = Model().to(device)
            model.load_state_dict(canonical)
            total = args.warmup_steps + args.measure_steps
            for step in range(total):
                rng = np.random.default_rng(derive_seed(args.seed, repetition, step, 0))
                prompts = rng.integers(0, vocab, size=(args.batch, args.prompt_len), dtype=np.int64)
                chunks = [chunk for chunk in np.array_split(prompts, min(args.workers, args.batch)) if len(chunk)]
                seeds = [derive_seed(args.seed, repetition, step, index) for index in range(len(chunks))]
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                started = time.perf_counter_ns()
                results = backend.run(model, chunks, seeds)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                duration = (time.perf_counter_ns() - started) / 1e9
                expected = [rollout(model, chunk, seed, args.gen_len) for chunk, seed in zip(chunks, seeds)]
                valid = all(np.array_equal(a[0], b[0]) and np.allclose(a[1], b[1], rtol=1e-5, atol=1e-6) for a, b in zip(results, expected))
                checks.append({"repetition": repetition, "step": step, "valid": valid})
                if not valid:
                    raise RuntimeError("semantic check failed")
                phase = "warmup" if step < args.warmup_steps else "measure"
                rows.append({"backend": args.backend, "repetition": repetition, "step": step, "phase": phase, "seconds": duration, "samples_per_second": args.batch / duration, "tokens_per_second": args.batch * args.gen_len / duration})
                gpu_rows.append({"repetition": repetition, "step": step, **gpu_memory_tree()})
    finally:
        backend.close()

    measured = [row for row in rows if row["phase"] == "measure"]
    seconds = summarize([row["seconds"] for row in measured])
    summary = {"backend": args.backend, "seconds": seconds, "samples_per_second": summarize([row["samples_per_second"] for row in measured]), "tokens_per_second": summarize([row["tokens_per_second"] for row in measured])}
    with (artifact / "samples.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader(); writer.writerows(rows)
    (artifact / "summary.json").write_text(json.dumps(summary, indent=2))
    (artifact / "semantic_checks.json").write_text(json.dumps(checks, indent=2))
    (artifact / "gpu_memory.json").write_text(json.dumps(gpu_rows, indent=2))
    manifest = {"command": sys.argv, "python": sys.version, "platform": platform.platform(), "torch": torch.__version__, "cuda": torch.version.cuda, "device": str(device), "git": git_metadata(), "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "arguments": vars(args) | {"artifact_dir": str(args.artifact_dir)}}
    (artifact / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
