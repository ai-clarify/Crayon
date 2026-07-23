#!/usr/bin/env python3
"""Full GSM8K evaluation through the Crayon pipeline.

Real dataset, real model, distributed: llm-actor workers generate answers with
the resident model, the llm-judge worker scores each completion against the
dataset gold (last integer must match). No learner, no weight broadcast — the
actors evaluate the base model as loaded.

    driver: chunk GSM8K into batches
    actors: llm.rollout(cfg{prompts, seeds=indices}) -> [{seed, completion}]
    judge:  llm.judge(rollout_output, golds)         -> [reward]
"""
import argparse
import json
import os
import subprocess
import time
import urllib.request
import uuid

import crayon

from crayon_llm_rl import connect, free_addr  # same-directory reuse

GSM8K_URL = (
    "https://raw.githubusercontent.com/openai/grade-school-math/master/"
    "grade_school_math/data/{split}.jsonl"
)


def load_gsm8k(split, workdir):
    path = os.path.join(workdir, f"gsm8k-{split}.jsonl")
    if not os.path.exists(path):
        urllib.request.urlretrieve(GSM8K_URL.format(split=split), path)
    items = []
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            gold = int(row["answer"].split("####")[1].strip().replace(",", ""))
            items.append({"question": row["question"], "gold": gold})
    return items


def prompt_of(question):
    return (
        f"{question}\nPlease reason step by step, and give the final answer "
        "as a plain number on the last line."
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--actors", type=int, default=2)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--wave", type=int, default=20, help="tasks per progress wave")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=0, help="0 = full split")
    ap.add_argument("--max-new-tokens", type=int, default=640)
    ap.add_argument("--model", required=True)
    ap.add_argument("--binary", default="target/release/crayon-cluster")
    ap.add_argument("--sidecar", default="benchmarks/llm_sidecar.py")
    ap.add_argument("--out", default="/tmp/crayon_gsm8k.json")
    ap.add_argument("--workdir", default="/tmp/crayon-gsm8k")
    a = ap.parse_args()
    os.makedirs(a.workdir, exist_ok=True)

    data = load_gsm8k(a.split, a.workdir)
    if a.limit:
        data = data[: a.limit]
    print(f"gsm8k {a.split}: {len(data)} problems", flush=True)

    coord = free_addr()
    procs = []

    def spawn(args, name, env=None):
        log = open(os.path.join(a.workdir, f"{name}.log"), "w")
        p = subprocess.Popen(args, stdout=log, stderr=log, env=dict(os.environ, **(env or {})))
        procs.append(p)

    spawn([a.binary, "coordinator", coord, "5000"], "coordinator")
    actor_env = {
        "CRAYON_LLM_SIDECAR": os.path.abspath(a.sidecar),
        "CRAYON_LLM_MODEL": a.model,
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    }
    for i in range(a.actors):
        spawn(
            [a.binary, "worker", coord, free_addr(), uuid.uuid4().hex, "1.0", "llm-actor"],
            f"actor-{i}",
            actor_env,
        )
    spawn([a.binary, "worker", coord, free_addr(), uuid.uuid4().hex, "1.0", "llm-judge"], "judge")

    try:
        client = connect(coord)
        while len(json.loads(client.workers())) < a.actors + 1:
            time.sleep(0.2)

        chunks = [data[i : i + a.batch] for i in range(0, len(data), a.batch)]
        correct = total = 0
        rows = []
        completions_path = os.path.join(a.workdir, "completions.jsonl")
        completions_file = open(completions_path, "w")
        run_start = time.perf_counter()

        for wave_start in range(0, len(chunks), a.wave):
            wave = chunks[wave_start : wave_start + a.wave]
            specs = []
            for w, chunk in enumerate(wave):
                base = (wave_start + w) * a.batch
                cfg = {
                    "version": 0,
                    "seeds": [base + j for j in range(len(chunk))],
                    "max_new_tokens": a.max_new_tokens,
                    "temperature": 0.0,
                    "prompts": [prompt_of(c["question"]) for c in chunk],
                }
                specs.append([json.dumps(cfg).encode()])
            rollout_tasks = client.submit_batch("llm", "rollout", 1, specs)
            judge_tasks = client.submit_batch(
                "llm",
                "judge",
                1,
                [
                    [
                        out,
                        json.dumps(
                            [
                                [(wave_start + w) * a.batch + j, chunk[j]["gold"]]
                                for j in range(len(chunk))
                            ]
                        ).encode(),
                    ]
                    for (w, chunk), (_, out) in zip(enumerate(wave), rollout_tasks)
                ],
            )
            rewards = [
                r
                for blob in client.get_many([o for _, o in judge_tasks], 3_600_000)
                for r in json.loads(blob)
            ]
            for blob in client.get_many([o for _, o in rollout_tasks], 3_600_000):
                for r in json.loads(blob):
                    completions_file.write(json.dumps(r) + "\n")
            completions_file.flush()
            for oid in [o for _, o in rollout_tasks] + [o for _, o in judge_tasks]:
                client.release(oid)

            correct += int(sum(rewards))
            total += len(rewards)
            elapsed = time.perf_counter() - run_start
            rate = total / elapsed
            print(
                f"{total:5}/{len(data)} | accuracy {correct / total:.4f}"
                f" | {rate:.2f} problems/s | eta {(len(data) - total) / rate / 60:.1f}min",
                flush=True,
            )
            rows.append({"done": total, "correct": correct, "elapsed_s": round(elapsed, 1)})

        total_s = time.perf_counter() - run_start
        summary = {
            "split": a.split,
            "problems": total,
            "correct": correct,
            "accuracy": round(correct / total, 4),
            "actors": a.actors,
            "batch": a.batch,
            "max_new_tokens": a.max_new_tokens,
            "model": a.model,
            "total_time_s": round(total_s, 1),
            "problems_per_s": round(total / total_s, 2),
            "rows": rows,
        }
        print("DONE", json.dumps({k: v for k, v in summary.items() if k != "rows"}))
        with open(a.out, "w") as f:
            json.dump(summary, f, indent=2)
    finally:
        for p in procs:
            p.kill()


if __name__ == "__main__":
    main()
