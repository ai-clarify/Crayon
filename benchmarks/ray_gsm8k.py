#!/usr/bin/env python3
"""Ray twin of crayon_gsm8k.py: full GSM8K evaluation, apples-to-apples.

Same model, same prompts, same judge rule (last integer, commas stripped, must
match the gold), same actor/judge/batch structure — only the runtime differs.
"""
import argparse
import json
import os
import re
import time
import urllib.request

import ray

GSM8K_URL = (
    "https://raw.githubusercontent.com/openai/grade-school-math/master/"
    "grade_school_math/data/{split}.jsonl"
)


def load_gsm8k(split, workdir):
    os.makedirs(workdir, exist_ok=True)
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


def last_integer(text):
    matches = re.findall(r"-?\d+", text.replace(",", ""))
    return int(matches[-1]) if matches and len(matches[-1]) < 12 else None


@ray.remote(num_gpus=0.4)
class RolloutActor:
    def __init__(self, model_path):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_path, padding_side="left")
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=torch.bfloat16
        ).cuda()

    def rollout(self, prompts, max_new_tokens):
        with self.torch.no_grad():
            enc = self.tok(prompts, return_tensors="pt", padding=True).to("cuda")
            out = self.model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=1.0,
                top_p=0.95,
                pad_token_id=self.tok.pad_token_id,
            )
        return self.tok.batch_decode(
            out[:, enc.input_ids.shape[1] :], skip_special_tokens=True
        )


@ray.remote
class Judge:
    def judge(self, completions, golds):
        return [float(last_integer(c) == g) for c, g in zip(completions, golds)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--actors", type=int, default=2)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--wave", type=int, default=20)
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="/tmp/ray_gsm8k.json")
    ap.add_argument("--workdir", default="/tmp/ray-gsm8k")
    a = ap.parse_args()

    data = load_gsm8k(a.split, a.workdir)
    if a.limit:
        data = data[: a.limit]
    print(f"gsm8k {a.split}: {len(data)} problems", flush=True)

    ray.init(ignore_reinit_error=True)
    actors = [RolloutActor.remote(a.model) for _ in range(a.actors)]
    judge = Judge.remote()

    chunks = [data[i : i + a.batch] for i in range(0, len(data), a.batch)]
    correct = total = 0
    rows = []
    completions_file = open(os.path.join(a.workdir, "completions.jsonl"), "w")
    run_start = time.perf_counter()

    for wave_start in range(0, len(chunks), a.wave):
        wave = chunks[wave_start : wave_start + a.wave]
        rollout_refs = [
            actors[w % a.actors].rollout.remote(
                [prompt_of(c["question"]) for c in chunk], a.max_new_tokens
            )
            for w, chunk in enumerate(wave)
        ]
        judge_refs = [
            judge.judge.remote(ref, [c["gold"] for c in chunk])
            for ref, chunk in zip(rollout_refs, wave)
        ]
        rewards = [r for rs in ray.get(judge_refs) for r in rs]
        for ref, chunk in zip(rollout_refs, wave):
            for c, completion in zip(chunk, ray.get(ref)):
                completions_file.write(
                    json.dumps({"gold": c["gold"], "completion": completion}) + "\n"
                )
        completions_file.flush()
        del rollout_refs, judge_refs

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


if __name__ == "__main__":
    main()
