#!/usr/bin/env python3
"""Resident LLM sidecar for Crayon's real RL pipeline.

One process per role instance, speaking newline-delimited JSON over
stdin/stdout. An actor worker drives it with `load` + `rollout`; the learner
(driver-attached) drives it with `learn` + `save`. The model stays resident on
the GPU across commands, so per-iteration cost is generation/backprop, not
model loading.

Commands:
  {"cmd":"init","model":PATH,"lr":F}                -> {"ok":true}
  {"cmd":"save","out":PATH}                          -> {"saved":PATH}
  {"cmd":"load","path":PATH,"version":N}             -> {"ok":true}
  {"cmd":"rollout","seeds":[..],"max_new_tokens":N}  -> {"rollouts":[{"seed":S,"completion":T}]}
  {"cmd":"learn","items":[{"prompt":P,"completion":C,"reward":R}],"out":PATH}
                                                     -> {"loss":F,"saved":PATH}

The task is arithmetic QA; prompts derive from a seed the same way in the Rust
judge (keep in sync): a = 50 + (seed >> 8) % 900, b = 50 + seed % 900.
"""
import json
import sys

import torch


def prompt_for(seed):
    a = 50 + (seed >> 8) % 900
    b = 50 + seed % 900
    return f"What is {a}+{b}? Answer with just the number."


class Sidecar:
    def __init__(self):
        self.model = None
        self.tok = None
        self.optimizer = None
        self.version = -1

    def init(self, model_path, lr):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained(model_path, padding_side="left")
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=torch.bfloat16
        ).cuda()
        if lr:
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        return {"ok": True}

    def _format(self, prompt):
        # Instruction/reasoning models (e.g. Qwen) answer well only through their
        # chat template; a base model has none, so fall back to the raw prompt.
        # rollout AND learn must format identically, or REINFORCE scores the
        # wrong token positions.
        if getattr(self.tok, "chat_template", None):
            return self.tok.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        return prompt

    def save(self, out):
        state = {k: v.cpu() for k, v in self.model.state_dict().items()}
        torch.save(state, out)
        return {"saved": out}

    def load(self, path, version):
        if version != self.version:
            self.model.load_state_dict(torch.load(path, map_location="cuda"))
            self.version = version
        return {"ok": True}

    @torch.no_grad()
    def rollout(self, seeds, max_new_tokens, temperature=1.0, prompts=None):
        prompts = prompts or [prompt_for(s) for s in seeds]
        prompts = [self._format(p) for p in prompts]
        enc = self.tok(prompts, return_tensors="pt", padding=True).to("cuda")
        sampling = (
            {"do_sample": True, "temperature": temperature, "top_p": 0.95}
            if temperature > 0
            else {"do_sample": False}
        )
        out = self.model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tok.pad_token_id,
            **sampling,
        )
        completions = self.tok.batch_decode(
            out[:, enc.input_ids.shape[1] :], skip_special_tokens=True
        )
        return {
            "rollouts": [
                {"seed": s, "completion": c} for s, c in zip(seeds, completions)
            ]
        }

    def learn(self, items, out, micro=8):
        # REINFORCE with a mean-reward baseline, backprop in micro-batches so
        # the V100 (shared with the actor sidecars) never holds the full-batch
        # activations + vocab logits at once.
        rewards = torch.tensor([i["reward"] for i in items], device="cuda")
        advantage = rewards - rewards.mean()
        self.optimizer.zero_grad()
        total_loss = 0.0
        for start in range(0, len(items), micro):
            chunk = items[start : start + micro]
            adv = advantage[start : start + micro]
            fmt = [self._format(i["prompt"]) for i in chunk]
            texts = [f + i["completion"] for f, i in zip(fmt, chunk)]
            prompt_lens = [len(self.tok(f)["input_ids"]) for f in fmt]
            enc = self.tok(texts, return_tensors="pt", padding=True).to("cuda")
            logits = self.model(**enc).logits[:, :-1]
            targets = enc.input_ids[:, 1:]
            token_lp = torch.log_softmax(logits.float(), dim=-1).gather(
                -1, targets.unsqueeze(-1)
            ).squeeze(-1)
            # Score only completion tokens (padding is on the left, so the
            # completion is the tail); normalize by length to keep scale stable.
            seq_lp = []
            for row, (ids, plen) in enumerate(zip(enc.input_ids, prompt_lens)):
                pad = int((ids == self.tok.pad_token_id).sum())
                lp = token_lp[row, pad + plen - 1 :]
                seq_lp.append(lp.sum() / max(len(lp), 1))
            loss = -(adv * torch.stack(seq_lp)).sum() / len(items)
            loss.backward()
            total_loss += float(loss.item())
            del enc, logits, targets, token_lp, seq_lp, loss
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        torch.cuda.empty_cache()
        self.save(out)
        return {"loss": total_loss, "saved": out}


def main():
    sidecar = Sidecar()
    handlers = {
        "init": lambda m: sidecar.init(m["model"], m.get("lr")),
        "save": lambda m: sidecar.save(m["out"]),
        "load": lambda m: sidecar.load(m["path"], m["version"]),
        "rollout": lambda m: sidecar.rollout(
            m["seeds"],
            m["max_new_tokens"],
            m.get("temperature", 1.0),
            m.get("prompts") or None,
        ),
        "learn": lambda m: sidecar.learn(m["items"], m["out"]),
    }
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        msg = json.loads(line)
        try:
            reply = handlers[msg["cmd"]](msg)
        except Exception as error:  # report, keep serving
            reply = {"error": f"{type(error).__name__}: {error}"}
        sys.stdout.write(json.dumps(reply) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
