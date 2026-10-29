"""Does a verify-and-retry loop actually help? Measured on arithmetic.

A tiny character-level transformer learns two-digit addition. We then compare
three ways of answering the same 200 held-out problems:

  1. greedy          — one answer, no loop
  2. self-consistency — sample k answers, take the majority
  3. verified retry  — sample answers until one passes the checker

For arithmetic the checker is exact, so a verifier loop can always succeed given
enough attempts. The interesting question is not *whether* it succeeds but *how
many attempts it costs*, which is what this script measures.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

CHARS = "0123456789+= "
STOI = {c: i for i, c in enumerate(CHARS)}
ITOS = {i: c for c, i in STOI.items()}
PAD = len(CHARS)
VOCAB = len(CHARS) + 1


MAX_OPERAND = 99


def make_problem(rng: random.Random) -> tuple[str, str]:
    """Always six characters in, two characters out.

    Operands are zero-padded to two digits and the answer is right-aligned to
    two, so every problem has the same shape and the model never has to learn
    variable-length alignment on top of arithmetic.
    """
    a, b = rng.randint(0, MAX_OPERAND), rng.randint(0, MAX_OPERAND)
    return f"{a:02d}+{b:02d}=", f"{a + b:>2}"


def encode(text: str, length: int) -> list[int]:
    ids = [STOI[c] for c in text][:length]
    return ids + [PAD] * (length - len(ids))


class TinyLM(nn.Module):
    def __init__(self, dim: int = 128, layers: int = 2, heads: int = 4, seq: int = 8):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, dim)
        self.pos = nn.Embedding(seq, dim)
        layer = nn.TransformerEncoderLayer(dim, heads, dim * 4, batch_first=True,
                                           dropout=0.0, norm_first=True)
        self.body = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.head = nn.Linear(dim, VOCAB)
        self.seq = seq

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        x = self.embed(tokens) + self.pos(positions)[None]
        mask = nn.Transformer.generate_square_subsequent_mask(tokens.shape[1],
                                                             device=tokens.device)
        x = self.body(x, mask=mask, is_causal=True)
        return self.head(x)


def train(steps: int, batch: int, lr: float = 3e-3, resume: bool = True) -> TinyLM:
    rng = random.Random(0)
    torch.manual_seed(0)
    model = TinyLM()
    if resume and Path("model.pt").exists():
        model.load_state_dict(torch.load("model.pt", weights_only=True))
        print("  resumed from model.pt")
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr)
    started = time.time()
    for step in range(1, steps + 1):
        prompts, answers = zip(*[make_problem(rng) for _ in range(batch)])
        # train on the whole string "12+34=46", next-token objective
        full = [encode(p + a, 8) for p, a in zip(prompts, answers)]
        inputs = torch.tensor([row[:-1] for row in full])
        targets = torch.tensor([row[1:] for row in full])
        logits = model(inputs)
        # weight the answer positions more heavily than the prompt
        weights = torch.ones_like(targets, dtype=torch.float32)
        weights[:, 6:] = 4.0
        loss = F.cross_entropy(logits.reshape(-1, VOCAB), targets.reshape(-1),
                               reduction="none")
        loss = (loss.view_as(weights) * weights).sum() / weights.sum()
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        if step % 200 == 0:
            print(f"  step {step}  loss {loss.item():.4f}  {time.time()-started:.0f}s",
                  flush=True)
            torch.save(model.state_dict(), "model.pt")
    return model


@torch.no_grad()
def sample_answer(model: TinyLM, prompt: str, temperature: float) -> str:
    tokens = torch.tensor([encode(prompt, 6)])
    generated = []
    for _ in range(2):
        logits = model(tokens)[:, -1, :] / max(temperature, 1e-6)
        logits[:, PAD] = float("-inf")  # never emit padding
        probs = torch.softmax(logits, dim=-1)
        nxt = torch.multinomial(probs, 1)
        generated.append(int(nxt))
        tokens = torch.cat([tokens, nxt], dim=1)
    return "".join(ITOS[i] for i in generated)


@torch.no_grad()
def greedy(model: TinyLM, prompt: str) -> str:
    tokens = torch.tensor([encode(prompt, 6)])
    generated = []
    for _ in range(2):
        logits = model(tokens)[:, -1, :].clone()
        logits[:, PAD] = float("-inf")
        nxt = logits.argmax(-1, keepdim=True)
        generated.append(int(nxt))
        tokens = torch.cat([tokens, nxt], dim=1)
    return "".join(ITOS[i] for i in generated)


def check(prompt: str, answer: str) -> bool:
    a, b = prompt.rstrip("=").split("+")
    try:
        return int(answer.strip()) == int(a) + int(b)
    except ValueError:
        return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", default=1200, type=int)
    parser.add_argument("--problems", default=200, type=int)
    parser.add_argument("--k", default=8, type=int)
    parser.add_argument("--out", default="results.json")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--max-operand", default=99, type=int)
    parser.add_argument("--temperature", default=0.8, type=float)
    args = parser.parse_args()

    global MAX_OPERAND
    MAX_OPERAND = args.max_operand
    weights = Path("model.pt")
    if args.eval_only and weights.exists():
        print("loading checkpoint")
        model = TinyLM()
        model.load_state_dict(torch.load(weights, weights_only=True))
    else:
        print("training")
        model = train(args.steps, 128)
        torch.save(model.state_dict(), weights)
    model.eval()

    rng = random.Random(1234)
    problems = [make_problem(rng) for _ in range(args.problems)]

    single = sum(check(p, greedy(model, p)) for p, _ in problems) / len(problems)

    vote_correct = 0
    retry_attempts: list[int] = []
    solved_within_k = 0

    for prompt, truth in problems:
        candidates = [sample_answer(model, prompt, args.temperature)
                      for _ in range(args.k)]
        majority = Counter(candidates).most_common(1)[0][0]
        if check(prompt, majority):
            vote_correct += 1

        for attempt, candidate in enumerate(candidates, start=1):
            if check(prompt, candidate):
                retry_attempts.append(attempt)
                solved_within_k += 1
                break
        else:
            retry_attempts.append(args.k + 1)

    n = len(problems)
    metrics = {
        "steps": args.steps,
        "problems": n,
        "k": args.k,
        "greedy_accuracy": single,
        "self_consistency_accuracy": vote_correct / n,
        "verified_retry_coverage": solved_within_k / n,
        "mean_attempts_to_solve": sum(retry_attempts) / n,
        "attempts_histogram": {
            str(i): retry_attempts.count(i) for i in range(1, args.k + 2)
        },
    }
    print(json.dumps(metrics, indent=2))
    Path(args.out).write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
