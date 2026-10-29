# Anviksha Verify Loop — does "retry until correct" actually work?

An experiment, in its own repository. No existing model is touched.

The question: before scaling to 1B then 1T parameters, should the first step be a
self-correcting transformer that keeps refining until the answer is 100 %
accurate?

This repository answers that with measurements, and with the memory arithmetic
that decides whether 1B parameters can fit in 100 MB at all.

---

## 1. The 100 MB constraint, in exact numbers

100 MB is 800 megabits. Every parameter has to fit inside that.

| Precision | 1B parameters | What fits in 100 MB |
|---|---|---|
| fp32 (32-bit) | 4.00 GB | 25 M |
| fp16 (16-bit) | 2.00 GB | 50 M |
| int8 (8-bit) | 1.00 GB | 100 M |
| int4 (4-bit) | 0.50 GB | 200 M |
| int2 (2-bit) | 0.25 GB | 400 M |
| int1 (1-bit) | 0.12 GB | 800 M |

**A 1B-parameter model cannot fit in 100 MB at any precision.** Even at one bit
per parameter it needs 125 MB, and a binary network at that scale loses far more
quality than the size saving is worth.

The honest target for a 100 MB dense model is **200 M parameters at 4-bit**, or
**100 M at 8-bit** if quality matters more than size. That is still a serious
model — roughly GPT-2 small class — and it is achievable on CPU for inference.

### And the training cost

| Model | FLOPs for 1T tokens | On 2 CPU cores (~10 GFLOP/s) |
|---|---|---|
| 1B parameters | 6.0 × 10²¹ | ~19,000 years |
| 1T parameters | 6.0 × 10²⁴ | ~19 million years |

Training a 1B model from scratch is a GPU-cluster job (thousands of GPU-hours).
Training a 1T model is a frontier-lab job (millions of GPU-hours). Neither is a
CPU or phone task, and no architecture choice changes that arithmetic.

**What is possible on CPU:** training models up to ~100 M parameters on modest
data, running 200 M-parameter models at 4-bit for inference, and fine-tuning
small models. The 1B target is reachable only by *distillation* from an existing
large model, not by training from scratch.

---

## 2. The verify-and-retry loop, measured

A tiny character-level transformer learns two-digit addition. Every problem has
the same shape (`03+05=` → ` 8`). Three ways of answering 200 held-out problems:

| Task | Model competence | Greedy | Self-consistency (k=8) | Verified retry (k=8) | Mean attempts |
|---|---|---|---|---|---|
| 1-digit addition | 100 % | 100 % | 100 % | 100 % | 1.00 |
| 2-digit addition | 49.5 % | 49.5 % | 49.5 % | **51.0 %** | 4.96 |

The retry loop bought **1.5 percentage points** for **5× the inference compute**.
The attempt histogram explains why:

```
2-digit: 100 problems solved on attempt 1
          2 problems solved on attempts 3-6
         98 problems never solved in 8 attempts
```

### The finding that matters

**The loop cannot create information the model does not have, and it cannot even
exploit the information the model does have unless the model produces diverse
samples.** This model is overconfident: at temperature 1.0 it emits the same
answer almost every time, so eight retries are eight copies of one guess. The
distribution is bimodal — either the model knows the problem and gets it first
try, or it has no idea and repeats the same wrong answer forever.

That is the core problem with "refine until 100 % correct":

1. **Retrying a deterministic model is pure waste.** Diversity must come first.
2. **"Until 100 % correct" needs an oracle.** For arithmetic there is one. For
   open-ended language there is not — nothing can tell the model when an answer
   is perfect, so the loop has no stopping condition and no gradient.
3. **The loop converts attempts into accuracy only along the tail.** If per-sample
   accuracy is p and samples are independent, k attempts give 1−(1−p)^k. At
   p = 0.495 and k = 8 that predicts 99.6 %; the measured 51 % shows the samples
   are nowhere near independent.

### Where the loop genuinely wins

The mechanism is real and powerful — but only where two conditions hold: a
checker exists, and the model is uncertain rather than confidently wrong. That
means code that can be executed against tests, maths that can be verified
symbolically, structured output that can be validated against a schema, and
retrieval that can be checked against a source. In those settings the loop is
worth 10-30 points, not 1.5.

---

## 3. What the first step should actually be

Not "a transformer that loops until 100 %". In order:

1. **Fix the size target to something real.** 200 M parameters at 4-bit is a
   100 MB model. Write that down as the spec, and stop planning for 1B in 100 MB.
2. **Build the verifier, not the loop.** The loop is trivial once a checker
   exists. The engineering is in the checker: what counts as correct, and how do
   you detect a wrong answer without a human.
3. **Train for calibration, not confidence.** A model that says "I don't know"
   is what makes retries useful. Label smoothing, temperature scaling and an
   explicit abstain token all help; raw cross-entropy does not.
4. **Only then add the loop**, and measure attempts-per-solve rather than
   accuracy alone, because attempts are the cost.
5. **Scale by distillation, not by scratch training.** A 200 M student trained
   on a large teacher's outputs is reachable on modest hardware. A 1B model
   trained from scratch is not.

---

## Run it

```bash
python src/arith_loop.py --steps 900 --max-operand 99 --problems 200 --k 8 \
    --temperature 1.0 --out results_2digit.json
python src/arith_loop.py --steps 800 --max-operand 9 --problems 200 --k 8 \
    --temperature 1.0 --out results_1digit.json
```

`results_*.json` hold the full metrics including the attempts histogram.

## Layout

```
src/arith_loop.py   tiny LM, three answering strategies, verified retry
configs/default.json
results_1digit.json, results_2digit.json
```

## What this does not prove

- One task, one model size, 200 problems. Arithmetic is the friendliest possible
  case for a verifier; language is the hardest.
- The 1.5-point gain is specific to a model with almost no output diversity. A
  model with calibrated uncertainty would show a larger gain — that is a
  prediction, not a result, and it needs the calibration experiment to confirm.
- Nothing here tests the 200 M-parameter target. That is the next experiment.
