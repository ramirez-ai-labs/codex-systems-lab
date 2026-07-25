# Light Fine-Tuning Lab ✅ Complete, end to end

This folder walks through fine-tuning `distilgpt2` on a small, real
Python-code dataset — a genuine end-to-end pipeline (data prep → training →
evaluation), not a simulation. Everything is CPU-friendly and intentionally
small so beginners can inspect every step.

---

## Why Fine-Tuning Matters

Fine-tuning is how a general-purpose model gets adapted to a specific
domain or style — the same idea behind coding assistants specialized on a
particular codebase or language. This lab demonstrates the *mechanics* of
that process end to end: preparing data, running the training loop,
saving the result, and measuring whether it actually improved.

---

## Pipeline Stages in This Folder

### 1️⃣ `prepare_dataset.py` — Building the Dataset ✅ Real data

Key ideas:
- Writes 31 training examples and 8 validation examples of short,
  hand-written Python functions to `data/train.jsonl` / `data/validation.jsonl`
- Deliberately tiny so beginners can read every single training example
- No synthetic generation or scraping — every snippet is hand-authored for
  this lab

### 2️⃣ `train.py` — Fine-Tuning the Model ✅ Real training, real save

Key ideas:
- Uses HuggingFace `Trainer` + AdamW on the 31 training examples, 2 epochs,
  batch size 2, entirely on CPU — training completes in under 2 minutes on
  this hardware
- `Trainer`'s own intermediate checkpointing is intentionally disabled
  (`save_strategy="no"`) for simplicity — a separate final
  `model.save_pretrained()` call saves the finished model to `outputs/`
- Loss drops from ~9.1 (step 1) to ~0.75 (final step) over 32 total steps —
  confirms the training loop is genuinely learning, not just running

### 3️⃣ `eval_code_perplexity.py` — Measuring the Improvement ✅ Real evaluation, exact match confirmed

Key ideas:
- Computes perplexity (how "surprised" the model is by held-out code) for
  both the base and fine-tuned model on the same 8 validation examples
- **Base model: 2989.72 perplexity. Fine-tuned: 38.23** — a ~78x
  improvement, and both numbers reproduced exactly on re-run
- This is a sanity check that the training loop works, not evidence of
  general code-generation quality — the dataset is tiny and repetitive, so
  the model largely learns to predict *this specific style* of function

---

## Training Parameters Explained

| Parameter | Why it matters |
|-----------|----------------|
| **Max sequence length (128 tokens)** | Truncates/pads each example to fit within memory limits. |
| **Batch size (2)** | Number of samples processed simultaneously; small batches prevent CPU RAM spikes. |
| **Epochs (2)** | Full passes over the dataset; a couple of passes are enough for this toy corpus. |
| **Learning rate (5e-5)** | Controls how big each weight update is; GPT-2 models stay stable at this scale. |
| **Optimizer (AdamW)** | Standard optimizer with weight decay, great default for transformers. |
| **Checkpoint saving (disabled)** | Skips `Trainer`'s intermediate checkpoints so beginners focus on the core loop — the final model is still saved separately, see stage 2 above. |

---

## How to Use This Section

Recommended order (each stage depends on the previous one's output):

```bash
python prepare_dataset.py
python train.py
python eval_code_perplexity.py
```

Run from inside `02_light_finetuning/`. See [`RESULTS.md`](RESULTS.md) for
the full narrative, training curve, and honest limitations.
