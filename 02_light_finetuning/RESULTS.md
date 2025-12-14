# 02_light_finetuning — Results

This document summarizes the outcome of the **Light Fine-Tuning Lab**, where we fine-tuned `distilgpt2` on a tiny Python dataset. The focus is learning the workflow rather than shipping a production-grade model.

---

## 1. Objective

This lab demonstrates an end-to-end fine-tuning loop:

- Prepare a small dataset.
- Tokenize and batch code snippets.
- Run Hugging Face’s `Trainer`.
- Evaluate with **perplexity**.

Everything is intentionally small, slow, and CPU-friendly so beginners can read every part of the pipeline.

---

## 2. Environment

- **Base model:** `distilgpt2`
- **Task:** Causal language modeling (next-token prediction)
- **Hardware:** CPU
- **Frameworks:** PyTorch + Hugging Face Transformers

---

## 3. Dataset Summary

Created via `prepare_dataset.py`.

```
data/
├── train.jsonl
└── validation.jsonl
```

Each line is a JSON object with a `text` field, e.g.

```json
{"text": "def is_even(number):\n    return number % 2 == 0"}
```

- **Training examples:** 8  
- **Validation examples:** 2  

The minimal size allows beginners to:

- Inspect every sample.
- Understand what the model memorizes.
- Run experiments locally without GPUs.

---

## 4. Training Configuration (`train.py`)

| Parameter            | Value          |
|----------------------|----------------|
| Max sequence length  | 128 tokens     |
| Batch size           | 2              |
| Epochs               | 2              |
| Learning rate        | 5e-5           |
| Optimizer            | AdamW (default)|
| Checkpoint saving    | Disabled       |

What each term means:

- **Max sequence length** – cap on how many tokens from each example we feed the model; keeps memory predictable.
- **Batch size** – how many samples the model sees at once; small batches run on CPUs with low RAM.
- **Epochs** – full passes over the dataset; two passes are enough for this toy corpus.
- **Learning rate** – step size for weight updates; `5e-5` is a stable default for GPT-2–style models.
- **Optimizer (AdamW)** – algorithm that updates weights using gradients while applying weight decay.
- **Checkpoint saving** – disabled to keep the script simple; no intermediate checkpoints are written.

Notes:

- Small batches keep memory usage low.
- Few epochs prevent overfitting to the tiny dataset.
- No checkpoints keeps the script simple for newcomers.

---

## 5. Training Behavior

What we observed:

- Training finished successfully on CPU.
- Loss dropped quickly because the dataset is tiny and repetitive.
- The model learned Python-style syntax patterns.

Illustrative trend:

1. Early steps → high loss (model is “surprised”).
2. Later steps → lower loss (model recognizes the dataset).

Interpretation: a decreasing loss confirms the fine-tuning loop works.

---

## 6. Evaluation Metric — Perplexity

We evaluated with `eval_code_perplexity.py`.

Plain-English explanation:

- Perplexity answers “How surprised is the model by this text?”
- Lower perplexity → better predictions.
- Higher perplexity → worse predictions.
- For code, it reflects familiarity with syntax/patterns, not correctness.

---

## 7. Evaluation Results

| Model               | Validation Perplexity |
|---------------------|-----------------------|
| Base `distilgpt2`   | ~4689                 |
| Fine-tuned model    | Not available         |

Why is the fine-tuned score missing?

- On some CPU-only setups, saving the fine-tuned weights fails because of a PyTorch/Transformers compatibility issue.
- Training still completed, and the evaluation script was verified independently.
- Highlighting this limitation mirrors real-world engineering hurdles.

---

## 8. Interpretation for Beginners

- Loss trending downward means the model fit the toy dataset.
- Perplexity is a sanity check, not a pass/fail grade.
- Tiny datasets lead to noisy metrics and limited improvements.
- Real gains require more diverse data and more rigorous evaluation (unit tests, functional checks, etc.).

---

## 9. Key Takeaways

By completing this lab we practiced:

- A full fine-tuning workflow (data → train → evaluate).
- Clear separation of scripts for preparation, training, and evaluation.
- Proper experiment documentation, including known limitations.
- Awareness of tooling constraints when working on CPU-only machines.

This mirrors how research notebooks evolve into engineering artifacts.

---

## 10. Suggested Next Steps

1. Expand the dataset to 50–500 Python functions.
2. Compare generated code snippets before vs. after fine-tuning.
3. Add basic unit tests for generated functions.
4. Experiment with parameter-efficient methods (LoRA, adapters).
5. Track multiple runs to understand variance.

---

_End of results._
