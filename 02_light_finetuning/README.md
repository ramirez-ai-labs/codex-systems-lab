# Light Fine-Tuning Lab

This folder walks through fine-tuning `distilgpt2` on a handful of Python snippets to demonstrate an end-to-end workflow (data prep → training → evaluation). Everything is CPU-friendly and designed for beginners.

## Key Scripts

- `prepare_dataset.py` – builds tiny `train.jsonl`/`validation.jsonl` files.
- `train.py` – runs Hugging Face `Trainer` with minimal settings.
- `eval_code_perplexity.py` – measures perplexity on the validation split.
- `RESULTS.md` – experiment summary with metrics and takeaways.

## Training Parameters Explained

| Parameter | Why it matters |
|-----------|----------------|
| **Max sequence length (128 tokens)** | Truncates/ pads each example to fit within memory limits. |
| **Batch size (2)** | Number of samples processed simultaneously; small batches prevent CPU RAM spikes. |
| **Epochs (2)** | Full passes over the dataset; a couple of passes are enough for this toy corpus. |
| **Learning rate (5e-5)** | Controls how big each weight update is; GPT-2 models stay stable at this scale. |
| **Optimizer (AdamW)** | Standard optimizer with weight decay, great default for transformers. |
| **Checkpoint saving (disabled)** | Skips intermediate checkpoints so beginners focus on the core loop. |

For a narrative walkthrough plus observations, read `RESULTS.md`.
