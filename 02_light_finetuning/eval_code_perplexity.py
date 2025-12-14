"""
Code Perplexity Evaluation (Beginner Friendly)
==============================================

This script evaluates a language model using **perplexity**, a standard
metric for language modeling.

Plain-English explanation:
--------------------------
Perplexity answers the question:

    "How surprised is the model by this text?"

- Lower perplexity  → model predicts the text better
- Higher perplexity → model struggles

IMPORTANT:
- Perplexity does NOT measure correctness
- It is a sanity check, not a full evaluation
- For code, perplexity mainly reflects syntax familiarity

This script:
1. Loads validation code examples
2. Computes perplexity for:
   - Base DistilGPT-2
   - Fine-tuned model (if available)
3. Compares results
"""

import math
import os
import torch
from typing import List

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


# ============================================================
# Configuration
# ============================================================

MODEL_NAME = "distilgpt2"

DATA_DIR = "data"
VALID_FILE = os.path.join(DATA_DIR, "validation.jsonl")

# Where the fine-tuned model *would* live
FINETUNED_MODEL_DIR = "outputs"

MAX_SEQ_LENGTH = 128


# ============================================================
# Device helper
# ============================================================

def get_device() -> torch.device:
    """Return CPU or GPU if available."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nRunning on device: {device}\n")
    return device


# ============================================================
# Load validation dataset
# ============================================================

def load_validation_data():
    """
    Load validation examples from JSONL.

    Expected format:
    { "text": "some python code..." }
    """
    print("Loading validation dataset...")

    dataset = load_dataset(
        "json",
        data_files={"validation": VALID_FILE},
    )

    print(f"Loaded {len(dataset['validation'])} validation examples\n")

    print("Sample validation examples:")
    for i in range(min(2, len(dataset["validation"]))):
        print(f"{i + 1}. {dataset['validation'][i]['text'][:60]}...")

    return dataset["validation"]


# ============================================================
# Tokenization
# ============================================================

def tokenize_validation_data(dataset, tokenizer):
    """
    Convert text into token IDs.

    Important for beginners:
    - Models do NOT read text
    - They read token IDs (numbers)
    """
    def tokenize(example):
        encoding = tokenizer(
            example["text"],
            truncation=True,
            padding="max_length",
            max_length=MAX_SEQ_LENGTH,
        )

        encoding["labels"] = encoding["input_ids"].copy()
        return encoding

    print("\nTokenizing validation dataset...")
    tokenized = dataset.map(tokenize, remove_columns=["text"])
    return tokenized


# ============================================================
# Check if fine-tuned model actually exists
# ============================================================

def finetuned_model_exists(model_dir: str) -> bool:
    """
    Check if a Hugging Face model checkpoint actually exists.
    """
    if not os.path.isdir(model_dir):
        return False

    for fname in ["pytorch_model.bin", "model.safetensors"]:
        if os.path.isfile(os.path.join(model_dir, fname)):
            return True

    return False


# ============================================================
# Perplexity computation
# ============================================================

def compute_perplexity(model, tokenized_dataset, device) -> float:
    """
    Compute perplexity over a dataset.

    Steps:
    1. Run model forward pass
    2. Collect loss
    3. Convert average loss → perplexity using exp()
    """
    model.eval()
    losses: List[float] = []

    with torch.no_grad():
        for example in tokenized_dataset:
            input_ids = torch.tensor(example["input_ids"]).unsqueeze(0).to(device)
            labels = torch.tensor(example["labels"]).unsqueeze(0).to(device)

            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            losses.append(loss.item())

    average_loss = sum(losses) / len(losses)
    perplexity = math.exp(average_loss)
    return perplexity


# ============================================================
# Main evaluation
# ============================================================

def evaluate():
    device = get_device()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    validation_data = load_validation_data()
    tokenized_validation = tokenize_validation_data(validation_data, tokenizer)

    # --------------------------------------------------------
    # Base model evaluation
    # --------------------------------------------------------
    print("\nEvaluating BASE model...")
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    base_model.to(device)

    base_ppl = compute_perplexity(base_model, tokenized_validation, device)
    print(f"Base model perplexity: {base_ppl:.2f}")

    # --------------------------------------------------------
    # Fine-tuned model evaluation (optional)
    # --------------------------------------------------------
    if finetuned_model_exists(FINETUNED_MODEL_DIR):
        print("\nEvaluating FINE-TUNED model...")
        finetuned_model = AutoModelForCausalLM.from_pretrained(FINETUNED_MODEL_DIR)
        finetuned_model.to(device)

        finetuned_ppl = compute_perplexity(
            finetuned_model,
            tokenized_validation,
            device,
        )
        print(f"Fine-tuned model perplexity: {finetuned_ppl:.2f}")

        improvement = base_ppl - finetuned_ppl
        print(f"\nPerplexity improvement: {improvement:.2f}")
    else:
        print(
            "\nFine-tuned model checkpoint not found.\n"
            "This is expected if model saving failed during training.\n"
            "Skipping fine-tuned evaluation."
        )

    print(
        "\nInterpretation for beginners:\n"
        "- Lower perplexity means the model predicts code more confidently\n"
        "- A small improvement is expected with tiny datasets\n"
        "- Large gains require more data or parameter-efficient tuning (LoRA)\n"
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    evaluate()
