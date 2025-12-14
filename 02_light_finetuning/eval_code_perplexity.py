"""
Code Perplexity Evaluation (Beginner Friendly)
==============================================

This script evaluates a language model using **perplexity**, a standard
metric for language modeling.

What perplexity measures (plain English):
----------------------------------------
Perplexity answers the question:

    "How surprised is the model by the text?"

- LOW perplexity  → the model predicts the text well
- HIGH perplexity → the model struggles

For code models:
- Lower perplexity usually means better understanding of syntax & patterns
- It does NOT guarantee functional correctness
- It IS a good first sanity check after fine-tuning

In this lab we:
1. Load a validation dataset of Python code
2. Compute perplexity for:
   - Base DistilGPT-2
   - Fine-tuned DistilGPT-2 (if available)
3. Compare the results

This script runs on CPU and finishes quickly.
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

# Where the fine-tuned model would live (optional)
FINETUNED_MODEL_DIR = "outputs"

MAX_SEQ_LENGTH = 128


# ============================================================
# Helper: Device
# ============================================================

def get_device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nRunning on device: {device}\n")
    return device


# ============================================================
# Load validation data
# ============================================================

def load_validation_data():
    """
    Load validation examples from JSONL.

    Each example must have:
        { "text": "some python code..." }
    """
    print("Loading validation dataset...")

    dataset = load_dataset(
        "json",
        data_files={"validation": VALID_FILE}
    )

    print(f"Loaded {len(dataset['validation'])} validation examples\n")

    print("Sample validation examples:")
    for i in range(min(3, len(dataset["validation"]))):
        print(f"{i+1}. {dataset['validation'][i]['text'][:60]}...")

    return dataset["validation"]


# ============================================================
# Tokenization
# ============================================================

def tokenize_examples(dataset, tokenizer):
    """
    Convert text into token IDs.

    Important:
    - Models operate on tokens, not raw text
    - We truncate long examples to keep runtime low
    """

    def tokenize(example):
        tokens = tokenizer(
            example["text"],
            truncation=True,
            padding="max_length",
            max_length=MAX_SEQ_LENGTH,
            return_tensors="pt",
        )
        tokens["labels"] = tokens["input_ids"].clone()
        return tokens

    tokenized = dataset.map(tokenize, remove_columns=["text"])
    return tokenized


# ============================================================
# Perplexity computation
# ============================================================

def compute_perplexity(model, tokenized_dataset, device) -> float:
    """
    Compute perplexity over a dataset.

    Steps:
    1. Run model forward pass
    2. Collect loss
    3. Convert loss → perplexity using exp()
    """
    model.eval()
    losses: List[float] = []

    with torch.no_grad():
        for example in tokenized_dataset:
            input_ids = example["input_ids"].to(device)
            labels = example["labels"].to(device)

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
    tokenized_validation = tokenize_examples(validation_data, tokenizer)

    # ----------------------------
    # Base model evaluation
    # ----------------------------
    print("\nEvaluating BASE model...")
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    base_model.to(device)

    base_ppl = compute_perplexity(base_model, tokenized_validation, device)
    print(f"Base model perplexity: {base_ppl:.2f}")

    # ----------------------------
    # Fine-tuned model evaluation (optional)
    # ----------------------------
    if os.path.exists(FINETUNED_MODEL_DIR):
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
        print("\nNo fine-tuned model found. Skipping fine-tuned evaluation.")

    print(
        "\nInterpretation for beginners:\n"
        "- Lower perplexity means the model predicts code more confidently\n"
        "- A small improvement is expected with tiny datasets\n"
        "- Large gains require more data, better objectives, or LoRA-style tuning\n"
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    evaluate()
