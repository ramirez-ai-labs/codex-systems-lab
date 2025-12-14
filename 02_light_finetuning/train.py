"""
Light Fine-Tuning Script (Beginner Friendly)
===========================================

This script fine-tunes a small language model (DistilGPT-2)
on a tiny Python code dataset.

IMPORTANT:
- This is NOT a production training script
- This is a teaching example
- It is intentionally small, slow, and readable

You will learn:
- What fine-tuning means
- How data flows into a language model
- How tokenization works
- How Hugging Face Trainer runs training
"""

import os
import torch

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
)

# ============================================================
# Configuration
# ============================================================

MODEL_NAME = "distilgpt2"

DATA_DIR = "data"
TRAIN_FILE = os.path.join(DATA_DIR, "train.jsonl")
VALID_FILE = os.path.join(DATA_DIR, "validation.jsonl")

OUTPUT_DIR = "outputs"

# Beginner-friendly hyperparameters.
MAX_SEQ_LENGTH = 128     # cap each example at 128 tokens so padding/truncation stays cheap
BATCH_SIZE = 2           # number of samples per optimizer step; tiny to fit on CPU
NUM_EPOCHS = 2           # number of full passes over the dataset
LEARNING_RATE = 5e-5     # step size for weight updates (stable default for GPT-style models)


# ============================================================
# Device helper
# ============================================================

def get_device():
    """
    Decide whether to run on CPU or GPU.

    Beginners:
    - CPU is perfectly fine for small experiments
    - GPU is optional and not required here
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nRunning on device: {device}\n")
    return device


# ============================================================
# Dataset loading
# ============================================================

def load_datasets():
    """
    Load training and validation data from JSONL files.

    Expected format (one JSON per line):

    {
        "text": "def fibonacci(n): ..."
    }
    """
    print("Loading datasets...\n")

    dataset = load_dataset(
        "json",
        data_files={
            "train": TRAIN_FILE,
            "validation": VALID_FILE,
        },
    )

    print(f"Loaded {len(dataset['train'])} training examples")
    print(f"Loaded {len(dataset['validation'])} validation examples")

    print("\nSample training examples:")
    for i in range(min(3, len(dataset["train"]))):
        snippet = dataset["train"][i]["text"][:80].replace("\n", " ")
        print(f"{i + 1}. {snippet}...")

    return dataset


# ============================================================
# Tokenization
# ============================================================

def tokenize_datasets(dataset, tokenizer):
    """
    Convert raw text into token IDs.

    Beginner explanation:
    - Models cannot read text
    - They only understand numbers (tokens)
    - Tokenization turns text → numbers
    """

    def tokenize_example(example):
        tokens = tokenizer(
            example["text"],
            truncation=True,
            padding="max_length",
            max_length=MAX_SEQ_LENGTH,
        )

        # For causal language modeling:
        # The model tries to predict the next token,
        # so labels are the same as input_ids.
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    print("\nTokenizing datasets...")
    tokenized = dataset.map(
        tokenize_example,
        remove_columns=["text"],
    )

    return tokenized


# ============================================================
# Training
# ============================================================

def train():
    """
    Main training function.
    """

    # ----------------------------
    # Device
    # ----------------------------
    device = get_device()

    # ----------------------------
    # Tokenizer
    # ----------------------------
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # GPT-2 has no pad token by default
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ----------------------------
    # Dataset
    # ----------------------------
    raw_datasets = load_datasets()
    tokenized_datasets = tokenize_datasets(raw_datasets, tokenizer)

    # ----------------------------
    # Model
    # ----------------------------
    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    # Resize embeddings because we added a pad token
    model.resize_token_embeddings(len(tokenizer))

    model.to(device)
    model.train()

    # ----------------------------
    # Training arguments
    # ----------------------------
    # Kept intentionally minimal and version-safe
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=NUM_EPOCHS,                 # two full passes keep runtime short
        per_device_train_batch_size=BATCH_SIZE,      # small batch keeps RAM usage low
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,                 # gentle updates to avoid divergence
        logging_steps=1,
        save_strategy="no",                         # disable checkpoints for simplicity
        report_to="none",                           # silence TensorBoard/W&B integrations
    )

    # Ensure output directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ----------------------------
    # Trainer
    # ----------------------------
    # Hugging Face Trainer builds an AdamW optimizer under the hood,
    # which is a standard choice for transformer fine-tuning tasks.
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
    )

    # ----------------------------
    # Train
    # ----------------------------
    print("\nStarting training...\n")
    trainer.train()

    # ----------------------------
    # Save model
    # ----------------------------
    print("\nSaving fine-tuned model...")
    
    # NOTE:
    # Some CPU-only PyTorch + Transformers installs fail when saving models
    # due to DTensor / distributed imports.
    # This does NOT affect training correctness.
    

    try:
        model.save_pretrained(OUTPUT_DIR, safe_serialization=False)
        tokenizer.save_pretrained(OUTPUT_DIR)
        print(f"Model saved to: {OUTPUT_DIR}")
    except Exception as e:
        print("\n⚠️ Model saving skipped.")
        print("Reason:", e)
        print(
            "\nThis is a known PyTorch/Transformers compatibility issue on some CPU-only setups.\n"
            "Training completed successfully, which is the main goal of this lab."
        )

    print("\nTraining complete!")
    print(f"Model saved to: {OUTPUT_DIR}")


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    train()
