"""
Prepare Dataset for Light Fine-Tuning
=====================================

This script creates a **very small, beginner-friendly dataset**
of Python code examples that we will later use to fine-tune
a language model.

Why this script exists:
-----------------------
Before training ANY model, we must decide:
1. What data we are training on
2. How that data is formatted
3. How it is split into train vs validation sets

This file focuses ONLY on data preparation.
No machine learning happens here yet.

High-level steps:
-----------------
1. Define a small set of Python code examples
2. Clean and normalize the text
3. Split into training and validation sets
4. Save the results to disk in JSONL format

JSONL (JSON Lines) is commonly used for LLM fine-tuning.
Each line is one training example.
"""

import json
import random
from pathlib import Path
from typing import List, Dict


# ============================================================
# Configuration
# ============================================================

# Where we will save the prepared dataset
DATA_DIR = Path("data")

# Output files
TRAIN_FILE = DATA_DIR / "train.jsonl"
VALID_FILE = DATA_DIR / "validation.jsonl"

# How much data goes into validation (20% is common)
VALIDATION_SPLIT = 0.2

# Random seed for reproducibility
RANDOM_SEED = 42


# ============================================================
# Step 1: Define raw Python code examples
# ============================================================

def get_raw_code_examples() -> List[str]:
    """
    Return a list of simple Python code snippets.

    These are intentionally small and readable so beginners
    can understand what the model is learning from.

    In real systems, this would come from:
    - GitHub repositories
    - Internal codebases
    - Curated datasets
    """
    return [
        "def fibonacci(n):\n    return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",

        "def factorial(n):\n    result = 1\n    for i in range(1, n + 1):\n        result *= i\n    return result",

        "def is_even(number):\n    return number % 2 == 0",

        "def reverse_list(items):\n    return items[::-1]",

        "def find_max(values):\n    max_value = values[0]\n    for v in values:\n        if v > max_value:\n            max_value = v\n    return max_value",

        "class Counter:\n    def __init__(self):\n        self.count = 0\n\n    def increment(self):\n        self.count += 1",

        "def square_numbers(numbers):\n    return [n * n for n in numbers]",

        "def greet(name):\n    print(f\"Hello, {name}!\")",

        "def sum_list(numbers):\n    total = 0\n    for n in numbers:\n        total += n\n    return total",

        "def is_palindrome(text):\n    cleaned = text.lower().replace(' ', '')\n    return cleaned == cleaned[::-1]",
    ]


# ============================================================
# Step 2: Clean and normalize text
# ============================================================

def clean_code(code: str) -> str:
    """
    Clean a single code snippet.

    For this beginner example, we only:
    - Strip extra whitespace
    - Ensure consistent newlines

    More advanced pipelines might:
    - Remove comments
    - Normalize indentation
    - Deduplicate examples
    """
    return code.strip()


def build_examples(raw_codes: List[str]) -> List[Dict[str, str]]:
    """
    Convert raw code strings into training examples.

    Each example is a dictionary with a single key: "text".

    This is the format expected by many Hugging Face
    fine-tuning scripts.
    """
    examples = []

    for code in raw_codes:
        cleaned_code = clean_code(code)
        examples.append({"text": cleaned_code})

    return examples


# ============================================================
# Step 3: Train / validation split
# ============================================================

def split_train_validation(
    examples: List[Dict[str, str]],
    validation_ratio: float
) -> (List[Dict[str, str]], List[Dict[str, str]]):
    """
    Split examples into training and validation sets.

    Why this matters:
    - Training data teaches the model
    - Validation data checks if the model generalizes

    We shuffle first so the split is random.
    """
    random.seed(RANDOM_SEED)
    random.shuffle(examples)

    split_index = int(len(examples) * (1 - validation_ratio))

    train_examples = examples[:split_index]
    valid_examples = examples[split_index:]

    return train_examples, valid_examples


# ============================================================
# Step 4: Save to JSONL files
# ============================================================

def save_jsonl(path: Path, examples: List[Dict[str, str]]) -> None:
    """
    Save a list of examples to a JSONL file.

    JSONL format:
    - One JSON object per line
    - Easy to stream and parse
    """
    with path.open("w", encoding="utf-8") as f:
        for example in examples:
            json.dump(example, f)
            f.write("\n")


# ============================================================
# Main execution
# ============================================================

def main():
    print("\nPreparing fine-tuning dataset...\n")

    # Create data directory if it doesn't exist
    DATA_DIR.mkdir(exist_ok=True)

    # Step 1: Load raw examples
    raw_codes = get_raw_code_examples()
    print(f"Loaded {len(raw_codes)} raw code examples.")

    # Step 2: Clean and format
    examples = build_examples(raw_codes)
    print("Cleaned and formatted examples.")

    # Step 3: Split dataset
    train_data, valid_data = split_train_validation(
        examples,
        VALIDATION_SPLIT
    )

    print(f"Training examples: {len(train_data)}")
    print(f"Validation examples: {len(valid_data)}")

    # Step 4: Save to disk
    save_jsonl(TRAIN_FILE, train_data)
    save_jsonl(VALID_FILE, valid_data)

    print("\nDataset preparation complete!")
    print(f"- Training file: {TRAIN_FILE}")
    print(f"- Validation file: {VALID_FILE}")
    
    # Show a few examples so beginners can see the format
    print("\nSample training examples:\n")

    num_examples_to_show = min(5, len(train_data))

    for i in range(num_examples_to_show):
        print(f"Example {i + 1}:")
        print(train_data[i])
        print("-" * 40)


if __name__ == "__main__":
    main()
