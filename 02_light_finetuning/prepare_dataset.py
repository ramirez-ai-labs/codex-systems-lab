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

        "def find_min(values):\n    min_value = values[0]\n    for v in values:\n        if v < min_value:\n            min_value = v\n    return min_value",

        "def average(numbers):\n    return sum(numbers) / len(numbers)",

        "def count_vowels(text):\n    return sum(1 for ch in text.lower() if ch in 'aeiou')",

        "def flatten(nested_list):\n    result = []\n    for item in nested_list:\n        result.extend(item)\n    return result",

        "def remove_duplicates(items):\n    return list(dict.fromkeys(items))",

        "def is_prime(n):\n    if n < 2:\n        return False\n    for i in range(2, int(n ** 0.5) + 1):\n        if n % i == 0:\n            return False\n    return True",

        "def gcd(a, b):\n    while b:\n        a, b = b, a % b\n    return a",

        "def bubble_sort(items):\n    items = items.copy()\n    n = len(items)\n    for i in range(n):\n        for j in range(n - i - 1):\n            if items[j] > items[j + 1]:\n                items[j], items[j + 1] = items[j + 1], items[j]\n    return items",

        "def binary_search(items, target):\n    low, high = 0, len(items) - 1\n    while low <= high:\n        mid = (low + high) // 2\n        if items[mid] == target:\n            return mid\n        elif items[mid] < target:\n            low = mid + 1\n        else:\n            high = mid - 1\n    return -1",

        "def word_count(text):\n    counts = {}\n    for word in text.split():\n        counts[word] = counts.get(word, 0) + 1\n    return counts",

        "def merge_dicts(a, b):\n    merged = a.copy()\n    merged.update(b)\n    return merged",

        "def chunk_list(items, size):\n    return [items[i:i + size] for i in range(0, len(items), size)]",

        "class Stack:\n    def __init__(self):\n        self.items = []\n\n    def push(self, item):\n        self.items.append(item)\n\n    def pop(self):\n        return self.items.pop()",

        "class Queue:\n    def __init__(self):\n        self.items = []\n\n    def enqueue(self, item):\n        self.items.append(item)\n\n    def dequeue(self):\n        return self.items.pop(0)",

        "def celsius_to_fahrenheit(celsius):\n    return (celsius * 9 / 5) + 32",

        "def clamp(value, low, high):\n    return max(low, min(value, high))",

        "def title_case(text):\n    return ' '.join(word.capitalize() for word in text.split())",

        "def unique_characters(text):\n    return len(set(text)) == len(text)",

        "def sum_of_digits(n):\n    return sum(int(digit) for digit in str(abs(n)))",

        "def rotate_list(items, k):\n    k = k % len(items)\n    return items[-k:] + items[:-k]",

        "def is_anagram(a, b):\n    return sorted(a.lower()) == sorted(b.lower())",

        "def range_sum(start, end):\n    return sum(range(start, end + 1))",

        "def to_snake_case(text):\n    return text.replace(' ', '_').lower()",

        "class Node:\n    def __init__(self, value):\n        self.value = value\n        self.next = None",

        "def linked_list_length(head):\n    count = 0\n    current = head\n    while current is not None:\n        count += 1\n        current = current.next\n    return count",

        "def filter_positive(numbers):\n    return [n for n in numbers if n > 0]",

        "def deep_get(d, keys, default=None):\n    for key in keys:\n        if isinstance(d, dict) and key in d:\n            d = d[key]\n        else:\n            return default\n    return d",

        "def moving_average(values, window):\n    return [sum(values[i:i + window]) / window for i in range(len(values) - window + 1)]",

        "def most_common(items):\n    counts = {}\n    for item in items:\n        counts[item] = counts.get(item, 0) + 1\n    return max(counts, key=counts.get)",
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
