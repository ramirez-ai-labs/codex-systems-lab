"""
Batch Size Benchmark
====================

This script measures how different *batch sizes* affect the speed of text 
generation for a small language model.

Beginners can use this to understand why batching improves throughput:

- Running one prompt at a time under-utilizes the model.
- Running several prompts *together* lets the model compute multiple outputs 
  in parallel.
- Even if the total time increases slightly, the *tokens per second* go way up.

This is how real AI systems (including coding assistants) serve many users at once.
"""

import time
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# We use a tiny model so beginners can run this on a laptop.
MODEL_NAME = "distilgpt2"

# A set of simple Python prompts to simulate code generation inputs.
PROMPTS = [
    "def fibonacci(n):",
    "def quicksort(arr):",
    "def factorial(n):",
    "def merge_sort(numbers):",
]


# ------------------------------------------------------------
# Helper: Build a prompt batch
# ------------------------------------------------------------
def build_prompt_batch(batch_size: int) -> List[str]:
    """
    Create a list of text prompts of length `batch_size`.

    We simply repeat the predefined prompts until we reach the requested size.
    This gives us realistic but simple inputs to test with.
    """
    prompts: List[str] = []

    # Keep copying prompts until the list is big enough.
    while len(prompts) < batch_size:
        prompts.extend(PROMPTS)

    return prompts[:batch_size]


# ------------------------------------------------------------
# Helper: Run generation for a batch
# ------------------------------------------------------------
def generate_batch(
    prompts: List[str],
    model,
    tokenizer,
    device: torch.device,
    max_new_tokens: int,
) -> float:
    """
    Generate text for a list of prompts and return the time it took.

    Steps:
    1. Tokenize all prompts AND pad them to the same length.
    2. Move the input tensors to the CPU/GPU device.
    3. Call model.generate() to actually produce new tokens.
    4. Measure the total time.

    This function does not return the generated text — only the latency.
    """
    tokenized_inputs = tokenizer(
        prompts, 
        return_tensors="pt",
        padding=True
    ).to(device)

    start_time = time.time()

    # Turn off gradients so inference runs faster.
    with torch.no_grad():
        model.generate(**tokenized_inputs, max_new_tokens=max_new_tokens)

    end_time = time.time()
    return end_time - start_time


# ------------------------------------------------------------
# Main benchmark function
# ------------------------------------------------------------
def benchmark_batch_sizes(batch_sizes, runs_per_size=5, max_new_tokens=32):
    """
    Measure how long each batch size takes on average.

    For each batch size:
    - Prepare a batch of prompts (e.g., 1, 2, 4, 8)
    - Run generation several times (for stable results)
    - Print the average latency
    """
    # Detect CPU or GPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nRunning on device: {device}\n")

    # Load tokenizer (converts raw text → tokens)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # GPT-2 models don't include a padding token → reuse EOS token.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Decoder-only models expect padding on the *left*.
    # This keeps important tokens aligned correctly.
    tokenizer.padding_side = "left"

    # Load model and move it to the device.
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)

    print("Batch Size Benchmark")
    print("====================\n")

    for batch_size in batch_sizes:
        latency_measurements = []

        for _ in range(runs_per_size):
            prompts = build_prompt_batch(batch_size)
            latency = generate_batch(prompts, model, tokenizer, device, max_new_tokens)
            latency_measurements.append(latency)

        avg_latency = sum(latency_measurements) / len(latency_measurements)
        print(f"Batch size {batch_size:>2}: {avg_latency:.4f} sec")

    print("\nBenchmark complete!\n")


# ------------------------------------------------------------
# Entry point
# ------------------------------------------------------------
if __name__ == "__main__":
    # These sizes show the effect clearly on both CPU and GPU.
    batch_sizes_to_test = [1, 2, 4, 8]
    benchmark_batch_sizes(batch_sizes_to_test)
