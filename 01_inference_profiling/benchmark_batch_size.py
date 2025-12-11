"""
Measure how different batch sizes affect text generation latency.
Beginners can use this to see why batching improves overall throughput.
"""

import time
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "distilgpt2"
PROMPTS = [
    "def fibonacci(n):",
    "def quicksort(arr):",
    "def factorial(n):",
    "def merge_sort(numbers):",
]


def build_prompt_batch(batch_size: int) -> List[str]:
    """
    Repeat the prompt list until it reaches the desired batch size.
    """
    prompts: List[str] = []
    while len(prompts) < batch_size:
        prompts.extend(PROMPTS)
    return prompts[:batch_size]


def generate_batch(prompts: List[str], model, tokenizer, device: torch.device, max_new_tokens: int) -> float:
    """
    Run generation for a list of prompts and return how long it took.
    """
    # Tokenize all prompts and move the tensors to the correct device.
    tokenized = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
    start_time = time.time()
    with torch.no_grad():
        model.generate(**tokenized, max_new_tokens=max_new_tokens)
    return time.time() - start_time


def benchmark_batch_sizes(batch_sizes, runs_per_size=5, max_new_tokens=32):
    """
    For each batch size, run several generations and report the average latency.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Tokenizer converts raw text prompts into token ids.
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # GPT-2 style models do not define a padding token, so reuse the EOS token.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Decoder-only models expect padding on the left when batching inputs.
    tokenizer.padding_side = "left"
    # Model is the neural network that predicts the next tokens.
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)

    for batch_size in batch_sizes:
        latencies = []
        for _ in range(runs_per_size):
            prompts = build_prompt_batch(batch_size)
            latency = generate_batch(prompts, model, tokenizer, device, max_new_tokens)
            latencies.append(latency)
        average_latency = sum(latencies) / len(latencies)
        print(f"Batch size {batch_size}: {average_latency:.4f} sec")


if __name__ == "__main__":
    batch_sizes_to_test = [1, 2, 4, 8]
    benchmark_batch_sizes(batch_sizes_to_test)
