"""
Batching Effects Demo
=====================

Goal: show beginners why serving several prompts together (a *batch*)
dramatically improves throughput compared to handling one prompt at a time.

Key ideas covered in this script:

1. A transformer can process multiple sequences simultaneously.
2. Even if the total latency per batch increases a bit, the
   **tokens-per-second** metric skyrockets with larger batches.
3. This is the strategy production models use to keep coding assistants
   and chatbots responsive for many users at once.
"""

import time
from typing import Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Configuration — tweak these if you want to experiment further.
# ---------------------------------------------------------------------------
MODEL_NAME = "distilgpt2"
# Reusable prompts that simulate simple coding requests.
PROMPT_LIBRARY = [
    "Write a Python function that returns the nth Fibonacci number.",
    "Implement a stack class with push and pop operations.",
    "Explain recursion using a factorial example.",
    "Clean up this snippet by adding comments and docstrings.",
]
BATCH_SIZES = [1, 2, 4, 8]
MAX_NEW_TOKENS = 24
RUNS_PER_BATCH = 2


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def build_prompts(batch_size: int) -> List[str]:
    """
    Create `batch_size` prompts by repeating entries from PROMPT_LIBRARY.

    We recycle prompts so the benchmark works even if the requested
    batch size is larger than the number of unique examples we wrote.
    """
    prompts: List[str] = []
    while len(prompts) < batch_size:
        prompts.extend(PROMPT_LIBRARY)
    return prompts[:batch_size]


def load_model_and_tokenizer(device: torch.device):
    """
    Load the tiny GPT-2 model and tokenizer.

    - Adds a padding token to avoid warnings.
    - Ensures the model config knows about that padding token.
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)
    model.config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer


def run_one_batch(
    prompts: List[str],
    model,
    tokenizer,
    device: torch.device,
) -> Dict[str, float]:
    """
    Generate text for a batch of prompts and capture latency + throughput.

    Returns a dictionary with:
        - latency: seconds for the full batch
        - total_new_tokens: actual number of tokens generated
        - tokens_per_second: throughput metric beginners can compare
    """
    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
    )
    encoded = {k: v.to(device) for k, v in encoded.items()}

    start = time.time()
    with torch.no_grad():
        outputs = model.generate(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            max_new_tokens=MAX_NEW_TOKENS,
            pad_token_id=tokenizer.pad_token_id,
        )
    latency = time.time() - start

    # Each row in outputs is prompt_tokens + new_tokens.
    prompt_length = encoded["input_ids"].shape[-1]
    generated_length = outputs.shape[-1]
    new_tokens_per_prompt = max(generated_length - prompt_length, 0)
    total_new_tokens = new_tokens_per_prompt * len(prompts)
    tokens_per_second = (
        total_new_tokens / latency if latency > 0 else float("inf")
    )

    return {
        "latency": latency,
        "total_new_tokens": total_new_tokens,
        "tokens_per_second": tokens_per_second,
    }


def benchmark_batches(batch_sizes: List[int], runs: int) -> List[Dict[str, float]]:
    """
    Loop over all requested batch sizes and record average stats.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n==============================")
    print("      BATCHING EFFECTS DEMO")
    print("==============================")
    print(f"Running on device: {device}\n")

    model, tokenizer = load_model_and_tokenizer(device)
    results: List[Dict[str, float]] = []

    for batch_size in batch_sizes:
        batch_latencies = []
        batch_throughputs = []
        batch_tokens = []

        for _ in range(runs):
            prompts = build_prompts(batch_size)
            stats = run_one_batch(prompts, model, tokenizer, device)
            batch_latencies.append(stats["latency"])
            batch_throughputs.append(stats["tokens_per_second"])
            batch_tokens.append(stats["total_new_tokens"])

        avg_latency = sum(batch_latencies) / len(batch_latencies)
        avg_throughput = sum(batch_throughputs) / len(batch_throughputs)
        avg_tokens = sum(batch_tokens) / len(batch_tokens)

        results.append(
            {
                "batch_size": batch_size,
                "avg_latency": avg_latency,
                "avg_tokens": avg_tokens,
                "tokens_per_second": avg_throughput,
            }
        )

    return results


def print_summary(results: List[Dict[str, float]]) -> None:
    """
    Display a beginner-friendly table that highlights the throughput gains.
    """
    print("Results (averaged over multiple runs):\n")
    header = f"{'Batch':>5} | {'Latency (s)':>12} | {'Tokens/batch':>13} | {'Tokens/sec':>11}"
    print(header)
    print("-" * len(header))
    for entry in results:
        print(
            f"{entry['batch_size']:>5} | "
            f"{entry['avg_latency']:>12.4f} | "
            f"{entry['avg_tokens']:>13.0f} | "
            f"{entry['tokens_per_second']:>11.1f}"
        )

    print(
        "\nTakeaway: batching makes the *tokens/sec* column explode upward,\n"
        "which is why large language models serve many users at once instead of\n"
        "processing each request in isolation."
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    results = benchmark_batches(BATCH_SIZES, RUNS_PER_BATCH)
    print_summary(results)


if __name__ == "__main__":
    main()
