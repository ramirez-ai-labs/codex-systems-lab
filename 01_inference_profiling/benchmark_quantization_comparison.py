"""
Quantization Comparison Benchmark
=================================

This script helps beginners understand how **model precision**
affects inference speed and memory usage.

We compare:
- FP32 (32-bit floating point, default)
- INT8 (8-bit integers, dynamically quantized)

Why this matters:
- Production LLM systems almost never run full FP32
- Lower precision = smaller models + faster inference (especially on CPU)
- Quantization is one of the biggest real-world optimizations

This script is CPU-only on purpose so anyone can run it.
"""

import time

import torch
from torch.quantization import quantize_dynamic
from transformers import AutoModelForCausalLM, AutoTokenizer


# ============================================================
# Configuration (safe defaults for laptops)
# ============================================================

MODEL_NAME = "distilgpt2"
PROMPT_TEXT = "def fibonacci(n):"
MAX_NEW_TOKENS = 24

# Keep this small so the demo runs quickly
RUNS_PER_VARIANT = 2


# ============================================================
# Helper: Load tokenizer
# ============================================================

def load_tokenizer():
    """
    Load a tokenizer and configure padding correctly.

    GPT-style models do not define a padding token by default,
    so we reuse the EOS (end-of-sequence) token.
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"
    return tokenizer


# ============================================================
# Helper: Load FP32 model
# ============================================================

def load_fp32_model(tokenizer):
    """
    Load the standard FP32 (full precision) model.

    This is the baseline most beginners start with.
    """
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()  # inference mode
    return model


# ============================================================
# Helper: Load INT8 quantized model
# ============================================================

def load_int8_model(tokenizer):
    """
    Load a dynamically quantized INT8 model.

    Dynamic quantization:
    - Converts Linear layers to INT8
    - Keeps activations in FP32
    - Works very well on CPUs
    """
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()

    quantized_model = quantize_dynamic(
        model,
        {torch.nn.Linear},
        dtype=torch.qint8,
    )

    return quantized_model


# ============================================================
# Helper: Estimate model size in MB
# ============================================================

def estimate_model_size_mb(model):
    """
    Rough estimate of model size by summing all parameter bytes.

    This helps beginners *see* the memory savings from quantization.
    """
    total_bytes = 0

    for parameter in model.parameters():
        total_bytes += parameter.numel() * parameter.element_size()

    return total_bytes / (1024 * 1024)


# ============================================================
# Benchmark: Run text generation
# ============================================================

def run_generation(model, tokenizer):
    """
    Generate text and measure how long it takes.

    Returns:
    - latency (seconds)
    - number of new tokens generated
    - tokens per second (throughput)
    """
    inputs = tokenizer(
        PROMPT_TEXT,
        return_tensors="pt",
    )

    start_time = time.time()

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            max_new_tokens=MAX_NEW_TOKENS,
            pad_token_id=tokenizer.pad_token_id,
        )

    latency = time.time() - start_time

    prompt_length = inputs["input_ids"].shape[-1]
    total_length = outputs.shape[-1]
    new_tokens = max(total_length - prompt_length, 0)

    tokens_per_second = new_tokens / latency if latency > 0 else 0.0

    return latency, new_tokens, tokens_per_second


# ============================================================
# Main benchmark loop
# ============================================================

def benchmark_models():
    """
    Run the benchmark for FP32 and INT8 models
    and collect average statistics.
    """
    tokenizer = load_tokenizer()
    device = torch.device("cpu")

    models = [
        ("FP32", load_fp32_model(tokenizer)),
        ("INT8", load_int8_model(tokenizer)),
    ]

    results = []

    for name, model in models:
        print(f"\nBenchmarking {name} model...")
        model.to(device)

        latencies = []
        throughputs = []
        new_tokens_list = []

        for _ in range(RUNS_PER_VARIANT):
            latency, new_tokens, tps = run_generation(model, tokenizer)
            latencies.append(latency)
            throughputs.append(tps)
            new_tokens_list.append(new_tokens)

        results.append({
            "name": name,
            "avg_latency": sum(latencies) / len(latencies),
            "avg_tokens_per_sec": sum(throughputs) / len(throughputs),
            "avg_new_tokens": sum(new_tokens_list) / len(new_tokens_list),
            "size_mb": estimate_model_size_mb(model),
        })

        model.to("cpu")

    return results


# ============================================================
# Print results
# ============================================================

def print_results(results):
    """
    Display a clean comparison table for beginners.
    """
    print("\n==============================")
    print("   QUANTIZATION COMPARISON")
    print("==============================\n")

    header = (
        f"{'Model':<8}"
        f"{'Latency (s)':>14}"
        f"{'Tokens/sec':>14}"
        f"{'New tokens':>14}"
        f"{'Size (MB)':>12}"
    )
    print(header)
    print("-" * len(header))

    for r in results:
        print(
            f"{r['name']:<8}"
            f"{r['avg_latency']:>14.4f}"
            f"{r['avg_tokens_per_sec']:>14.2f}"
            f"{r['avg_new_tokens']:>14.0f}"
            f"{r['size_mb']:>12.1f}"
        )

    print(
        "\nTakeaways:\n"
        "- INT8 models are significantly smaller in memory.\n"
        "- CPU inference is usually faster with INT8.\n"
        "- This is why production systems rarely deploy FP32 models.\n"
    )


# ============================================================
# Entry point
# ============================================================

def main():
    print("\nQuantization demo (CPU-only for fairness and accessibility).")
    results = benchmark_models()
    print_results(results)
    print("Benchmark complete!\n")


if __name__ == "__main__":
    main()
