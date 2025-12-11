"""
BEGINNER-FRIENDLY CPU vs GPU Inference Benchmark
================================================

This script shows how long it takes a small language model (DistilGPT-2)
to generate text on different devices (CPU and GPU).

It is intentionally written with beginners in mind:
- no compact Python shortcuts
- simple variable names
- step-by-step comments explaining *why* each line exists
- gentle introduction to model loading, tokenization, and generation

You can run this file directly:

    python benchmark_cpu_vs_gpu.py

If your machine has a GPU (NVIDIA + CUDA), the script will test both CPU and GPU.
Otherwise, it will run only on the CPU.
"""

import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# We use a very small model so beginners can experiment on any machine
MODEL_NAME = "distilgpt2"

# A short piece of code for the model to complete
PROMPT_TEXT = "def fibonacci(n):"


def measure_generation_time(device_name, number_of_runs=5, max_new_tokens=32):
    """
    Measure how long it takes the model to generate text on one device.

    Parameters:
        device_name (str): "cpu" or "cuda"
        number_of_runs (int): how many times we repeat the measurement
        max_new_tokens (int): how many new tokens the model should generate

    Returns:
        float: average number of seconds per generation
    """

    # ----------------------------------------------------------
    # 1. Load the tokenizer (turns words into tokens)
    # ----------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # ----------------------------------------------------------
    # 2. Load the model itself (the neural network)
    # ----------------------------------------------------------
    # .to(device_name) moves the model onto CPU or GPU
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model = model.to(device_name)

    # ----------------------------------------------------------
    # 3. Convert the prompt string into token IDs
    # ----------------------------------------------------------
    encoded_inputs = tokenizer(PROMPT_TEXT, return_tensors="pt")

    # Move token IDs onto the same device as the model
    encoded_inputs = encoded_inputs.to(device_name)

    # ----------------------------------------------------------
    # 4. Repeatedly measure generation time
    # ----------------------------------------------------------
    run_times = []

    for run_index in range(number_of_runs):

        # Record the start time
        start_time = time.time()

        # Turn off gradient tracking since we are only doing inference
        with torch.no_grad():
            model.generate(
                input_ids=encoded_inputs["input_ids"],
                attention_mask=encoded_inputs["attention_mask"],
                max_new_tokens=max_new_tokens
            )

        # Record how long this generation took
        end_time = time.time()
        elapsed_time = end_time - start_time

        run_times.append(elapsed_time)

    # ----------------------------------------------------------
    # 5. Compute the average time across all runs
    # ----------------------------------------------------------
    average_time = sum(run_times) / len(run_times)
    return average_time


def main():
    """
    Main entry point for running the benchmark.
    """

    print("\n==============================")
    print(" CPU vs GPU Inference Benchmark")
    print("==============================\n")

    # Always include CPU, since every machine has one
    devices_to_test = ["cpu"]

    # Check whether this machine has a CUDA-capable GPU
    if torch.cuda.is_available():
        devices_to_test.append("cuda")
        print("A CUDA-enabled GPU was detected! Testing CPU and GPU...\n")
    else:
        print("No GPU detected. Running on CPU only.\n")

    # Dictionary to store results
    results = {}

    # Run the benchmark for each device
    for device_name in devices_to_test:
        print(f"Running inference on: {device_name.upper()}")

        average_latency = measure_generation_time(
            device_name=device_name,
            number_of_runs=5,
            max_new_tokens=32
        )

        results[device_name] = average_latency
        print(f"Average time on {device_name.upper()}: {average_latency:.4f} seconds\n")

    # ----------------------------------------------------------
    # Final summary
    # ----------------------------------------------------------
    print("\n==============================")
    print("        FINAL RESULTS")
    print("==============================\n")

    for device_name, latency in results.items():
        print(f"{device_name.upper()}: {latency:.4f} seconds")

    print("\nBenchmark complete!\n")


if __name__ == "__main__":
    main()
