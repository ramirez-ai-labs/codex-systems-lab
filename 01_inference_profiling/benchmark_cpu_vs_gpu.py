"""

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
Small script that times how long a simple text generation task takes on each
available device so beginners can see the difference between CPU and GPU.
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

def run(device, runs=5, max_new_tokens=32):
    """
    Load the model and tokenizer on the requested device, run several
    generations, and return the average time per run.
    """
    # Tokenizer: turns raw text into token ids the model understands.
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    # Model: the neural network that predicts the next tokens.
    model = AutoModelForCausalLM.from_pretrained(MODEL).to(device)
    # Inputs: tokenized prompt tensors moved onto the same device as the model.
    inputs = tokenizer(PROMPT, return_tensors="pt").to(device)

    latencies = []
    for _ in range(runs):
        # Capture the exact time right before generation begins.
        start_time = time.time()
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=max_new_tokens)
        # Store how long this iteration took so we can average later.
        latencies.append(time.time() - start_time)
    return sum(latencies) / len(latencies)

def main():
    # Always include the CPU because every machine has at least one.
    devices = ["cpu"]
    if torch.cuda.is_available():
        # Only add the GPU if PyTorch can see one on this machine.
        devices.append("cuda")

    # Dictionary to store results
    results = {}
    for device in devices:
        # Measure the average latency for the current device.
        average_latency = run(device)
        results[device] = average_latency
        print(f"{device}: {average_latency:.4f} sec")

if __name__ == "__main__":
    main()
