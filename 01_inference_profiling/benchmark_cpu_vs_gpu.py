"""
Small script that times how long a simple text generation task takes on each
available device so beginners can see the difference between CPU and GPU.
"""

import time
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL = "distilgpt2"
PROMPT = "def fibonacci(n):"

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

    results = {}
    for device in devices:
        # Measure the average latency for the current device.
        average_latency = run(device)
        results[device] = average_latency
        print(f"{device}: {average_latency:.4f} sec")

if __name__ == "__main__":
    main()
