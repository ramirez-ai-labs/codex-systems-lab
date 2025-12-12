"""
KV Cache Analysis Benchmark
===========================

This script helps beginners understand one of the most important LLM
performance concepts:

    - The FIRST generated token is slow
    - The NEXT tokens are MUCH faster (thanks to the KV cache)

We fix all tokenizer warnings by providing an explicit attention mask.

Why an attention mask?
----------------------
An attention mask tells the model WHICH tokens are real input
and WHICH are padding. Without it, the model may behave unpredictably,
especially when padding token == eos token (common in GPT models).
"""

import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "distilgpt2"
PROMPT_TEXT = "def fibonacci(n):"
NUM_NEXT_TOKENS = 15


def tokenize_prompt(tokenizer, device):
    """
    Convert raw text into model-ready tensors.
    Ensures:
    - input_ids
    - attention_mask
    are both created and moved to the correct device.
    """
    encoded = tokenizer(
        PROMPT_TEXT,
        return_tensors="pt",
        padding=True
    )

    # Move to device (CPU or GPU)
    encoded = {key: value.to(device) for key, value in encoded.items()}
    return encoded


def measure_first_token_latency(model, tokenizer, device):
    """
    Measure how long it takes for the model to generate the FIRST token.
    This is slow because the model must read and process the entire prompt.
    """
    encoded = tokenize_prompt(tokenizer, device)

    start_time = time.time()
    with torch.no_grad():
        model.generate(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            max_new_tokens=1
        )
    return time.time() - start_time


def measure_next_token_latency(model, tokenizer, device, num_tokens):
    """
    Measure the average time for the model to generate SUBSEQUENT tokens.
    These are faster because the KV-cache remembers previous computations.
    """
    encoded = tokenize_prompt(tokenizer, device)

    # Warmup: generate the first token to populate KV cache
    with torch.no_grad():
        output = model.generate(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            max_new_tokens=1
        )

    # Prepare updated input for next-token generation loop
    input_ids = output
    attention_mask = torch.ones_like(input_ids).to(device)

    token_times = []

    for _ in range(num_tokens):
        start_time = time.time()

        with torch.no_grad():
            output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=1
            )

        end_time = time.time()
        token_times.append(end_time - start_time)

        # Update inputs for next iteration
        input_ids = output
        attention_mask = torch.ones_like(input_ids).to(device)

    return sum(token_times) / len(token_times)


def main():
    print("\n==============================")
    print("        KV CACHE TEST")
    print("==============================\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}\n")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)
    # Hugging Face warns when model.config.pad_token_id is unset even if the tokenizer has one.
    model.config.pad_token_id = tokenizer.pad_token_id

    print("Measuring first-token latency...")
    first_time = measure_first_token_latency(model, tokenizer, device)
    print(f"⏱ First token latency : {first_time:.4f} sec\n")

    print("Measuring next-token latency...")
    next_time = measure_next_token_latency(model, tokenizer, device, NUM_NEXT_TOKENS)
    print(f"⚡ Next-token latency  : {next_time:.4f} sec (average)\n")

    print("==============================")
    print("         SUMMARY")
    print("==============================\n")
    print(f"First token latency : {first_time:.4f} s")
    print(f"Next-token latency  : {next_time:.4f} s (avg)")
    print(f"Speedup factor      : {first_time / next_time:.1f}x\n")

    print("Why this matters:")
    print("- Coding assistants feel instant because next-token latency is tiny.")
    print("- First-token latency dominates chat UX.")
    print("- KV cache is one of the biggest speed improvements in LLMs.\n")

    print("Benchmark complete!\n")


if __name__ == "__main__":
    main()
