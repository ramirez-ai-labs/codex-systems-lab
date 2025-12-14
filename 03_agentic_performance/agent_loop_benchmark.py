"""
Agent Loop Benchmark (Beginner Friendly)
=======================================

This script demonstrates the *core idea* behind agentic systems:

    think → act → reflect

It compares:
1. A single LLM call (baseline)
2. A simple multi-step agent loop

Goal:
-----
Show that *multi-step reasoning is slower* because:
- Each step triggers a new model inference
- First-token latency repeats every step
- Latency compounds quickly

This is the foundation for understanding:
- Tool calling
- Retries
- Error recovery
- Agent orchestration systems

Nothing fancy here — clarity over cleverness.
"""

import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# ============================================================
# Configuration
# ============================================================

MODEL_NAME = "distilgpt2"
PROMPT = "Write a Python function that checks if a number is prime."
MAX_NEW_TOKENS = 64

# Number of agent steps
AGENT_STEPS = 3


# ============================================================
# Device helper
# ============================================================

def get_device():
    """
    Decide whether to use CPU or GPU.

    Beginner note:
    - CPU is totally fine for this demo
    - GPU is optional
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nRunning on device: {device}\n")
    return device


# ============================================================
# Model + tokenizer loading
# ============================================================

def load_model_and_tokenizer(device):
    """
    Load a small language model and tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # GPT-2 models don't define a padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()

    return model, tokenizer


# ============================================================
# Single inference (baseline)
# ============================================================

def run_single_inference(model, tokenizer, device):
    """
    Run ONE model generation.

    This represents:
    - A normal chatbot response
    - A non-agentic LLM call
    """
    inputs = tokenizer(PROMPT, return_tensors="pt").to(device)
    print(f"Prompt length (tokens): {inputs['input_ids'].shape[-1]}")

    start_time = time.time()
    with torch.no_grad():
        model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=MAX_NEW_TOKENS,
            pad_token_id=tokenizer.pad_token_id,
        )
    latency = time.time() - start_time

    return latency


# ============================================================
# Agent loop (think → act → reflect)
# ============================================================

def run_agent_loop(model, tokenizer, device, steps):
    """
    Run a very simple agent loop.

    Each step does:
    1. THINK   → reason about the task
    2. ACT     → generate output
    3. REFLECT → evaluate and continue

    IMPORTANT:
    Each step triggers a *new* model inference.
    """
    context = PROMPT
    step_latencies = []

    for step in range(steps):
        print(f"\n--- Agent Step {step + 1} ---")

        step_prompt = (
            f"{context}\n\n"
            "Think about the problem.\n"
            "Write a solution.\n"
            "Reflect briefly on correctness.\n"
        )

        inputs = tokenizer(step_prompt, return_tensors="pt").to(device)
        print(f"Prompt length (tokens): {inputs['input_ids'].shape[-1]}")

        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=MAX_NEW_TOKENS,
                pad_token_id=tokenizer.pad_token_id,
            )
        step_latency = time.time() - start_time
        step_latencies.append(step_latency)

        print(f"Step latency: {step_latency:.4f} sec")

        # Append output back into context (simulating memory)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        context = generated_text

    total_latency = sum(step_latencies)
    return total_latency, step_latencies


# ============================================================
# Main benchmark
# ============================================================

def main():
    print("\n==============================")
    print("  AGENT LOOP BENCHMARK")
    print("==============================")

    device = get_device()
    model, tokenizer = load_model_and_tokenizer(device)

    # ----------------------------
    # Baseline: single inference
    # ----------------------------
    print("Running single inference (baseline)...")
    single_latency = run_single_inference(model, tokenizer, device)
    print(f"Single inference latency: {single_latency:.4f} sec")

    # ----------------------------
    # Agent loop
    # ----------------------------
    print("\nRunning agent loop...")
    agent_latency, step_latencies = run_agent_loop(
        model,
        tokenizer,
        device,
        AGENT_STEPS,
    )

    # ----------------------------
    # Summary
    # ----------------------------
    print("\n==============================")
    print("           SUMMARY")
    print("==============================\n")

    print(f"Single inference latency : {single_latency:.4f} sec")
    print(f"Agent loop latency       : {agent_latency:.4f} sec")
    print(f"Agent steps              : {AGENT_STEPS}")

    slowdown = agent_latency / single_latency if single_latency > 0 else float("inf")
    print(f"\nAgent loop is ~{slowdown:.1f}x slower than single inference")
    print(
        "Note:\n"
        "Single inference generates a longer answer in one pass.\n"
        "Agent steps generate smaller chunks but repeat setup costs."
    )

    print(
        "\nBeginner takeaway:\n"
        "- Each agent step repeats first-token latency\n"
        "- Latency compounds linearly with steps\n"
        "- This is why agent systems feel slower than chat\n"
        "- All optimizations (tools, caching, batching) try to fight this"
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()
