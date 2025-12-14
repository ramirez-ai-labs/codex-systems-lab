"""
Tool Latency Simulation (Beginner Friendly)
===========================================

This script demonstrates **why tools dominate agent latency**.

In real agent systems (Copilot, ChatGPT tools, Codex agents), the model does NOT
just "think" once. Instead, it often:

    1. Thinks (LLM inference)
    2. Calls a tool (API, database, code execution)
    3. Waits for the tool to respond
    4. Thinks again using the tool result

This script answers:

- What is a "tool" in an agent system?
- Why tools often cost MORE time than the LLM itself
- Why fast models can still feel slow
- Why async, batching, and caching matter in production

IMPORTANT:
----------
This script does NOT call real APIs.
We simulate tools using `time.sleep()` so beginners can run this anywhere.
"""

import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ============================================================
# Configuration
# ============================================================

MODEL_NAME = "distilgpt2"
PROMPT = "Explain how binary search works."
MAX_NEW_TOKENS = 40

# Simulated tool latency (seconds)
# These numbers are realistic for network / API calls
TOOL_LATENCY_SECONDS = 1.5

# Number of agent steps
AGENT_STEPS = 3


# ============================================================
# Helper: device selection
# ============================================================

def get_device() -> torch.device:
    """
    Choose CPU or GPU if available.

    Beginners:
    - CPU is fine for understanding behavior
    - GPU only reduces LLM time, NOT tool time
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nRunning on device: {device}\n")
    return device


# ============================================================
# Helper: load model
# ============================================================

def load_model_and_tokenizer(device):
    """
    Load a tiny language model and tokenizer.

    This is the "thinking engine" of the agent.
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    model.to(device)
    model.eval()

    return model, tokenizer


# ============================================================
# Simulated LLM inference
# ============================================================

def run_llm_step(model, tokenizer, device, prompt: str) -> float:
    """
    Run ONE LLM generation step and measure latency.

    This simulates the agent "thinking".
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    start_time = time.time()
    with torch.no_grad():
        model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            max_new_tokens=MAX_NEW_TOKENS,
            pad_token_id=tokenizer.pad_token_id,
        )
    latency = time.time() - start_time
    return latency


# ============================================================
# Simulated tool call
# ============================================================

def run_tool_call() -> float:
    """
    Simulate a tool call.

    In real systems, this could be:
    - Web search
    - Code execution
    - Database query
    - API call

    Here we just sleep to simulate waiting.
    """
    start_time = time.time()
    time.sleep(TOOL_LATENCY_SECONDS)
    return time.time() - start_time


# ============================================================
# Main experiment
# ============================================================

def run_experiment():
    device = get_device()
    model, tokenizer = load_model_and_tokenizer(device)

    print("==============================")
    print(" TOOL LATENCY SIMULATION")
    print("==============================\n")

    # --------------------------------------------------------
    # Baseline: LLM only
    # --------------------------------------------------------
    print("Running LLM-only inference (no tools)...")
    llm_only_latency = run_llm_step(model, tokenizer, device, PROMPT)
    print(f"LLM-only latency: {llm_only_latency:.4f} sec\n")

    # --------------------------------------------------------
    # Agent loop with tools
    # --------------------------------------------------------
    print("Running agent loop with tool calls...\n")

    total_llm_time = 0.0
    total_tool_time = 0.0

    for step in range(1, AGENT_STEPS + 1):
        print(f"--- Agent Step {step} ---")

        llm_latency = run_llm_step(model, tokenizer, device, PROMPT)
        print(f"LLM thinking time : {llm_latency:.4f} sec")
        total_llm_time += llm_latency

        tool_latency = run_tool_call()
        print(f"Tool call time    : {tool_latency:.4f} sec\n")
        total_tool_time += tool_latency

    agent_total_latency = total_llm_time + total_tool_time

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------
    print("==============================")
    print("           SUMMARY")
    print("==============================\n")

    print(f"LLM-only latency        : {llm_only_latency:.4f} sec")
    print(f"Agent LLM time (total)  : {total_llm_time:.4f} sec")
    print(f"Agent tool time (total) : {total_tool_time:.4f} sec")
    print(f"Agent total latency     : {agent_total_latency:.4f} sec\n")
    slowdown_ratio = agent_total_latency / llm_only_latency if llm_only_latency else float("inf")
    print(f"Agent total vs LLM-only : ~{slowdown_ratio:.1f}x slower")

    tool_fraction = total_tool_time / agent_total_latency
    print(f"Tool latency fraction   : {tool_fraction:.0%}")
    print(
        "\nNote:\n"
        "- Single LLM-only inference includes generating many tokens in one pass.\n"
        "- Agent steps generate fewer tokens per step but repeat the setup cost.\n"
        "- First-token latency is paid repeatedly in agent loops."
    )

    print(
        "\nBeginner takeaways:\n"
        "- Tool calls often dominate total agent latency\n"
        "- Faster models do NOT fix slow tools\n"
        "- This is why async execution and batching matter\n"
        "- Most agent optimizations focus on tool orchestration, not the LLM\n"
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    run_experiment()
