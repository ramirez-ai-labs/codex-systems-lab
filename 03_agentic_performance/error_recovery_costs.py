"""
Error Recovery Costs (Beginner Friendly)
=======================================

This script demonstrates how **tool failures and retries**
dramatically increase agent latency.

Key idea:
---------
Retries feel cheap, but they are NOT.

Every retry:
- Re-runs the LLM "thinking" step
- Re-runs the tool call
- Adds unpredictable latency

This is why agent systems sometimes feel fast…
and sometimes feel painfully slow.

What this script shows:
-----------------------
1. A tool that randomly fails
2. An agent that retries on failure
3. How retries multiply total latency
"""

import random
import time
import torch


# ============================================================
# Configuration
# ============================================================

NUM_AGENT_STEPS = 3
MAX_RETRIES_PER_STEP = 2

# Simulated latencies (seconds)
LLM_LATENCY_RANGE = (0.8, 1.4)
TOOL_LATENCY_RANGE = (1.2, 1.6)

# Probability that a tool call fails
TOOL_FAILURE_PROBABILITY = 0.4


# ============================================================
# Helpers
# ============================================================

def get_device() -> torch.device:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nRunning on device: {device}\n")
    print("Note: Tool failures are random to simulate real-world systems.\n")
    return device


def simulate_llm_thinking() -> float:
    """
    Simulate LLM reasoning latency.
    """
    latency = random.uniform(*LLM_LATENCY_RANGE)
    time.sleep(latency)
    return latency


def simulate_tool_call() -> (bool, float):
    """
    Simulate a tool call that may fail.

    Returns:
        success (bool)
        latency (float)
    """
    latency = random.uniform(*TOOL_LATENCY_RANGE)
    time.sleep(latency)

    success = random.random() > TOOL_FAILURE_PROBABILITY
    return success, latency


# ============================================================
# Agent loop with retries
# ============================================================

def run_agent_with_retries():
    print("==============================")
    print("  ERROR RECOVERY COSTS DEMO")
    print("==============================\n")

    total_llm_time = 0.0
    total_tool_time = 0.0
    total_retries = 0

    start_time = time.time()

    for step in range(1, NUM_AGENT_STEPS + 1):
        print(f"--- Agent Step {step} ---")

        attempt = 0
        success = False

        while not success and attempt <= MAX_RETRIES_PER_STEP:
            attempt += 1

            print(f"Attempt {attempt}:")

            # LLM reasoning
            llm_time = simulate_llm_thinking()
            total_llm_time += llm_time
            print(f"  LLM thinking time : {llm_time:.3f} sec")

            # Tool execution
            success, tool_time = simulate_tool_call()
            total_tool_time += tool_time
            print(f"  Tool call time    : {tool_time:.3f} sec")

            if success:
                print("  ✅ Tool succeeded\n")
            else:
                total_retries += 1
                print("  ❌ Tool failed — retrying...\n")

                if attempt > MAX_RETRIES_PER_STEP:
                    print("  🚨 Max retries exceeded — giving up on this step\n")

    total_latency = time.time() - start_time

    return {
        "total_latency": total_latency,
        "llm_time": total_llm_time,
        "tool_time": total_tool_time,
        "retries": total_retries,
    }


# ============================================================
# Summary
# ============================================================

def print_summary(stats):
    print("==============================")
    print("           SUMMARY")
    print("==============================\n")

    print(f"Agent steps             : {NUM_AGENT_STEPS}")
    print(f"Total retries           : {stats['retries']}")
    print(f"Total LLM time          : {stats['llm_time']:.2f} sec")
    print(f"Total tool time         : {stats['tool_time']:.2f} sec")
    print(f"Total agent latency     : {stats['total_latency']:.2f} sec\n")

    retry_penalty = stats["llm_time"] + stats["tool_time"]
    print(f"Estimated retry penalty : ~{retry_penalty:.2f} sec\n")

    print("Beginner takeaways:")
    print("- Retries replay BOTH LLM and tool latency")
    print("- A single failure can double total time")
    print("- Latency becomes unpredictable with failures")
    print("- Real agent systems spend huge effort avoiding retries")
    print("- This is why reliability matters as much as speed")


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    get_device()
    stats = run_agent_with_retries()
    print_summary(stats)
