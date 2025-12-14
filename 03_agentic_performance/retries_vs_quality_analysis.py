"""
Retries vs Quality Analysis (Beginner Friendly)
===============================================

This script demonstrates one of the MOST important tradeoffs
in agent systems:

    "Should we retry if something fails?"

Retries can:
- Improve quality (more chances to succeed)
- DESTROY latency (each retry replays LLM + tool time)

Real agent systems must balance:
- Speed
- Reliability
- Quality

This script simulates that tradeoff in a simple, visual way.
"""

import random
import time

# ============================================================
# Configuration
# ============================================================

# Simulated latency (seconds)
LLM_LATENCY_RANGE = (0.8, 1.3)
TOOL_LATENCY_RANGE = (1.2, 1.6)

# Probability that a single attempt succeeds
BASE_SUCCESS_PROB = 0.55

# How much success probability improves per retry
RETRY_QUALITY_BOOST = 0.15

# Maximum retries to test
MAX_RETRIES = 3

# Number of agent steps per run
AGENT_STEPS = 3

# Random seed for reproducibility
random.seed(42)


# ============================================================
# Helpers
# ============================================================

def simulate_llm_thinking() -> float:
    """Simulate LLM inference latency."""
    delay = random.uniform(*LLM_LATENCY_RANGE)
    time.sleep(delay)
    return delay


def simulate_tool_call() -> float:
    """Simulate external tool latency."""
    delay = random.uniform(*TOOL_LATENCY_RANGE)
    time.sleep(delay)
    return delay


def attempt_step(success_probability: float):
    """
    Simulate ONE attempt at completing a step.

    Returns:
    - success (bool)
    - llm_time
    - tool_time
    """
    llm_time = simulate_llm_thinking()
    tool_time = simulate_tool_call()
    success = random.random() < success_probability
    return success, llm_time, tool_time


# ============================================================
# Core Simulation
# ============================================================

def run_agent_with_retries(max_retries: int):
    """
    Run an agent with a fixed retry budget.

    Returns:
    - total_latency
    - successful_steps
    """
    total_llm_time = 0.0
    total_tool_time = 0.0
    successful_steps = 0

    success_probability = BASE_SUCCESS_PROB

    for step in range(1, AGENT_STEPS + 1):
        print(f"\n--- Agent Step {step} ---")

        for attempt in range(max_retries + 1):
            print(f"Attempt {attempt + 1}:")

            success, llm_time, tool_time = attempt_step(success_probability)

            total_llm_time += llm_time
            total_tool_time += tool_time

            print(f"  LLM time  : {llm_time:.2f} sec")
            print(f"  Tool time : {tool_time:.2f} sec")

            if success:
                print("  ✅ Step succeeded")
                successful_steps += 1
                break
            else:
                print("  ❌ Step failed")

        # Each retry slightly improves odds next time
        success_probability = min(
            success_probability + RETRY_QUALITY_BOOST, 0.95
        )

    total_latency = total_llm_time + total_tool_time
    return total_latency, successful_steps


# ============================================================
# Main Analysis
# ============================================================

def main():
    print("\n==============================")
    print(" RETRIES vs QUALITY ANALYSIS")
    print("==============================\n")

    results = []

    for retries in range(MAX_RETRIES + 1):
        print(f"\n==============================")
        print(f" Running with max retries = {retries}")
        print(f"==============================")

        latency, success_count = run_agent_with_retries(retries)

        results.append({
            "retries": retries,
            "latency": latency,
            "successes": success_count,
        })

    # ========================================================
    # Summary
    # ========================================================
    print("\n==============================")
    print("           SUMMARY")
    print("==============================\n")

    print(f"{'Retries':>7} | {'Latency (s)':>12} | {'Successful Steps':>17}")
    print("-" * 45)

    for r in results:
        print(
            f"{r['retries']:>7} | "
            f"{r['latency']:>12.2f} | "
            f"{r['successes']:>17}"
        )

    print(
        "\nBeginner takeaways:\n"
        "- Retries improve success, but with diminishing returns\n"
        "- Each retry replays BOTH LLM and tool latency\n"
        "- Past 1–2 retries, latency explodes faster than quality improves\n"
        "- This is why real agent systems aggressively cap retries\n"
        "- Fast failures + graceful degradation beat endless retries\n"
    )


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    main()
