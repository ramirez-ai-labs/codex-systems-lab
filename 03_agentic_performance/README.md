# Agentic Performance Lab

This section explores the **performance costs of agent-based AI systems**.

Unlike single-turn chat models, agent systems:
- reason in multiple steps
- call external tools
- retry on failure
- recover from errors

Each of these behaviors improves *capability* — but also increases *latency*.

The goal of this lab is to make those tradeoffs **concrete and measurable**.

---

## Why Agent Performance Matters

Modern AI coding systems (e.g. Codex-style agents) are not just LLM calls.
They are **loops** that combine:

- LLM reasoning
- tool execution (search, code, APIs)
- retries and error handling
- orchestration logic

This lab shows why agent systems:
- feel slower than chat
- have unpredictable latency
- require careful systems design

---

## Experiments in This Folder

### 1️⃣ `agent_loop_benchmark.py` — What is an Agent Loop?

Demonstrates the **baseline cost** of multi-step reasoning.

Key ideas:
- An agent is a loop of *think → act → reflect*
- Each step reintroduces first-token latency
- Latency compounds linearly with step count

This is the foundation for everything else.

---

### 2️⃣ `tool_latency_simulation.py` — Why Tools Dominate Latency

Simulates an agent that calls external tools.

Key ideas:
- Tool calls often take longer than the LLM itself
- Faster models do NOT fix slow tools
- Tool orchestration is the real bottleneck in agent systems

This explains why async execution and batching matter.

---

### 3️⃣ `error_recovery_costs.py` — The Hidden Cost of Failures

Simulates tool failures and retries.

Key ideas:
- Retries replay BOTH LLM and tool latency
- A single failure can double total execution time
- Reliability is as important as raw speed

This mirrors real-world agent behavior under failure.

---

### 4️⃣ `retries_vs_quality_analysis.py` — Diminishing Returns

Explores the tradeoff between retries and success.

Key ideas:
- More retries increase success probability
- Latency grows faster than quality improves
- Past 1–2 retries, returns diminish sharply

This is why production systems aggressively cap retries.

---

## How to Use This Section

Recommended order:

```bash
python agent_loop_benchmark.py
python tool_latency_simulation.py
python error_recovery_costs.py
python retries_vs_quality_analysis.py
