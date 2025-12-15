
---

# 📊 `03_agentic_performance/RESULTS.md`

# Agentic Performance — Results & Takeaways

This document summarizes what we observed when running the agent performance
experiments in this folder.

All experiments were run on CPU with simulated latency to highlight
**relative costs**, not absolute speed.

---

## 1️⃣ Agent Loop vs Single Inference

**Observation:**
- A single inference is faster than an agent loop
- Each agent step adds additional latency

Example result:
- Single inference: ~4.8 sec
- 3-step agent loop: ~6.5 sec

**Why this happens:**
- Each step reintroduces first-token latency
- Latency grows roughly linearly with step count

**Key insight:**
> Multi-step reasoning is inherently slower than single-turn generation.

---

## 2️⃣ Tool Latency Dominates Total Time

**Observation:**
- Tool calls often take more time than the LLM
- In many runs, tools accounted for ~50–60% of total latency

Example breakdown:
- LLM thinking time: ~3.4 sec
- Tool execution time: ~4.5 sec
- Total agent latency: ~7.9 sec

**Key insight:**
> Faster models do not fix slow tools.

This is why real agent systems focus heavily on:
- async execution
- batching tool calls
- minimizing tool usage

---

## 3️⃣ Error Recovery Is Expensive

**Observation:**
- A single tool failure significantly increases total latency
- Retries replay BOTH LLM and tool costs

Example:
- One failed attempt nearly doubled total execution time

**Key insight:**
> Reliability matters as much as speed.

Agent systems pay a heavy price for retries.

---

## 4️⃣ Retries vs Quality Tradeoff

**Observation:**
- Allowing retries improves success probability
- Latency increases rapidly with retry count
- Past 1–2 retries, gains diminish

Example summary:

| Max Retries | Total Latency (s) | Successful Steps |
|------------|------------------|------------------|
| 0          | ~7.2             | 3                |
| 1          | ~6.6             | 3                |
| 2          | ~9.9             | 3                |
| 3          | ~10.6            | 3                |

**Key insight:**
> More retries ≠ better system.

Production systems cap retries and favor fast failure.

---

## Final Takeaways (Beginner-Friendly)

- Agent systems are slower because they repeat work
- Tool calls dominate latency
- Retries multiply cost unpredictably
- Most performance wins come from orchestration, not models
- “Smart” agents require strong systems engineering

---

## Why This Matters for AI Coding Systems

This section explains why real-world coding agents:
- feel slower than chat
- require async, caching, and batching
- invest heavily in infra and reliability
- trade raw intelligence for predictable UX

Understanding these costs is essential for building
scalable AI-assisted developer tools.
