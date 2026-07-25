# 📊 Agentic Performance — Results & Takeaways

This document summarizes what was actually measured (or simulated, where
noted) when running the four scripts in this folder — real console output,
not hand-typed illustrative numbers. Re-running these scripts on different
hardware, or even on this same machine, will produce different exact
numbers: two of the four scripts use unseeded wall-clock timing and/or
`random.uniform()` without a fixed seed, so only the *shape* of the result
is expected to reproduce, not the precise figures. Each section below says
explicitly which case it is.

---

## Hardware Snapshot

- Machine: MacBook Air — Intel(R) Core(TM) i5-8210Y @ 1.60GHz
- CPU cores: 4 logical
- GPU: ❌ None detected
- Model: `distilgpt2` (for the two scripts that call it)

---

## 1️⃣ Agent Loop vs Single Inference — `agent_loop_benchmark.py` ✅ Real inference, wall-clock varies

**Real DistilGPT-2 inference calls**, no simulated latency anywhere in this
script. Run with `python 03_agentic_performance/agent_loop_benchmark.py`.

| Metric | Value |
| --- | --- |
| Single inference latency | 2.4961 sec |
| Agent loop latency (3 steps) | 10.1055 sec |
| Slowdown | ~4.0x |

Raw console output:

```text
Running on device: cpu

Running single inference (baseline)...
Prompt length (tokens): 12
Single inference latency: 2.4961 sec

Running agent loop...

--- Agent Step 1 ---
Prompt length (tokens): 32
Step latency: 2.4712 sec

--- Agent Step 2 ---
Prompt length (tokens): 116
Step latency: 3.7581 sec

--- Agent Step 3 ---
Prompt length (tokens): 200
Step latency: 3.8763 sec

==============================
           SUMMARY
==============================

Single inference latency : 2.4961 sec
Agent loop latency       : 10.1055 sec
Agent steps              : 3

Agent loop is ~4.0x slower than single inference
```

**Why this happens:** each step's prompt grows (the previous step's output
is appended as context), so later steps generate against a longer prompt
and take longer — on top of paying first-token latency freshly each step.

**Note on variance:** this script measures real wall-clock time with no
fixed seed for timing (there's nothing to seed — it's real inference), so
re-running it produces different exact numbers each time depending on OS
scheduling and thermal state. A separate run during this same review
measured 2.35s / 7.18s (~3.1x); both runs agree on the qualitative finding
(the loop is meaningfully slower) but not on the precise multiplier.

**Key insight:** multi-step reasoning is inherently slower than single-turn
generation, and the gap compounds as context grows.

---

## 2️⃣ Tool Latency Dominates Total Time — `tool_latency_simulation.py` ⚠️ Real LLM calls, simulated tool calls

**Real DistilGPT-2 inference for the "thinking" step; the "tool" step is a
fixed `time.sleep(1.5)`, not a real API call** — the script's own docstring
says so explicitly. Run with
`python 03_agentic_performance/tool_latency_simulation.py`.

| Metric | Value |
| --- | --- |
| LLM-only latency (baseline) | 1.4389 sec |
| Agent LLM time (3 steps, total) | 3.8869 sec |
| Agent tool time (3 steps, total) | 4.5041 sec |
| Agent total latency | 8.3910 sec |
| Slowdown vs. LLM-only | ~5.8x |
| Tool latency fraction | 54% |

Raw console output:

```text
Running on device: cpu

==============================
 TOOL LATENCY SIMULATION
==============================

Running LLM-only inference (no tools)...
LLM-only latency: 1.4389 sec

Running agent loop with tool calls...

--- Agent Step 1 ---
LLM thinking time : 1.2695 sec
Tool call time    : 1.5003 sec

--- Agent Step 2 ---
LLM thinking time : 1.3700 sec
Tool call time    : 1.5016 sec

--- Agent Step 3 ---
LLM thinking time : 1.2474 sec
Tool call time    : 1.5022 sec

==============================
           SUMMARY
==============================

LLM-only latency        : 1.4389 sec
Agent LLM time (total)  : 3.8869 sec
Agent tool time (total) : 4.5041 sec
Agent total latency     : 8.3910 sec

Agent total vs LLM-only : ~5.8x slower
Tool latency fraction   : 54%
```

**Key insight:** even with a modest fixed tool latency (1.5s/call, chosen
to be "realistic for network/API calls" per the script's own comment, not
measured from a real API), tool time is roughly on par with LLM time here
and would dominate further if tool latency were higher — which is common
for real search/code-execution tools.

---

## 3️⃣ Error Recovery Is Expensive — `error_recovery_costs.py` 🎲 Fully simulated, unseeded

**No model or tool is called, real or otherwise.** LLM and tool latency are
both `random.uniform()` plus `time.sleep()`, and tool failure is a random
draw (`TOOL_FAILURE_PROBABILITY = 0.4`) with **no fixed random seed** — this
script's exact numbers are not reproducible between runs, only the shape
of the tradeoff is. Run with
`python 03_agentic_performance/error_recovery_costs.py`.

| Metric | Value (this run) |
| --- | --- |
| Agent steps | 3 |
| Total retries | 2 |
| Total LLM time | 6.19 sec |
| Total tool time | 6.61 sec |
| Total agent latency | 12.83 sec |

Raw console output:

```text
Running on device: cpu

Note: Tool failures are random to simulate real-world systems.

==============================
  ERROR RECOVERY COSTS DEMO
==============================

--- Agent Step 1 ---
Attempt 1:
  LLM thinking time : 1.100 sec
  Tool call time    : 1.204 sec
  ✅ Tool succeeded

--- Agent Step 2 ---
Attempt 1:
  LLM thinking time : 1.139 sec
  Tool call time    : 1.353 sec
  ❌ Tool failed — retrying...

Attempt 2:
  LLM thinking time : 1.307 sec
  Tool call time    : 1.276 sec
  ✅ Tool succeeded

--- Agent Step 3 ---
Attempt 1:
  LLM thinking time : 1.248 sec
  Tool call time    : 1.561 sec
  ❌ Tool failed — retrying...

Attempt 2:
  LLM thinking time : 1.398 sec
  Tool call time    : 1.220 sec
  ✅ Tool succeeded

==============================
           SUMMARY
==============================

Agent steps             : 3
Total retries           : 2
Total LLM time          : 6.19 sec
Total tool time         : 6.61 sec
Total agent latency     : 12.83 sec

Estimated retry penalty : ~12.81 sec
```

**Key insight:** 2 of 3 steps needed a retry in this run (a 40%-per-attempt
failure rate makes that likely, not unusual), and each retry replayed both
the simulated LLM and tool cost in full. Since nothing is seeded, a
different run could show 0 retries or 3 — the takeaway is about the *shape*
of the cost (retries replay full cost, not incremental cost), not this
run's specific retry count.

---

## 4️⃣ Retries vs Quality Tradeoff — `retries_vs_quality_analysis.py` 🎲 Fully simulated, seeded (`random.seed(42)`)

**No model or tool is called.** Like the previous script, latency is
`random.uniform()` + `time.sleep()` — but this one fixes `random.seed(42)`,
so unlike `error_recovery_costs.py`, re-running it reproduces the exact
numbers below. Run with
`python 03_agentic_performance/retries_vs_quality_analysis.py`.

| Max Retries | Total Latency (s) | Successful Steps |
|------------|------------------|------------------|
| 0          | 7.22             | 3                |
| 1          | 6.56             | 3                |
| 2          | 9.89             | 3                |
| 3          | 10.57            | 3                |

Raw console output (summary table):

```text
Retries |  Latency (s) |  Successful Steps
---------------------------------------------
      0 |         7.22 |                 3
      1 |         6.56 |                 3
      2 |         9.89 |                 3
      3 |        10.57 |                 3
```

**Key insight:** all 3 steps succeeded even at 0 retries in this seeded
run — success count staying flat while latency still climbs from retries=1
to retries=3 shows the model's built-in `RETRY_QUALITY_BOOST` wasn't
actually needed here to hit 3/3, so the added latency at higher retry
budgets bought nothing on this particular seed. That's itself a useful,
honest data point: the "quality" side of this tradeoff depends heavily on
the random seed and failure probabilities chosen, while the "latency
grows with retries" side is much more robust.

---

## Final Takeaways (Beginner-Friendly)

- Agent systems are slower because they repeat work — confirmed by real
  inference measurement in section 1, not just simulation.
- Tool calls can dominate latency — section 2 measured tools at 54% of
  total time with a modest fixed 1.5s tool latency; real tools are often
  slower still.
- Retries multiply cost unpredictably — sections 3 and 4 show this in
  simulation, with 3 being reproducible (seeded) and 4 not (unseeded).
- Most performance wins come from orchestration, not models.
- "Smart" agents require strong systems engineering.

---

## Why This Matters for AI Coding Systems

This section explains why real-world coding agents:
- feel slower than chat
- require async, caching, and batching
- invest heavily in infra and reliability
- trade raw intelligence for predictable UX

Understanding these costs is essential for building
scalable AI-assisted developer tools.

---

## Limitations

- **Sections 1 and 2 make real model calls; sections 3 and 4 make none.**
  See [`README.md`](README.md) for the same real-vs-simulated breakdown at
  the script level.
- **Section 2's tool latency (1.5s) is a fixed constant chosen to be
  "realistic," not measured from any real API.**
- **Sections 1 and 3 have no fixed seed**, so their exact numbers will
  differ on re-run; only section 4 (`random.seed(42)`) is exactly
  reproducible.
- **All numbers are from a single run each** (except where noted), on one
  specific low-power CPU — not averaged over multiple runs the way
  `01_inference_profiling`'s benchmarks are.
