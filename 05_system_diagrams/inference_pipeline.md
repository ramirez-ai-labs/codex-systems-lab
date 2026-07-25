# Inference Pipeline — Sequence Diagram

This diagram walks through a single inference request against the
`distilgpt2` model as actually run in
[`01_inference_profiling`](../01_inference_profiling/), stage by stage, with
the real measured latencies from
[`RESULTS_kv_cache_analysis.md`](../01_inference_profiling/RESULTS_kv_cache_analysis.md)
attached to each stage.

```mermaid
sequenceDiagram
    participant Client
    participant Tokenizer
    participant Model as distilgpt2 (CPU)
    participant KVCache as KV Cache

    Client->>Tokenizer: prompt: "def fibonacci(n):"
    Tokenizer->>Model: input_ids

    rect rgb(255, 235, 235)
    Note over Model,KVCache: First forward pass — no cache yet.<br/>Full prompt attention computed from scratch.
    Model->>KVCache: store K/V for every prompt token
    Model-->>Client: token 1
    Note over Client: measured: 4.4217s (first-token latency)
    end

    rect rgb(235, 255, 235)
    loop each subsequent token
        Model->>KVCache: reuse cached K/V, compute only the new token
        Model-->>Client: token N
    end
    Note over Client: measured: 0.0919s avg (next-token latency)
    end

    Note over Client,KVCache: Speedup factor: 48.1x (first token vs. cached next token)
```

## What's Real vs. What's a Simplification

- **The latency numbers are real**, taken directly from
  `01_inference_profiling/benchmark_kv_cache_analysis.py`'s output on a
  MacBook Air (i5-8210Y, CPU-only, no GPU) — see the linked RESULTS file for
  the raw console output and hardware snapshot.
- **The diagram itself is a simplification.** It shows the conceptual
  request path (tokenize → first pass → cached decode loop) that the
  benchmark script measures; it does not show batching, request queuing, or
  any serving infrastructure — those are separate concerns covered in
  [`batching_architecture.mmd`](batching_architecture.mmd) and
  [`container_orchestration.md`](container_orchestration.md).
- No production serving stack (vLLM, TGI, etc.) is depicted or implied —
  this is the bare HuggingFace `generate()` loop that the benchmark scripts
  actually call.

## Why This Matters

The 48x gap between first-token and next-token latency is why chat and
coding-assistant UX is dominated by **time-to-first-token**, not steady-state
throughput — once the KV cache is warm, additional tokens are comparatively
cheap. This is the same insight
[`03_agentic_performance`](../03_agentic_performance/README.md) builds on:
every new agent step (think → act → reflect) pays a fresh first-token cost,
which is why agent loops feel slower than a single chat turn even when the
model is identical.
