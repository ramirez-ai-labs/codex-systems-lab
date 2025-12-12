# KV Cache Analysis Results

This experiment shows how much faster a transformer model becomes after the key–value (KV) cache is populated. The cache stores attention states from earlier tokens, so later tokens can be generated without recomputing the entire history a core optimization behind responsive coding assistants and chatbots.

---

## Hardware Snapshot

- Machine: **Intel(R) Core(TM) i5-8210Y @ 1.60 GHz** (MacBook Air)
- CPU cores: **4 logical**
- GPU: **Not available** (`torch.cuda.is_available()` → `False`)
- Model: **distilgpt2**
- Prompt: `def fibonacci(n):`
- Max new tokens: **32**
- Averaged next-token runs: **10**

---

## Benchmark Command

```bash
python 01_inference_profiling/kv_cache_analysis.py
```

---

## Observed Latency

| Metric                     | Time (seconds) |
|---------------------------|----------------|
| First-token latency       | 4.4217         |
| Next-token latency (avg)  | 0.0919         |
| Speedup factor            | 48.1×          |

---

## Raw Console Output

```
==============================
        KV CACHE TEST
==============================

Running on device: cpu

Measuring first-token latency...
Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.
⏱ First token latency : 4.4217 sec

Measuring next-token latency...
Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.
...
⚡ Next-token latency  : 0.0919 sec (average)

==============================
         SUMMARY
==============================

First token latency : 4.4217 s
Next-token latency  : 0.0919 s (avg)
Speedup factor      : 48.1x

Why this matters:
- Coding assistants feel instant because next-token latency is tiny.
- First-token latency dominates chat UX.
- KV cache is one of the biggest speed improvements in LLMs.

Benchmark complete!
```

---

## What Is the KV Cache?

Transformers compute three internal vectors per token: queries (Q), keys (K), and values (V). During the first generation step, the model must calculate and store all K/V tensors for the entire prompt. On subsequent steps, it reuses the cached K/V tensors and only computes the incremental pieces for the new token. This short-term memory is what enables rapid token-by-token generation.

---

## Why This Matters for Coding Models

- Coding assistants feel “instant” because chat UX is dominated by next-token latency.
- Production systems rely on KV caching to keep per-token costs predictable.
- Measuring the gap between first- and next-token times clarifies UX bottlenecks.

---

## Suggested Follow-Up Experiments

1. Run the same benchmark on a GPU-enabled machine to quantify hardware effects.
2. Plot latency vs. number of warmed tokens to visualize cache benefits.
3. Compare models of different sizes or architectures to see how KV caching scales.
