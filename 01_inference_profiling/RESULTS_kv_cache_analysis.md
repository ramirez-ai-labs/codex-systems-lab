# KV Cache Analysis Results

This experiment shows how much faster a transformer model becomes after the key–value (KV) cache is populated. The cache stores attention states from earlier tokens, so later tokens can be generated without recomputing the entire history a core optimization behind responsive coding assistants and chatbots.

> 🚨 **Update: the headline claim below (48.1x speedup) does not currently
> reproduce.** Four fresh runs measured a speedup factor of 1.2x, 0.9x,
> 0.6x, and 1.2x — essentially no consistent difference between first-token
> and next-token latency, and one run even measured next-token latency as
> *slower*. This isn't just environment drift: there's a real methodology
> gap in this script (see [Limitations](#-limitations)) that likely means
> it was never correctly isolating the effect it claims to measure. The
> original 4.42s / 0.09s / 48.1x numbers are preserved in
> [Historical Record](#-historical-record-original-claim) below (search for
> that heading if your viewer doesn't jump there), but should not be
> treated as a reproducible finding.

---

## Hardware Snapshot

- Machine: **Intel(R) Core(TM) i5-8210Y @ 1.60 GHz** (MacBook Air)
- CPU cores: **4 logical**
- GPU: **Not available** (`torch.cuda.is_available()` → `False`)
- Model: **distilgpt2**
- Prompt: `def fibonacci(n):`
- Max new tokens: **32**
- Averaged next-token runs: **10**

---

## Benchmark Command

```bash
python 01_inference_profiling/benchmark_kv_cache_analysis.py
```

---

## Observed Latency (four fresh runs)

| Run | First-token (s) | Next-token avg (s) | Speedup factor |
| --- | --- | --- | --- |
| 1 | 0.0786 | 0.0660 | 1.2x |
| 2 | 0.0750 | 0.0813 | 0.9x |
| 3 | 0.1779 | 0.2925 | 0.6x |
| 4 | 0.0992 | 0.0843 | 1.2x |

No run came remotely close to the originally documented 48.1x. First-token
and next-token latency are consistently in the same ballpark — sometimes
next-token is faster, sometimes slower.

Raw console output (run 4):

```text
==============================
        KV CACHE TEST
==============================

Running on device: cpu

Measuring first-token latency...
⏱ First token latency : 0.0992 sec

Measuring next-token latency...
⚡ Next-token latency  : 0.0843 sec (average)

==============================
         SUMMARY
==============================

First token latency : 0.0992 s
Next-token latency  : 0.0843 s (avg)
Speedup factor      : 1.2x

Why this matters:
- Coding assistants feel instant because next-token latency is tiny.
- First-token latency dominates chat UX.
- KV cache is one of the biggest speed improvements in LLMs.

Benchmark complete!
```

---

## What Is the KV Cache? (the concept — still real)

Transformers compute three internal vectors per token: queries (Q), keys (K), and values (V). During the first generation step, the model must calculate and store all K/V tensors for the entire prompt. On subsequent steps, it reuses the cached K/V tensors and only computes the incremental pieces for the new token. This short-term memory is what enables rapid token-by-token generation, and it's a real, well-documented mechanism in production LLM serving — the problem is specifically that **this script doesn't demonstrate it**, not that the concept is wrong.

---

## 🧩 Limitations

- **Methodology gap: no cache is actually persisted across the measured calls.** `measure_next_token_latency()` calls `model.generate(max_new_tokens=1)` in a loop, once per "next token." Each of those is an independent top-level `generate()` invocation — Hugging Face's KV cache lives inside a single `generate()` call and is discarded when that call returns. So every iteration of the loop recomputes attention over the entire (growing) sequence from scratch, the same as the "first token" measurement does. The script was very likely never isolating cached-vs-uncached attention cost the way its docstring describes; a correct version would need one continuous multi-token `generate()` call (or manual `past_key_values` reuse across steps) to actually exercise the cache.
- **This means the historical 48.1x number likely reflected something else** — plausibly one-time process/model warm-up cost on whatever run produced it (JIT/kernel selection, thread-pool spin-up, memory allocation), rather than a genuine per-step cache benefit, since the two functions being compared perform structurally the same operation.
- **High latency variance on this hardware** (see [`RESULTS_cpu_vs_gpu.md`](RESULTS_cpu_vs_gpu.md)) makes small absolute differences here hard to interpret even before the methodology issue.
- **Small samples**: `NUM_NEXT_TOKENS = 15` per run, 4 runs total — enough to see the effect isn't there, not a rigorous statistical characterization.

---

## Why This Matters for Coding Models

- The KV cache concept itself is real and matters — production coding assistants do get most of their responsiveness from not recomputing attention over the whole prompt on every token.
- This specific script, as currently written, doesn't demonstrate that — which is itself a useful lesson: a benchmark's *methodology* has to actually exercise the mechanism it claims to measure, and "the numbers came out dramatic" isn't enough evidence that it did.
- Measuring the gap between first- and next-token latency *correctly* would require a single continuous generation call, not a loop of independent one-token calls.

---

## 📜 Historical Record (original claim)

Preserved for audit purposes — this is what this file originally reported,
before the methodology issue above was found. Do not treat these numbers as
current or reproducible.

| Metric | Time (seconds) |
|---------------------------|----------------|
| First-token latency       | 4.4217         |
| Next-token latency (avg)  | 0.0919         |
| Speedup factor            | 48.1×          |

Original raw console output:

```text
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

Benchmark complete!
```

---

## Suggested Follow-Up Experiments

1. Rewrite this script to measure token-by-token latency **within one continuous `model.generate()` call** (e.g., via a `StoppingCriteria` or manual greedy-decoding loop with `past_key_values` passed explicitly between steps), so it actually exercises cache reuse.
2. Once fixed, re-run on this same hardware and see whether a genuine, reproducible speedup appears.
3. Run the corrected version on a GPU-enabled machine to quantify hardware effects.
4. Compare models of different sizes or architectures to see how KV caching scales, once the measurement itself is trustworthy.
