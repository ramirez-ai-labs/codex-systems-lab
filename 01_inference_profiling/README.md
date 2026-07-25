# Inference Profiling Lab ⚠️ Measured, real hardware — two findings didn't hold up

This folder explores **why LLM inference feels fast or slow**, using real
`distilgpt2` inference runs on a MacBook Air (i5-8210Y, CPU-only, no GPU
detected). Every number here comes from actually running the script, not
estimation — but re-running these benchmarks multiple times surfaced real
problems in two of them, which are documented below rather than quietly
patched over.

---

## Why Inference Profiling Matters

Before you can reason about batching, caching, or quantization tradeoffs in
a production system, you need honest answers to simpler questions:

- How slow is a single request, really, on ordinary hardware?
- Where does that time actually go?
- Which "obvious" optimizations hold up when you actually measure them?

This lab answers those by running the model and recording what happens,
including when a benchmark's own methodology turns out to be broken.

---

## Experiments in This Folder

### 1️⃣ `benchmark_cpu_vs_gpu.py` — Why GPUs Matter ✅ Measured, high variance

Makes real DistilGPT-2 `generate()` calls and times them directly.

Key ideas:
- CPU-only generation of 32 tokens took **~1–7 seconds across six fresh
  runs** on this machine — a 7x spread for the identical script and code
- The originally documented 2.17s sits near the median of that range, so
  it wasn't wrong — just presented with more precision than this hardware
  can actually deliver run to run
- A GPU would typically cut this latency by 5–20x and be far more
  consistent run to run than this CPU is

See [`RESULTS_cpu_vs_gpu.md`](RESULTS_cpu_vs_gpu.md) for the full spread,
mean/median, and raw console output.

### 2️⃣ `benchmark_batching_effects.py` — Why Batching Improves Throughput ✅ Measured

Makes real DistilGPT-2 `generate()` calls across batch sizes 1/2/4/8.

Key ideas:
- Tokens/sec throughput climbs from **~32 (batch 1) to ~113 (batch 8)**,
  even though per-batch latency also rises
- A single request doesn't get *faster* from batching — the hardware gets
  radically more efficient at serving many requests at once
- This is why production systems batch concurrent requests instead of
  processing them one at a time

See [`RESULTS_batch_size.md`](RESULTS_batch_size.md) for the full table and
a note on the run-to-run variance observed in a second capture.

### 3️⃣ `benchmark_kv_cache_analysis.py` — Why First-Token Latency Is Slow 🚨 Documented finding retracted

Makes real DistilGPT-2 `generate()` calls — but with a methodology gap that
means it isn't measuring what it claims to.

Key ideas:
- The original claim was a **48.1x speedup** from first-token to cached
  next-token latency — a real, well-documented LLM-serving phenomenon
- **Four fresh runs measured 0.6x–1.2x instead** — essentially no
  consistent cache benefit, and one run even showed next-token latency as
  *slower* than first-token
- Root cause: each "next token" measurement calls an independent
  `generate(max_new_tokens=1)`, and Hugging Face's KV cache doesn't persist
  across separate `generate()` calls — so this script never actually
  exercised cache reuse
- The underlying concept is real; this specific benchmark just doesn't
  demonstrate it (yet) — see the results file for what a correct version
  would need

See [`RESULTS_kv_cache_analysis.md`](RESULTS_kv_cache_analysis.md) for the
full writeup, the original numbers (preserved in a Historical Record
section for audit purposes), and the methodology fix needed.

### 4️⃣ `benchmark_quantization_comparison.py` — Why Lower Precision Speeds Up CPUs ✅ Measured

Makes real FP32 vs. INT8 dynamically-quantized `generate()` calls.

Key ideas:
- INT8 was consistently **~1.5x faster** than FP32 across repeated runs —
  this one held up on re-measurement
- The script's reported model-size numbers are wrong and shouldn't be
  trusted: PyTorch's quantized `nn.Linear` stores weights in packed buffers
  that the size estimator never counts, so it silently reports the same
  FP32-sized number for both models
- Trust the latency/throughput columns; don't trust the size column until
  the estimator is fixed to walk packed quantized weights

See [`RESULTS_quantization.md`](RESULTS_quantization.md) for the full
breakdown and known limitation.

---

## How to Use This Section

Recommended order:

```bash
python benchmark_cpu_vs_gpu.py
python benchmark_batching_effects.py
python benchmark_kv_cache_analysis.py
python benchmark_quantization_comparison.py
```

Run from inside `01_inference_profiling/` (or prefix each command with
`01_inference_profiling/` if running from the repo root). Each script can
also be run independently — none depend on another's output.
