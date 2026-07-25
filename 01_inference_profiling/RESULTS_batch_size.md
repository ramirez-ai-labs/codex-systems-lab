# Batch Size Benchmark — How Batch Size Affects Throughput

This benchmark shows how generating multiple prompts at once ("batching")
can speed up inference by better utilizing the model's parallel compute.

> ⚠️ **Update:** this file previously documented an older version of
> `benchmark_batching_effects.py` that no longer exists in this repo — a
> single repeated prompt, 5 runs per batch size, latency-only reporting.
> The script now uses 4 rotating prompts, 2 runs per batch size, and
> reports **tokens/sec throughput** alongside latency, which is the more
> meaningful metric for a batching experiment. Rewritten below to match
> what the script actually does and outputs today.

---

## 🚀 How to Run

```bash
python 01_inference_profiling/benchmark_batching_effects.py
```

---

## 🖥️ Hardware Snapshot

- Machine: MacBook Air — Intel(R) Core(TM) i5-8210Y @ 1.60GHz
- CPU cores: 4 logical
- GPU: ❌ None detected
- Model: `distilgpt2`

---

## ⚡ Observed Latency & Throughput

24 new tokens per prompt, averaged over 2 runs per batch size, using 4
rotating prompts from the script's `PROMPT_LIBRARY` (recycled to fill
larger batches):

| Batch | Latency (s) | Tokens/batch | Tokens/sec |
| ----- | ------------ | ------------ | ---------- |
| 1     | 0.7414       | 24           | 32.4       |
| 2     | 1.2256       | 48           | 39.2       |
| 4     | 1.3375       | 96           | 71.8       |
| 8     | 1.6938       | 192          | 113.4      |

Raw console output:

```text
==============================
      BATCHING EFFECTS DEMO
==============================
Running on device: cpu

Results (averaged over multiple runs):

Batch |  Latency (s) |  Tokens/batch |  Tokens/sec
--------------------------------------------------
    1 |       0.7414 |            24 |        32.4
    2 |       1.2256 |            48 |        39.2
    4 |       1.3375 |            96 |        71.8
    8 |       1.6938 |           192 |       113.4

Takeaway: batching makes the *tokens/sec* column explode upward,
which is why large language models serve many users at once instead of
processing each request in isolation.
```

**Run-to-run variance note:** a second run showed the same overall shape
(tokens/sec far higher at batch 8 than batch 1) but wasn't perfectly
monotonic in the middle — batch 4 measured *lower* tokens/sec than batch 2
in that run (30.0 vs. 33.4). Absolute latency and throughput numbers on
this CPU vary noticeably run to run (see
[`RESULTS_cpu_vs_gpu.md`](RESULTS_cpu_vs_gpu.md) for a much starker example
of this same hardware's variance); the throughput-scales-with-batch-size
*trend* is the reliable takeaway, not any individual figure in this table.

---

## 📘 Beginner Interpretation

- Per-batch latency does increase with batch size (0.74s → 1.69s from
  batch 1 to batch 8), but not nearly as fast as the number of tokens
  processed per batch does (24 → 192, an 8x increase) — so **tokens/sec
  throughput climbs from ~32 to ~113**, roughly a 3.5x improvement.
- This is the actual reason production systems batch requests: not because
  a single request gets *faster*, but because the hardware processes many
  requests' tokens for barely more latency than one request alone costs.
- On CPU, gains taper off and get noisier at larger batch sizes (see the
  variance note above); a GPU would show a cleaner, larger effect since
  its parallelism headroom is much greater than 4 CPU cores.

Takeaway: batch size doesn't make one request faster — it makes the
hardware radically more efficient at serving many requests at once.
