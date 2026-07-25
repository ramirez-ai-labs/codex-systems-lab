# CPU vs GPU — Inference Latency Benchmark

This experiment measures how long a small language model (`distilgpt2`)
takes to generate text on different compute devices. Beginners can use this
to understand why GPUs matter for LLM workloads.

> ⚠️ **Update:** this benchmark turned out to have much higher run-to-run
> variance on this hardware than a single number suggests. See
> [Observed Latency](#-observed-latency) below — six fresh runs ranged from
> 1.05s to 7.39s for the *same* script, same machine, same code. Treat any
> single number here (including the original 2.17s) as one sample from a
> wide distribution, not a precise measurement.

---

## 🚀 How to Run

```bash
python 01_inference_profiling/benchmark_cpu_vs_gpu.py
```

---

## 🖥️ Hardware Snapshot

- Machine: MacBook Air
- CPU: Intel(R) Core(TM) i5-8210Y @ 1.60GHz (4 logical cores)
- GPU available? ❌ No (PyTorch did not detect CUDA)

---

## ⚡ Observed Latency

Average generation time for 32 new tokens (5 runs, per the script's own
internal averaging) — but run the *whole script* six times back-to-back and
you get:

| Run | Avg Latency (sec) |
| --- | --- |
| 1 | 1.0491 |
| 2 | 1.0596 |
| 3 | 2.3124 |
| 4 | 2.8782 |
| 5 | 3.7327 |
| 6 | 7.3948 |

- **Mean:** ~3.07s
- **Median:** ~2.60s
- **Range:** 1.05s – 7.39s (a 7x spread)

The original documented value (2.1687s, captured under an older `torch<2.2`
pin — see [Limitations](#-limitations)) sits comfortably inside this range,
close to the median. It was never "wrong," but presenting it alone implied
a precision this benchmark doesn't have on this hardware.

Raw console output (run 6, the slowest):

```text
Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.
Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.
Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.
Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.
Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.

==============================
 CPU vs GPU Inference Benchmark
==============================

No GPU detected. Running on CPU only.

Running inference on: CPU
Average time on CPU: 7.3948 seconds


==============================
        FINAL RESULTS
==============================

CPU: 7.3948 seconds

Benchmark complete!
```

---

## 📘 Beginner Interpretation

- This low-power CPU needs somewhere around 1–7 seconds to generate 32
  tokens with DistilGPT2 — and which end of that range you land on for any
  given run is unpredictable on this hardware (likely thermal throttling
  and/or OS scheduling noise on a fanless Y-series chip, not something this
  script controls for).
- Larger coding models would be much slower on CPU unless quantized or
  batched, and would likely show the same kind of run-to-run variance.
- A GPU typically reduces the latency by 5–20×, so rerun this on a CUDA
  host to compare — and expect a GPU run to be far more consistent
  run-to-run than this CPU figure is.

Takeaway: CPU-only inference is fine for demos, but high-throughput
assistants rely on GPUs — and on CPU specifically, don't trust a single run
of this benchmark as a stable number; look at the spread.

---

## 🧩 Limitations

- **High variance, small sample.** Six runs is enough to see the spread is
  real, not enough to characterize its distribution precisely. The
  "5 runs averaged" the script does internally isn't enough to smooth out
  variance of this magnitude.
- **The original 2.1687s was captured while `requirements.txt` pinned
  `torch<2.2`.** That pin was later relaxed to `torch>=2.0,<2.4` (torch
  `<2.2` has no Python 3.12 wheels); `.venv` now runs torch 2.2.2, a
  version that wasn't installable under the old pin. Given the variance
  documented above, there isn't clear evidence this changed the typical
  latency — but it's a real environment difference worth naming.
- **No control for background load or thermal state.** This is a fanless
  low-power CPU (i5-8210Y); sustained or back-to-back benchmark runs likely
  hit thermal throttling, which would show up exactly as the kind of
  spread observed here.
