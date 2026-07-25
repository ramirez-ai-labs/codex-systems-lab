# Quantization Comparison — FP32 vs. INT8

This benchmark compares full-precision (FP32) inference against dynamically
quantized INT8 inference on CPU.

---

## 🚀 How to Run

```bash
python 01_inference_profiling/benchmark_quantization_comparison.py
```

---

## 🖥️ Hardware Snapshot

- Machine: MacBook Air — Intel(R) Core(TM) i5-8210Y @ 1.60GHz
- CPU cores: 4 logical
- GPU: ❌ None detected
- Model: `distilgpt2`

---

## ⚡ Observed Latency & Throughput

Average over 2 runs, 24 new tokens generated per run:

| Model | Latency (s) | Tokens/sec | Reported Size (MB) |
| ----- | ----------- | ---------- | ------------------- |
| FP32  | 0.8389      | 29.30      | 312.5                |
| INT8  | 0.5351      | 44.85      | 312.5                |

Raw console output:

```
Quantization demo (CPU-only for fairness and accessibility).

Benchmarking FP32 model...

Benchmarking INT8 model...

==============================
   QUANTIZATION COMPARISON
==============================

Model      Latency (s)    Tokens/sec    New tokens   Size (MB)
--------------------------------------------------------------
FP32            0.8389         29.30            24       312.5
INT8            0.5351         44.85            24       312.5

Benchmark complete!
```

---

## 📘 Interpretation

- **Latency/throughput are real and consistent with expectations**: INT8 was
  ~1.5x faster (44.85 vs. 29.30 tokens/sec), matching the general rule that
  dynamic quantization speeds up CPU inference.
- **The size column is misleading and should not be trusted as-is.** Both
  variants report an identical 312.5 MB. `estimate_model_size_mb()` sums
  `parameter.numel() * parameter.element_size()` over `model.parameters()`,
  but PyTorch's dynamically quantized `nn.Linear` replacement stores its
  weights as packed buffers (`_packed_params`), not as `nn.Parameter` — so
  the quantized weights never show up in that iteration and the size
  estimate silently falls back to counting only the non-quantized
  parameters, which are identical between the two models. The script's own
  printed takeaway ("INT8 models are significantly smaller in memory") is
  **not actually demonstrated by this script's measurement** — it's a true
  general fact about quantization, but this benchmark doesn't measure it
  correctly.

---

## 🧩 Known Limitation

To get an accurate INT8 size, the estimator would need to walk quantized
modules' `_packed_params` (via `torch.ao.nn.quantized.dynamic.Linear`) or
serialize both models to disk with `torch.save` and compare file sizes.
Not fixed here — flagged so the number isn't taken at face value.

---

Takeaway: trust the latency/throughput numbers from this benchmark; don't
trust the size column until the estimator is fixed to account for packed
quantized weights.
