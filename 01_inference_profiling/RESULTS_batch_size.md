# Batch Size Benchmark — How Batch Size Affects Throughput

This benchmark shows how generating multiple prompts at once (“batching”)
can speed up inference by better utilizing the model’s parallel compute.

---

## 🚀 How to Run

```bash
python 01_inference_profiling/benchmark_batch_size.py
```

---

## 🖥️ Hardware Snapshot

- Machine: MacBook Air — Intel(R) Core(TM) i5-8210Y @ 1.60GHz
- CPU cores: 4 logical
- GPU: ❌ None detected
- Model: `distilgpt2`

---

## ⚡ Observed Latency

Average latency per batch (5 runs per size):

| Batch Size | Avg Latency (sec) |
| ---------- | ----------------- |
| 1          | 1.5233            |
| 2          | 1.3181            |
| 4          | 1.4766            |
| 8          | 2.4881            |

Raw console output:

```
Running on device: cpu

Batch Size Benchmark
====================

Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.
Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.
Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.
Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.
Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.
Batch size  1: 1.5233 sec
Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.
Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.
Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.
Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.
Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.
Batch size  2: 1.3181 sec
Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.
Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.
Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.
Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.
Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.
Batch size  4: 1.4766 sec
Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.
Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.
Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.
Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.
Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.
Batch size  8: 2.4881 sec

Benchmark complete!
```

---

## 📘 Beginner Interpretation

- Batching multiple prompts slightly increases latency per batch on CPU, but total throughput improves because more tokens are processed at once.
- Batch size 2 was faster than batch size 1 on this machine, showing the benefit of parallelization.
- Larger batches (4 or 8) start to slow down due to limited CPU cores and memory bandwidth; GPUs handle large batches more efficiently.

Takeaway: Even on CPU, moderate batching boosts throughput—but GPUs unlock the real gains.
