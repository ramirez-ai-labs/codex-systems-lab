# CPU vs GPU — Inference Latency Benchmark

This experiment measures how long a small language model (`distilgpt2`)
takes to generate text on different compute devices. Beginners can use this
to understand why GPUs matter for LLM workloads.

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

Average generation time for 32 new tokens (5 runs):

| Device | Avg Latency (sec) |
| ------ | ----------------- |
| cpu    | 2.1687            |

Raw console output:

```
Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.
Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.
Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.
Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.
Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.
cpu: 2.1687 sec
```

---

## 📘 Beginner Interpretation

- This low-power CPU needs ~2.17 seconds to generate 32 tokens with DistilGPT2.
- Larger coding models would be much slower on CPU unless quantized or batched.
- A GPU typically reduces the latency by 5–20×, so rerun this on a CUDA host to compare.

Takeaway: CPU-only inference is fine for demos, but high-throughput assistants rely on GPUs.
