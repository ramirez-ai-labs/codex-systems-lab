# 📊 01_inference_profiling — CPU vs GPU Benchmark Results

This document shows the results of running the beginner-friendly inference benchmark:

```bash
python 01_inference_profiling/benchmark_cpu_vs_gpu.py
```

The goal is to help beginners clearly understand how **device choice (CPU vs GPU)** affects the speed of text generation in AI models.

---

# 🖥️ Hardware Snapshot (Auto-Detected)

- **Machine:** MacBook Air  
- **CPU:** Intel(R) Core(TM) i5-8210Y @ 1.60GHz  
- **Logical CPU Cores:** 4  
- **GPU Available:** ❌ *No GPU detected*  
  - (`torch.cuda.is_available()` returned `False`)

> Since this machine does not have a CUDA-enabled GPU, the script ran only the CPU benchmark.

---

# ⚙️ Benchmark Configuration

| Setting | Value |
|--------|-------|
| **Model:** | `distilgpt2` |
| **Prompt:** | `"def fibonacci(n):"` |
| **Runs per device:** | 5 |
| **Max new tokens:** | 32 |

---

# 🚀 Results: CPU Latency

Average time (across 5 runs):

| Device | Avg Latency (sec) |
|--------|-------------------|
| **CPU** | **1.3938 sec** |

---

# 🧾 Raw Console Output

```
==============================
 CPU vs GPU Inference Benchmark
==============================

No GPU detected. Running on CPU only.

Running inference on: CPU
Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.
Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.
Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.
Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.
Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.
Average time on CPU: 1.3938 seconds


==============================
        FINAL RESULTS
==============================

CPU: 1.3938 seconds

Benchmark complete!
```

---

# 🧐 What These Numbers Mean (Beginner-Friendly)

### ✔ 1.39 seconds per generation  
On a lightweight laptop CPU, transformer models run slowly.  
For interactive tools (like Copilot-style autocomplete), users usually expect:

- **50–150 ms** for autocomplete  
- **200–400 ms** for multi-line suggestions  

So **1.39 seconds** is too slow for production‑grade responsiveness.

---

# ❌ Why No GPU Results?

This MacBook Air uses **Intel integrated graphics**, and:

- PyTorch **cannot use CUDA** on Intel GPUs  
- Apple GPU acceleration (`mps`) only works on **M1/M2/M3 Macs**

To compare CPU vs GPU, rerun the script on:

- A desktop with an **NVIDIA GPU**
- A cloud GPU instance
- An Apple Silicon Mac (`mps` backend)

---

# 📈 Expected Performance on Other Devices

| Device | Typical Time |
|--------|--------------|
| Laptop CPU | 1–3 sec |
| Apple M1/M2 GPU | 0.10–0.30 sec |
| NVIDIA RTX 3060 | 0.05–0.15 sec |
| NVIDIA A100 | 0.01–0.03 sec |

---

# 🧠 Key Insights

- CPUs can run LLMs, but **slowly**.  
- GPUs provide **20–50× speedups**.  
- Larger models scale latency significantly.  
- Real systems rely on **KV caching**, **quantization**, **batching**, and optimized runtimes.

---

# 🔧 Future Enhancements

Potential extensions:

- Batch size benchmarking  
- KV cache comparison  
- Quantization profiling  
- Apple MPS support  
- Token throughput graphs  

---

# ✅ Summary

- **CPU average latency:** 1.3938 seconds  
- **GPU:** not available  
- Great beginner demonstration of how hardware affects LLM inference.

---
