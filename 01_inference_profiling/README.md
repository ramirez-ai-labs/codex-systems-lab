# Inference Profiling Lab ✅ Measured, real hardware

This folder explores **why LLM inference feels fast or slow**, using real
`distilgpt2` inference runs on a MacBook Air (i5-8210Y, CPU-only, no GPU
detected). Every number below is measured, not estimated — see each
`RESULTS_*.md` for raw console output, the full hardware snapshot, and (for
two of these four) an honest account of why the numbers didn't hold up on
re-measurement.

## Experiments (recommended order)

1. `benchmark_cpu_vs_gpu.py` → [`RESULTS_cpu_vs_gpu.md`](RESULTS_cpu_vs_gpu.md)
   - Why GPUs matter for inference — ~1–7s avg for 32 tokens on this CPU;
     six fresh runs showed a 7x spread, so treat any single number
     (including this range) as one sample from high hardware variance, not
     a precise figure

2. `benchmark_batching_effects.py` → [`RESULTS_batch_size.md`](RESULTS_batch_size.md)
   - Why batching improves throughput — tokens/sec climbs from ~32 (batch 1)
     to ~113 (batch 8) even though per-batch latency also rises

3. `benchmark_kv_cache_analysis.py` → [`RESULTS_kv_cache_analysis.md`](RESULTS_kv_cache_analysis.md)
   - ⚠️ The documented "48.1x speedup" does not reproduce — four fresh runs
     measured ~0.6–1.2x (no consistent cache benefit at all). Turned out to
     be a real methodology gap in the script, not just noise — see the
     results file's Limitations section

4. `benchmark_quantization_comparison.py` → [`RESULTS_quantization.md`](RESULTS_quantization.md)
   - Why lower precision speeds up CPUs — INT8 ~1.5x faster than FP32; the script's reported model-size numbers are inaccurate and shouldn't be trusted (documented as a known limitation in the results file)

Each script can be run independently; commands above match this repo's actual filenames.
