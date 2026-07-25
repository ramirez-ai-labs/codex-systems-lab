# Inference Profiling Lab ✅ Measured, real hardware

This folder explores **why LLM inference feels fast or slow**, using real
`distilgpt2` inference runs on a MacBook Air (i5-8210Y, CPU-only, no GPU
detected). Every number below is measured, not estimated — see each
`RESULTS_*.md` for raw console output and the full hardware snapshot.

## Experiments (recommended order)

1. `benchmark_cpu_vs_gpu.py` → [`RESULTS_cpu_vs_gpu.md`](RESULTS_cpu_vs_gpu.md)
   - Why GPUs matter for inference — 2.17s avg for 32 tokens on this CPU

2. `benchmark_batching_effects.py` → [`RESULTS_batch_size.md`](RESULTS_batch_size.md)
   - Why batching improves throughput — batch size 2 fastest (1.32s avg), batch size 8 slows down (2.49s avg, CPU-bound)

3. `benchmark_kv_cache_analysis.py` → [`RESULTS_kv_cache_analysis.md`](RESULTS_kv_cache_analysis.md)
   - Why first-token latency is slow — 48.1x speedup from first-token (4.42s) to cached next-token (0.09s) latency

4. `benchmark_quantization_comparison.py` → [`RESULTS_quantization.md`](RESULTS_quantization.md)
   - Why lower precision speeds up CPUs — INT8 ~1.5x faster than FP32; the script's reported model-size numbers are inaccurate and shouldn't be trusted (documented as a known limitation in the results file)

Each script can be run independently; commands above match this repo's actual filenames.
