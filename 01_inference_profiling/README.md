# Inference Profiling Lab

This folder explores **why LLM inference feels fast or slow**.

## Experiments (recommended order)

1. `benchmark_cpu_vs_gpu.py`
   - Why GPUs matter for inference

2. `benchmark_batch_size.py`
   - Why batching improves throughput

3. `benchmark_kv_cache.py`
   - Why first-token latency is slow

4. `benchmark_quantization.py`
   - Why lower precision speeds up CPUs

Each script can be run independently.
Results are recorded in RESULTS.md files.
