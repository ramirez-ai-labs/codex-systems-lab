# 📘 **codex-systems-lab**

*A hands-on lab exploring the systems and performance foundations behind modern AI models and agentic workflows.*

This repository contains experiments measuring real inference behavior — latency, batching,
KV-cache effects, and agent-loop overhead — plus a working end-to-end LLM fine-tuning pipeline.
It's intentionally CPU-friendly and structured like a research notebook: each experiment has
its own script and a `RESULTS.md` documenting what was actually run, including where things
didn't fully work.

---

# 🔍 **Why This Lab Exists**

Modern AI systems for coding, assistance, and automation combine model inference, context
routing, agentic loop orchestration, tool calling, and performance-sensitive deployment. This
repo demonstrates hands-on measurement and experimentation across that stack — not just
reading about how it works, but running it and recording what actually happens, including
the parts that didn't go as planned.

---

# 🧭 **Repository Structure**

```
codex-systems-lab/
│
├── 01_inference_profiling/          → Inference speed, batching, KV cache, quantization (measured, real hardware)
├── 02_light_finetuning/             → End-to-end LLM fine-tuning pipeline on Python code (functional, one open issue)
├── 03_agentic_performance/          → Agent loop benchmarks (part measured, part simulated — see below)
├── 04_research_reproductions/       → Planned — not yet built
├── 05_system_diagrams/              → Planned — not yet built
└── README.md                        → This file
```

---

# 🚀 **How to Get Started**

## 1. Clone the repo

```bash
git clone https://github.com/ramirez-ai-labs/codex-systems-lab
cd codex-systems-lab
```

## 2. Set up the virtual environment (recommended)

```bash
python -m venv .venv
.\.venv\Scripts\Activate  # Windows
pip install -r requirements.txt
```

## 3. Run your first experiment

```bash
python 01_inference_profiling/benchmark_cpu_vs_gpu.py
```

Results appear in `01_inference_profiling/RESULTS_cpu_vs_gpu.md`.

---

# 📊 **What's Actually Here**

## **01 — Inference Profiling** ✅ Measured, real hardware

Real DistilGPT-2 inference runs on a MacBook Air (i5-8210Y, CPU-only, no GPU detected),
with raw console output and machine specs recorded in each results file:

* `benchmark_cpu_vs_gpu.py` → `RESULTS_cpu_vs_gpu.md` — 2.17s avg for 32 tokens on CPU
* `benchmark_batching_effects.py` → `RESULTS_batch_size.md` — throughput across batch sizes 1/2/4/8
* `benchmark_kv_cache_analysis.py` → `RESULTS_kv_cache_analysis.md` — 48.1x speedup from first-token to cached next-token latency
* `benchmark_quantization_comparison.py` — FP32 vs. INT8 dynamic quantization comparison. Script is functional; results file not yet captured.

## **02 — Light Fine-Tuning** ✅ Complete, end to end

Full fine-tuning of DistilGPT-2 on 31 Python-function training examples using HuggingFace
`Trainer` + AdamW — data prep, tokenization, training, checkpoint save, and perplexity
evaluation all run end to end on CPU. Validation perplexity drops from 2989.72 (base model)
to 38.23 (fine-tuned), a full before/after comparison in `RESULTS.md`. The earlier
checkpoint-save failure was a `requirements.txt` bug (`torch<2.2` has no wheels for
Python 3.12), not a fundamental compatibility issue — fixed by relaxing the pin.
LoRA and functional-correctness testing are scoped as next steps, not yet implemented.

## **03 — Agentic Performance** ⚠️ Part measured, part simulated

* `agent_loop_benchmark.py` and `tool_latency_simulation.py` make real DistilGPT-2 inference
  calls to measure think→act→reflect overhead and tool-latency dominance — genuine measurement.
* `error_recovery_costs.py` and `retries_vs_quality_analysis.py` model retry/failure costs
  using `random.uniform()` and `time.sleep()`, not a real model or real tool — this models the
  *shape* of the cost tradeoff, it doesn't measure a real system. Both scripts are transparent
  about this in their own docstrings.

## **04 — Research Reproductions** 📋 Planned

One subfolder scaffold exists (`paper_1_edit_distance_vs_acceptance`) with a README describing
the intended reproduction. The notebook and results have not been built yet.

## **05 — Systems Diagrams** 📋 Planned

Scaffolded but empty. Diagrams for inference pipelines, batching, and agent control flow are
planned, not yet created.

---

# 🧠 **Skills Demonstrated**

* Real inference measurement: latency, batching, KV-cache, quantization tradeoffs
* End-to-end LLM fine-tuning workflow (data → train → evaluate) with honest documentation of a real infra failure
* Agent-loop cost modeling, distinguishing measured latency from simulated cost structure
* Clear experiment documentation, including what didn't work

---

# 📬 **Roadmap**

* Capture quantization comparison results
* Build out `04_research_reproductions` and `05_system_diagrams`
* Add LoRA fine-tuning and functional-correctness testing to the fine-tuning lab

---

# 🔗 Author

**Victor Ramirez**
AI Architect & Developer Experience Engineer
Portfolio: [https://ramirezailabs.com/](https://ramirezailabs.com/)
GitHub: [https://github.com/ramirez-ai-labs](https://github.com/ramirez-ai-labs)
LinkedIn: [https://linkedin.com/in/victor-hugo-ramirez-mids](https://linkedin.com/in/victor-hugo-ramirez-mids)

---
