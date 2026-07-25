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
├── 02_light_finetuning/             → End-to-end LLM fine-tuning pipeline on Python code (complete, real before/after eval)
├── 03_agentic_performance/          → Agent loop benchmarks (part measured, part simulated — see below)
├── 04_research_reproductions/       → paper_1 and paper_2 implemented (synthetic, real output)
├── 05_system_diagrams/              → Diagrams grounded in real measurements, plus one labeled proposal
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
* `benchmark_quantization_comparison.py` → `RESULTS_quantization.md` — FP32 vs. INT8 dynamic quantization; INT8 was ~1.5x faster, but the script's reported model-size numbers are inaccurate (documented as a known limitation in the results file, not silently trusted)

## **02 — Light Fine-Tuning** ✅ Complete, end to end

Full fine-tuning of DistilGPT-2 on 31 Python-function training examples using HuggingFace
`Trainer` + AdamW — data prep, tokenization, training, checkpoint save, and perplexity
evaluation all run end to end on CPU. Validation perplexity drops from 2989.72 (base model)
to 38.23 (fine-tuned), a full before/after comparison in `RESULTS.md`. The earlier
checkpoint-save failure was a `requirements.txt` bug (`torch<2.2` has no wheels for
Python 3.12), not a fundamental compatibility issue — fixed by relaxing the pin.
LoRA and functional-correctness testing are scoped as next steps, not yet implemented.

## **03 — Agentic Performance** ⚠️ Part measured, part simulated

* `agent_loop_benchmark.py` makes real DistilGPT-2 inference calls to measure think→act→reflect
  overhead — genuine measurement. `tool_latency_simulation.py` also makes real inference calls
  for the "think" step, but stands in a fixed `time.sleep()` for the "tool" step instead of
  calling a real tool — its own docstring says so explicitly ("This script does NOT call real
  APIs"). So the LLM-side cost is measured; the tool-side cost is a constant, not measured.
* `error_recovery_costs.py` and `retries_vs_quality_analysis.py` model retry/failure costs
  using `random.uniform()` and `time.sleep()` for both the LLM and tool steps — no model or
  tool is called at all, real or otherwise. This models the *shape* of the cost tradeoff, it
  doesn't measure a real system. Both docstrings call themselves simulations, but neither
  spells out as explicitly as `tool_latency_simulation.py` does that no real model is involved.

## **04 — Research Reproductions** ✅ Both implemented (synthetic)

`paper_1_edit_distance_vs_acceptance` is implemented: `replicate_experiment.ipynb` runs end
to end (`jupyter nbconvert --to notebook --execute`) and `RESULTS.md` reports the real output
— acceptance rate falls from 97.8% at very-low edit distance to 0.0% at high edit distance.
This is a genuine measurement of a *simulation*, not of real developer behavior: the data is
synthetic and the acceptance-probability curve is a hand-picked logistic function, not fit to
any real dataset — `RESULTS.md` spells out that limitation explicitly.

`paper_2_rag_scaling_laws` is also implemented: its notebook measures real Recall@k over a
synthetic embedding corpus (recall rises with k but with diminishing returns, and drops
sharply as corpus size grows), then layers a hand-picked "context dilution" function on top
to simulate downstream answer accuracy, which peaks around k=10–50 and then falls off.
`RESULTS.md` is explicit that only the recall measurement is real — the dilution/accuracy
curve is invented, not fit to any real RAG system.

## **05 — System Diagrams** ✅ Implemented (one is a labeled proposal)

Three Mermaid diagrams, each grounded in numbers already measured elsewhere in this repo:

* `inference_pipeline.md` — sequence diagram of a single request (tokenize → first forward
  pass → KV-cache-backed decode loop), annotated with the real 4.42s first-token vs. 0.09s
  cached next-token latency from `01_inference_profiling`.
* `batching_architecture.mmd` — flowchart of request batching, annotated with the real
  per-batch-size latencies from `RESULTS_batch_size.md` (batch size 2 fastest at 1.32s avg,
  batch size 8 slowest at 2.49s avg on this CPU-only machine).
* `container_orchestration.md` — a **proposed** containerized deployment connecting the
  model-serving, batching, and agent-orchestration pieces measured elsewhere in this lab.
  Explicitly marked as a design sketch: no Dockerfile, Compose file, or Kubernetes manifest
  exists anywhere in this repo.

---

# 🧠 **Skills Demonstrated**

* Real inference measurement: latency, batching, KV-cache, quantization tradeoffs
* End-to-end LLM fine-tuning workflow (data → train → evaluate) with honest documentation of a real infra failure
* Agent-loop cost modeling, distinguishing measured latency from simulated cost structure
* Research reproduction pipeline: hypothesis → synthetic data → sampled outcomes → binned measurement → visualization
* System diagramming that stays grounded in real measurements, with proposed-but-unbuilt designs clearly labeled as such
* Clear experiment documentation, including what didn't work and what's synthetic vs. real

---

# 📬 **Roadmap**

* Fix the quantization benchmark's model-size estimator (packed quantized weights aren't counted)
* Fit paper_1's acceptance model, and paper_2's dilution model, to real or published data instead of hand-picked constants
* Turn `05_system_diagrams/container_orchestration.md` from a proposal into a real Dockerfile + Kubernetes manifest, and measure actual deployment behavior
* Add LoRA fine-tuning and functional-correctness testing to the fine-tuning lab

---

# 🔗 Author

**Victor Ramirez**
AI Architect & Developer Experience Engineer
Portfolio: [https://ramirezailabs.com/](https://ramirezailabs.com/)
GitHub: [https://github.com/ramirez-ai-labs](https://github.com/ramirez-ai-labs)
LinkedIn: [https://linkedin.com/in/victor-hugo-ramirez-mids](https://linkedin.com/in/victor-hugo-ramirez-mids)
License: [MIT](LICENSE)

---
