# 📘 **codex-systems-lab**

*A hands-on research and engineering lab exploring the systems, performance, and training foundations behind modern AI models and agentic workflows.*

This repository contains a curated set of experiments designed to demonstrate the technical depth expected in modern AI research and engineering work. It covers:

* **Inference performance optimization**
* **Light LLM fine-tuning for code generation and task-specific adaptation**
* **Agentic system behavior & latency profiling**
* **Reproductions of research findings**
* **Systems diagrams for LLM inference & agentic workflows**

This lab blends **ML research**, **systems engineering**, and **developer-experience insights**—mirroring the full-stack thinking required to build and evaluate practical AI systems.

---

# 🔍 **Why This Lab Exists**

Modern AI systems for coding, assistance, and automation combine:

* Large-scale model inference
* Context routing and retrieval
* Agentic loop orchestration
* Tool calling
* Performance-sensitive deployment environments
* Model evaluation and experiment design

This repo demonstrates the ability to reason **end-to-end across that entire stack**.

It is intentionally structured like a research notebook + engineering diagnostics toolkit.

---

# 🧭 **Repository Structure**

```
codex-systems-lab/
│
├── 01_inference_profiling/          → Inference speed, batching, quantization, KV cache effects
├── 02_light_finetuning/             → Small-scale LLM fine-tuning on Python code
├── 03_agentic_performance/          → Agent loop benchmarks & tool-use simulations
├── 04_research_reproductions/       → Small-scale replications of ML research studies
├── 05_system_diagrams/              → Mermaid diagrams: inference pipelines, batching, agent flows
└── README.md                        → This file
```

Each directory includes:

* **Python scripts**
* **RESULTS.md** with findings
* **Notebooks or plots (when relevant)**

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

For example, measure CPU vs GPU inference:

```bash
python 01_inference_profiling/benchmark_cpu_vs_gpu.py
```

Results will appear in:

```
01_inference_profiling/RESULTS.md
```

---

# 📊 **Included Experiment Categories**

## **01 — Inference Profiling**

Experiments include:

* CPU vs GPU latency
* Batch size effects
* KV cache impact analysis
* Quantization comparison (fp32 → fp16 → int8 → gguf)

These scripts illustrate the tradeoffs between **speed, cost, resource usage, and model throughput**.

---

## **02 — Light Fine-Tuning**

Small-scale code-specific fine-tuning examples:

* LoRA fine-tuning on Python functions
* Perplexity evaluation on code
* Functional correctness checks
* Dataset creation workflow

This section shows practical ML training experience without requiring expensive GPUs.

---

## **03 — Agentic Performance**

Simulated agent loops to understand:

* Multi-step reasoning cost
* Tool latency accumulation
* Retry strategies
* Error recovery cost

These experiments mirror real-world agentic coding systems (e.g., planning, tool-use, self-repair).

---

## **04 — Research Reproductions**

Small replications of published findings, such as:

* Edit-distance vs acceptance rate
* RAG retrieval degradation effects
* Prompting strategy comparisons
* Scaling effects (small-model approximations)

This section demonstrates **research literacy, experiment design, and reproducibility**.

---

## **05 — Systems Diagrams**

Includes Mermaid diagrams for:

* LLM inference pipeline
* Batching architecture
* Agent control-flow loop
* Context routing

These diagrams summarize the conceptual model behind the experiments.

---

# 🧪 **Example Output**

```text
CPU: 0.4231 sec
CUDA: 0.0784 sec

Batch size 1 → 0.42 sec
Batch size 4 → 0.29 sec
Batch size 8 → 0.23 sec
```

Output tables & notes live in each folder’s `RESULTS.md`.

---

# 🧠 **Skills Demonstrated**

This repository highlights capabilities valuable across **applied AI engineering and research roles**:

* LLM inference mechanics
* Model optimization and throughput diagnostics
* Experimental design & empirical analysis
* Light fine-tuning and evaluation
* Agent loop modeling & performance tuning
* Systems thinking across ML + infrastructure
* Clear communication of findings

---

# 📬 **Questions or Improvements?**

This is an evolving lab. Future additions will include:

* Advanced quantization
* Function-calling benchmarks
* Multimodal agent loops
* Model-based debugging analysis

---

# 🔗 Author

**Victor Ramirez**
AI Architect & Developer Experience Engineer
Portfolio: [https://ramirezailabs.com/](https://ramirezailabs.com/)
GitHub: [https://github.com/ramirez-ai-labs](https://github.com/ramirez-ai-labs)
LinkedIn: [https://linkedin.com/in/victor-hugo-ramirez-mids](https://linkedin.com/in/victor-hugo-ramirez-mids)

---
