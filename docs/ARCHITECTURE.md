# Architecture: codex-systems-lab

This document explains how the five sections of this lab connect and build on each other.

---

## **High-Level Data Flow**

```
Real Hardware Environment
    ↓
01. Inference Profiling
    ├─ Measures: latency, throughput, KV-cache, quantization
    ├─ Outputs: RESULTS_*.md, performance data
    └─ Feeds into: 05 (System Diagrams)
    ↓
02. Light Fine-Tuning
    ├─ Measures: training time, validation perplexity before/after
    ├─ Outputs: trained model, RESULTS.md
    └─ Demonstrates: complete training pipeline
    ↓
03. Agentic Performance
    ├─ Measures: agent loop overhead, tool latency impact
    ├─ Mix of: measured (inference) + simulated (tool calls)
    ├─ Outputs: RESULTS.md, latency breakdowns
    └─ Feeds into: 05 (System Diagrams)
    ↓
04. Research Reproductions
    ├─ Implements: published paper experiments
    ├─ Uses: synthetic data (labeled as such)
    ├─ Outputs: Jupyter notebooks, RESULTS.md
    └─ Purpose: demonstrate reproducibility methodology
    ↓
05. System Diagrams
    ├─ Integrates: real measurements from 01, 03
    ├─ Proposes: container orchestration (not implemented)
    └─ Serves as: bridge to production deployment
```

---

## **Module Breakdown**

### **01: Inference Profiling** 
**Purpose:** Establish ground truth about model performance on real hardware.

**Files:**
- `benchmark_cpu_vs_gpu.py` → Single-token latency comparison
- `benchmark_batching_effects.py` → Throughput at different batch sizes
- `benchmark_kv_cache_analysis.py` → Token generation speedup (methodological insights)
- `benchmark_quantization_comparison.py` → FP32 vs INT8 tradeoffs

**Outputs:** RESULTS_*.md files with:
- Raw measurements
- Run-to-run variance
- Hardware specifications
- Honest documentation of failures

**Key Finding:** Real measurement beats theory. Variance is real data.

---

### **02: Light Fine-Tuning**
**Purpose:** Demonstrate end-to-end training pipeline with honest failure documentation.

**Files:**
- `prepare_dataset.py` → Load 31 Python examples
- `train.py` → HuggingFace Trainer + validation
- `eval_code_perplexity.py` → Compute before/after perplexity

**Outputs:** 
- Trained model checkpoint
- RESULTS.md with perplexity drop (2989.72 → 38.23)
- Documentation of real failures (torch version issues)

**Key Teaching:** Real pipelines have failures. Document them.

---

### **03: Agentic Performance**
**Purpose:** Measure cost of agentic workflows; highlight measured vs. simulated components.

**Files:**
- `agent_loop_benchmark.py` → Measured: real think→act→reflect loops
- `tool_latency_simulation.py` → Mixed: measured inference + simulated tool latency
- `error_recovery_costs.py` → Simulated: models retry/failure costs
- `retries_vs_quality_analysis.py` → Simulated: tradeoff analysis

**Key Distinction:** Every script's docstring states measured vs. simulated status.

**Outputs:** RESULTS.md with:
- Actual loop multiplier (3.0x slower than single inference)
- Tool latency breakdown
- Retry cost estimates

---

### **04: Research Reproductions**
**Purpose:** Demonstrate scientific reproducibility; practice implementing papers.

**Files:**
- `paper_1_edit_distance_vs_acceptance/` → Synthetic acceptance-rate curve
- `paper_2_rag_scaling_laws/` → Synthetic recall + accuracy modeling

**Key Learning:** Synthetic benchmarks are valuable when clearly labeled. These implement paper methodology on toy data to teach the concepts.

---

### **05: System Diagrams**
**Purpose:** Integrate learnings from 01 & 03 into architectural views.

**Files:**
- `inference_pipeline.md` → Sequence diagram with real latency numbers from 01
- `batching_architecture.mmd` → Flowchart with measured throughput from 01
- `container_orchestration.md` → Proposed (not yet built) deployment architecture

**Design Philosophy:** Diagrams anchor to measurements, not theory. Proposals are explicitly labeled.

---

## **Data Dependencies**

```
01_inference_profiling/
    ├─ Produces: RESULTS_cpu_vs_gpu.md, RESULTS_batch_size.md, etc.
    └─ Used by: 05_system_diagrams (latency annotations)

02_light_finetuning/
    ├─ Produces: trained model, RESULTS.md
    └─ Standalone (not consumed by other sections)

03_agentic_performance/
    ├─ Produces: RESULTS.md with loop overhead
    └─ Used by: 05_system_diagrams (agent cost breakdown)

04_research_reproductions/
    ├─ Produces: Jupyter notebooks, RESULTS.md
    └─ Demonstrates: reproducibility methodology (not consumed downstream)

05_system_diagrams/
    ├─ Consumes: Measurements from 01 & 03
    ├─ Proposes: Deployment architecture (not implemented)
    └─ Purpose: Integrative view
```

---

## **Hardware Environment**

All measurements assume:
- **Default:** CPU-only (no GPU)
- **Model:** DistilGPT-2 (small, CPU-friendly)
- **Environment:** Local machine (not cloud)

Results will vary by hardware. Document your environment in RESULTS.md if reproducing.

---

## **Extension Points**

### **To Add a New Benchmark:**
1. Create new script in appropriate section (01-04)
2. Follow template: setup → run → analyze → output
3. Create companion RESULTS_*.md documenting methodology, failures, variance
4. If results inform a system diagram, update 05

### **To Add a New Section:**
Would require architectural changes. Current 5-section structure is intentional:
- Sections 1-4 are independent (can run in any order)
- Section 5 integrates findings from 1 & 3

---

## **Testing & Validation**

Each script can be run independently:
```bash
python 01_inference_profiling/benchmark_cpu_vs_gpu.py
python 02_light_finetuning/train.py
python 03_agentic_performance/agent_loop_benchmark.py
```

No scripts depend on outputs from previous sections (except diagrams in 05, which reference measurements).

---

## **Key Design Decisions**

| Decision | Rationale |
|----------|-----------|
| CPU-only | Lower barrier to entry; demonstrates algorithmic efficiency |
| Small model (DistilGPT-2) | Fast iteration, clear measurement variance |
| Multiple RESULTS_*.md files | Distributed documentation; each benchmark owns its findings |
| Measured vs. simulated labeled | Builds credibility; readers know what to trust |
| Section 5 as proposals | Keeps architectural thinking separate from validation |

---

## **Related Work**

This lab complements `codex-evaluation-benchmark`:
- **Systems Lab** (this repo): How models *behave* (latency, throughput, tradeoffs)
- **Evaluation Benchmark** (other repo): How models *impact developers* (acceptance, productivity, quality)

Together, they form a closed loop: measure systems → evaluate impact → iterate.

---

## **Future Directions**

1. **Implement container_orchestration.md** into real Dockerfile + K8s manifests
2. **Measure tool latency** (currently simulated in 03)
3. **Fit paper models to real data** (currently hand-picked constants in 04)
4. **Add LoRA fine-tuning** (04 roadmap item)
5. **Connect to evaluation benchmark** — measure if system improvements → better developer outcomes
