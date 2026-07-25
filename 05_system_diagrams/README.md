# 05 — System Diagrams ⚠️ Implemented — one headline number since retracted

Visual diagrams connecting the measurements and simulations from
`01_inference_profiling`, `02_light_finetuning`, and `03_agentic_performance`
into a single picture of how an inference/agent system fits together.

```bash
05_system_diagrams/
├── README.md                       ← This file
├── inference_pipeline.md            ← Implemented — sequence diagram; headline latency claim since retracted
├── batching_architecture.mmd        ← Implemented — real measured throughput, flowchart
└── container_orchestration.md       ← Implemented — proposed deployment design, clearly marked as not built
```

---

## Why These Diagrams Exist

Numbers in `RESULTS.md` files are easy to skim past. These diagrams put the
same measurements into a shape that shows *where* in a request's lifecycle
each cost shows up — per-batch-size throughput, and (as a proposal) how
those pieces would sit inside a production deployment alongside the agent
loop from `03_agentic_performance`.

---

## Diagrams in This Folder

### 1️⃣ `inference_pipeline.md` — Request Sequence Diagram 🚨 Headline number retracted

A sequence diagram of a single inference request (tokenize → first forward
pass → KV-cache-backed decode loop).

Key ideas:
- Originally annotated with a measured 4.42s first-token vs. 0.09s cached
  next-token latency (48.1x speedup) from `01_inference_profiling`
- **That speedup does not reproduce** — re-running the underlying benchmark
  repeatedly shows ~0.6–1.2x, essentially no cache benefit, traced to a
  real methodology gap in the script (see
  [`RESULTS_kv_cache_analysis.md`](../01_inference_profiling/RESULTS_kv_cache_analysis.md#-limitations))
- The diagram still shows the general mechanism, which is real and
  well-documented in LLM serving — but the timing annotations were removed
  since this repo's own measurement of them isn't currently trustworthy

### 2️⃣ `batching_architecture.mmd` — Batching Flowchart ✅ Implemented

A flowchart of how concurrent prompts get grouped into a batch before the
forward pass.

Key ideas:
- Annotated with real per-batch-size throughput from
  [`RESULTS_batch_size.md`](../01_inference_profiling/RESULTS_batch_size.md):
  tokens/sec climbs from ~32 (batch 1) to ~113 (batch 8) on this CPU-only
  machine, even as per-batch latency also rises
- An earlier version of this diagram cited latency-only numbers from an
  older version of the underlying script that no longer exists — updated
  to match its current behavior

### 3️⃣ `container_orchestration.md` — Proposed Deployment Architecture ✅ Implemented (as a proposal, not built)

A flowchart proposing how the model-serving, batching, and
agent-orchestration pieces measured elsewhere in this lab would compose
into a containerized production deployment.

Key ideas:
- **This one is explicitly a design sketch** — there is no Dockerfile,
  Compose file, or Kubernetes manifest anywhere in this repo
- Each box is grounded in a real number from this lab's benchmarks (or
  honestly flagged where the underlying benchmark doesn't hold up), but the
  deployment itself was never built or run
- Autoscaling behavior is asserted as a plausible policy, not measured —
  this repo never ran a multi-pod or multi-node experiment

---

## How to View These Diagrams

All diagrams use [Mermaid](https://mermaid.js.org/) syntax. GitHub renders
` ```mermaid ` fences in `.md` files natively — `inference_pipeline.md` and
`container_orchestration.md` will render inline in GitHub's file browser.
`batching_architecture.mmd` is raw Mermaid source (no markdown wrapper) —
view it with the [Mermaid Live Editor](https://mermaid.live/) or a
Mermaid-aware editor extension, or validate/render it locally:

```bash
npx -y @mermaid-js/mermaid-cli -i batching_architecture.mmd -o batching_architecture.svg
```

---

_See the top-level [README.md](../README.md) for how this section fits into
the overall repo._
