# 05 — System Diagrams

Visual diagrams connecting the measurements and simulations from
`01_inference_profiling`, `02_light_finetuning`, and `03_agentic_performance`
into a single picture of how an inference/agent system fits together.

```bash
05_system_diagrams/
├── README.md                       ← This file
├── inference_pipeline.md            ← Implemented — real measured latencies, sequence diagram
├── batching_architecture.mmd        ← Implemented — real measured latencies, flowchart
└── container_orchestration.md       ← Implemented — proposed deployment design, clearly marked as not built
```

## inference_pipeline.md ✅ Implemented

A sequence diagram of a single inference request (tokenize → first forward
pass → KV-cache-backed decode loop), annotated with the real latency numbers
measured in
[`01_inference_profiling/RESULTS_kv_cache_analysis.md`](../01_inference_profiling/RESULTS_kv_cache_analysis.md):
4.42s first-token latency vs. 0.09s cached next-token latency, a 48.1x gap.

## batching_architecture.mmd ✅ Implemented

A flowchart of how concurrent prompts get grouped into a batch before the
forward pass, annotated with the real per-batch-size latencies measured in
[`01_inference_profiling/RESULTS_batch_size.md`](../01_inference_profiling/RESULTS_batch_size.md):
batch size 2 was fastest on this CPU-only machine (1.32s avg); batch size 8
was slowest (2.49s avg), CPU-bound rather than benefiting from
parallelization.

## container_orchestration.md ✅ Implemented (as a proposal, not built)

A flowchart proposing how the model-serving, batching, and
agent-orchestration pieces measured elsewhere in this lab would compose into
a containerized production deployment. **This one is explicitly a design
sketch** — there is no Dockerfile, Compose file, or Kubernetes manifest
anywhere in this repo. The diagram grounds each box in a real number from
this lab's benchmarks, but the deployment itself was never built or run.

---

## Why These Diagrams Exist

Numbers in `RESULTS.md` files are easy to skim past. These diagrams put the
same measurements into a shape that shows *where* in a request's lifecycle
each cost shows up — first-token vs. cached-token latency, per-batch-size
throughput, and (as a proposal) how those pieces would sit inside a
production deployment alongside the agent loop from `03_agentic_performance`.

## Rendering

All diagrams use [Mermaid](https://mermaid.js.org/) syntax. GitHub renders
` ```mermaid ` fences in `.md` files natively. `batching_architecture.mmd`
is raw Mermaid source (no markdown wrapper) — view it with the
[Mermaid Live Editor](https://mermaid.live/) or a Mermaid-aware editor
extension if your viewer doesn't render `.mmd` files directly.

---

_See the top-level [README.md](../README.md) for how this section fits into
the overall repo._
