# 05 — System Diagrams

Visual diagrams connecting the measurements and simulations from
`01_inference_profiling`, `02_light_finetuning`, and `03_agentic_performance`
into a single picture of how an inference/agent system fits together.

```bash
05_system_diagrams/
├── README.md                       ← This file
├── inference_pipeline.md            ← Implemented — sequence diagram; headline latency claim since retracted
├── batching_architecture.mmd        ← Implemented — real measured latencies, flowchart
└── container_orchestration.md       ← Implemented — proposed deployment design, clearly marked as not built
```

## inference_pipeline.md ⚠️ Implemented — headline number since retracted

A sequence diagram of a single inference request (tokenize → first forward
pass → KV-cache-backed decode loop). Originally annotated with a measured
4.42s first-token vs. 0.09s cached next-token latency (48.1x speedup) from
[`01_inference_profiling/RESULTS_kv_cache_analysis.md`](../01_inference_profiling/RESULTS_kv_cache_analysis.md).
That speedup does not reproduce — re-running the benchmark repeatedly shows
~0.6–1.2x, essentially no cache benefit, traced to a real methodology gap
in the script (see the results file's Limitations section). The diagram
still shows the general mechanism, which is real and well-documented in LLM
serving, but the timing annotations have been removed since this repo's own
measurement of them isn't currently trustworthy.

## batching_architecture.mmd ✅ Implemented

A flowchart of how concurrent prompts get grouped into a batch before the
forward pass, annotated with the real per-batch-size throughput measured in
[`01_inference_profiling/RESULTS_batch_size.md`](../01_inference_profiling/RESULTS_batch_size.md):
tokens/sec climbs from ~32 (batch 1) to ~113 (batch 8) on this CPU-only
machine, even as per-batch latency also rises. (An earlier version of this
diagram cited latency-only numbers from an older version of the underlying
script that no longer exists — updated to match its current behavior.)

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
each cost shows up — per-batch-size throughput, and (as a proposal) how
these pieces would sit inside a production deployment alongside the agent
loop from `03_agentic_performance`. (The first-token-vs-cached-token
latency diagram was meant to do the same for KV-cache effects, but this
repo's own benchmark of that effect doesn't currently hold up — see
`inference_pipeline.md` above.)

## Rendering

All diagrams use [Mermaid](https://mermaid.js.org/) syntax. GitHub renders
` ```mermaid ` fences in `.md` files natively. `batching_architecture.mmd`
is raw Mermaid source (no markdown wrapper) — view it with the
[Mermaid Live Editor](https://mermaid.live/) or a Mermaid-aware editor
extension if your viewer doesn't render `.mmd` files directly.

---

_See the top-level [README.md](../README.md) for how this section fits into
the overall repo._
