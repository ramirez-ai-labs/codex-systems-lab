# 04 — Research Reproductions ✅ Both implemented (synthetic)

Small, beginner-friendly reproductions of documented findings from
AI-assisted coding and RAG research, using synthetic data and hand-picked
models rather than real telemetry or published datasets.

```bash
04_research_reproductions/
├── README.md                                 ← This file
├── findings_summary.md                       ← Planned, empty
├── paper_1_edit_distance_vs_acceptance/       ← Implemented, notebook runs end to end
└── paper_2_rag_scaling_laws/                  ← Implemented, notebook runs end to end
```

---

## Why These Reproductions Matter

Papers and blog posts report findings; reproducing the *shape* of a finding
— even on synthetic data — forces you to actually build the measurement
pipeline: hypothesis → data → sampled outcomes → binned measurement →
visualization. That pipeline is the reusable skill; the specific numbers
below are illustrative, not a replacement for the real data or the real
paper.

---

## Reproductions in This Folder

### 1️⃣ `paper_1_edit_distance_vs_acceptance` — Edit Distance vs. Acceptance Rate ✅ Implemented

`replicate_experiment.ipynb` runs end to end
(`jupyter nbconvert --to notebook --execute`); every number below is real
output from that run, re-executed and confirmed to match exactly.

Key ideas:
- Reproduces the finding that AI code suggestions closer to the developer's
  final code (lower edit distance) are more likely to be accepted
- **Acceptance rate falls from 97.8% at very-low edit distance to 0.0% at
  high edit distance** — a real, reproducible measurement of a simulation
- The data is synthetic and the acceptance-probability curve is a
  hand-picked logistic function, **not fit to any real dataset** —
  `RESULTS.md` spells out that limitation explicitly

See its own [README.md](paper_1_edit_distance_vs_acceptance/README.md) for
the full writeup and
[RESULTS.md](paper_1_edit_distance_vs_acceptance/RESULTS.md) for the
complete measured output.

### 2️⃣ `paper_2_rag_scaling_laws` — Retrieval Recall & Answer Accuracy vs. k ✅ Implemented

`replicate_experiment.ipynb` runs end to end; every number below is real
output, re-executed and confirmed to match exactly.

Key ideas:
- Reproduces the pattern that retrieval recall improves with k (retrieved
  chunks) but with diminishing returns, and that a larger corpus needs a
  larger k for the same recall — **a real measurement** over synthetic
  embeddings, not a hand-typed curve
- Downstream answer accuracy does **not** keep climbing with k — it peaks
  and then falls off as irrelevant retrieved chunks dilute the signal
- The recall measurement is real; the accuracy-vs-k relationship is built
  on top of a hand-picked "context dilution" model, **not fit to any real
  RAG system** — `RESULTS.md` spells out that limitation explicitly

See its own [README.md](paper_2_rag_scaling_laws/README.md) for the full
writeup and [RESULTS.md](paper_2_rag_scaling_laws/RESULTS.md) for the
complete measured output.

---

## How to Use This Section

Each notebook is independent and can be run/re-executed on its own:

```bash
jupyter nbconvert --to notebook --execute --inplace paper_1_edit_distance_vs_acceptance/replicate_experiment.ipynb
jupyter nbconvert --to notebook --execute --inplace paper_2_rag_scaling_laws/replicate_experiment.ipynb
```

Both are intentionally small and CPU-friendly, matching the rest of this
lab. See the top-level [README.md](../README.md) for how this section fits
into the overall repo.
