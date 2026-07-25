# 04 — Research Reproductions

Small, beginner-friendly reproductions of findings from AI-assisted coding
research. Both are scaffolded; **neither has an implemented notebook yet.**

```bash
04_research_reproductions/
├── README.md                                 ← This file
├── findings_summary.md                       ← Planned, empty
├── paper_1_edit_distance_vs_acceptance/       ← Scaffolded, has a hypothesis writeup
└── paper_2_rag_scaling_laws/                  ← Scaffolded, empty
```

## paper_1_edit_distance_vs_acceptance

Reproduces the finding that AI code suggestions closer to the developer's
final code (lower edit distance) are more likely to be accepted. See its own
[README.md](paper_1_edit_distance_vs_acceptance/README.md) for the full
writeup and [RESULTS.md](paper_1_edit_distance_vs_acceptance/RESULTS.md) for
the hypothesized outcome — clearly marked as not yet measured, since
`replicate_experiment.ipynb` is still empty.

## paper_2_rag_scaling_laws

Planned reproduction of RAG retrieval-quality scaling effects. Not started —
no README, notebook, or results yet.

---

_Both reproductions are intentionally small and CPU-friendly, matching the
rest of this lab. See the top-level [README.md](../README.md) for how this
section fits into the overall repo._
