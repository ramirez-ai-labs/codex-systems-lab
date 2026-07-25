# 04 — Research Reproductions

Small, beginner-friendly reproductions of findings from AI-assisted coding
research.

```bash
04_research_reproductions/
├── README.md                                 ← This file
├── findings_summary.md                       ← Planned, empty
├── paper_1_edit_distance_vs_acceptance/       ← Implemented, notebook runs end to end
└── paper_2_rag_scaling_laws/                  ← Scaffolded, empty
```

## paper_1_edit_distance_vs_acceptance ✅ Implemented

Reproduces the finding that AI code suggestions closer to the developer's
final code (lower edit distance) are more likely to be accepted, using
synthetic data and a hand-picked acceptance model. See its own
[README.md](paper_1_edit_distance_vs_acceptance/README.md) for the full
writeup and [RESULTS.md](paper_1_edit_distance_vs_acceptance/RESULTS.md) for
the real measured output — acceptance rate falls from 97.8% at very-low edit
distance to 0.0% at high edit distance, but the results section is explicit
that this shape was built into the model by construction, not discovered
from any real dataset.

## paper_2_rag_scaling_laws 📋 Planned

Planned reproduction of RAG retrieval-quality scaling effects. Not started —
no README, notebook, or results yet.

---

_Both reproductions are intentionally small and CPU-friendly, matching the
rest of this lab. See the top-level [README.md](../README.md) for how this
section fits into the overall repo._
