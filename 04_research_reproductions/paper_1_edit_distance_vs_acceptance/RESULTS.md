# Hypothesized Results: Edit Distance vs Acceptance Rate

> ⚠️ **Status: not yet run.** `replicate_experiment.ipynb` in this folder is empty —
> no code has been written or executed for this reproduction. Everything below
> describes the *expected* outcome, drafted ahead of implementation as a design
> sketch for what the notebook should produce. Treat it as a hypothesis, not a
> finding. This file previously lived (misfiled) under `paper_2_rag_scaling_laws/`
> with no indication it was unexecuted — moved here and relabeled for accuracy.

## Experiment Summary

This experiment is intended to test the relationship between **edit distance** and
**acceptance rate** for AI-generated code suggestions.

**Core question:**

> *How does the amount of editing required to reach a final solution affect whether developers accept AI suggestions?*

The plan is to simulate:

- AI-generated code suggestions
- Developer-edited final code
- Acceptance decisions based on how close the suggestion was to the final result

Closeness would be measured using **edit distance**, a simple string-based metric.

---

## Expected Findings (Hypothesis)

### 1. Acceptance Should Drop as Edit Distance Increases

The expected core result:

> **Lower edit distance → higher acceptance probability**

When suggestions are close to the final solution, developers should be far more
likely to accept them — this is the general pattern reported in real-world AI
coding telemetry (GitHub Copilot studies, Codex productivity research), which is
what this reproduction is trying to demonstrate on synthetic data, not what it
has actually shown yet.

| Edit Distance (Relative) | Expected Acceptance Trend |
| --- | --- |
| Very low (near exact) | Very high |
| Moderate | Sharp drop-off |
| High | Rarely accepted |

---

### 2. Small Improvements Should Yield Big Gains

The hypothesis:

> **Reducing edit distance slightly could dramatically improve acceptance.**

Going from *"mostly correct"* → *"nearly correct"* is expected to noticeably
raise acceptance likelihood, even though both are technically "incorrect" —
developers are expected to prefer the suggestion that saves more effort.

If this holds, it would explain why prompt tuning, retrieval improvements, and
minor decoding optimizations can have outsized impact on developer productivity.

---

### 3. Diminishing Returns After "Good Enough"

Once suggestions reach a **low edit distance**, further improvements are expected
to have limited impact — developers accepting "close enough" suggestions rather
than requiring perfection.

---

## Conceptual Shape (Not Real Data)

```text
Acceptance Rate
│\
│ \
│  \
│   \____
│        \__
└────────────────── Edit Distance
```

This is an illustration of the expected relationship shape, not a plot of
measured output — no data has been generated yet.

---

## Limitations

- Nothing here is measured — the notebook does not exist yet
- Planned data is synthetic, not real developer telemetry
- Planned metric is string-based edit distance, not semantic similarity
- No modeling of developer intent, context, or latency is planned

---

## Next Steps

1. Implement `replicate_experiment.ipynb`: generate synthetic suggestions,
   compute edit distance, simulate acceptance decisions
2. Replace this file's hypothesized numbers with actual measured output
3. Compare edit distance with semantic similarity metrics
4. Add latency as a second axis
