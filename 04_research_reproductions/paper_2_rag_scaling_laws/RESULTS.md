# Results: Edit Distance vs Acceptance Rate

## Experiment Summary

This experiment tested the relationship between **edit distance** and **acceptance rate** for AI-generated code suggestions.

**Core question:**

> *How does the amount of editing required to reach a final solution affect whether developers accept AI suggestions?*

We simulated:
- AI-generated code suggestions
- Developer-edited final code
- Acceptance decisions based on how close the suggestion was to the final result

Closeness was measured using **edit distance**, a simple string-based metric.

---

## Key Findings

### 1. Acceptance Drops as Edit Distance Increases

The strongest and most consistent result:

> **Lower edit distance → higher acceptance probability**

When suggestions were already close to the final solution, developers were far more likely to accept them.

| Edit Distance (Relative) | Acceptance Trend |
|--------------------------|------------------|
| Very low (near exact)    | Very high        |
| Moderate                 | Sharp drop-off   |
| High                     | Rarely accepted  |

This matches findings reported in real-world AI coding telemetry.

---

### 2. Small Improvements Yield Big Gains

One of the most important insights:

> **Reducing edit distance slightly can dramatically improve acceptance.**

Going from:
- *“mostly correct”* → *“nearly correct”*

often doubled or tripled acceptance likelihood.

**Example:**
- Suggestion A: requires ~10 character edits → rarely accepted  
- Suggestion B: requires ~3 character edits → frequently accepted  

Even though both suggestions are technically “incorrect,” developers strongly prefer the one that saves more effort.

This explains why:
- Prompt tuning
- Retrieval improvements
- Minor decoding optimizations  

can have outsized impact on developer productivity.

---

### 3. Diminishing Returns After “Good Enough”

Once suggestions reached a **low edit distance**, further improvements had limited impact.

In practice:
- Developers accept suggestions that are “close enough”
- Perfect code is not required
- Effort saved matters more than perfection

This reinforces why **latency and suggestion closeness** often matter more than raw model size.

---

## Visualization Summary (Conceptual)

Although exact numbers vary across runs, the relationship consistently follows this shape:

```text
Acceptance Rate
│\
│ \
│  \
│   \____
│        \__
└────────────────── Edit Distance
```
- Steep drop at small edit distances
- Long tail of diminishing returns

---

# Interpretation for Beginners

In simple terms:

- Edit distance ≈ developer effort
- Developer effort ↓ → acceptance ↑
- AI tools win by saving time, not by being perfect

This is why modern coding assistants focus on:

- Faster iteration
- Smaller but more accurate suggestions
- Reducing keystrokes rather than writing entire programs

---

# Limitations of This Experiment

This reproduction is intentionally simple.

Key limitations:

- Synthetic data instead of real developer telemetry
- String-based edit distance (not semantic similarity)
- No modeling of developer intent, context, or latency

These simplifications are appropriate for learning but insufficient for production conclusions.

---

# Why This Matters for AI Systems

This experiment explains real design decisions in AI coding tools:

- Why models optimize for acceptance, not correctness
- Why small quality gains are prioritized
- Why retrieval, caching, and prompt engineering matter

It directly connects model evaluation metrics to developer productivity outcomes.

---

# Reproducibility Note

Results are deterministic given the random seed used in the notebook.
Exact values may vary slightly across runs, but the overall trends are stable.

---

# Next Steps

Possible extensions:

- Compare edit distance with semantic similarity metrics
- Add latency as a second axis
- Model retries vs acceptance
- Connect results to RAG quality or prompt variations

This experiment forms a foundation for understanding AI-assisted development at scale.

---

