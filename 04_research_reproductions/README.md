# Paper Reproduction: Edit Distance vs Acceptance Rate

## 📄 What This Experiment Is About

This folder reproduces a **core finding from AI-assisted coding research**:

> **The closer an AI-generated code suggestion is to the final accepted code,  
> the more likely a developer is to accept it.**

That “closeness” is measured using **edit distance** — a simple but powerful
string-based metric.

This experiment helps answer:
- Why small improvements in code quality matter
- Why “almost right” suggestions are often accepted
- How evaluation metrics connect to real developer behavior

---

## 🧠 Key Concepts (Beginner-Friendly)

### What is *Edit Distance*?

Edit distance measures **how many small changes** are needed to turn one string
into another.

Examples:
- `foo()` → `foo()` → edit distance = **0**
- `foo()` → `bar()` → edit distance = **3**
- One deletion, insertion, or replacement = **1 edit**

In code generation:
- Low edit distance → model output is very close to what the developer wants
- High edit distance → developer has to rewrite a lot → likely rejected

---

### What is *Acceptance Rate*?

Acceptance rate is the fraction of suggestions that developers **keep** instead
of deleting or rewriting.

Example:
- 100 suggestions shown
- 65 accepted
→ acceptance rate = **65%**

---

## 🔬 What This Reproduction Demonstrates

We recreate the relationship:

| Edit Distance | Acceptance Likelihood |
|--------------|-----------------------|
| Very low     | Very high             |
| Medium       | Mixed                 |
| High         | Very low              |

This mirrors real findings from:
- GitHub Copilot studies
- Codex productivity research
- IDE telemetry analyses

---

## 🧪 Files in This Folder

```bash
paper_1_edit_distance_vs_acceptance/
├── README.md ← You are here
├── replicate_experiment.ipynb ← Step-by-step experiment
└── RESULTS.md ← Findings and interpretation
```

---

## 📓 `replicate_experiment.ipynb`

The notebook walks through:

1. Creating **synthetic code suggestions**
2. Simulating **developer-edited final code**
3. Computing **edit distance**
4. Mapping distance → acceptance probability
5. Visualizing the relationship

Everything is:
- Small
- CPU-friendly
- Heavily commented
- Designed for learning, not scale

---

## 📊 Expected Outcome (Intuition)

You should see:
- Acceptance rate drops sharply as edit distance increases
- Most “wins” come from **small improvements**
- Diminishing returns after a certain quality threshold

This explains why:
- Minor model improvements can produce big DX gains
- Perfect code is not required for usefulness
- Evaluation should focus on *closeness*, not perfection

---

## 🧩 Why This Matters for AI Coding Systems

This experiment connects **metrics → human behavior**:

- Edit distance → developer effort
- Developer effort → acceptance
- Acceptance → productivity gains

Modern systems use this insight to:
- Tune decoding strategies
- Rank suggestions
- Decide when to show or hide completions

---

## ⚠️ Important Notes

- This is a **conceptual reproduction**, not an exact paper replication
- Data is synthetic to keep the lab lightweight
- The goal is understanding, not leaderboard performance

---

## 🎓 Beginner Takeaway

> AI coding models don’t need to be perfect.  
> They need to be *close enough* to save time.

Edit distance is one of the simplest ways to measure that closeness.

---

## 🔗 Next Steps

After this experiment, consider:
- Comparing edit distance to semantic similarity
- Adding latency or retry cost into acceptance modeling
- Connecting this to RAG quality or prompt strategies

When you’re ready, move on to:

```bash
paper_2_rag_scaling_laws/
```
