# Contributing to codex-systems-lab

Thank you for interest in contributing to this systems measurement lab! This guide explains how to add new benchmarks, reproduce experiments, and document your work.

---

## **Quick Start for Contributors**

1. Fork/clone the repo
2. Create a feature branch: `git checkout -b feature/my-benchmark`
3. Add your benchmark in the appropriate numbered section (01-05)
4. Run and document results
5. Submit pull request with RESULTS.md

---

## **Adding New Benchmarks**

### **Which Section?**

- **01_inference_profiling/** — Latency, throughput, batching, quantization, KV-cache
- **02_light_finetuning/** — Fine-tuning pipelines, training, evaluation
- **03_agentic_performance/** — Agent loop overhead, tool latency, retry costs
- **04_research_reproductions/** — Reproducing published papers or findings
- **05_system_diagrams/** — System architecture and performance diagrams

### **Benchmark Template**

Create a new script following this pattern:

```python
"""
Benchmark: [Name]
================
Brief description of what is measured and why.

Methodology:
- What is being tested?
- Hardware/environment assumptions
- How many runs?
- What metric is collected?

Cautions:
- Known limitations or gotchas
- Results may vary by hardware
"""

from pathlib import Path
import time
import json

# ===============================================================
# SETUP
# ===============================================================

def setup():
    """Initialize model, data, environment."""
    # Load model, prepare data, etc.
    pass

# ===============================================================
# BENCHMARK
# ===============================================================

def run_benchmark(runs: int = 3):
    """Run the benchmark multiple times."""
    results = []
    
    for run_num in range(runs):
        start = time.time()
        # ... actual work ...
        elapsed = time.time() - start
        results.append(elapsed)
        print(f"Run {run_num+1}: {elapsed:.2f}s")
    
    return results

# ===============================================================
# ANALYSIS & OUTPUT
# ===============================================================

def analyze_results(results):
    """Compute statistics and document findings."""
    import numpy as np
    
    return {
        "mean": float(np.mean(results)),
        "std": float(np.std(results)),
        "min": float(np.min(results)),
        "max": float(np.max(results)),
        "runs": len(results)
    }

def main():
    print("Running benchmark...")
    setup()
    results = run_benchmark(runs=3)
    stats = analyze_results(results)
    
    print("\nResults:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Save to JSON
    output_file = Path(__file__).parent / f"{Path(__file__).stem}_results.json"
    with open(output_file, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\n✅ Wrote results to {output_file}")

if __name__ == "__main__":
    main()
```

---

## **Documentation: RESULTS.md**

Every benchmark must have a companion `RESULTS_*.md` file documenting:

```markdown
# Benchmark: [Name]

## Setup
- Hardware: CPU/GPU specs
- Python version, key libraries
- Model or dataset info

## Methodology
- What was measured
- How many runs
- Why this metric matters

## Results
[Raw numbers, tables, or figures]

## Findings
- What you learned
- Performance implications
- Surprising results

## Known Limitations
- What didn't work
- Assumptions that may not hold
- Where this would fail

## Next Steps
- How to improve this benchmark
- Related measurements to try
```

**Key principle:** Document failures and variance as prominently as successes. A benchmark that documents why it failed is more valuable than one that hides edge cases.

---

## **Code Style**

- Use type hints on function signatures
- Write docstrings with sections: Description, Args, Returns
- Add inline comments explaining *why*, not just *what*
- Use descriptive variable names (e.g., `latency_ms` not `l`)
- Separate logic into sections with `# ====== SECTION NAME ======` headers

---

## **Testing Your Work**

Before submitting:

1. **Run your benchmark:** `python my_benchmark.py`
2. **Verify output files:** Check RESULTS_*.md and JSON outputs exist
3. **Test on different hardware** (if possible) to catch environment-specific issues
4. **Document variance:** Run multiple times and report spread

---

## **Commit Message Guidelines**

```
[type] Brief description

Longer explanation of what and why.

Example:
[feature] Add KV-cache latency benchmark

Measures single-token vs multi-token inference latency
with and without KV-cache to quantify real speedup.
Results show 0.6-1.2x, not the claimed 48x.
```

Types: `[feature]`, `[fix]`, `[docs]`, `[refactor]`, `[test]`

---

## **Asking Questions**

- **Setup issues?** Check hardware specs in RESULTS.md files—your environment might differ
- **Benchmark design?** Open an issue describing what you want to measure and why
- **Results surprising?** Document it! Unexpected findings are valuable

---

## **Code of Conduct**

- Be honest about limitations and failures
- Prefer measured results over theory
- Document assumptions explicitly
- Welcome contributions from all backgrounds

Thank you for helping build rigorous system measurement!
