# A/B Testing Toolkit (Design + Analysis + Guardrails)

A practical, reproducible A/B testing repo that shows both **how to design experiments** and **how to analyze them safely** (sanity checks, variance reduction, robust CIs). Includes an end-to-end case study plus reusable code you can drop into future projects.

## What’s in this repo

- **End-to-end experiment analysis** (notebook + script)
- **Reusable A/B testing utilities**
  - Frequentist: difference-in-proportions, t-test, bootstrap CIs
  - Bayesian: Beta–Binomial for conversion (optional)
  - **CUPED** variance reduction (pre-period covariate)
  - **Sanity checks**: Sample Ratio Mismatch (SRM), balance checks
- **Power & sample size** helpers (MDE, duration estimation)
- **Synthetic data generator** so anyone can run the repo without private data
- **Tests + CI** to validate stats functions

> This repo is meant for product analytics / experimentation workflows (conversion, revenue/ARPU, retention-style metrics).

---

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .

abtest
# or
abtest --n 10000 --p-control 0.10 --p-treat 0.105

```

## Repo Setup

```

ab-testing/
├── README.md
├── .gitignore
├── .github/
│   └── workflows/
│       └── ci.yml
├── data/
│   ├── sample.csv
│   └── README.md
├── notebooks/
│   └── 01_end_to_end_ab_test.ipynb
├── abtest/
│   ├── __init__.py
│   ├── metrics.py
│   ├── stats.py
│   ├── sanity.py
│   ├── cuped.py
│   ├── power.py
│   └── utils.py
├── scripts/
│   ├── generate_data.py
│   └── run_analysis.py
├── docs/
│   ├── experiment_design.md
│   ├── metric_definitions.md
│   └── pitfalls_checklist.md
└── tests/
    ├── test_stats.py
    ├── test_cuped.py
    └── test_sanity.py

```

## Quick end-to-end example (A/B test with confidence interval)

Assume you have event-level data like:

| user_id | variant | converted |
|--------:|:--------|----------:|
| 1       | control | 0         |
| 2       | control | 1         |
| 3       | treatment | 0       |
| 4       | treatment | 1       |

```python
import pandas as pd
from abtest import analyze_proportions  # adjust import to your package

# 1) Load data
df = pd.read_csv("data/experiment.csv")

# 2) Define metric
# Here the metric is conversion rate: mean(converted) per variant

# 3) Run test
result = analyze_proportions(
    df=df,
    variant_col="variant",
    outcome_col="converted",
    control_label="control",
    treatment_label="treatment",
    alpha=0.05,
)

# 4) Print decision + CI
print(f"Control CR:   {result.control_rate:.4f}")
print(f"Treatment CR: {result.treatment_rate:.4f}")
print(f"Lift:         {result.lift:.2%}")
print(f"95% CI:       [{result.ci_low:.2%}, {result.ci_high:.2%}]")
print(f"p-value:      {result.p_value:.4g}")
print(f"Decision:     {result.decision}")
