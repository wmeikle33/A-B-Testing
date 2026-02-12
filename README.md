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
pip install -r requirements.txt

### 1) Setup

ab-testing/
├── README.md
├── requirements.txt
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
