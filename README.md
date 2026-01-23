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

### 1) Setup
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
