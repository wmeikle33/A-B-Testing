# Statistical Assumptions & Methods

## Scope
This test uses: z-test for proportions, Welch t-test for continuous metrics, Mann–Whitney U and bootstrap CIs as robustness checks, CUPED for variance reduction, and optional sequential monitoring.

## Experiment-level assumptions
- Randomization is correct (SRM p ≥ 0.01).
- No interference (SUTVA), stable unit, no cross-assignment.
- No differential attrition; consistent exposure rules.

## Metric assumptions
- Definitions/windows pre-specified; no leakage.
- Heavy tails handled via winsorization/log or bootstrap.

## Methods

### Proportions (primary)
- Test: two-sample z (pooled).
- Report: p1, p2, diff, Wilson CIs, z, p, n/arm.
- If small n: exact/binomial or bootstrap.

### Continuous (revenue)
- Test: Welch t-test.
- If heavy-tailed: bootstrap CI or MWU; note winzor/log if applied.

### CUPED (variance reduction)
- Covariate: pre-period metric (defined here: …).
- Report θ and variance reduction %; adjusted Welch results.

### Sequential (if used)
- Spending plan: O’Brien–Fleming (k looks).
- Report look index, boundary, stop/no stop.

## Multiple comparisons
- Pre-specified segments only; control via BH/FDR (or Bonferroni).
- Report adjusted p-values.

## Clustered designs (if any)
- Analyze at cluster level or use cluster-robust SE; state ICC considerations.

## Diagnostics we will run
- SRM χ²; balance (std diff < 0.1); duplicates and cross-assignment counts;
- Distribution plots for revenue; outlier policy.

## Power & α
- α = 0.05 (two-sided). Power = 80%. MDE planned at baseline p0 = … ; n/arm = …

## Reporting format
For each metric: n_A, n_B, effect (B−A), 95% CI, p, method, notes on assumptions/flags, decision vs guardrails.
