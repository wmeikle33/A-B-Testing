# Experiment Design: <Feature / Test Name>

## 1. Overview & Goal
- **Goal:** <business outcome> by changing <user behavior>.
- **Link(s):** Product spec, tracking doc, dashboard.

## 2. Hypotheses
- **H0:** No difference in <primary metric> between A and B.
- **H1:** Variant B increases <primary metric> vs A.

## 3. Randomization & Exposure
- **Unit:** <user / session / account>
- **Bucketing:** hash(user_id) → arm (sticky across sessions)
- **Eligibility:** users that <criteria>; exclusions: <bot/QA/staff/etc>
- **Exposure rule:** count a user in arm X after <first eligible event>

## 4. Variants & Allocation
- **Arms:** A (control), B (treatment)
- **Split:** 50/50; **Ramp:** 10% → 50% → 100% (daily)

## 5. Metrics
### 5.1 Primary
- **Name:** <e.g., Conversion (7d)>
- **Definition:** users with event `<purchase>` within 7 days / eligible users
- **Direction:** higher is better

### 5.2 Guardrails
- Error rate (≤ baseline + 0.2pp), P95 latency (≤ baseline + 5%), Uninstall rate (no increase)

### 5.3 Secondary
- Revenue per user (7d), Session length, Retention D7 (report only)

## 6. Data & Logging
- **Events:** `<view>`, `<click>`, `<purchase>` with fields: user_id, ts, device, price, arm
- **Joins:** user_id, day bucket
- **Source:** warehouse tables <links> / schema <links>

## 7. Statistical Plan
- **Tests:** z-test for proportions; Welch t-test for revenue; MWU/Bootstrap if non-normal/outliers
- **CI:** 95% two-sided; **α:** 0.05
- **Variance reduction:** CUPED with covariate = pre-period revenue (last 28d)

## 8. Power & Sample Size
- **Baseline:** p0 = 0.08; **MDE:** +1.2pp
- **Power:** 80%; **α:** 0.05 two-sided
- **Required n/arm:** <calc> ; **Expected duration:** <X> days @ <traffic/day>

## 9. Diagnostics & Data Quality
- **SRM check:** χ² p < 0.01 ⇒ pause & investigate
- **Balance:** standardized diff < 0.1 for key covariates
- **Duplicates/cross-assignment:** must be 0 after filtering

## 10. Monitoring & Stopping
- **Policy:** <No peeking> OR <5 looks with O’Brien–Fleming spending>
- **Early stop rule:** |z| ≥ boundary at a look ⇒ stop; else run full duration

## 11. Decision Criteria
- Ship if: 95% CI of (B−A) on primary > 0 **and** all guardrails within thresholds.
- Otherwise: do not ship; document learnings and next steps.

## 12. Risks & Mitigations
- Seasonality (run ≥ full weeks); novelty (monitor post-ship 2 weeks); bot traffic (filters)

## 13. Privacy & Ethics
- No PII in exports; comply with <GDPR/CCPA/…> where applicable.

## 14. Rollout Plan
- After “ship” decision: 10% → 50% → 100%, with rollback if guardrail violated.

## 15. Heterogeneity (Pre-specified)
- Segments: device {mobile, desktop}, geo {US, EU, APAC}; adjust for multiple testing (BH/FDR).

## 16. Timeline, Roles, Approvals
- **Owner:** <name> | **Analyst:** <name> | **Reviewer:** <name>
- **Planned run:** <start> → <end>
- **Version history:** link to PRs / doc revisions
