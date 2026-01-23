"""
abtest/stats.py

Core statistical utilities for A/B testing.

Dependencies:
  - numpy
  - scipy

What’s included:
  - Conversion (proportions): effect size, z-test, Wald CI
  - Continuous metrics (means): effect size, Welch t-test, t-based CI
  - Bootstrap: percentile CI + bootstrap p-value for mean difference
  - Ratio metrics: delta-method CI for ratio-of-means (optional but useful)
  - Multiple testing: Benjamini–Hochberg FDR
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal, Optional, Tuple, Dict, Any, List

import numpy as np
from scipy import stats


# -------------------------
# Small helpers
# -------------------------

def _as_1d_float(x: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(x), dtype=float)
    if arr.ndim != 1:
        raise ValueError("Input must be 1D.")
    return arr

def _safe_rel_lift(treat: float, control: float) -> float:
    if control == 0:
        return np.inf if treat != 0 else 0.0
    return (treat - control) / control

def _two_sided_alpha(alpha: float, two_sided: bool) -> float:
    if not (0 < alpha < 1):
        raise ValueError("alpha must be in (0,1)")
    return alpha / 2 if two_sided else alpha


# -------------------------
# Proportions (conversion)
# -------------------------

@dataclass(frozen=True)
class ProportionResult:
    p_control: float
    p_treat: float
    abs_lift: float
    rel_lift: float
    z: float
    p_value: float
    ci: Tuple[float, float]
    alpha: float
    method: str

def proportion_effect(x_control: int, n_control: int,
                      x_treat: int, n_treat: int) -> Dict[str, float]:
    if n_control <= 0 or n_treat <= 0:
        raise ValueError("n_control and n_treat must be > 0")
    p_c = x_control / n_control
    p_t = x_treat / n_treat
    return {
        "p_control": p_c,
        "p_treat": p_t,
        "abs_lift": p_t - p_c,
        "rel_lift": _safe_rel_lift(p_t, p_c),
    }

def ztest_proportions(
    x_control: int, n_control: int,
    x_treat: int, n_treat: int,
    alpha: float = 0.05,
    alternative: Literal["two-sided", "less", "greater"] = "two-sided",
    pooled: bool = True,
) -> ProportionResult:
    """
    Z-test for difference in proportions.
    - pooled=True uses pooled variance under H0 (typical for hypothesis testing)
    - CI uses unpooled standard error (common in reporting)
    """
    eff = proportion_effect(x_control, n_control, x_treat, n_treat)
    p_c, p_t = eff["p_control"], eff["p_treat"]
    diff = eff["abs_lift"]

    # Standard error for test
    if pooled:
        p_pool = (x_control + x_treat) / (n_control + n_treat)
        se_test = np.sqrt(p_pool * (1 - p_pool) * (1 / n_control + 1 / n_treat))
    else:
        se_test = np.sqrt(p_c * (1 - p_c) / n_control + p_t * (1 - p_t) / n_treat)

    if se_test == 0:
        z = 0.0
        p_value = 1.0
    else:
        z = diff / se_test
        if alternative == "two-sided":
            p_value = 2 * stats.norm.sf(abs(z))
        elif alternative == "greater":
            p_value = stats.norm.sf(z)
        elif alternative == "less":
            p_value = stats.norm.cdf(z)
        else:
            raise ValueError("alternative must be 'two-sided', 'less', or 'greater'")

    # Wald CI (unpooled)
    two_sided = (alternative == "two-sided")
    a = _two_sided_alpha(alpha, two_sided)
    zcrit = stats.norm.ppf(1 - a)
    se_ci = np.sqrt(p_c * (1 - p_c) / n_control + p_t * (1 - p_t) / n_treat)
    ci = (diff - zcrit * se_ci, diff + zcrit * se_ci)

    return ProportionResult(
        p_control=p_c,
        p_treat=p_t,
        abs_lift=diff,
        rel_lift=eff["rel_lift"],
        z=float(z),
        p_value=float(p_value),
        ci=(float(ci[0]), float(ci[1])),
        alpha=alpha,
        method=f"z-test proportions (pooled={pooled}), Wald CI",
    )


# -------------------------
# Means (continuous metrics)
# -------------------------

@dataclass(frozen=True)
class MeanResult:
    mean_control: float
    mean_treat: float
    abs_lift: float
    rel_lift: float
    t: float
    df: float
    p_value: float
    ci: Tuple[float, float]
    alpha: float
    method: str

def mean_effect(x_control: Iterable[float], x_treat: Iterable[float]) -> Dict[str, float]:
    xc = _as_1d_float(x_control)
    xt = _as_1d_float(x_treat)
    mc = float(np.mean(xc))
    mt = float(np.mean(xt))
    return {
        "mean_control": mc,
        "mean_treat": mt,
        "abs_lift": mt - mc,
        "rel_lift": _safe_rel_lift(mt, mc),
    }

def welch_ttest_means(
    x_control: Iterable[float],
    x_treat: Iterable[float],
    alpha: float = 0.05,
    alternative: Literal["two-sided", "less", "greater"] = "two-sided",
) -> MeanResult:
    """
    Welch's t-test for difference in means + t-based CI.
    """
    xc = _as_1d_float(x_control)
    xt = _as_1d_float(x_treat)
    if len(xc) < 2 or len(xt) < 2:
        raise ValueError("Need at least 2 observations per group for Welch t-test.")

    eff = mean_effect(xc, xt)
    diff = eff["abs_lift"]

    # scipy ttest supports alternative
    t_res = stats.ttest_ind(xt, xc, equal_var=False, alternative=alternative)  # treat - control
    t_stat = float(t_res.statistic)
    p_value = float(t_res.pvalue)

    # Welch-Satterthwaite df + CI
    vc = np.var(xc, ddof=1) / len(xc)
    vt = np.var(xt, ddof=1) / len(xt)
    se = np.sqrt(vc + vt)

    # df
    numerator = (vc + vt) ** 2
    denom = (vc ** 2) / (len(xc) - 1) + (vt ** 2) / (len(xt) - 1)
    df = float(numerator / denom) if denom > 0 else float(len(xc) + len(xt) - 2)

    two_sided = (alternative == "two-sided")
    a = _two_sided_alpha(alpha, two_sided)
    tcrit = stats.t.ppf(1 - a, df)

    ci = (diff - tcrit * se, diff + tcrit * se)

    return MeanResult(
        mean_control=eff["mean_control"],
        mean_treat=eff["mean_treat"],
        abs_lift=diff,
        rel_lift=eff["rel_lift"],
        t=t_stat,
        df=df,
        p_value=p_value,
        ci=(float(ci[0]), float(ci[1])),
        alpha=alpha,
        method="Welch t-test + t CI",
    )


# -------------------------
# Bootstrap (robust inference)
# -------------------------

@dataclass(frozen=True)
class BootstrapResult:
    abs_lift: float
    rel_lift: float
    ci: Tuple[float, float]
    p_value: Optional[float]
    n_boot: int
    method: str

def bootstrap_ci_diff_means(
    x_control: Iterable[float],
    x_treat: Iterable[float],
    alpha: float = 0.05,
    n_boot: int = 5000,
    seed: Optional[int] = None,
    two_sided: bool = True,
) -> BootstrapResult:
    """
    Percentile bootstrap CI for mean difference (treat - control).
    Robust for heavy-tailed metrics like revenue.
    """
    xc = _as_1d_float(x_control)
    xt = _as_1d_float(x_treat)
    if len(xc) == 0 or len(xt) == 0:
        raise ValueError("Empty input.")

    rng = np.random.default_rng(seed)

    mc = np.mean(xc)
    mt = np.mean(xt)
    diff_obs = float(mt - mc)
    rel = float(_safe_rel_lift(mt, mc))

    boot_diffs = np.empty(n_boot, dtype=float)
    n_c, n_t = len(xc), len(xt)

    for b in range(n_boot):
        bc = rng.choice(xc, size=n_c, replace=True)
        bt = rng.choice(xt, size=n_t, replace=True)
        boot_diffs[b] = np.mean(bt) - np.mean(bc)

    a = _two_sided_alpha(alpha, two_sided)
    lo = float(np.quantile(boot_diffs, a))
    hi = float(np.quantile(boot_diffs, 1 - a))

    return BootstrapResult(
        abs_lift=diff_obs,
        rel_lift=rel,
        ci=(lo, hi),
        p_value=None,
        n_boot=n_boot,
        method="Percentile bootstrap CI for mean diff",
    )

def bootstrap_pvalue_diff_means(
    x_control: Iterable[float],
    x_treat: Iterable[float],
    n_boot: int = 5000,
    seed: Optional[int] = None,
    two_sided: bool = True,
) -> float:
    """
    Simple bootstrap p-value for mean difference under the null via permutation.
    (Permutation test is often preferable for p-values.)

    Returns a p-value for H0: mean_treat == mean_control.
    """
    xc = _as_1d_float(x_control)
    xt = _as_1d_float(x_treat)
    rng = np.random.default_rng(seed)

    diff_obs = float(np.mean(xt) - np.mean(xc))

    combined = np.concatenate([xc, xt])
    n_c = len(xc)

    more_extreme = 0
    for _ in range(n_boot):
        rng.shuffle(combined)  # in-place
        bc = combined[:n_c]
        bt = combined[n_c:]
        diff_b = float(np.mean(bt) - np.mean(bc))
        if two_sided:
            if abs(diff_b) >= abs(diff_obs):
                more_extreme += 1
        else:
            if diff_b >= diff_obs:
                more_extreme += 1

    # +1 smoothing for stability
    return (more_extreme + 1) / (n_boot + 1)


# -------------------------
# Ratio metrics (optional)
# -------------------------

@dataclass(frozen=True)
class RatioResult:
    ratio_control: float
    ratio_treat: float
    abs_lift: float
    rel_lift: float
    ci: Tuple[float, float]
    alpha: float
    method: str

def delta_method_ci_ratio_of_means(
    num_control: Iterable[float], den_control: Iterable[float],
    num_treat: Iterable[float], den_treat: Iterable[float],
    alpha: float = 0.05,
    two_sided: bool = True,
) -> RatioResult:
    """
    Delta-method CI for difference in ratio-of-means:
      R = mean(num)/mean(den)
    Common for CTR-like metrics when you have numerator/denominator per unit.

    Assumes i.i.d units; for clustered/session data, you'd want cluster-robust SEs.
    """
    nc = _as_1d_float(num_control)
    dc = _as_1d_float(den_control)
    nt = _as_1d_float(num_treat)
    dt = _as_1d_float(den_treat)
    if len(nc) != len(dc) or len(nt) != len(dt):
        raise ValueError("Numerator and denominator arrays must match lengths per group.")

    # ratio-of-means
    mu_nc, mu_dc = float(np.mean(nc)), float(np.mean(dc))
    mu_nt, mu_dt = float(np.mean(nt)), float(np.mean(dt))
    if mu_dc == 0 or mu_dt == 0:
        raise ValueError("Mean denominator is zero; ratio undefined.")

    r_c = mu_nc / mu_dc
    r_t = mu_nt / mu_dt
    diff = r_t - r_c

    # Delta method for ratio mean(num)/mean(den)
    # Approx Var(r) ≈ (1/mu_den^2)*Var(num̄) + (mu_num^2/mu_den^4)*Var(den̄) - 2*(mu_num/mu_den^3)*Cov(num̄, den̄)
    def ratio_var(num: np.ndarray, den: np.ndarray) -> float:
        n = len(num)
        mu_n = float(np.mean(num))
        mu_d = float(np.mean(den))
        var_nbar = float(np.var(num, ddof=1) / n)
        var_dbar = float(np.var(den, ddof=1) / n)
        cov = float(np.cov(num, den, ddof=1)[0, 1] / n)
        return (var_nbar / (mu_d ** 2)
                + (mu_n ** 2) * var_dbar / (mu_d ** 4)
                - 2 * mu_n * cov / (mu_d ** 3))

    var_rc = ratio_var(nc, dc)
    var_rt = ratio_var(nt, dt)
    se = math.sqrt(max(0.0, var_rc + var_rt))

    a = _two_sided_alpha(alpha, two_sided)
    zcrit = stats.norm.ppf(1 - a)
    ci = (diff - zcrit * se, diff + zcrit * se)

    return RatioResult(
        ratio_control=r_c,
        ratio_treat=r_t,
        abs_lift=diff,
        rel_lift=_safe_rel_lift(r_t, r_c),
        ci=(float(ci[0]), float(ci[1])),
        alpha=alpha,
        method="Delta-method CI for ratio-of-means diff",
    )


# -------------------------
# Multiple testing
# -------------------------

def benjamini_hochberg(p_values: Iterable[float], q: float = 0.1) -> Dict[str, Any]:
    """
    Benjamini–Hochberg FDR control.

    Returns:
      - rejected: boolean array aligned to input order
      - cutoff_p: largest p-value that is rejected (or None)
      - q: target FDR
    """
    p = np.asarray(list(p_values), dtype=float)
    if p.ndim != 1:
        raise ValueError("p_values must be 1D.")
    if not (0 < q < 1):
        raise ValueError("q must be in (0,1).")

    m = len(p)
    order = np.argsort(p)
    p_sorted = p[order]
    thresh = q * (np.arange(1, m + 1) / m)

    passed = p_sorted <= thresh
    if not np.any(passed):
        return {"rejected": np.zeros(m, dtype=bool), "cutoff_p": None, "q": q}

    k = int(np.max(np.where(passed)[0]))
    cutoff = float(p_sorted[k])

    rejected_sorted = p_sorted <= cutoff
    rejected = np.zeros(m, dtype=bool)
    rejected[order] = rejected_sorted

    return {"rejected": rejected, "cutoff_p": cutoff, "q": q}
