"""
abtest/cuped.py

CUPED (Controlled-experiment Using Pre-Experiment Data)
Variance reduction for A/B testing using a pre-period covariate.

References:
  Y_adj = Y - theta * (X - mean(X))
  theta = Cov(Y, X) / Var(X)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple, Dict, Any, Literal

import numpy as np
from scipy import stats


def _as_1d_float(x: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(x), dtype=float)
    if arr.ndim != 1:
        raise ValueError("Input must be 1D.")
    return arr


@dataclass(frozen=True)
class CupedFit:
    theta: float
    x_mean: float
    corr_xy: float
    var_y: float
    var_y_adj: float
    var_reduction: float  # fraction reduction, e.g. 0.23 means 23% lower variance


def estimate_theta(x: Iterable[float], y: Iterable[float]) -> float:
    """
    theta = Cov(Y, X) / Var(X)
    """
    x = _as_1d_float(x)
    y = _as_1d_float(y)
    if len(x) != len(y):
        raise ValueError("x and y must have same length")
    if len(x) < 2:
        return 0.0

    vx = np.var(x, ddof=1)
    if vx == 0:
        return 0.0

    cov = np.cov(x, y, ddof=1)[0, 1]
    return float(cov / vx)


def cuped_adjust(
    y: Iterable[float],
    x: Iterable[float],
    theta: Optional[float] = None,
    center: Literal["global"] = "global",
) -> Tuple[np.ndarray, CupedFit]:
    """
    Returns (y_adj, fit_info)

    center="global": y_adj = y - theta*(x - mean(x))
    """
    x = _as_1d_float(x)
    y = _as_1d_float(y)
    if len(x) != len(y):
        raise ValueError("x and y must have same length")
    if len(x) < 2:
        # nothing meaningful to do
        y_adj = y.copy()
        fit = CupedFit(
            theta=0.0,
            x_mean=float(np.mean(x)) if len(x) else 0.0,
            corr_xy=float(np.nan),
            var_y=float(np.var(y)) if len(y) else 0.0,
            var_y_adj=float(np.var(y)) if len(y) else 0.0,
            var_reduction=0.0,
        )
        return y_adj, fit

    if theta is None:
        theta = estimate_theta(x, y)

    if center != "global":
        raise ValueError("Only center='global' is supported in this template.")

    x_mean = float(np.mean(x))
    y_adj = y - theta * (x - x_mean)

    corr = float(np.corrcoef(x, y)[0, 1]) if np.std(x, ddof=1) > 0 and np.std(y, ddof=1) > 0 else float(np.nan)
    var_y = float(np.var(y, ddof=1))
    var_y_adj = float(np.var(y_adj, ddof=1))
    var_reduction = 0.0 if var_y == 0 else float((var_y - var_y_adj) / var_y)

    fit = CupedFit(
        theta=float(theta),
        x_mean=x_mean,
        corr_xy=corr,
        var_y=var_y,
        var_y_adj=var_y_adj,
        var_reduction=var_reduction,
    )
    return y_adj, fit


@dataclass(frozen=True)
class CupedABResult:
    theta: float
    control_mean: float
    treat_mean: float
    abs_lift: float
    rel_lift: float
    t: float
    df: float
    p_value: float
    ci: Tuple[float, float]
    var_reduction: float


def cuped_ab_test(
    y_control: Iterable[float], x_control: Iterable[float],
    y_treat: Iterable[float], x_treat: Iterable[float],
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> CupedABResult:
    """
    Convenience wrapper:
      - estimate theta using pooled data
      - adjust outcomes in both groups
      - run Welch t-test on adjusted outcomes
      - report effect + CI + variance reduction
    """
    yc = _as_1d_float(y_control)
    xc = _as_1d_float(x_control)
    yt = _as_1d_float(y_treat)
    xt = _as_1d_float(x_treat)

    x_all = np.concatenate([xc, xt])
    y_all = np.concatenate([yc, yt])

    theta = estimate_theta(x_all, y_all)
    y_adj_all, fit = cuped_adjust(y_all, x_all, theta=theta)

    # split back
    y_adj_c = y_adj_all[: len(yc)]
    y_adj_t = y_adj_all[len(yc):]

    mc = float(np.mean(y_adj_c))
    mt = float(np.mean(y_adj_t))
    diff = mt - mc
    rel = np.inf if mc == 0 else diff / mc

    # Welch t-test
    t_res = stats.ttest_ind(y_adj_t, y_adj_c, equal_var=False, alternative=alternative)
    t_stat = float(t_res.statistic)
    p_value = float(t_res.pvalue)

    # CI with Welch-Satterthwaite df
    vc = np.var(y_adj_c, ddof=1) / len(y_adj_c)
    vt = np.var(y_adj_t, ddof=1) / len(y_adj_t)
    se = np.sqrt(vc + vt)

    numerator = (vc + vt) ** 2
    denom = (vc ** 2) / (len(y_adj_c) - 1) + (vt ** 2) / (len(y_adj_t) - 1)
    df = float(numerator / denom) if denom > 0 else float(len(y_adj_c) + len(y_adj_t) - 2)

    two_sided = (alternative == "two-sided")
    a = alpha / 2 if two_sided else alpha
    tcrit = stats.t.ppf(1 - a, df)
    ci = (diff - tcrit * se, diff + tcrit * se)

    return CupedABResult(
        theta=float(theta),
        control_mean=mc,
        treat_mean=mt,
        abs_lift=float(diff),
        rel_lift=float(rel),
        t=t_stat,
        df=df,
        p_value=p_value,
        ci=(float(ci[0]), float(ci[1])),
        var_reduction=float(fit.var_reduction),
    )
