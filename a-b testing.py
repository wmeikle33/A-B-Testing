import math
from dataclasses import dataclass
from typing import Literal, Tuple
import numpy as np
from scipy.stats import norm

Alt = Literal["two-sided", "larger", "smaller"]

@dataclass
class ABResult:
    p_A: float
    p_B: float
    lift_abs: float          # B - A (percentage points in 0..1 scale)
    lift_rel: float          # (B/A - 1)
    z: float
    p_value: float
    ci_diff: Tuple[float, float]  # CI for (p_B - p_A)
    cohen_h: float

def _z_pvalue(z: float, alternative: Alt) -> float:
    if alternative == "two-sided":
        return 2 * (1 - norm.cdf(abs(z)))
    if alternative == "larger":   # H1: pB > pA
        return 1 - norm.cdf(z)
    if alternative == "smaller":  # H1: pB < pA
        return norm.cdf(z)
    raise ValueError("bad alternative")

def two_proportion_ztest(conv_A: int, n_A: int, conv_B: int, n_B: int,
                         alpha: float = 0.05, alternative: Alt = "two-sided") -> ABResult:
    """Classical pooled z-test for proportions. Balanced or not; small-sample -> prefer exact test."""
    if not (0 <= conv_A <= n_A and 0 <= conv_B <= n_B):
        raise ValueError("counts must be within sample sizes")
    pA = conv_A / n_A
    pB = conv_B / n_B
    # pooled SE under H0 (pA == pB)
    p_pool = (conv_A + conv_B) / (n_A + n_B)
    se = math.sqrt(p_pool * (1 - p_pool) * (1/n_A + 1/n_B))
    z = (pB - pA) / se if se > 0 else 0.0
    pval = _z_pvalue(z, alternative)

    # CI for difference using unpooled SE (Wald); for small n use Newcombe/Wilson instead
    se_unpooled = math.sqrt(pA*(1-pA)/n_A + pB*(1-pB)/n_B)
    zc = norm.ppf(1 - alpha/2)
    ci = ( (pB - pA) - zc*se_unpooled, (pB - pA) + zc*se_unpooled )

    # Cohen's h (effect size for proportions)
    def _phi(p): return 2*math.asin(math.sqrt(p))
    h = _phi(pB) - _phi(pA)

    lift_abs = pB - pA
    lift_rel = (pB / pA - 1.0) if pA > 0 else float("inf")
    return ABResult(p_A=pA, p_B=pB, lift_abs=lift_abs, lift_rel=lift_rel, z=z, p_value=pval, ci_diff=ci, cohen_h=h)
