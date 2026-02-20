from dataclasses import dataclass

@dataclass
class ABTestContinuousResult:
    mean_A: float
    mean_B: float
    diff: float
    rel_diff: float
    t_stat: float
    p_value: float
    ci_low: float
    ci_high: float
    n_A: int
    n_B: int

def run_ab_test_continuous(
    df: pd.DataFrame,
    group_col: str,
    metric_col: str,
    control_label="A",
    treatment_label="B",
    alpha=0.05,
) -> ABTestContinuousResult:
    # Extract data
    A = df.loc[df[group_col] == control_label, metric_col].values
    B = df.loc[df[group_col] == treatment_label, metric_col].values

    # Basic stats
    mean_A = A.mean()
    mean_B = B.mean()
    n_A, n_B = A.size, B.size
    std_A = A.std(ddof=1)
    std_B = B.std(ddof=1)

    # Welch's t-test
    t_stat, p_value = stats.ttest_ind(B, A, equal_var=False)

    # Effect sizes
    diff = mean_B - mean_A
    rel_diff = diff / mean_A if mean_A != 0 else np.nan

    # SE and Welch df
    se_diff = np.sqrt(std_A**2 / n_A + std_B**2 / n_B)
    df_welch = (std_A**2 / n_A + std_B**2 / n_B)**2 / (
        (std_A**2 / n_A)**2 / (n_A - 1) + (std_B**2 / n_B)**2 / (n_B - 1)
    )

    t_crit = stats.t.ppf(1 - alpha/2, df_welch)
    ci_low = diff - t_crit * se_diff
    ci_high = diff + t_crit * se_diff

    return ABTestContinuousResult(
        mean_A=mean_A,
        mean_B=mean_B,
        diff=diff,
        rel_diff=rel_diff,
        t_stat=t_stat,
        p_value=p_value,
        ci_low=ci_low,
        ci_high=ci_high,
        n_A=n_A,
        n_B=n_B,
    )

    lift_abs = pB - pA
    lift_rel = (pB / pA - 1.0) if pA > 0 else float("inf")
    return ABResult(p_A=pA, p_B=pB, lift_abs=lift_abs, lift_rel=lift_rel, z=z, p_value=pval, ci_diff=ci, cohen_h=h)

from dataclasses import dataclass

@dataclass
class PairedTTestResult:
    mean_A: float
    mean_B: float
    mean_diff: float
    t_stat: float
    p_value: float
    ci_low: float
    ci_high: float
    n: int

def run_paired_t_test(A, B, alpha=0.05) -> PairedTTestResult:
    A = np.asarray(A)
    B = np.asarray(B)
    if A.shape != B.shape:
        raise ValueError("A and B must have the same shape for a paired t-test.")
    
    diff = B - A
    n = diff.size
    mean_A = A.mean()
    mean_B = B.mean()
    mean_diff = diff.mean()
    std_diff = diff.std(ddof=1)
    
    # t-statistic and p-value (two-sided)
    t_stat, p_value = stats.ttest_rel(B, A)
    
    # CI for the mean difference
    df = n - 1
    se_diff = std_diff / np.sqrt(n)
    t_crit = stats.t.ppf(1 - alpha/2, df)
    ci_low = mean_diff - t_crit * se_diff
    ci_high = mean_diff + t_crit * se_diff
    
    return PairedTTestResult(
        mean_A=mean_A,
        mean_B=mean_B,
        mean_diff=mean_diff,
        t_stat=t_stat,
        p_value=p_value,
        ci_low=ci_low,
        ci_high=ci_high,
        n=n,
    )

import numpy as np
from scipy import stats

def independent_t_test(A, B, alpha=0.05):
    A = np.asarray(A)
    B = np.asarray(B)

    mean_A = A.mean()
    mean_B = B.mean()
    n_A = A.size
    n_B = B.size
    std_A = A.std(ddof=1)
    std_B = B.std(ddof=1)

    # Welch's t-test
    t_stat, p_value = stats.ttest_ind(B, A, equal_var=False)

    # Difference (B - A)
    diff = mean_B - mean_A
    rel_diff = diff / mean_A if mean_A != 0 else np.nan

    # Standard error of diff
    se_diff = np.sqrt(std_A**2 / n_A + std_B**2 / n_B)

    # Welch–Satterthwaite degrees of freedom
    df = (std_A**2 / n_A + std_B**2 / n_B)**2 / (
        (std_A**2 / n_A)**2 / (n_A - 1) + (std_B**2 / n_B)**2 / (n_B - 1)
    )

    # Critical t for two-sided CI
    t_crit = stats.t.ppf(1 - alpha / 2, df)
    ci_low = diff - t_crit * se_diff
    ci_high = diff + t_crit * se_diff

    return {
        "mean_A": mean_A,
        "mean_B": mean_B,
        "diff_B_minus_A": diff,
        "relative_diff": rel_diff,
        "t_stat": t_stat,
        "p_value": p_value,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "n_A": n_A,
        "n_B": n_B,
        "df": df,
    }

def anova():
