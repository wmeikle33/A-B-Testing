"""
abtest/sanity.py

Sanity checks for A/B experiments:
  - SRM (sample ratio mismatch)
  - assignment integrity (unit appears in >1 variant)
  - balance checks (covariates)
  - missingness and basic data validation
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats


# -------------------------
# SRM (Sample Ratio Mismatch)
# -------------------------

@dataclass(frozen=True)
class SRMResult:
    chi2: float
    p_value: float
    observed: Dict[str, int]
    expected: Dict[str, float]

def srm_chisquare(observed_counts: Dict[str, int],
                  expected_shares: Dict[str, float]) -> SRMResult:
    """
    Chi-square goodness-of-fit test comparing observed group counts
    to expected allocation shares.
    """
    groups = list(observed_counts.keys())
    obs = np.array([observed_counts[g] for g in groups], dtype=float)

    shares = np.array([expected_shares[g] for g in groups], dtype=float)
    if np.any(shares < 0) or not np.isclose(shares.sum(), 1.0):
        raise ValueError("expected_shares must be non-negative and sum to 1.")

    total = obs.sum()
    exp = shares * total

    chi2, p = stats.chisquare(f_obs=obs, f_exp=exp)

    return SRMResult(
        chi2=float(chi2),
        p_value=float(p),
        observed={g: int(observed_counts[g]) for g in groups},
        expected={g: float(expected_shares[g]) for g in groups},
    )

def srm_from_df(df: pd.DataFrame,
                variant_col: str = "variant",
                expected_shares: Optional[Dict[str, float]] = None) -> SRMResult:
    counts = df[variant_col].value_counts(dropna=False).to_dict()
    if expected_shares is None:
        # default to equal split across observed groups
        k = len(counts)
        expected_shares = {g: 1.0 / k for g in counts.keys()}
    return srm_chisquare(counts, expected_shares)


# -------------------------
# Assignment integrity
# -------------------------

@dataclass(frozen=True)
class AssignmentIntegrity:
    n_rows: int
    n_units: int
    n_duplicate_units: int
    n_multi_variant_units: int

def check_unique_assignment(df: pd.DataFrame,
                            unit_col: str = "unit_id",
                            variant_col: str = "variant") -> AssignmentIntegrity:
    """
    Ensures each unit_id is assigned to exactly one variant.
    """
    n_rows = len(df)
    n_units = df[unit_col].nunique(dropna=False)

    # duplicate units: unit appears multiple times (could be ok if event-level table)
    dup_counts = df.groupby(unit_col)[variant_col].count()
    n_duplicate_units = int((dup_counts > 1).sum())

    # multi-variant units: same unit shows up under multiple variants (bad)
    var_per_unit = df.groupby(unit_col)[variant_col].nunique()
    n_multi_variant_units = int((var_per_unit > 1).sum())

    return AssignmentIntegrity(
        n_rows=n_rows,
        n_units=int(n_units),
        n_duplicate_units=n_duplicate_units,
        n_multi_variant_units=n_multi_variant_units,
    )


# -------------------------
# Balance checks
# -------------------------

def standardized_mean_diff(x_control: np.ndarray, x_treat: np.ndarray) -> float:
    """
    SMD = (mean_t - mean_c) / pooled_std
    Rule of thumb: |SMD| < 0.1 is often considered acceptable balance.
    """
    x_control = np.asarray(x_control, dtype=float)
    x_treat = np.asarray(x_treat, dtype=float)
    mc, mt = np.mean(x_control), np.mean(x_treat)
    sc, st = np.std(x_control, ddof=1), np.std(x_treat, ddof=1)
    pooled = np.sqrt((sc**2 + st**2) / 2.0)
    if pooled == 0:
        return 0.0
    return float((mt - mc) / pooled)

@dataclass(frozen=True)
class BalanceRow:
    covariate: str
    control_mean: float
    treat_mean: float
    abs_diff: float
    smd: float
    p_value: float

def balance_report(df: pd.DataFrame,
                   covariates: Sequence[str],
                   variant_col: str = "variant",
                   control_label: str = "control",
                   treat_label: str = "treatment") -> List[BalanceRow]:
    """
    For numeric covariates, reports mean diff, SMD, and Welch t-test p-value.
    """
    rows: List[BalanceRow] = []

    dfc = df[df[variant_col] == control_label]
    dft = df[df[variant_col] == treat_label]

    for col in covariates:
        xc = dfc[col].dropna().to_numpy(dtype=float)
        xt = dft[col].dropna().to_numpy(dtype=float)

        if len(xc) < 2 or len(xt) < 2:
            p = np.nan
        else:
            p = float(stats.ttest_ind(xt, xc, equal_var=False).pvalue)

        mc = float(np.mean(xc)) if len(xc) else np.nan
        mt = float(np.mean(xt)) if len(xt) else np.nan
        smd = standardized_mean_diff(xc, xt) if len(xc) and len(xt) else np.nan

        rows.append(BalanceRow(
            covariate=col,
            control_mean=mc,
            treat_mean=mt,
            abs_diff=(mt - mc) if np.isfinite(mc) and np.isfinite(mt) else np.nan,
            smd=smd,
            p_value=p,
        ))
    return rows


# -------------------------
# Missingness + basic validation
# -------------------------

def missingness_by_variant(df: pd.DataFrame,
                           cols: Sequence[str],
                           variant_col: str = "variant") -> pd.DataFrame:
    """
    Returns a table: missing rate per column per variant.
    """
    out = []
    for v, g in df.groupby(variant_col):
        row = {"variant": v}
        for c in cols:
            row[c] = float(g[c].isna().mean())
        out.append(row)
    return pd.DataFrame(out)

def validate_variant_labels(df: pd.DataFrame,
                            variant_col: str = "variant",
                            allowed: Sequence[str] = ("control", "treatment")) -> None:
    bad = set(df[variant_col].dropna().unique()) - set(allowed)
    if bad:
        raise ValueError(f"Unexpected variant labels: {sorted(bad)}")

def check_metric_ranges(df: pd.DataFrame,
                        rules: Dict[str, Tuple[Optional[float], Optional[float]]]) -> Dict[str, int]:
    """
    rules example: {"revenue": (0, None), "latency_ms": (0, 30000)}
    Returns count of out-of-range rows per metric.
    """
    counts: Dict[str, int] = {}
    for col, (lo, hi) in rules.items():
        x = df[col]
        bad = pd.Series(False, index=df.index)
        if lo is not None:
            bad |= (x < lo)
        if hi is not None:
            bad |= (x > hi)
        counts[col] = int(bad.sum())
    return counts
