"""
abtest/utils.py

Utility functions used across the repo:
  - Data validation and splitting
  - Aggregation to randomization unit
  - Simple robust transforms (winsorization)
  - Reproducibility helpers
  - Formatting for reports
"""

from __future__ import annotations
from dataclasses import asdict, is_dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# -------------------------
# Validation / coercion
# -------------------------

def check_columns(df: pd.DataFrame, required: Sequence[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

def assert_two_variants(df: pd.DataFrame,
                        variant_col: str = "variant",
                        control_label: str = "control",
                        treat_label: str = "treatment") -> None:
    vals = set(df[variant_col].dropna().unique())
    expected = {control_label, treat_label}
    if vals != expected:
        raise ValueError(f"Expected variants {expected}, got {vals}")

def coerce_numeric(df: pd.DataFrame, cols: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out

def dropna_for_cols(df: pd.DataFrame, cols: Sequence[str]) -> Tuple[pd.DataFrame, Dict[str, int]]:
    before = len(df)
    out = df.dropna(subset=list(cols))
    after = len(out)
    return out, {"dropped_rows": before - after, "before": before, "after": after}


# -------------------------
# Variant splitting / aggregation
# -------------------------

def split_by_variant(df: pd.DataFrame,
                     variant_col: str = "variant",
                     control_label: str = "control",
                     treat_label: str = "treatment") -> Tuple[pd.DataFrame, pd.DataFrame]:
    dc = df[df[variant_col] == control_label].copy()
    dt = df[df[variant_col] == treat_label].copy()
    return dc, dt

def aggregate_to_unit(df: pd.DataFrame,
                      unit_col: str,
                      variant_col: str,
                      metric_cols: Sequence[str],
                      agg: str = "sum") -> pd.DataFrame:
    """
    If your raw data is event-level, aggregate to the randomization unit (often user_id).
    Example:
      df_user = aggregate_to_unit(df_events, "user_id", "variant", ["revenue", "converted"], agg="sum")
    """
    check_columns(df, [unit_col, variant_col] + list(metric_cols))
    grouped = df.groupby([unit_col, variant_col], as_index=False)[list(metric_cols)].agg(agg)
    return grouped


# -------------------------
# Robust transforms
# -------------------------

def winsorize(x: Iterable[float], p: float = 0.01) -> np.ndarray:
    """
    Clip extremes at p and 1-p quantiles.
    """
    arr = np.asarray(list(x), dtype=float)
    if arr.size == 0:
        return arr
    lo = np.quantile(arr, p)
    hi = np.quantile(arr, 1 - p)
    return np.clip(arr, lo, hi)

def log1p_safe(x: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(x), dtype=float)
    return np.log1p(np.maximum(arr, 0.0))


# -------------------------
# Reproducibility
# -------------------------

def rng(seed: Optional[int] = None) -> np.random.Generator:
    return np.random.default_rng(seed)


# -------------------------
# Reporting / formatting
# -------------------------

def as_report_dict(obj) -> Dict:
    """
    Convert dataclass or dict-like result to a plain dict for JSON/printing.
    """
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, dict):
        return obj
    raise TypeError("Expected dataclass or dict.")

def fmt_pct(x: float, digits: int = 2) -> str:
    return f"{100.0 * x:.{digits}f}%"

def fmt_float(x: float, digits: int = 4) -> str:
    return f"{x:.{digits}f}"

def fmt_pvalue(p: float) -> str:
    if p < 1e-4:
        return "<1e-4"
    return f"{p:.4f}"

def fmt_ci(ci: Tuple[float, float], digits: int = 4) -> str:
    lo, hi = ci
    return f"[{lo:.{digits}f}, {hi:.{digits}f}]"
