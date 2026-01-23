# tests/test_sanity.py
import numpy as np
import pandas as pd
import pytest

from abtest.sanity import (
    srm_chisquare,
    srm_from_df,
    check_unique_assignment,
    balance_report,
    missingness_by_variant,
    validate_variant_labels,
    check_metric_ranges,
)


def test_srm_chisquare_passes_for_expected_split():
    # observed close to 50/50
    observed = {"control": 500, "treatment": 500}
    expected = {"control": 0.5, "treatment": 0.5}
    res = srm_chisquare(observed, expected)
    assert res.p_value > 0.05


def test_srm_chisquare_flags_srm():
    # heavily skewed away from 50/50
    observed = {"control": 700, "treatment": 300}
    expected = {"control": 0.5, "treatment": 0.5}
    res = srm_chisquare(observed, expected)
    assert res.p_value < 1e-6


def test_srm_from_df_defaults_equal_split():
    df = pd.DataFrame({
        "unit_id": range(10),
        "variant": ["control"] * 5 + ["treatment"] * 5
    })
    res = srm_from_df(df, variant_col="variant")
    assert res.p_value > 0.05


def test_check_unique_assignment_flags_multi_variant_units():
    # unit 1 appears in BOTH variants => bad
    df = pd.DataFrame({
        "unit_id": [1, 1, 2, 3, 4],
        "variant": ["control", "treatment", "control", "control", "treatment"],
        "converted": [0, 1, 0, 1, 0],
    })
    out = check_unique_assignment(df, unit_col="unit_id", variant_col="variant")
    assert out.n_units == 4
    assert out.n_multi_variant_units == 1  # unit_id=1
    assert out.n_duplicate_units >= 1


def test_balance_report_basic_numeric():
    rng = np.random.default_rng(0)
    n = 200

    df = pd.DataFrame({
        "variant": ["control"] * n + ["treatment"] * n,
        "age": np.concatenate([rng.normal(30, 5, n), rng.normal(30, 5, n)]),
        "pre_revenue": np.concatenate([rng.normal(10, 3, n), rng.normal(10, 3, n)]),
    })

    rows = balance_report(df, covariates=["age", "pre_revenue"])
    assert len(rows) == 2
    # SMD should be small on average because both groups from same distribution
    for r in rows:
        assert abs(r.smd) < 0.25  # loose threshold for randomness


def test_missingness_by_variant():
    df = pd.DataFrame({
        "variant": ["control", "control", "treatment", "treatment"],
        "revenue": [1.0, np.nan, 2.0, np.nan],
        "pre_metric": [np.nan, np.nan, 5.0, 6.0],
    })
    miss = missingness_by_variant(df, cols=["revenue", "pre_metric"])
    # should have two rows (control/treatment)
    assert set(miss["variant"]) == {"control", "treatment"}

    control_row = miss[miss["variant"] == "control"].iloc[0]
    treat_row = miss[miss["variant"] == "treatment"].iloc[0]

    assert control_row["revenue"] == pytest.approx(0.5)       # 1 missing out of 2
    assert control_row["pre_metric"] == pytest.approx(1.0)    # 2 missing out of 2
    assert treat_row["revenue"] == pytest.approx(0.5)
    assert treat_row["pre_metric"] == pytest.approx(0.0)      # none missing


def test_validate_variant_labels_raises_on_bad_label():
    df = pd.DataFrame({"variant": ["control", "treatment", "oops"]})
    with pytest.raises(ValueError):
        validate_variant_labels(df, variant_col="variant", allowed=("control", "treatment"))


def test_check_metric_ranges_counts_bad_rows():
    df = pd.DataFrame({
        "revenue": [10.0, -1.0, 5.0, 999.0],
        "latency_ms": [100.0, 40000.0, 200.0, -5.0],
    })
    rules = {
        "revenue": (0.0, 500.0),       # revenue must be between 0 and 500
        "latency_ms": (0.0, 30000.0),  # latency 0..30000
    }
    bad = check_metric_ranges(df, rules)
    assert bad["revenue"] == 2        # -1, 999
    assert bad["latency_ms"] == 2     # 40000, -5
