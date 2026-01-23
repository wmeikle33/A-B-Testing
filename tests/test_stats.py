import numpy as np
import pytest

from abtest.stats import (
    proportion_effect,
    ztest_proportions,
    mean_effect,
    welch_ttest_means,
    bootstrap_ci_diff_means,
    bootstrap_pvalue_diff_means,
    benjamini_hochberg,
)


def test_proportion_effect_basic():
    # control: 20/100, treat: 30/100
    eff = proportion_effect(20, 100, 30, 100)
    assert eff["p_control"] == pytest.approx(0.20)
    assert eff["p_treat"] == pytest.approx(0.30)
    assert eff["abs_lift"] == pytest.approx(0.10)
    assert eff["rel_lift"] == pytest.approx(0.10 / 0.20)


def test_ztest_proportions_strong_signal():
    # Big difference => should be significant
    res = ztest_proportions(200, 1000, 260, 1000, alpha=0.05, alternative="two-sided", pooled=True)
    assert res.abs_lift == pytest.approx(0.06, abs=1e-12)
    assert res.p_value < 0.05
    lo, hi = res.ci
    assert lo < res.abs_lift < hi


def test_ztest_proportions_no_signal():
    # Same rate => p-value should be large
    res = ztest_proportions(250, 1000, 250, 1000, alpha=0.05)
    assert abs(res.abs_lift) < 1e-12
    assert res.p_value > 0.2  # loose threshold, stable


def test_mean_effect_basic():
    xc = [1, 2, 3]
    xt = [2, 3, 4]
    eff = mean_effect(xc, xt)
    assert eff["mean_control"] == pytest.approx(2.0)
    assert eff["mean_treat"] == pytest.approx(3.0)
    assert eff["abs_lift"] == pytest.approx(1.0)
    assert eff["rel_lift"] == pytest.approx(0.5)


def test_welch_ttest_detects_shift():
    rng = np.random.default_rng(0)
    xc = rng.normal(loc=0.0, scale=1.0, size=500)
    xt = rng.normal(loc=0.3, scale=1.2, size=500)  # different mean + variance

    res = welch_ttest_means(xc, xt, alpha=0.05)
    assert res.abs_lift == pytest.approx(np.mean(xt) - np.mean(xc), abs=1e-12)
    assert res.p_value < 0.05
    lo, hi = res.ci
    assert lo < res.abs_lift < hi


def test_welch_ttest_no_shift():
    rng = np.random.default_rng(1)
    xc = rng.normal(loc=0.0, scale=1.0, size=800)
    xt = rng.normal(loc=0.0, scale=1.0, size=800)

    res = welch_ttest_means(xc, xt, alpha=0.05)
    assert res.p_value > 0.05  # not always huge, but usually > 0.05 with n=800


def test_bootstrap_ci_contains_true_effect_simple():
    # Construct data where true mean diff is exactly 1.0 with small noise
    rng = np.random.default_rng(42)
    xc = rng.normal(loc=10.0, scale=0.5, size=400)
    xt = rng.normal(loc=11.0, scale=0.5, size=400)

    out = bootstrap_ci_diff_means(xc, xt, alpha=0.05, n_boot=3000, seed=123)
    lo, hi = out.ci
    # True effect is 1.0
    assert lo < 1.0 < hi


def test_bootstrap_permutation_pvalue_small_for_large_effect():
    rng = np.random.default_rng(7)
    xc = rng.normal(loc=0.0, scale=1.0, size=300)
    xt = rng.normal(loc=0.8, scale=1.0, size=300)

    p = bootstrap_pvalue_diff_means(xc, xt, n_boot=3000, seed=99, two_sided=True)
    assert p < 0.01


def test_bh_fdr_known_example():
    # Classic toy set: expect first 3 rejected at q=0.1
    pvals = [0.001, 0.01, 0.04, 0.06, 0.20]
    res = benjamini_hochberg(pvals, q=0.1)
    rejected = res["rejected"]
    # BH thresholds: (i/m)q => [0.02,0.04,0.06,0.08,0.10]
    # sorted p: 0.001,0.01,0.04,0.06,0.2 => first 4 pass actually (0.06 <= 0.08)
    assert rejected.sum() == 4
    assert res["cutoff_p"] == pytest.approx(0.06)


def test_inputs_are_1d():
    # stats functions should reject 2D arrays
    xc = np.zeros((2, 2))
    xt = np.zeros((2, 2))
    with pytest.raises(ValueError):
        welch_ttest_means(xc, xt)
