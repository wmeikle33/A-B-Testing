# tests/test_cuped.py
import numpy as np
import pytest

from abtest.cuped import estimate_theta, cuped_adjust, cuped_ab_test


def test_estimate_theta_close_to_true():
    """
    If Y = a*X + noise, then theta should be ~a.
    """
    rng = np.random.default_rng(0)
    n = 5000
    a_true = 2.5

    x = rng.normal(0.0, 1.0, size=n)
    noise = rng.normal(0.0, 0.5, size=n)
    y = a_true * x + noise

    theta = estimate_theta(x, y)
    assert theta == pytest.approx(a_true, rel=0.05)  # within ~5%


def test_cuped_reduces_variance_when_correlated():
    rng = np.random.default_rng(1)
    n = 4000
    a = 1.8

    x = rng.normal(0.0, 1.0, size=n)
    y = a * x + rng.normal(0.0, 1.0, size=n)

    y_adj, fit = cuped_adjust(y, x)
    assert fit.corr_xy > 0.5  # should be strongly correlated
    assert fit.var_y_adj < fit.var_y
    assert fit.var_reduction > 0.2  # at least 20% reduction (should hold with this setup)

    # Adjusted outcome should be less correlated with X
    corr_after = np.corrcoef(x, y_adj)[0, 1]
    assert abs(corr_after) < abs(fit.corr_xy)


def test_cuped_does_not_change_when_uncorrelated():
    rng = np.random.default_rng(2)
    n = 3000

    x = rng.normal(0.0, 1.0, size=n)
    y = rng.normal(0.0, 1.0, size=n)  # independent

    y_adj, fit = cuped_adjust(y, x)
    # With no correlation, theta ~ 0, variance reduction should be near 0
    assert abs(fit.theta) < 0.1
    assert abs(fit.var_reduction) < 0.05


def test_cuped_adjust_invariant_to_shift_in_x():
    """
    Shifting X by a constant should not change Y_adj except for numerical noise,
    because CUPED uses (X - mean(X)).
    """
    rng = np.random.default_rng(3)
    n = 2000

    x = rng.normal(0.0, 1.0, size=n)
    y = 1.2 * x + rng.normal(0.0, 1.0, size=n)

    y_adj1, fit1 = cuped_adjust(y, x)
    y_adj2, fit2 = cuped_adjust(y, x + 10.0)  # shift X

    assert fit1.theta == pytest.approx(fit2.theta, rel=1e-6, abs=1e-6)
    # y_adj should be (almost) identical
    assert np.allclose(y_adj1, y_adj2, atol=1e-8)


def test_cuped_ab_test_detects_known_treatment_effect():
    """
    Construct a case where treatment adds a constant delta to Y.
    CUPED should still estimate ~delta and typically improve precision.
    """
    rng = np.random.default_rng(4)
    n = 2000
    delta = 0.25
    a = 1.5

    # Pre-period covariate X
    xc = rng.normal(0.0, 1.0, size=n)
    xt = rng.normal(0.0, 1.0, size=n)

    # Post metric Y correlated with X, plus treatment shift
    yc = a * xc + rng.normal(0.0, 1.0, size=n)
    yt = a * xt + rng.normal(0.0, 1.0, size=n) + delta

    res = cuped_ab_test(yc, xc, yt, xt, alpha=0.05, alternative="two-sided")

    # Estimated lift should be close to delta
    assert res.abs_lift == pytest.approx(delta, abs=0.07)
    # Typically significant with n=2000 and variance reduction
    assert res.p_value < 0.05
    lo, hi = res.ci
    assert lo < res.abs_lift < hi


def test_estimate_theta_handles_zero_variance_x():
    x = np.ones(1000)
    y = np.arange(1000, dtype=float)
    theta = estimate_theta(x, y)
    assert theta == 0.0
