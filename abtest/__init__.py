"""
abtest: A/B testing utilities (sanity checks, statistical tests, and power analysis).

Public API is re-exported here for convenience.
"""

from importlib.metadata import PackageNotFoundError, version as _version

# ---- Version ----
try:
    __version__ = _version("abtest")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"

# ---- Re-exports (edit to match your actual functions/classes) ----
# Stats / inference
from .stats import (  # noqa: F401
    ztest_proportions,
    welch_ttest_means,
    mannwhitney_u_test,
    bootstrap_ci,
)

# Power / sample size
from .power import (  # noqa: F401
    required_sample_size_proportions,
    required_sample_size_means,
)

# Sanity checks
from .sanity import (  # noqa: F401
    srm_chi2_test,
    check_balance,
    check_missingness,
)

__all__ = [
    "__version__",
    # stats
    "ztest_proportions",
    "welch_ttest_means",
    "mannwhitney_u_test",
    "bootstrap_ci",
    # power
    "required_sample_size_proportions",
    "required_sample_size_means",
    # sanity
    "srm_chi2_test",
    "check_balance",
    "check_missingness",
]
