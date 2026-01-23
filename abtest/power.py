from dataclasses import dataclass
import math

@dataclass
class SampleSizeResult:
    n_control: int
    n_treat: int
    n_total: int

def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def norm_ppf(p: float) -> float:
    # Approximation (Acklam). Good enough for power calcs.
    # You can replace with scipy.stats.norm.ppf if using SciPy.
    if not (0.0 < p < 1.0):
        raise ValueError("p must be in (0,1)")
    # ... (implementation omitted here if you prefer SciPy)
    raise NotImplementedError

def z_alpha(alpha: float, two_sided: bool = True) -> float:
    a = alpha / 2.0 if two_sided else alpha
    return norm_ppf(1.0 - a)

def z_beta(power: float) -> float:
    return norm_ppf(power)

def sample_size_proportions(
    p0: float,
    mde_abs: float | None = None,
    mde_rel: float | None = None,
    alpha: float = 0.05,
    power: float = 0.8,
    two_sided: bool = True,
    ratio: float = 1.0,   # nt / nc
) -> SampleSizeResult:
    if (mde_abs is None) == (mde_rel is None):
        raise ValueError("Provide exactly one of mde_abs or mde_rel")

    p1 = p0 + (mde_abs if mde_abs is not None else p0 * mde_rel)
    if not (0 < p0 < 1 and 0 < p1 < 1):
        raise ValueError("p0 and p1 must be in (0,1)")

    za = z_alpha(alpha, two_sided)
    zb = z_beta(power)

    # Unpooled variance under H1 (common for planning)
    v = p0 * (1 - p0) + p1 * (1 - p1) / ratio
    delta = abs(p1 - p0)

    nc = ( (za + zb) ** 2 * v ) / (delta ** 2)
    nc = math.ceil(nc)
    nt = math.ceil(nc * ratio)
    return SampleSizeResult(nc, nt, nc + nt)

def sample_size_means(
    sigma: float,
    mde: float,
    alpha: float = 0.05,
    power: float = 0.8,
    two_sided: bool = True,
    ratio: float = 1.0,   # nt / nc
) -> SampleSizeResult:
    za = z_alpha(alpha, two_sided)
    zb = z_beta(power)
    if sigma <= 0 or mde <= 0:
        raise ValueError("sigma and mde must be > 0")

    # Var(mean_diff) = sigma^2/nc + sigma^2/nt = sigma^2*(1/nc + 1/(ratio*nc))
    factor = (1.0 + 1.0 / ratio)
    nc = ( (za + zb) ** 2 * (sigma ** 2) * factor ) / (mde ** 2)
    nc = math.ceil(nc)
    nt = math.ceil(nc * ratio)
    return SampleSizeResult(nc, nt, nc + nt)

def effective_sigma_cuped(sigma: float, rho: float) -> float:
    if not (-1 <= rho <= 1):
        raise ValueError("rho must be in [-1, 1]")
    return sigma * math.sqrt(max(0.0, 1.0 - rho * rho))

def duration_days(n_total: int, daily_units: int, allocation: float = 1.0) -> float:
    if daily_units <= 0 or n_total <= 0:
        raise ValueError("n_total and daily_units must be > 0")
    if not (0 < allocation <= 1):
        raise ValueError("allocation must be in (0,1]")
    return n_total / (daily_units * allocation)
