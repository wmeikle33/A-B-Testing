from __future__ import annotations
import hashlib
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd

def _rng(seed: Optional[int] = None) -> np.random.Generator:
    return np.random.default_rng(seed)

def _sticky_hash_prob(user_id: int, salt: str = "abtest") -> float:
    """Hash user_id -> [0,1) deterministically."""
    h = hashlib.md5(f"{salt}:{user_id}".encode()).hexdigest()
    return int(h[:8], 16) / 16**8

def assign_arms(
    user_ids: np.ndarray,
    split: float = 0.5,
    method: str = "random",
    seed: Optional[int] = 123,
    salt: str = "abtest",
    srm_pp: float = 0.0,
) -> np.ndarray:
    """
    Assign users to A/B.
    - method='random': RNG-based (reproducible with seed)
    - method='hash'  : sticky hash(user_id)
    - srm_pp: sample-ratio mismatch (e.g., 0.05 shifts 5pp from A->B)
    """
    if method not in {"random", "hash"}:
        raise ValueError("method must be 'random' or 'hash'")
    split_eff = np.clip(split - srm_pp, 0.0, 1.0)  # SRM shifts mass to B
    if method == "random":
        r = _rng(seed).random(len(user_ids))
    else:
        r = np.array([_sticky_hash_prob(int(u), salt) for u in user_ids])
    return np.where(r < split_eff, "A", "B")

def generate_ab(
    n: int = 10000,
    pA: float = 0.08,                   # baseline conversion in A
    uplift_pp: float = 0.012,           # absolute uplift (B vs A) in conversion prob
    rev_mu_A: float = 2.2,              # lognormal mean for converters in A
    rev_uplift_mult: float = 1.05,      # revenue multiplicative uplift in B (if converted)
    hetero_sd: float = 0.20,            # user-level heterogeneity on logit scale
    pre_corr: float = 0.30,             # correlation between pre_metric and revenue
    seed: Optional[int] = 42,
    # segments
    device_probs: Tuple[float, float] = (0.5, 0.5),  # desktop, mobile
    geo_probs: Tuple[float, float, float] = (0.4, 0.3, 0.3),  # US, EU, APAC
    # assignment and misbehavior
    split: float = 0.5,
    assign_method: str = "hash",        # 'hash' (sticky) or 'random'
    srm_pp: float = 0.0,                # sample ratio mismatch to B in percentage points
    contamination_rate: float = 0.0,    # fraction of users appearing in both arms
    duplicates_rate: float = 0.0,       # fraction of rows that are dupes (same user repeated)
    attrition_rate: float = 0.0,        # fraction of users dropped post-assignment (uniformly)
    # cluster option (store/team) to simulate ICC
    n_clusters: int = 0,                # 0 = no clustering
    cluster_sd: float = 0.25,           # cluster random effect on logit scale
    # time window
    days: int = 7,
) -> pd.DataFrame:
    """
    Return a user-level dataframe with:
    user_id, group, convert, revenue, pre_metric, device, geo, exposure_ts, purchase_ts (NaT if not converted)

    You can deliberately inject SRM, contamination, duplicates, and attrition to demo diagnostics.
    """
    rng = _rng(seed)
    user_id = np.arange(1, n + 1)

    # Segments
    device = rng.choice(["desktop", "mobile"], size=n, p=device_probs)
    geo = rng.choice(["US", "EU", "APAC"], size=n, p=geo_probs)

    # Optional clustering (random intercept per cluster on logit)
    if n_clusters > 0:
        clusters = rng.integers(0, n_clusters, size=n)
        cluster_effect = rng.normal(0.0, cluster_sd, size=n_clusters)
        c_eff = cluster_effect[clusters]
    else:
        clusters = np.zeros(n, dtype=int)
        c_eff = 0.0

    # Sticky/random assignment with optional SRM
    group = assign_arms(user_id, split=split, method=assign_method, seed=seed, srm_pp=srm_pp)

    # User heterogeneity on logit scale
    hetero = rng.normal(0.0, hetero_sd, size=n)

    # Baseline conversion prob for A (logit), then convert to prob
    # logit(p) = log(p / (1-p)) ; invert with sigmoid
    def sigmoid(x): return 1 / (1 + np.exp(-x))
    logit_pA = np.log(pA / max(1e-8, (1 - pA))) + hetero + c_eff
    pA_user = sigmoid(logit_pA)

    # Treatment: add absolute uplift_pp to B users, then clip to [0.001, 0.999]
    p = pA_user.copy()
    p[group == "B"] = np.clip(p[group == "B"] + uplift_pp, 0.001, 0.999)

    convert = rng.binomial(1, p, size=n)

    # Revenue: only for converters; heavy-tailed lognormal
    # Let B have a multiplicative uplift on the log-mean
    mu = rev_mu_A + (np.log(rev_uplift_mult) if rev_uplift_mult > 0 else 0.0) * (group == "B")
    sigma = 0.5
    revenue = np.where(
        convert == 1,
        rng.lognormal(mean=mu, sigma=sigma, size=n),
        0.0,
    )

    # Pre-period covariate (for CUPED): correlated with revenue
    # pre_metric = alpha * revenue + noise, rescaled to be non-negative
    noise = rng.normal(0, 1, size=n)
    pre_metric = np.maximum(0.0, pre_corr * revenue + noise)

    # Simple timestamps (UTC) within 'days' window
    start = np.datetime64("2025-01-01")
    exposure_ts = start + rng.integers(0, days, size=n).astype("timedelta64[D]")
    # purchase occurs same day or next 3 days (if converted)
    purchase_offset = rng.integers(0, 4, size=n).astype("timedelta64[D]")
    purchase_ts = np.where(convert == 1, (exposure_ts + purchase_offset).astype("datetime64[ns]"), "NaT").astype("datetime64[ns]")

    df = pd.DataFrame({
        "user_id": user_id,
        "group": group,
        "convert": convert,
        "revenue": revenue,
        "pre_metric": pre_metric,
        "device": device,
        "geo": geo,
        "cluster": clusters,
        "exposure_ts": exposure_ts.astype("datetime64[ns]"),
        "purchase_ts": purchase_ts,
    })

    # Inject attrition (drop random users post-assignment)
    if attrition_rate > 0:
        keep_mask = rng.random(len(df)) >= attrition_rate
        df = df.loc[keep_mask].reset_index(drop=True)

    # Inject contamination (small fraction of users appear in both arms)
    if contamination_rate > 0:
        m = int(len(df) * contamination_rate)
        if m > 0:
            sample_ids = rng.choice(df["user_id"].unique(), size=m, replace=False)
            dup_rows = df[df["user_id"].isin(sample_ids)].copy()
            dup_rows["group"] = np.where(dup_rows["group"] == "A", "B", "A")
            # slightly shift timestamps to look realistic
            dup_rows["exposure_ts"] = dup_rows["exposure_ts"] + np.timedelta64(1, "D")
            dup_rows["purchase_ts"] = dup_rows["purchase_ts"] + np.timedelta64(1, "D")
            df = pd.concat([df, dup_rows], ignore_index=True)

    # Inject duplicates (exact row repeats)
    if duplicates_rate > 0:
        k = int(len(df) * duplicates_rate)
        if k > 0:
            dup = df.sample(k, replace=True, random_state=seed)
            df = pd.concat([df, dup], ignore_index=True)

    # Sort by exposure_ts then user_id for readability
    df = df.sort_values(["exposure_ts", "user_id"]).reset_index(drop=True)
    return df
