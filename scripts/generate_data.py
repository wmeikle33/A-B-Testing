#!/usr/bin/env python3
"""
Generate synthetic A/B testing data.

Creates a CSV with per-user rows:
- user_id
- variant (control/treatment)
- device (desktop/mobile)
- country (US/CA/GB/AU/IN)
- exposure_ts (ISO timestamp)
- converted (0/1)
- revenue (float, 0 if not converted)

Usage:
  python scripts/generate_data.py --n 50000 --out data/ab_test.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd


def logistic(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Generate synthetic A/B test data.")
    ap.add_argument("--n", type=int, default=50_000, help="Number of users.")
    ap.add_argument("--seed", type=int, default=7, help="Random seed.")
    ap.add_argument("--out", type=str, default="data/ab_test.csv", help="Output CSV path.")
    ap.add_argument("--treatment-lift", type=float, default=0.06,
                    help="Relative lift in conversion odds for treatment (approx).")
    ap.add_argument("--start-date", type=str, default="2026-01-01",
                    help="Start date (YYYY-MM-DD) for exposure timestamps.")
    ap.add_argument("--days", type=int, default=14, help="Number of days of traffic.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    n = args.n

    # --- Assignment / covariates ---
    user_id = np.arange(1, n + 1)
    variant = rng.choice(["control", "treatment"], size=n, p=[0.5, 0.5])

    device = rng.choice(["desktop", "mobile"], size=n, p=[0.45, 0.55])
    country = rng.choice(["US", "CA", "GB", "AU", "IN"], size=n, p=[0.45, 0.12, 0.12, 0.08, 0.23])

    # Exposure timestamps across a date range
    start = datetime.fromisoformat(args.start_date).replace(tzinfo=timezone.utc)
    minutes_in_range = args.days * 24 * 60
    offsets = rng.integers(0, minutes_in_range, size=n)
    exposure_ts = [start + timedelta(minutes=int(m)) for m in offsets]

    # --- Conversion model ---
    # Baseline log-odds tuned to ~10% overall conversion before covariates
    base_log_odds = -2.2

    # Covariate effects (log-odds space)
    device_effect = np.where(device == "mobile", -0.15, 0.0)   # mobile slightly worse
    country_effect = np.select(
        [country == "US", country == "CA", country == "GB", country == "AU", country == "IN"],
        [0.10, 0.05, 0.03, 0.02, -0.10],
        default=0.0
    )

    # Treatment effect as a relative lift in odds (approx)
    treat_effect = np.where(variant == "treatment", np.log(1.0 + args.treatment_lift), 0.0)

    # Add user heterogeneity (random noise)
    user_noise = rng.normal(0.0, 0.35, size=n)

    log_odds = base_log_odds + device_effect + country_effect + treat_effect + user_noise
    p_convert = logistic(log_odds)

    converted = rng.binomial(1, p_convert, size=n).astype(int)

    # --- Revenue model ---
    # If converted, revenue ~ lognormal; also allow treatment to slightly increase AOV
    aov_base = 3.4  # log-space mean ~ exp(3.4) ~ 30
    aov_sigma = 0.7

    aov_lift = np.where(variant == "treatment", 1.03, 1.00)  # small AOV lift
    revenue = np.zeros(n, dtype=float)
    conv_idx = converted == 1
    revenue[conv_idx] = (rng.lognormal(mean=aov_base, sigma=aov_sigma, size=conv_idx.sum()) * aov_lift[conv_idx])

    df = pd.DataFrame(
        {
            "user_id": user_id,
            "variant": variant,
            "device": device,
            "country": country,
            "exposure_ts": [ts.isoformat() for ts in exposure_ts],
            "converted": converted,
            "revenue": np.round(revenue, 2),
        }
    ).sort_values("exposure_ts", kind="mergesort")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    # --- Quick sanity summary ---
    summary = (
        df.groupby("variant")
          .agg(users=("user_id", "count"),
               conversions=("converted", "sum"),
               conversion_rate=("converted", "mean"),
               revenue_total=("revenue", "sum"),
               revenue_per_user=("revenue", "mean"))
    )
    print(f"Wrote {len(df):,} rows -> {out_path}")
    print(summary.to_string(float_format=lambda x: f"{x:0.4f}"))


if __name__ == "__main__":
    main()

