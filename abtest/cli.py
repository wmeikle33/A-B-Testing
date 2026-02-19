from __future__ import annotations

import argparse
from dataclasses import asdict
import numpy as np

from .stats import ztest_proportions

def main() -> None:
    ap = argparse.ArgumentParser(prog="abtest", description="Run a tiny A/B test demo.")
    ap.add_argument("--n", type=int, default=5000, help="Users per variant")
    ap.add_argument("--p-control", type=float, default=0.10, help="Control conversion rate")
    ap.add_argument("--p-treat", type=float, default=0.11, help="Treatment conversion rate")
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    x_c = int(rng.binomial(args.n, args.p_control))
    x_t = int(rng.binomial(args.n, args.p_treat))

    res = ztest_proportions(x_c, args.n, x_t, args.n, alpha=args.alpha, alternative="two-sided")
    print("Inputs:", {"n": args.n, "x_control": x_c, "x_treat": x_t})
    print("Result:", asdict(res))
