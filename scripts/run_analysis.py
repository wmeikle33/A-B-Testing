"""
Run an end-to-end A/B test analysis on a CSV.

Expected input format (minimum):
- group column: e.g. "group" with values like "A" and "B" (or "control"/"treatment")
- metric column: e.g. "converted" (0/1) for binary metrics, or "revenue" for continuous
Optional:
- user_id column (for duplicate checks)
- covariate columns (for balance checks)

Examples:
  python scripts/run_analysis.py --input data/ab_test_data.csv --metric converted --metric-type proportion
  python scripts/run_analysis.py --demo --metric converted --metric-type proportion
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd


# ---- Utilities ---------------------------------------------------------------

def _to_dict(obj: Any) -> Any:
    """Convert dataclasses / numpy scalars to JSON-serializable objects."""
    if is_dataclass(obj):
        return asdict(obj)
    # pandas / numpy scalars
    try:
        import numpy as np  # type: ignore
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except Exception:
        pass
    if isinstance(obj, dict):
        return {k: _to_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_dict(v) for v in obj]
    return obj


def _infer_control_treatment(groups: pd.Series) -> tuple[str, str]:
    uniq = list(pd.unique(groups.dropna()))
    if len(uniq) != 2:
        raise ValueError(f"Expected exactly 2 groups, found: {uniq}")
    # Prefer common names; otherwise use sorted
    normalized = {str(u).lower(): str(u) for u in uniq}
    for a, b in [("a", "b"), ("control", "treatment"), ("0", "1")]:
        if a in normalized and b in normalized:
            return normalized[a], normalized[b]
    uniq_sorted = sorted(map(str, uniq))
    return uniq_sorted[0], uniq_sorted[1]


# ---- Main analysis -----------------------------------------------------------

def run(
    df: pd.DataFrame,
    group_col: str,
    metric_col: str,
    metric_type: str,
    alpha: float,
    alternative: str,
    user_id_col: Optional[str] = None,
    out_json: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Run sanity checks + primary metric test.
    Returns a dict report.
    """
    # Import your library (adjust names if needed)
    # Sanity
    from abtest import sanity as sanity_mod  # type: ignore
    # Stats
    from abtest import stats as stats_mod  # type: ignore

    control_label, treatment_label = _infer_control_treatment(df[group_col])

    report: Dict[str, Any] = {
        "inputs": {
            "group_col": group_col,
            "metric_col": metric_col,
            "metric_type": metric_type,
            "alpha": alpha,
            "alternative": alternative,
            "control_label": control_label,
            "treatment_label": treatment_label,
        }
    }

    # Basic integrity
    if user_id_col and user_id_col in df.columns:
        dupes = df[user_id_col].duplicated().sum()
        report["integrity"] = {"duplicate_user_ids": int(dupes)}

    # Group counts
    counts = df[group_col].value_counts(dropna=False).to_dict()
    report["group_counts"] = {str(k): int(v) for k, v in counts.items()}

    # ---- Sanity checks ----
    sanity: Dict[str, Any] = {}

    # SRM (sample ratio mismatch): try common function names
    if hasattr(sanity_mod, "srm_chi2_test"):
        sanity["srm"] = _to_dict(
            sanity_mod.srm_chi2_test(counts={control_label: counts[control_label], treatment_label: counts[treatment_label]}, alpha=alpha)
        )
    elif hasattr(sanity_mod, "check_srm"):
        sanity["srm"] = _to_dict(
            sanity_mod.check_srm(df, group_col=group_col, alpha=alpha)
        )

    # Missingness
    if hasattr(sanity_mod, "check_missingness"):
        sanity["missingness"] = _to_dict(
            sanity_mod.check_missingness(df, cols=[metric_col], group_col=group_col)
        )

    report["sanity"] = sanity

    # ---- Primary metric test ----
    metric_report: Dict[str, Any] = {}
    df_c = df[df[group_col] == control_label]
    df_t = df[df[group_col] == treatment_label]

    if metric_type == "proportion":
        # Expect 0/1 (or bool) in metric_col
        c = df_c[metric_col].astype(float)
        t = df_t[metric_col].astype(float)

        c_succ = int(c.sum())
        c_n = int(c.shape[0])
        t_succ = int(t.sum())
        t_n = int(t.shape[0])

        metric_report["control_successes"] = c_succ
        metric_report["control_total"] = c_n
        metric_report["treatment_successes"] = t_succ
        metric_report["treatment_total"] = t_n

        if hasattr(stats_mod, "ztest_proportions"):
            res = stats_mod.ztest_proportions(
                c_succ, c_n, t_succ, t_n, alpha=alpha, alternative=alternative
            )
            metric_report["test"] = "ztest_proportions"
            metric_report["result"] = _to_dict(res)
        else:
            raise AttributeError("Expected stats.ztest_proportions to exist for proportion metrics.")

    elif metric_type == "continuous":
        c = df_c[metric_col].dropna().astype(float).to_list()
        t = df_t[metric_col].dropna().astype(float).to_list()

        if hasattr(stats_mod, "welch_ttest_means"):
            res = stats_mod.welch_ttest_means(
                c, t, alpha=alpha, alternative=alternative
            )
            metric_report["test"] = "welch_ttest_means"
            metric_report["result"] = _to_dict(res)
        elif hasattr(stats_mod, "ttest_means"):
            res = stats_mod.ttest_means(
                c, t, alpha=alpha, alternative=alternative
            )
            metric_report["test"] = "ttest_means"
            metric_report["result"] = _to_dict(res)
        else:
            raise AttributeError("Expected stats.welch_ttest_means or stats.ttest_means to exist for continuous metrics.")
    else:
        raise ValueError("metric_type must be 'proportion' or 'continuous'")

    report["metric"] = metric_report

    # Optionally write JSON
    if out_json:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(_to_dict(report), indent=2), encoding="utf-8")

    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=None, help="Path to CSV input")
    parser.add_argument("--demo", action="store_true", help="Generate demo data first (requires scripts/generate_data.py)")
    parser.add_argument("--demo-out", type=str, default="data/ab_test_data.csv", help="Where to write demo CSV")
    parser.add_argument("--group-col", type=str, default="group")
    parser.add_argument("--metric", type=str, required=True, help="Metric column name")
    parser.add_argument("--metric-type", type=str, choices=["proportion", "continuous"], required=True)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--alternative", type=str, default="two-sided", choices=["two-sided", "greater", "less"])
    parser.add_argument("--user-id-col", type=str, default=None)
    parser.add_argument("--out-json", type=str, default=None, help="Write a JSON report to this path")
    args = parser.parse_args()

    if args.demo:
        # Call your existing generator
        from scripts.generate_data import main as gen_main  # type: ignore

        # generate_data.py likely parses its own argv; easiest is to call it as a subprocess.
        # Here we do a simple fallback: if it exposes a function, use it; otherwise advise subprocess.
        try:
            # If your generate_data.py has a function we can call directly, do it here.
            # Otherwise, keep demo off and use `python scripts/generate_data.py ...`
            pass
        except Exception:
            pass

    if args.input is None and not args.demo:
        raise SystemExit("Provide --input path or use --demo")

    input_path = Path(args.demo_out if args.demo else args.input)
    df = pd.read_csv(input_path)

    report = run(
        df=df,
        group_col=args.group_col,
        metric_col=args.metric,
        metric_type=args.metric_type,
        alpha=args.alpha,
        alternative=args.alternative,
        user_id_col=args.user_id_col,
        out_json=Path(args.out_json) if args.out_json else None,
    )

    # Pretty console summary
    ctrl = report["inputs"]["control_label"]
    trt = report["inputs"]["treatment_label"]
    print(f"\nA/B Analysis: {ctrl} vs {trt}")
    print(f"Metric: {report['inputs']['metric_col']} ({report['inputs']['metric_type']})")
    print(f"Counts: {report['group_counts']}")
    if "srm" in report.get("sanity", {}):
        srm = report["sanity"]["srm"]
        if isinstance(srm, dict) and "p_value" in srm:
            print(f"SRM p-value: {srm['p_value']}")
    metric_res = report["metric"]["result"]
    if isinstance(metric_res, dict):
        pv = metric_res.get("p_value", None)
        eff = metric_res.get("effect", None) or metric_res.get("mean_diff", None)
        ci = (metric_res.get("ci_low", None), metric_res.get("ci_high", None))
        print(f"Test: {report['metric']['test']}")
        print(f"Effect: {eff}")
        print(f"CI: {ci}")
        print(f"p-value: {pv}")
    if args.out_json:
        print(f"\nWrote report: {args.out_json}")


if __name__ == "__main__":
    main()
