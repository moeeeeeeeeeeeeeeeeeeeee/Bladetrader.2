"""
Backtest post-earnings strategies on live-validated event data.

Reads data/case4_earnings_validation.json (from validate_case4_earnings.py).
Writes data/case4_backtest_summary.json with PnL metrics for:
- baseline momentum heuristic
- enhanced sentiment/spillover heuristic
- sklearn model (out-of-sample test fold)
- walk-forward model (expanding window)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

load_dotenv(ROOT / ".env")

from finhack.research.case4_backtest import run_full_backtest, write_backtest_summary
from finhack.research.constants import (
    DEFAULT_HOLDOUT_DAYS,
    DEFAULT_MIN_CONFIDENCE,
    DEFAULT_ROUND_TRIP_COST_BPS,
    DEFAULT_TARGET_TRADES_PER_MONTH,
    DEFAULT_TOP_K_LONG_PER_FOLD,
    DEFAULT_TOP_K_SHORT_PER_FOLD,
    DEFAULT_TRAIN_RATIO,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Case 4 earnings backtest (live validation input)")
    parser.add_argument(
        "--validation-path",
        default="data/case4_earnings_validation.json",
        help="Path to validation JSON from validate_case4_earnings.py",
    )
    parser.add_argument(
        "--output-path",
        default="data/case4_backtest_summary.json",
        help="Where to write backtest summary JSON",
    )
    parser.add_argument("--train-ratio", type=float, default=DEFAULT_TRAIN_RATIO)
    parser.add_argument("--cost-bps", type=float, default=DEFAULT_ROUND_TRIP_COST_BPS)
    parser.add_argument("--min-confidence", type=float, default=DEFAULT_MIN_CONFIDENCE)
    parser.add_argument(
        "--top-k-long",
        type=int,
        default=DEFAULT_TOP_K_LONG_PER_FOLD,
        help="High-conviction lane: top-K longs per walk-forward fold (default 1).",
    )
    parser.add_argument(
        "--top-k-short",
        type=int,
        default=DEFAULT_TOP_K_SHORT_PER_FOLD,
        help="High-conviction lane: top-K shorts per walk-forward fold (default 1).",
    )
    parser.add_argument(
        "--target-trades-per-month",
        type=float,
        default=DEFAULT_TARGET_TRADES_PER_MONTH,
        help="Threshold-tuned lane: tune train-only confidence cutoff to hit this trade cadence.",
    )
    parser.add_argument(
        "--holdout-days",
        type=int,
        default=DEFAULT_HOLDOUT_DAYS,
        help="Sacred holdout window (calendar days) reserved from all tuning and walk-forward.",
    )
    parser.add_argument(
        "--feature-cache",
        default="data/case4_feature_frame_cache.pkl",
        help="Pickle cache for the leakage-safe feature frame; speeds up repeated runs.",
    )
    parser.add_argument(
        "--rebuild-features",
        action="store_true",
        help="Force re-computation of the feature frame even if the cache file exists.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-step progress prints.",
    )
    args = parser.parse_args()

    validation_path = ROOT / args.validation_path
    if not validation_path.exists():
        raise SystemExit(
            f"Missing {validation_path}. Run: python scripts/validate_case4_earnings.py"
        )

    feature_cache_path = (ROOT / args.feature_cache) if args.feature_cache else None
    payload = run_full_backtest(
        validation_path,
        train_ratio=args.train_ratio,
        round_trip_cost_bps=args.cost_bps,
        min_confidence=args.min_confidence,
        top_k_long_per_fold=args.top_k_long,
        top_k_short_per_fold=args.top_k_short,
        target_trades_per_month=args.target_trades_per_month,
        holdout_days=args.holdout_days,
        feature_cache_path=feature_cache_path,
        rebuild_features=args.rebuild_features,
        progress=not args.quiet,
    )
    out_path = ROOT / args.output_path
    write_backtest_summary(payload, out_path)
    print(f"Wrote: {out_path.as_posix()}")

    summary: dict[str, object] = {
        "events_labeled": payload["events_labeled"],
        "events_labeled_with_holdout": payload.get("events_labeled_with_holdout"),
        "test_events": payload["parameters"]["test_events"],
        "best_strategy": payload["best_out_of_sample_strategy"],
        "holdout_events": payload["parameters"].get("holdout_events"),
        "holdout_days": payload["parameters"].get("holdout_days"),
    }
    for name, block in payload["strategies"].items():
        m = block["metrics"]
        summary[name] = {
            "trades": m["trades"],
            "hit_rate": m["hit_rate"],
            "cumulative_return_pct": m["cumulative_return_pct"],
            "sharpe_ratio": m["sharpe_ratio"],
            "max_drawdown_pct": m["max_drawdown_pct"],
        }
    fi = payload.get("feature_importance") or {}
    summary["feature_importance_enhanced_top3"] = (
        [f["feature"] for f in (fi.get("enhanced") or {}).get("features_ranked", [])[:3]]
        if fi.get("enhanced") else None
    )
    holdout = payload.get("locked_holdout_report")
    if holdout:
        summary["locked_holdout"] = {
            "events": holdout["holdout_events"],
            "window": f"{holdout['holdout_start_utc']} -> {holdout['holdout_end_utc']}",
            "top_k": {
                "trades": holdout["top_k"]["metrics"]["trades"],
                "hit_rate": holdout["top_k"]["metrics"]["hit_rate"],
                "cumulative_return_pct": holdout["top_k"]["metrics"]["cumulative_return_pct"],
            },
            "threshold_tuned": {
                "trades": holdout["threshold_tuned"]["metrics"]["trades"],
                "hit_rate": holdout["threshold_tuned"]["metrics"]["hit_rate"],
                "cumulative_return_pct": holdout["threshold_tuned"]["metrics"]["cumulative_return_pct"],
                "threshold_from_working_set": holdout["threshold_tuned"]["threshold_from_working_set"],
            },
        }
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
