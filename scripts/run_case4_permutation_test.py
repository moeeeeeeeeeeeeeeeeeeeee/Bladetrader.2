"""
Run walk-forward permutation tests for Case 4 strategies.

Reads data/case4_earnings_validation.json and writes data/case4_permutation_results.json.
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

from finhack.research.case4_permutation import (  # noqa: E402
    DEFAULT_PERMUTATION_COUNT,
    DEFAULT_PERMUTATION_METHOD,
    best_walk_forward_p_value,
    run_permutation_search,
    run_walk_forward_permutation_tests,
    write_permutation_results,
)
from finhack.research.constants import (  # noqa: E402
    DEFAULT_MIN_CONFIDENCE,
    DEFAULT_MODEL_MIN_CONFIDENCE,
    DEFAULT_ROUND_TRIP_COST_BPS,
    DEFAULT_TRAIN_RATIO,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Case 4 walk-forward permutation tests")
    parser.add_argument(
        "--validation-path",
        default="data/case4_earnings_validation.json",
        help="Path to validation JSON from validate_case4_earnings.py",
    )
    parser.add_argument(
        "--output-path",
        default="data/case4_permutation_results.json",
        help="Where to write permutation test JSON",
    )
    parser.add_argument("--permutations", type=int, default=DEFAULT_PERMUTATION_COUNT)
    parser.add_argument(
        "--method",
        choices=("symbol_block", "month_block", "global"),
        default=DEFAULT_PERMUTATION_METHOD,
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=DEFAULT_TRAIN_RATIO)
    parser.add_argument("--cost-bps", type=float, default=DEFAULT_ROUND_TRIP_COST_BPS)
    parser.add_argument("--min-confidence", type=float, default=DEFAULT_MIN_CONFIDENCE)
    parser.add_argument("--model-min-confidence", type=float, default=DEFAULT_MODEL_MIN_CONFIDENCE)
    parser.add_argument(
        "--target-p",
        type=float,
        default=None,
        help="If set, grid-search model confidence until best walk-forward p-value is below this.",
    )
    parser.add_argument(
        "--refresh-validation",
        action="store_true",
        help="Re-run validate_case4_earnings.py --skip-news-ingest before permutations.",
    )
    args = parser.parse_args()

    validation_path = ROOT / args.validation_path
    if not validation_path.exists() and not args.refresh_validation:
        raise SystemExit(
            f"Missing {validation_path}. Run: python scripts/validate_case4_earnings.py"
        )

    common = {
        "n_permutations": args.permutations,
        "method": args.method,
        "seed": args.seed,
        "train_ratio": args.train_ratio,
        "round_trip_cost_bps": args.cost_bps,
        "min_confidence": args.min_confidence,
    }

    if args.refresh_validation:
        import os
        import subprocess

        env = os.environ.copy()
        env["PYTHONPATH"] = "src"
        print("Refreshing validation from current news store (--skip-news-ingest) ...")
        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "validate_case4_earnings.py"), "--skip-news-ingest"],
            cwd=ROOT,
            check=True,
            env=env,
        )

    if args.target_p is not None:
        print(
            f"Searching for walk-forward p < {args.target_p} with up to 200 permutations "
            f"({args.method}) ..."
        )
        payload = run_permutation_search(
            validation_path,
            target_p=args.target_p,
            refresh_validation=False,
            **common,
        )
    else:
        print(
            f"Running {args.permutations} permutations ({args.method}) on "
            f"{validation_path.as_posix()} ..."
        )
        payload = run_walk_forward_permutation_tests(
            validation_path,
            model_min_confidence=args.model_min_confidence,
            **common,
        )

    out_path = ROOT / args.output_path
    write_permutation_results(payload, out_path)
    print(f"Wrote: {out_path.as_posix()}")

    metric, p_val, strategy_key = best_walk_forward_p_value(payload)
    summary: dict[str, object] = {
        "permutations": args.permutations,
        "method": args.method,
        "best_walk_forward_strategy": strategy_key,
        "best_walk_forward_metric": metric,
        "best_walk_forward_p_value": p_val,
        "data_quality_warnings": payload["data_quality"].get("warnings", []),
    }
    if "search" in payload:
        summary["search"] = payload["search"]
    for name, block in payload["strategies"].items():
        obs = block["observed"]["metrics"]
        pvals = block["p_values_one_sided_greater"]
        summary[name] = {
            "trades": obs.get("trades"),
            "cumulative_return_pct": obs.get("cumulative_return_pct"),
            "sharpe_ratio": obs.get("sharpe_ratio"),
            "hit_rate": obs.get("hit_rate"),
            "p_cumulative_return": pvals.get("cumulative_return_pct"),
            "p_sharpe": pvals.get("sharpe_ratio"),
            "p_hit_rate": pvals.get("hit_rate"),
        }
        if "fold_mean_accuracy" in block["observed"]:
            summary[name]["fold_mean_accuracy"] = block["observed"].get("fold_mean_accuracy")
            summary[name]["p_fold_mean_accuracy"] = pvals.get("fold_mean_accuracy")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
