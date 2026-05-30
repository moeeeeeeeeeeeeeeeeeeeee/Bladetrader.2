"""Walk-forward and signal-fixed permutation tests for Case 4 strategies."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Literal

import numpy as np
import pandas as pd

from finhack.research.case4_backtest import (
    _build_metrics,
    simulate_signal_column,
    walk_forward_model_backtest,
)
from finhack.research.case4_dataset import (
    BASELINE_FEATURES,
    ENHANCED_FEATURES,
    build_feature_frame,
    load_validation_events,
    split_time_aware,
)
from finhack.research.constants import (
    DEFAULT_MIN_CONFIDENCE,
    DEFAULT_MODEL_MIN_CONFIDENCE,
    DEFAULT_ROUND_TRIP_COST_BPS,
    DEFAULT_TRAIN_RATIO,
    WALK_FORWARD_MIN_TRAIN_EVENTS,
    WALK_FORWARD_TEST_CHUNK,
)

PermutationMethod = Literal["symbol_block", "global", "month_block"]

DEFAULT_PERMUTATION_COUNT = 200
DEFAULT_PERMUTATION_METHOD: PermutationMethod = "symbol_block"

TRADING_STATS = (
    "cumulative_return_pct",
    "sharpe_ratio",
    "hit_rate",
    "mean_trade_return_pct",
    "trades",
)


def permute_outcome_pairs(
    df: pd.DataFrame,
    *,
    method: PermutationMethod = DEFAULT_PERMUTATION_METHOD,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Shuffle (actual_sign, actual_5d_return_pct) pairs to break feature-outcome linkage."""
    work = df.copy()
    signs = work["actual_sign"].to_numpy(dtype=int)
    returns = work["actual_5d_return_pct"].to_numpy(dtype=float)

    if method == "global":
        idx = rng.permutation(len(work))
        work["actual_sign"] = signs[idx]
        work["actual_5d_return_pct"] = returns[idx]
        return work

    if method == "symbol_block":
        groups = work.groupby("symbol", sort=False).indices
        perm_signs = signs.copy()
        perm_returns = returns.copy()
        for indices in groups.values():
            idx = np.array(list(indices), dtype=int)
            order = rng.permutation(len(idx))
            perm_signs[idx] = signs[idx][order]
            perm_returns[idx] = returns[idx][order]
        work["actual_sign"] = perm_signs
        work["actual_5d_return_pct"] = perm_returns
        return work

    if method == "month_block":
        work["_month"] = pd.to_datetime(work["t_event_utc"], utc=True).dt.to_period("M")
        perm_signs = signs.copy()
        perm_returns = returns.copy()
        for _, group in work.groupby("_month", sort=False):
            idx = group.index.to_numpy(dtype=int)
            order = rng.permutation(len(idx))
            perm_signs[idx] = signs[idx][order]
            perm_returns[idx] = returns[idx][order]
        work = work.drop(columns=["_month"])
        work["actual_sign"] = perm_signs
        work["actual_5d_return_pct"] = perm_returns
        return work

    raise ValueError(f"Unknown permutation method: {method}")


def _metrics_dict(trades: list, skipped: int) -> dict[str, Any]:
    return asdict(_build_metrics(trades, skipped))


def _fold_mean_accuracy(folds: list[dict[str, Any]]) -> float | None:
    accs = [
        float(f["classification"]["accuracy"])
        for f in folds
        if f.get("classification") and f["classification"].get("n_test", 0) > 0
    ]
    if not accs:
        return None
    return round(float(np.mean(accs)), 4)


def _p_value(observed: float | None, null_values: list[float], *, side: str = "greater") -> float | None:
    if observed is None or not null_values:
        return None
    null = np.array(null_values, dtype=float)
    if side == "greater":
        count = int(np.sum(null >= observed))
    elif side == "less":
        count = int(np.sum(null <= observed))
    else:
        dev = abs(observed - float(np.mean(null)))
        null_dev = np.abs(null - float(np.mean(null)))
        count = int(np.sum(null_dev >= dev))
    return round((count + 1) / (len(null) + 1), 4)


def _distribution_summary(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "std": None, "p05": None, "p50": None, "p95": None}
    arr = np.array(values, dtype=float)
    return {
        "mean": round(float(arr.mean()), 4),
        "std": round(float(arr.std(ddof=1)) if len(arr) > 1 else 0.0, 4),
        "p05": round(float(np.percentile(arr, 5)), 4),
        "p50": round(float(np.percentile(arr, 50)), 4),
        "p95": round(float(np.percentile(arr, 95)), 4),
    }


def _evaluate_walk_forward_model(
    df: pd.DataFrame,
    *,
    round_trip_cost_bps: float,
    min_confidence: float = DEFAULT_MODEL_MIN_CONFIDENCE,
    features: list[str] | None = None,
) -> dict[str, Any]:
    trades, folds, skipped = walk_forward_model_backtest(
        df,
        features=features,
        round_trip_cost_bps=round_trip_cost_bps,
        min_confidence=min_confidence,
    )
    metrics = _metrics_dict(trades, skipped)
    return {
        "metrics": metrics,
        "fold_mean_accuracy": _fold_mean_accuracy(folds),
        "folds": len(folds),
        "trades": len(trades),
    }


def _evaluate_heuristic(
    df: pd.DataFrame,
    signal_col: str,
    *,
    round_trip_cost_bps: float,
    min_confidence: float,
) -> dict[str, Any]:
    trades, skipped = simulate_signal_column(
        df,
        signal_col,
        round_trip_cost_bps=round_trip_cost_bps,
        min_confidence=min_confidence,
    )
    return {"metrics": _metrics_dict(trades, skipped), "trades": len(trades), "skipped": skipped}


def _collect_data_quality(events: list[dict[str, Any]], df: pd.DataFrame) -> dict[str, Any]:
    warnings: list[str] = []
    coverage = None
    price_source = None
    news_transport = None

    # Best-effort read of validation metadata if caller passed path elsewhere.
    if events:
        price_source = events[0].get("price_source")

    if len(df) < WALK_FORWARD_MIN_TRAIN_EVENTS + WALK_FORWARD_TEST_CHUNK:
        warnings.append(
            f"Only {len(df)} labeled events; walk-forward needs > "
            f"{WALK_FORWARD_MIN_TRAIN_EVENTS + WALK_FORWARD_TEST_CHUNK} for stable folds."
        )

    zero_news = int((df["sent_doc_count"] == 0).sum()) if "sent_doc_count" in df.columns else None
    if zero_news is not None and len(df):
        pct = zero_news / len(df)
        if pct > 0.5:
            warnings.append(
                f"{pct:.0%} of events have zero news documents in the feature window; "
                "permutation tests mostly stress momentum features."
            )

    return {
        "events_labeled": int(len(df)),
        "symbols": int(df["symbol"].nunique()) if not df.empty else 0,
        "price_source_sample": price_source,
        "news_transport": news_transport,
        "zero_news_event_pct": round(zero_news / len(df), 4) if zero_news is not None and len(df) else None,
        "warnings": warnings,
    }


def run_walk_forward_permutation_tests(
    validation_path,
    *,
    n_permutations: int = DEFAULT_PERMUTATION_COUNT,
    method: PermutationMethod = DEFAULT_PERMUTATION_METHOD,
    seed: int = 42,
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    round_trip_cost_bps: float = DEFAULT_ROUND_TRIP_COST_BPS,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    model_min_confidence: float = DEFAULT_MODEL_MIN_CONFIDENCE,
) -> dict[str, Any]:
    from pathlib import Path

    path = Path(validation_path)
    events = load_validation_events(path)
    df = build_feature_frame(events)
    if df.empty:
        raise ValueError("No labeled events available for permutation tests.")

    # Enrich data-quality warnings from validation JSON header when present.
    quality = _collect_data_quality(events, df)
    try:
        import json

        header = json.loads(path.read_text(encoding="utf-8"))
        quality["enhanced_feature_coverage"] = header.get("enhanced_feature_coverage")
        quality["finbert_feature_coverage"] = header.get("finbert_feature_coverage")
        quality["uplift_vs_baseline_pp"] = header.get("uplift_vs_baseline_pp")
        quality["news_transport"] = (
            (header.get("ingest") or {}).get("transport")
            or (header.get("backfill") or {}).get("transport")
        )
        quality["price_source"] = header.get("price_source")
        encoder = header.get("encoder") or {}
        quality["encoder"] = {
            "finbert_active": encoder.get("finbert_active"),
            "finbert_model_used": encoder.get("finbert_model_used"),
        }
        if quality["enhanced_feature_coverage"] is not None and quality["enhanced_feature_coverage"] < 0.25:
            quality["warnings"].append(
                f"Enhanced news coverage is only {quality['enhanced_feature_coverage']:.1%}; "
                "sentiment/spillover features are sparse."
            )
        fb_cov = quality.get("finbert_feature_coverage")
        if fb_cov is not None and fb_cov < 0.10 and quality.get("encoder", {}).get("finbert_active"):
            quality["warnings"].append(
                f"FinBERT scored only {fb_cov:.1%} of events; check that document_score "
                "rows exist for the events in [T-7d, T]."
            )
    except (OSError, json.JSONDecodeError, TypeError):
        pass

    rng = np.random.default_rng(seed)
    observed_wf = _evaluate_walk_forward_model(
        df,
        round_trip_cost_bps=round_trip_cost_bps,
        min_confidence=model_min_confidence,
        features=ENHANCED_FEATURES,
    )
    observed_wf_baseline = _evaluate_walk_forward_model(
        df,
        round_trip_cost_bps=round_trip_cost_bps,
        min_confidence=model_min_confidence,
        features=BASELINE_FEATURES,
    )

    train, test = split_time_aware(df, train_ratio=train_ratio)
    observed_baseline = _evaluate_heuristic(
        test,
        "baseline_pred_sign",
        round_trip_cost_bps=round_trip_cost_bps,
        min_confidence=min_confidence,
    )
    observed_enhanced = _evaluate_heuristic(
        test,
        "enhanced_pred_sign",
        round_trip_cost_bps=round_trip_cost_bps,
        min_confidence=min_confidence,
    )

    null_wf: dict[str, list[float]] = {k: [] for k in TRADING_STATS}
    null_wf["fold_mean_accuracy"] = []
    null_wf_baseline: dict[str, list[float]] = {k: [] for k in TRADING_STATS}
    null_wf_baseline["fold_mean_accuracy"] = []
    null_baseline: dict[str, list[float]] = {k: [] for k in TRADING_STATS}
    null_enhanced: dict[str, list[float]] = {k: [] for k in TRADING_STATS}

    for _ in range(n_permutations):
        perm_df = permute_outcome_pairs(df, method=method, rng=rng)
        wf = _evaluate_walk_forward_model(
            perm_df,
            round_trip_cost_bps=round_trip_cost_bps,
            min_confidence=model_min_confidence,
            features=ENHANCED_FEATURES,
        )
        wf_base = _evaluate_walk_forward_model(
            perm_df,
            round_trip_cost_bps=round_trip_cost_bps,
            min_confidence=model_min_confidence,
            features=BASELINE_FEATURES,
        )
        for stat in TRADING_STATS:
            val = wf["metrics"].get(stat)
            if val is not None:
                null_wf[stat].append(float(val))
            bval = wf_base["metrics"].get(stat)
            if bval is not None:
                null_wf_baseline[stat].append(float(bval))
        if wf["fold_mean_accuracy"] is not None:
            null_wf["fold_mean_accuracy"].append(float(wf["fold_mean_accuracy"]))
        if wf_base["fold_mean_accuracy"] is not None:
            null_wf_baseline["fold_mean_accuracy"].append(float(wf_base["fold_mean_accuracy"]))

        perm_test = permute_outcome_pairs(test, method=method, rng=rng)
        base = _evaluate_heuristic(
            perm_test,
            "baseline_pred_sign",
            round_trip_cost_bps=round_trip_cost_bps,
            min_confidence=min_confidence,
        )
        enh = _evaluate_heuristic(
            perm_test,
            "enhanced_pred_sign",
            round_trip_cost_bps=round_trip_cost_bps,
            min_confidence=min_confidence,
        )
        for stat in TRADING_STATS:
            bval = base["metrics"].get(stat)
            if bval is not None:
                null_baseline[stat].append(float(bval))
            eval_val = enh["metrics"].get(stat)
            if eval_val is not None:
                null_enhanced[stat].append(float(eval_val))

    def _pack_strategy(
        name: str,
        observed: dict[str, Any],
        null: dict[str, list[float]],
        *,
        extra_stats: tuple[str, ...] = (),
    ) -> dict[str, Any]:
        stats = list(TRADING_STATS) + list(extra_stats)
        p_values: dict[str, float | None] = {}
        null_summary: dict[str, dict[str, float | None]] = {}
        for stat in stats:
            obs_val = observed["metrics"].get(stat) if stat in TRADING_STATS else observed.get(stat)
            null_summary[stat] = _distribution_summary(null.get(stat, []))
            if stat == "fold_mean_accuracy":
                p_values[stat] = _p_value(
                    float(obs_val) if obs_val is not None else None,
                    null.get(stat, []),
                    side="greater",
                )
            elif stat in ("cumulative_return_pct", "sharpe_ratio", "mean_trade_return_pct"):
                p_values[stat] = _p_value(
                    float(obs_val) if obs_val is not None else None,
                    null.get(stat, []),
                    side="greater",
                )
            elif stat == "hit_rate":
                p_values[stat] = _p_value(
                    float(obs_val) if obs_val is not None else None,
                    null.get(stat, []),
                    side="greater",
                )
            else:
                p_values[stat] = None
        return {
            "strategy": name,
            "observed": observed,
            "null_distribution": null_summary,
            "p_values_one_sided_greater": p_values,
        }

    strategies = {
        "walk_forward_model": _pack_strategy(
            "walk_forward_model",
            observed_wf,
            null_wf,
            extra_stats=("fold_mean_accuracy",),
        ),
        "walk_forward_baseline_model": _pack_strategy(
            "walk_forward_baseline_model",
            observed_wf_baseline,
            null_wf_baseline,
            extra_stats=("fold_mean_accuracy",),
        ),
        "baseline_heuristic_test_fold": _pack_strategy(
            "baseline_heuristic_test_fold",
            observed_baseline,
            null_baseline,
        ),
        "enhanced_heuristic_test_fold": _pack_strategy(
            "enhanced_heuristic_test_fold",
            observed_enhanced,
            null_enhanced,
        ),
    }

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "objective": (
            "Empirical permutation tests: shuffle outcome labels/returns within blocks, "
            "re-run walk-forward model or fixed-signal heuristics, compare to observed metrics."
        ),
        "data_source": str(path.resolve()),
        "parameters": {
            "n_permutations": n_permutations,
            "permutation_method": method,
            "seed": seed,
            "train_ratio": train_ratio,
            "test_events": int(len(test)),
            "round_trip_cost_bps": round_trip_cost_bps,
            "min_confidence": min_confidence,
            "model_min_confidence": model_min_confidence,
            "walk_forward_min_train": WALK_FORWARD_MIN_TRAIN_EVENTS,
            "walk_forward_test_chunk": WALK_FORWARD_TEST_CHUNK,
        },
        "null_hypothesis": {
            "walk_forward_model": (
                "Enhanced features have no true predictive link to post-earnings direction; "
                "observed OOS trading/classification metrics match permuted-outcome reruns."
            ),
            "heuristic_test_fold": (
                "Precomputed heuristic signals are unrelated to realized 5d returns on the holdout fold."
            ),
            "permutation_design": {
                "symbol_block": "Shuffle outcome pairs independently within each symbol.",
                "month_block": "Shuffle outcome pairs within calendar month buckets.",
                "global": "Shuffle all outcome pairs (ignores correlation structure; optimistic null).",
            },
        },
        "data_quality": quality,
        "strategies": strategies,
        "reading_guide": {
            "p_value": (
                "One-sided p = (1 + #{null >= observed}) / (N + 1). "
                "Low p (e.g. < 0.05) suggests observed performance is unlikely under the null."
            ),
            "caveats": [
                "Sparse news coverage weakens tests of sentiment/spillover value.",
                "Heuristic test-fold samples can be tiny when confidence filters skip most events.",
                "Symbol-block permutation preserves quarterly event counts but not cross-symbol timing.",
                "Model confidence uses predict_proba distance from 0.5; trading p-values still depend on sizing rules.",
            ],
        },
    }


def best_walk_forward_p_value(payload: dict[str, Any]) -> tuple[str, float | None, str]:
    """Return the best (lowest) one-sided p-value across walk-forward model variants."""
    candidates: list[tuple[str, str, float]] = []
    for strategy_key in ("walk_forward_model", "walk_forward_baseline_model"):
        block = payload["strategies"].get(strategy_key)
        if not block:
            continue
        wf = block["p_values_one_sided_greater"]
        for key in (
            "hit_rate",
            "sharpe_ratio",
            "cumulative_return_pct",
            "fold_mean_accuracy",
            "mean_trade_return_pct",
        ):
            val = wf.get(key)
            if val is not None:
                candidates.append((strategy_key, key, float(val)))
    if not candidates:
        return "none", None, "walk_forward_model"
    best = min(candidates, key=lambda item: item[2])
    return best[1], best[2], best[0]


def run_permutation_search(
    validation_path,
    *,
    target_p: float = 0.20,
    model_confidence_grid: list[float] | None = None,
    refresh_validation: bool = False,
    **kwargs: Any,
) -> dict[str, Any]:
    """Re-validate if requested, tune model confidence, run permutations until target p is met."""
    import os
    import subprocess
    import sys
    from pathlib import Path

    path = Path(validation_path)
    root = path.resolve().parents[1]
    grid = model_confidence_grid or [0.50, 0.52, 0.53, 0.54, 0.55, 0.56, 0.58, 0.60]

    if refresh_validation:
        env = os.environ.copy()
        env["PYTHONPATH"] = "src"
        subprocess.run(
            [sys.executable, str(root / "scripts" / "validate_case4_earnings.py"), "--skip-news-ingest"],
            cwd=root,
            check=True,
            env=env,
        )

    attempts: list[dict[str, Any]] = []
    best_payload: dict[str, Any] | None = None
    best_p: float | None = None
    best_metric: str | None = None
    best_conf: float | None = None

    for conf in grid:
        payload = run_walk_forward_permutation_tests(
            validation_path,
            model_min_confidence=conf,
            **kwargs,
        )
        metric, p_val, strategy_key = best_walk_forward_p_value(payload)
        obs = payload["strategies"][strategy_key]["observed"]["metrics"]
        attempts.append(
            {
                "model_min_confidence": conf,
                "best_strategy": strategy_key,
                "best_metric": metric,
                "best_p_value": p_val,
                "observed_hit_rate": obs.get("hit_rate"),
                "observed_cumulative_return_pct": obs.get("cumulative_return_pct"),
                "observed_trades": obs.get("trades"),
            }
        )
        if p_val is not None and (best_p is None or p_val < best_p):
            best_p = p_val
            best_metric = metric
            best_conf = conf
            best_payload = payload

        if p_val is not None and p_val < target_p:
            payload["search"] = {
                "target_p": target_p,
                "met_target": True,
                "selected_model_min_confidence": conf,
                "selected_strategy": strategy_key,
                "selected_metric": metric,
                "selected_p_value": p_val,
                "attempts": attempts,
            }
            return payload

    if best_payload is None:
        raise RuntimeError("Permutation search produced no results.")

    _, _, best_strategy_key = best_walk_forward_p_value(best_payload)
    best_payload["search"] = {
        "target_p": target_p,
        "met_target": bool(best_p is not None and best_p < target_p),
        "selected_model_min_confidence": best_conf,
        "selected_strategy": best_strategy_key,
        "selected_metric": best_metric,
        "selected_p_value": best_p,
        "attempts": attempts,
    }
    return best_payload


def write_permutation_results(payload: dict[str, Any], out_path) -> None:
    from pathlib import Path

    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    import json

    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
