"""Event-level and walk-forward backtests for post-earnings Case 4 strategies."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression

from finhack.research.case4_dataset import (
    ENHANCED_FEATURES,
    BASELINE_FEATURES,
    _select_features,
    build_feature_frame,
    fit_predict,
    load_validation_events,
    event_key,
    split_time_aware,
)
from finhack.research.model_store import fit_model, predict_proba
from finhack.research.constants import (
    DEFAULT_HOLDOUT_DAYS,
    DEFAULT_MAX_POSITION_WEIGHT,
    DEFAULT_MIN_CONFIDENCE,
    DEFAULT_MIN_POSITION_WEIGHT,
    DEFAULT_MODEL_MIN_CONFIDENCE,
    DEFAULT_PERMUTATION_IMPORTANCE_REPEATS,
    DEFAULT_PERMUTATION_IMPORTANCE_SEED,
    DEFAULT_ROUND_TRIP_COST_BPS,
    DEFAULT_TARGET_TRADES_PER_MONTH,
    DEFAULT_TOP_K_LONG_PER_FOLD,
    DEFAULT_TOP_K_SHORT_PER_FOLD,
    DEFAULT_TRAIN_RATIO,
    WALK_FORWARD_MIN_TRAIN_EVENTS,
    WALK_FORWARD_TEST_CHUNK,
)

SignalSource = Literal["baseline", "enhanced_heuristic", "model_enhanced"]


@dataclass
class TradeRecord:
    symbol: str
    t_event_utc: str
    signal: int
    actual_5d_return_pct: float
    gross_return_pct: float
    net_return_pct: float
    confidence: float
    position_weight: float
    won: bool


@dataclass
class PathTradeRecord:
    symbol: str
    t_event_utc: str
    signal: int
    path_return_pct: float
    net_return_pct: float
    exit_reason: str
    stop_hit: bool
    target_hit: bool
    confidence: float
    position_weight: float
    won: bool


@dataclass
class StrategyMetrics:
    trades: int
    skipped: int
    hit_rate: float | None
    mean_trade_return_pct: float | None
    median_trade_return_pct: float | None
    cumulative_return_pct: float | None
    max_drawdown_pct: float | None
    sharpe_ratio: float | None
    total_wins: int
    total_losses: int
    stop_hit_rate: float | None = None
    target_hit_rate: float | None = None
    mean_holding_days: float | None = None


def _position_weight(
    confidence: float,
    *,
    min_weight: float = DEFAULT_MIN_POSITION_WEIGHT,
    max_weight: float = DEFAULT_MAX_POSITION_WEIGHT,
) -> float:
    conf = max(0.0, min(1.0, float(confidence)))
    return round(min(max_weight, max(min_weight, min_weight + conf * (max_weight - min_weight))), 4)


def _sign_to_signal(raw: int) -> int:
    if raw > 0:
        return 1
    if raw < 0:
        return -1
    return 0


def _trade_gross(signal: int, actual_return_pct: float) -> float:
    return float(signal) * float(actual_return_pct)


def _apply_costs(gross_pct: float, round_trip_cost_bps: float) -> float:
    cost_pct = round_trip_cost_bps / 100.0
    return gross_pct - cost_pct


def _max_drawdown_pct(equity_curve: list[float]) -> float | None:
    if len(equity_curve) < 2:
        return None
    peak = equity_curve[0]
    max_dd = 0.0
    for val in equity_curve:
        peak = max(peak, val)
        if peak > 0:
            dd = ((peak - val) / peak) * 100.0
            max_dd = max(max_dd, dd)
    return round(max_dd, 4)


def _sharpe_ratio(returns_pct: list[float], periods_per_year: float = 50.0) -> float | None:
    if len(returns_pct) < 2:
        return None
    arr = np.array(returns_pct, dtype=float)
    std = float(arr.std(ddof=1))
    if std == 0:
        return None
    mean = float(arr.mean())
    annualized = (mean / std) * math.sqrt(periods_per_year)
    return round(annualized, 4)


def _build_metrics(trades: list[TradeRecord], skipped: int) -> StrategyMetrics:
    if not trades:
        return StrategyMetrics(
            trades=0,
            skipped=skipped,
            hit_rate=None,
            mean_trade_return_pct=None,
            median_trade_return_pct=None,
            cumulative_return_pct=None,
            max_drawdown_pct=None,
            sharpe_ratio=None,
            total_wins=0,
            total_losses=0,
        )

    net_returns = [t.net_return_pct * t.position_weight for t in trades]
    wins = sum(1 for r in net_returns if r > 0)
    losses = sum(1 for r in net_returns if r <= 0)

    equity = 100.0
    curve = [equity]
    for r in net_returns:
        equity *= 1.0 + (r / 100.0)
        curve.append(equity)

    cumulative = round((equity / 100.0 - 1.0) * 100.0, 4)
    return StrategyMetrics(
        trades=len(trades),
        skipped=skipped,
        hit_rate=round(wins / len(trades), 4),
        mean_trade_return_pct=round(float(np.mean(net_returns)), 4),
        median_trade_return_pct=round(float(np.median(net_returns)), 4),
        cumulative_return_pct=cumulative,
        max_drawdown_pct=_max_drawdown_pct(equity_curve=curve),
        sharpe_ratio=_sharpe_ratio(net_returns),
        total_wins=wins,
        total_losses=losses,
    )


def simulate_signal_column(
    df: pd.DataFrame,
    signal_col: str,
    *,
    confidence_col: str = "enhanced_confidence",
    round_trip_cost_bps: float = DEFAULT_ROUND_TRIP_COST_BPS,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
) -> tuple[list[TradeRecord], int]:
    trades: list[TradeRecord] = []
    skipped = 0
    for _, row in df.iterrows():
        signal = _sign_to_signal(int(row.get(signal_col, 0)))
        confidence = float(row.get(confidence_col, 0.0))
        if signal == 0:
            skipped += 1
            continue
        if confidence < min_confidence:
            skipped += 1
            continue
        actual = float(row["actual_5d_return_pct"])
        gross = _trade_gross(signal, actual)
        net = _apply_costs(gross, round_trip_cost_bps)
        weight = _position_weight(confidence)
        trades.append(
            TradeRecord(
                symbol=str(row["symbol"]),
                t_event_utc=str(row["t_event_utc"]),
                signal=signal,
                actual_5d_return_pct=round(actual, 4),
                gross_return_pct=round(gross, 4),
                net_return_pct=round(net, 4),
                confidence=round(confidence, 4),
                position_weight=weight,
                won=net > 0,
            )
        )
    return trades, skipped


def _event_signal_confidence(row: dict[str, Any]) -> float:
    conf = float(row.get("enhanced_confidence", 0.0) or 0.0)
    if conf > 0:
        return conf
    pre = abs(float(row.get("baseline_pre_7d_return_pct", 0.0) or 0.0))
    return min(1.0, 0.15 + pre / 12.0)


def simulate_path_trades(
    events: list[dict[str, Any]],
    *,
    signal_key: str = "enhanced_pred_sign",
    round_trip_cost_bps: float = DEFAULT_ROUND_TRIP_COST_BPS,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
) -> tuple[list[PathTradeRecord], int]:
    """Simulate trades using pre-computed stop/target path exits from validation."""
    trades: list[PathTradeRecord] = []
    skipped = 0
    for row in events:
        signal = _sign_to_signal(int(row.get(signal_key, 0)))
        confidence = _event_signal_confidence(row)
        path_ret = row.get("path_return_pct")
        if signal == 0:
            skipped += 1
            continue
        if confidence < min_confidence:
            skipped += 1
            continue
        if path_ret is None:
            skipped += 1
            continue
        gross = float(path_ret)
        net = _apply_costs(gross, round_trip_cost_bps)
        weight = _position_weight(confidence)
        trades.append(
            PathTradeRecord(
                symbol=str(row.get("symbol", "")),
                t_event_utc=str(row.get("t_event_utc", "")),
                signal=signal,
                path_return_pct=round(gross, 4),
                net_return_pct=round(net, 4),
                exit_reason=str(row.get("path_exit_reason", "time")),
                stop_hit=bool(row.get("path_stop_hit")),
                target_hit=bool(row.get("path_target_hit")),
                confidence=round(confidence, 4),
                position_weight=weight,
                won=net > 0,
            )
        )
    return trades, skipped


def _build_path_metrics(trades: list[PathTradeRecord], skipped: int) -> StrategyMetrics:
    if not trades:
        return StrategyMetrics(
            trades=0,
            skipped=skipped,
            hit_rate=None,
            mean_trade_return_pct=None,
            median_trade_return_pct=None,
            cumulative_return_pct=None,
            max_drawdown_pct=None,
            sharpe_ratio=None,
            total_wins=0,
            total_losses=0,
            stop_hit_rate=None,
            target_hit_rate=None,
            mean_holding_days=None,
        )
    net_returns = [t.net_return_pct * t.position_weight for t in trades]
    wins = sum(1 for r in net_returns if r > 0)
    losses = sum(1 for r in net_returns if r <= 0)
    equity = 100.0
    curve = [equity]
    for r in net_returns:
        equity *= 1.0 + (r / 100.0)
        curve.append(equity)
    stop_hits = sum(1 for t in trades if t.stop_hit)
    target_hits = sum(1 for t in trades if t.target_hit)
    return StrategyMetrics(
        trades=len(trades),
        skipped=skipped,
        hit_rate=round(wins / len(trades), 4),
        mean_trade_return_pct=round(float(np.mean(net_returns)), 4),
        median_trade_return_pct=round(float(np.median(net_returns)), 4),
        cumulative_return_pct=round((equity / 100.0 - 1.0) * 100.0, 4),
        max_drawdown_pct=_max_drawdown_pct(equity_curve=curve),
        sharpe_ratio=_sharpe_ratio(net_returns),
        total_wins=wins,
        total_losses=losses,
        stop_hit_rate=round(stop_hits / len(trades), 4),
        target_hit_rate=round(target_hits / len(trades), 4),
        mean_holding_days=None,
    )


def simulate_model_signals(
    df: pd.DataFrame,
    predictions: np.ndarray,
    probabilities: np.ndarray | None = None,
    *,
    round_trip_cost_bps: float = DEFAULT_ROUND_TRIP_COST_BPS,
    min_confidence: float = DEFAULT_MODEL_MIN_CONFIDENCE,
) -> tuple[list[TradeRecord], int]:
    work = df.copy()
    # Binary classifier: 0 = bearish (short), 1 = bullish (long).
    mapped = np.where(predictions.astype(int) >= 1, 1, -1)
    work["model_pred_sign"] = mapped
    if probabilities is not None:
        proba = np.asarray(probabilities, dtype=float)
        work["model_confidence"] = np.maximum(proba, 1.0 - proba)
    else:
        work["model_confidence"] = 0.55
    return simulate_signal_column(
        work,
        "model_pred_sign",
        confidence_col="model_confidence",
        round_trip_cost_bps=round_trip_cost_bps,
        min_confidence=min_confidence,
    )


def walk_forward_model_backtest(
    df: pd.DataFrame,
    *,
    features: list[str] | None = None,
    min_train: int = WALK_FORWARD_MIN_TRAIN_EVENTS,
    test_chunk: int = WALK_FORWARD_TEST_CHUNK,
    round_trip_cost_bps: float = DEFAULT_ROUND_TRIP_COST_BPS,
    min_confidence: float = DEFAULT_MODEL_MIN_CONFIDENCE,
) -> tuple[list[TradeRecord], list[dict[str, Any]], int]:
    feature_cols = features or ENHANCED_FEATURES
    all_trades: list[TradeRecord] = []
    folds: list[dict[str, Any]] = []
    skipped = 0

    ordered = df.sort_values("t_event_utc").reset_index(drop=True)
    n = len(ordered)
    if n <= min_train:
        return [], [], 0

    for start in range(min_train, n, test_chunk):
        end = min(start + test_chunk, n)
        train = ordered.iloc[:start].copy()
        test = ordered.iloc[start:end].copy()
        if test.empty:
            break
        preds, proba, fold_metrics = fit_predict(train, test, feature_cols)
        fold_trades, fold_skipped = simulate_model_signals(
            test,
            preds,
            proba,
            round_trip_cost_bps=round_trip_cost_bps,
            min_confidence=min_confidence,
        )
        skipped += fold_skipped
        all_trades.extend(fold_trades)
        folds.append(
            {
                "train_events": int(len(train)),
                "test_events": int(len(test)),
                "test_start_utc": str(test.iloc[0]["t_event_utc"]),
                "test_end_utc": str(test.iloc[-1]["t_event_utc"]),
                "classification": fold_metrics,
                "trades": len(fold_trades),
            }
        )
    return all_trades, folds, skipped


def _fit_logistic_for_ranking(
    train: pd.DataFrame,
    eval_df: pd.DataFrame,
    features: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]:
    """Fit on train and score both train and eval. Returns (train_proba,
    eval_preds, eval_proba, eval_signals, used_model).

    Despite the legacy name this routes through ``model_store.fit_model``
    so LightGBM is preferred over logistic when available.
    """
    x_train = _select_features(train, features)
    y_train = train["actual_sign"].to_numpy(dtype=int)
    x_eval = _select_features(eval_df, features)
    if len(np.unique(y_train)) < 2 or x_train.shape[1] == 0:
        eval_mom = (
            eval_df["baseline_pre_7d_return_pct"].to_numpy(dtype=float)
            if "baseline_pre_7d_return_pct" in eval_df.columns
            else np.zeros(len(eval_df))
        )
        train_mom = (
            train["baseline_pre_7d_return_pct"].to_numpy(dtype=float)
            if "baseline_pre_7d_return_pct" in train.columns
            else np.zeros(len(train))
        )
        eval_preds = (eval_mom > 0).astype(int)
        eval_proba = np.clip(0.5 + np.abs(eval_mom) / 24.0, 0.5, 0.95)
        train_proba = np.clip(0.5 + np.abs(train_mom) / 24.0, 0.5, 0.95)
        eval_signals = np.where(eval_preds >= 1, 1, -1)
        return train_proba, eval_preds, eval_proba, eval_signals, False
    model = fit_model(x_train, y_train, features)
    train_proba = predict_proba(model, x_train)
    eval_proba = predict_proba(model, x_eval)
    eval_preds = (eval_proba >= 0.5).astype(int)
    eval_signals = np.where(eval_preds >= 1, 1, -1)
    return train_proba, eval_preds, eval_proba, eval_signals, True


def _trade_record_from_row(
    row: pd.Series,
    signal: int,
    confidence: float,
    *,
    round_trip_cost_bps: float,
    position_weight: float = 1.0,
) -> TradeRecord:
    actual = float(row["actual_5d_return_pct"])
    gross = _trade_gross(signal, actual)
    net = _apply_costs(gross, round_trip_cost_bps)
    return TradeRecord(
        symbol=str(row["symbol"]),
        t_event_utc=str(row["t_event_utc"]),
        signal=int(signal),
        actual_5d_return_pct=round(actual, 4),
        gross_return_pct=round(gross, 4),
        net_return_pct=round(net, 4),
        confidence=round(float(confidence), 4),
        position_weight=float(position_weight),
        won=net > 0,
    )


def select_top_k_signals(
    signals: np.ndarray,
    confidences: np.ndarray,
    *,
    k_long: int,
    k_short: int,
) -> list[int]:
    """Return iloc indices of top-K longs + top-K shorts ranked by confidence (desc)."""
    long_idx = np.where(signals == 1)[0]
    short_idx = np.where(signals == -1)[0]
    long_pick = long_idx[np.argsort(-confidences[long_idx])][: max(0, int(k_long))]
    short_pick = short_idx[np.argsort(-confidences[short_idx])][: max(0, int(k_short))]
    return [int(i) for i in long_pick.tolist() + short_pick.tolist()]


def walk_forward_top_k_backtest(
    df: pd.DataFrame,
    *,
    features: list[str] | None = None,
    min_train: int = WALK_FORWARD_MIN_TRAIN_EVENTS,
    test_chunk: int = WALK_FORWARD_TEST_CHUNK,
    k_long_per_fold: int = DEFAULT_TOP_K_LONG_PER_FOLD,
    k_short_per_fold: int = DEFAULT_TOP_K_SHORT_PER_FOLD,
    round_trip_cost_bps: float = DEFAULT_ROUND_TRIP_COST_BPS,
) -> tuple[list[TradeRecord], list[dict[str, Any]], int]:
    """Walk-forward; per fold, trade only top-K longs + top-K shorts by model confidence.

    Equal-weighted positions (no confidence sizing) so the metric reflects pure selectivity.
    """
    feature_cols = features or ENHANCED_FEATURES
    ordered = df.sort_values("t_event_utc").reset_index(drop=True)
    n = len(ordered)
    if n <= min_train:
        return [], [], 0

    all_trades: list[TradeRecord] = []
    folds: list[dict[str, Any]] = []
    skipped = 0

    for start in range(min_train, n, test_chunk):
        end = min(start + test_chunk, n)
        train = ordered.iloc[:start].copy()
        test = ordered.iloc[start:end].copy()
        if test.empty:
            break
        _, _, test_proba, test_signals, _ = _fit_logistic_for_ranking(train, test, feature_cols)
        confidences = np.maximum(test_proba, 1.0 - test_proba)
        picks = select_top_k_signals(
            test_signals,
            confidences,
            k_long=k_long_per_fold,
            k_short=k_short_per_fold,
        )
        fold_trades = [
            _trade_record_from_row(
                test.iloc[i],
                int(test_signals[i]),
                float(confidences[i]),
                round_trip_cost_bps=round_trip_cost_bps,
                position_weight=1.0,
            )
            for i in picks
        ]
        all_trades.extend(fold_trades)
        skipped += max(0, len(test) - len(fold_trades))
        folds.append(
            {
                "train_events": int(len(train)),
                "test_events": int(len(test)),
                "test_start_utc": str(test.iloc[0]["t_event_utc"]),
                "test_end_utc": str(test.iloc[-1]["t_event_utc"]),
                "trades": len(fold_trades),
                "min_confidence_in_fold": round(
                    float(min(t.confidence for t in fold_trades)), 4
                ) if fold_trades else None,
            }
        )
    return all_trades, folds, skipped


def walk_forward_threshold_tuned_backtest(
    df: pd.DataFrame,
    *,
    features: list[str] | None = None,
    min_train: int = WALK_FORWARD_MIN_TRAIN_EVENTS,
    test_chunk: int = WALK_FORWARD_TEST_CHUNK,
    target_trades_per_month: float = DEFAULT_TARGET_TRADES_PER_MONTH,
    round_trip_cost_bps: float = DEFAULT_ROUND_TRIP_COST_BPS,
) -> tuple[list[TradeRecord], list[dict[str, Any]], int]:
    """Walk-forward; per fold, pick a confidence threshold on TRAIN that would have yielded
    `target_trades_per_month` and apply that threshold blindly to TEST. Equal-weighted positions.
    """
    feature_cols = features or ENHANCED_FEATURES
    ordered = df.sort_values("t_event_utc").reset_index(drop=True)
    n = len(ordered)
    if n <= min_train:
        return [], [], 0

    all_trades: list[TradeRecord] = []
    folds: list[dict[str, Any]] = []
    skipped = 0

    for start in range(min_train, n, test_chunk):
        end = min(start + test_chunk, n)
        train = ordered.iloc[:start].copy()
        test = ordered.iloc[start:end].copy()
        if test.empty:
            break
        train_proba, _, test_proba, test_signals, _ = _fit_logistic_for_ranking(
            train, test, feature_cols
        )
        train_confidences = np.maximum(train_proba, 1.0 - train_proba)
        test_confidences = np.maximum(test_proba, 1.0 - test_proba)

        train_start = pd.to_datetime(train.iloc[0]["t_event_utc"])
        train_end = pd.to_datetime(train.iloc[-1]["t_event_utc"])
        train_months = max(1.0, (train_end - train_start).days / 30.0)
        target_n = max(1, int(round(target_trades_per_month * train_months)))
        target_n = min(target_n, len(train_confidences))

        sorted_desc = np.sort(train_confidences)[::-1]
        threshold = float(sorted_desc[target_n - 1])

        fold_trades: list[TradeRecord] = []
        for i in range(len(test)):
            if test_confidences[i] < threshold:
                continue
            fold_trades.append(
                _trade_record_from_row(
                    test.iloc[i],
                    int(test_signals[i]),
                    float(test_confidences[i]),
                    round_trip_cost_bps=round_trip_cost_bps,
                    position_weight=1.0,
                )
            )
        all_trades.extend(fold_trades)
        skipped += max(0, len(test) - len(fold_trades))
        folds.append(
            {
                "train_events": int(len(train)),
                "test_events": int(len(test)),
                "test_start_utc": str(test.iloc[0]["t_event_utc"]),
                "test_end_utc": str(test.iloc[-1]["t_event_utc"]),
                "threshold_from_train": round(threshold, 4),
                "trades": len(fold_trades),
            }
        )
    return all_trades, folds, skipped


def partition_holdout(
    df: pd.DataFrame, *, holdout_days: int = DEFAULT_HOLDOUT_DAYS
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split df into (working, holdout) where holdout is the most recent `holdout_days`.

    Holdout is sacred: nothing tunes against it. Only used for the final locked-holdout report.
    """
    if df.empty or holdout_days <= 0:
        return df.copy(), df.iloc[:0].copy()
    last_dt = pd.to_datetime(df["t_event_utc"].max())
    cutoff = last_dt - pd.Timedelta(days=int(holdout_days))
    working = df[pd.to_datetime(df["t_event_utc"]) <= cutoff].reset_index(drop=True)
    holdout = df[pd.to_datetime(df["t_event_utc"]) > cutoff].reset_index(drop=True)
    return working, holdout


def evaluate_locked_holdout(
    working_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    *,
    features: list[str] | None = None,
    k_long: int = DEFAULT_TOP_K_LONG_PER_FOLD,
    k_short: int = DEFAULT_TOP_K_SHORT_PER_FOLD,
    target_trades_per_month: float = DEFAULT_TARGET_TRADES_PER_MONTH,
    round_trip_cost_bps: float = DEFAULT_ROUND_TRIP_COST_BPS,
    max_trade_preview: int = 50,
) -> dict[str, Any] | None:
    """Fit on working_df, evaluate on holdout_df for top-K and threshold-tuned lanes.

    Returns None when either side is empty.
    """
    if working_df.empty or holdout_df.empty:
        return None
    feature_cols = features or ENHANCED_FEATURES

    train_proba, _, test_proba, test_signals, used_model = _fit_logistic_for_ranking(
        working_df, holdout_df, feature_cols
    )
    train_confidences = np.maximum(train_proba, 1.0 - train_proba)
    test_confidences = np.maximum(test_proba, 1.0 - test_proba)

    holdout_days_span = max(
        1.0,
        (pd.to_datetime(holdout_df["t_event_utc"].max())
         - pd.to_datetime(holdout_df["t_event_utc"].min())).days,
    )
    holdout_months = max(holdout_days_span / 30.0, 1.0 / 30.0)

    target_top_k_long = max(1, int(round(k_long * holdout_months)))
    target_top_k_short = max(1, int(round(k_short * holdout_months)))
    picks = select_top_k_signals(
        test_signals,
        test_confidences,
        k_long=target_top_k_long,
        k_short=target_top_k_short,
    )
    top_k_trades = [
        _trade_record_from_row(
            holdout_df.iloc[i],
            int(test_signals[i]),
            float(test_confidences[i]),
            round_trip_cost_bps=round_trip_cost_bps,
            position_weight=1.0,
        )
        for i in picks
    ]
    top_k_skipped = max(0, len(holdout_df) - len(top_k_trades))

    working_start = pd.to_datetime(working_df["t_event_utc"].min())
    working_end = pd.to_datetime(working_df["t_event_utc"].max())
    working_months = max(1.0, (working_end - working_start).days / 30.0)
    target_n = max(1, int(round(target_trades_per_month * working_months)))
    target_n = min(target_n, len(train_confidences))
    threshold = float(np.sort(train_confidences)[::-1][target_n - 1])

    threshold_trades: list[TradeRecord] = []
    for i in range(len(holdout_df)):
        if test_confidences[i] < threshold:
            continue
        threshold_trades.append(
            _trade_record_from_row(
                holdout_df.iloc[i],
                int(test_signals[i]),
                float(test_confidences[i]),
                round_trip_cost_bps=round_trip_cost_bps,
                position_weight=1.0,
            )
        )
    threshold_skipped = max(0, len(holdout_df) - len(threshold_trades))

    return {
        "holdout_events": int(len(holdout_df)),
        "holdout_start_utc": str(holdout_df["t_event_utc"].min()),
        "holdout_end_utc": str(holdout_df["t_event_utc"].max()),
        "holdout_days_span": int(holdout_days_span),
        "used_model": bool(used_model),
        "top_k": {
            "k_long_per_month": int(k_long),
            "k_short_per_month": int(k_short),
            "k_long_applied": int(target_top_k_long),
            "k_short_applied": int(target_top_k_short),
            "metrics": asdict(_build_metrics(top_k_trades, top_k_skipped)),
            "trades_preview": [asdict(t) for t in top_k_trades[-max_trade_preview:]],
        },
        "threshold_tuned": {
            "target_trades_per_month": float(target_trades_per_month),
            "threshold_from_working_set": round(threshold, 6),
            "metrics": asdict(_build_metrics(threshold_trades, threshold_skipped)),
            "trades_preview": [asdict(t) for t in threshold_trades[-max_trade_preview:]],
        },
    }


def compute_feature_importance(
    df: pd.DataFrame,
    features: list[str],
    *,
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    n_repeats: int = DEFAULT_PERMUTATION_IMPORTANCE_REPEATS,
    seed: int = DEFAULT_PERMUTATION_IMPORTANCE_SEED,
) -> dict[str, Any] | None:
    """Permutation feature importance using TRAIN slice only (no leakage into test/holdout).

    Logistic regression is intentionally used here even when LightGBM is the
    production model: ``permutation_importance`` works with any sklearn-style
    estimator and the ranking is what matters, not the absolute score.
    """
    train, _ = split_time_aware(df, train_ratio=train_ratio)
    if len(train) < 20:
        return None
    x = _select_features(train, features)
    y = train["actual_sign"].to_numpy(dtype=int)
    if len(np.unique(y)) < 2 or x.shape[1] == 0:
        return None
    means = x.mean(axis=0)
    stds = x.std(axis=0)
    stds = np.where(stds < 1e-9, 1.0, stds)
    x_std = (x - means) / stds
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(x_std, y)
    result = permutation_importance(
        clf, x_std, y, n_repeats=int(n_repeats), random_state=int(seed), scoring="accuracy"
    )
    order = np.argsort(-result.importances_mean)
    ranked = [
        {
            "feature": features[i],
            "importance_mean": round(float(result.importances_mean[i]), 6),
            "importance_std": round(float(result.importances_std[i]), 6),
            "rank": int(rank + 1),
        }
        for rank, i in enumerate(order)
    ]
    return {
        "method": "sklearn.inspection.permutation_importance (accuracy)",
        "scope": "train_fold_only",
        "train_events": int(len(train)),
        "n_repeats": int(n_repeats),
        "seed": int(seed),
        "features_ranked": ranked,
    }


def run_full_backtest(
    validation_path: Path,
    *,
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    round_trip_cost_bps: float = DEFAULT_ROUND_TRIP_COST_BPS,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    max_trade_preview: int = 50,
    top_k_long_per_fold: int = DEFAULT_TOP_K_LONG_PER_FOLD,
    top_k_short_per_fold: int = DEFAULT_TOP_K_SHORT_PER_FOLD,
    target_trades_per_month: float = DEFAULT_TARGET_TRADES_PER_MONTH,
    holdout_days: int = DEFAULT_HOLDOUT_DAYS,
    feature_cache_path: Path | None = None,
    rebuild_features: bool = False,
    progress: bool = False,
) -> dict[str, Any]:
    events = load_validation_events(validation_path)
    if rebuild_features and feature_cache_path is not None and feature_cache_path.exists():
        feature_cache_path.unlink()
    df_full = build_feature_frame(events, cache_path=feature_cache_path, progress=progress)
    if df_full.empty:
        raise ValueError("No labeled events available for backtest.")

    df, holdout_df = partition_holdout(df_full, holdout_days=holdout_days)
    if df.empty:
        raise ValueError(
            f"Holdout window ({holdout_days}d) consumes the entire dataset; reduce --holdout-days."
        )

    train, test = split_time_aware(df, train_ratio=train_ratio)
    model_preds, model_proba, model_test_metrics = fit_predict(train, test, ENHANCED_FEATURES)
    baseline_preds, _baseline_proba, baseline_model_metrics = fit_predict(train, test, BASELINE_FEATURES)

    baseline_trades, baseline_skipped = simulate_signal_column(
        test,
        "baseline_pred_sign",
        round_trip_cost_bps=round_trip_cost_bps,
        min_confidence=min_confidence,
    )
    enhanced_trades, enhanced_skipped = simulate_signal_column(
        test,
        "enhanced_pred_sign",
        round_trip_cost_bps=round_trip_cost_bps,
        min_confidence=min_confidence,
    )
    model_trades, model_skipped = simulate_model_signals(
        test,
        model_preds,
        model_proba,
        round_trip_cost_bps=round_trip_cost_bps,
        min_confidence=DEFAULT_MODEL_MIN_CONFIDENCE,
    )
    wf_trades, wf_folds, wf_skipped = walk_forward_model_backtest(
        df,
        features=ENHANCED_FEATURES,
        round_trip_cost_bps=round_trip_cost_bps,
        min_confidence=DEFAULT_MODEL_MIN_CONFIDENCE,
    )
    wf_base_trades, wf_base_folds, wf_base_skipped = walk_forward_model_backtest(
        df,
        features=BASELINE_FEATURES,
        round_trip_cost_bps=round_trip_cost_bps,
        min_confidence=DEFAULT_MODEL_MIN_CONFIDENCE,
    )

    hc_topk_trades, hc_topk_folds, hc_topk_skipped = walk_forward_top_k_backtest(
        df,
        features=ENHANCED_FEATURES,
        k_long_per_fold=top_k_long_per_fold,
        k_short_per_fold=top_k_short_per_fold,
        round_trip_cost_bps=round_trip_cost_bps,
    )
    hc_topk_base_trades, hc_topk_base_folds, hc_topk_base_skipped = walk_forward_top_k_backtest(
        df,
        features=BASELINE_FEATURES,
        k_long_per_fold=top_k_long_per_fold,
        k_short_per_fold=top_k_short_per_fold,
        round_trip_cost_bps=round_trip_cost_bps,
    )
    hc_thr_trades, hc_thr_folds, hc_thr_skipped = walk_forward_threshold_tuned_backtest(
        df,
        features=ENHANCED_FEATURES,
        target_trades_per_month=target_trades_per_month,
        round_trip_cost_bps=round_trip_cost_bps,
    )
    hc_thr_base_trades, hc_thr_base_folds, hc_thr_base_skipped = walk_forward_threshold_tuned_backtest(
        df,
        features=BASELINE_FEATURES,
        target_trades_per_month=target_trades_per_month,
        round_trip_cost_bps=round_trip_cost_bps,
    )

    all_events = events
    events_by_key = {
        event_key(str(e.get("symbol", "")), e.get("t_event_utc")): e for e in all_events
    }
    test_events = [
        events_by_key[key]
        for key in (event_key(str(row["symbol"]), row["t_event_utc"]) for _, row in test.iterrows())
        if key in events_by_key
    ]
    enhanced_path_trades, enhanced_path_skipped = simulate_path_trades(
        test_events,
        signal_key="enhanced_pred_sign",
        round_trip_cost_bps=round_trip_cost_bps,
        min_confidence=min_confidence,
    )
    baseline_path_trades, baseline_path_skipped = simulate_path_trades(
        test_events,
        signal_key="baseline_pred_sign",
        round_trip_cost_bps=round_trip_cost_bps,
        min_confidence=min_confidence,
    )

    def _pack(
        name: SignalSource,
        trades: list[TradeRecord],
        skipped: int,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        metrics = _build_metrics(trades, skipped)
        payload: dict[str, Any] = {
            "strategy": name,
            "metrics": asdict(metrics),
            "recent_trades": [asdict(t) for t in trades[-max_trade_preview:]],
        }
        if extra:
            payload.update(extra)
        return payload

    def _pack_path(
        name: str,
        trades: list[PathTradeRecord],
        skipped: int,
    ) -> dict[str, Any]:
        metrics = _build_path_metrics(trades, skipped)
        return {
            "strategy": name,
            "metrics": asdict(metrics),
            "recent_trades": [asdict(t) for t in trades[-max_trade_preview:]],
            "exit_mode": "stop_target_daily_path",
        }

    best_strategy = "enhanced_heuristic"
    best_cum = -999.0
    results_map: dict[str, dict[str, Any]] = {}
    for key, trades, skipped, extra in (
        ("baseline", baseline_trades, baseline_skipped, None),
        ("enhanced_heuristic", enhanced_trades, enhanced_skipped, None),
        (
            "model_enhanced",
            model_trades,
            model_skipped,
            {"classification_on_test_fold": model_test_metrics},
        ),
        (
            "walk_forward_model",
            wf_trades,
            wf_skipped,
            {"folds": wf_folds, "features": ENHANCED_FEATURES},
        ),
        (
            "walk_forward_baseline_model",
            wf_base_trades,
            wf_base_skipped,
            {"folds": wf_base_folds, "features": BASELINE_FEATURES},
        ),
        (
            "high_conviction_topk_enhanced",
            hc_topk_trades,
            hc_topk_skipped,
            {
                "folds": hc_topk_folds,
                "features": ENHANCED_FEATURES,
                "selection_rule": (
                    f"per_fold_top_{top_k_long_per_fold}_long_plus_top_{top_k_short_per_fold}_short_by_confidence"
                ),
                "position_sizing": "equal_weight",
            },
        ),
        (
            "high_conviction_topk_baseline",
            hc_topk_base_trades,
            hc_topk_base_skipped,
            {
                "folds": hc_topk_base_folds,
                "features": BASELINE_FEATURES,
                "selection_rule": (
                    f"per_fold_top_{top_k_long_per_fold}_long_plus_top_{top_k_short_per_fold}_short_by_confidence"
                ),
                "position_sizing": "equal_weight",
            },
        ),
        (
            "high_conviction_threshold_enhanced",
            hc_thr_trades,
            hc_thr_skipped,
            {
                "folds": hc_thr_folds,
                "features": ENHANCED_FEATURES,
                "selection_rule": (
                    f"train_tuned_threshold_targeting_{target_trades_per_month}_trades_per_month"
                ),
                "position_sizing": "equal_weight",
            },
        ),
        (
            "high_conviction_threshold_baseline",
            hc_thr_base_trades,
            hc_thr_base_skipped,
            {
                "folds": hc_thr_base_folds,
                "features": BASELINE_FEATURES,
                "selection_rule": (
                    f"train_tuned_threshold_targeting_{target_trades_per_month}_trades_per_month"
                ),
                "position_sizing": "equal_weight",
            },
        ),
    ):
        packed = _pack(key, trades, skipped, extra)  # type: ignore[arg-type]
        results_map[key] = packed
        cum = packed["metrics"].get("cumulative_return_pct")
        if cum is not None and cum > best_cum:
            best_cum = float(cum)
            best_strategy = key

    for path_key, path_trades, path_skipped in (
        ("enhanced_path_stop_target", enhanced_path_trades, enhanced_path_skipped),
        ("baseline_path_stop_target", baseline_path_trades, baseline_path_skipped),
    ):
        packed = _pack_path(path_key, path_trades, path_skipped)
        results_map[path_key] = packed

    feature_importance_section = {
        "enhanced": compute_feature_importance(df, ENHANCED_FEATURES, train_ratio=train_ratio),
        "baseline": compute_feature_importance(df, BASELINE_FEATURES, train_ratio=train_ratio),
    }

    locked_holdout_report = evaluate_locked_holdout(
        df,
        holdout_df,
        features=ENHANCED_FEATURES,
        k_long=top_k_long_per_fold,
        k_short=top_k_short_per_fold,
        target_trades_per_month=target_trades_per_month,
        round_trip_cost_bps=round_trip_cost_bps,
        max_trade_preview=max_trade_preview,
    )

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "objective": (
            "Simulate post-earnings trades: fixed 5d hold, stop/target paths, "
            "and high-conviction selective lanes (top-K + train-tuned threshold). "
            "Locked holdout is reported separately and never used for tuning."
        ),
        "data_source": str(validation_path),
        "events_labeled": int(len(df)),
        "events_labeled_with_holdout": int(len(df_full)),
        "parameters": {
            "train_ratio": train_ratio,
            "test_events": int(len(test)),
            "train_events": int(len(train)),
            "round_trip_cost_bps": round_trip_cost_bps,
            "min_confidence": min_confidence,
            "position_sizing": "confidence_weighted (legacy lanes); equal_weight (high-conviction lanes)",
            "min_position_weight": DEFAULT_MIN_POSITION_WEIGHT,
            "max_position_weight": DEFAULT_MAX_POSITION_WEIGHT,
            "hold_days": 5,
            "entry": "close_on_earnings_date_T",
            "exit": "close_T_plus_5_trading_days",
            "path_exit": "first_touch_stop_or_target_else_T_plus_5_close",
            "high_conviction": {
                "top_k_long_per_fold": int(top_k_long_per_fold),
                "top_k_short_per_fold": int(top_k_short_per_fold),
                "target_trades_per_month": float(target_trades_per_month),
            },
            "holdout_days": int(holdout_days),
            "holdout_events": int(len(holdout_df)),
        },
        "leakage_controls": {
            "feature_window": "[T-7d, T]",
            "test_fold": "Chronological last portion of events; no random shuffle.",
            "walk_forward": "Expanding train window; predict only on future event chunks.",
            "high_conviction_topk": "Selection happens per-fold using only that fold's model output on TEST events; train labels never seen.",
            "high_conviction_threshold": "Threshold derived from TRAIN-fold confidences only; applied blindly to TEST.",
            "locked_holdout": (
                f"Most recent {int(holdout_days)} calendar days held out from all tuning, model "
                "selection, and walk-forward. Only used once for the final holdout report."
            ),
        },
        "strategies": results_map,
        "best_out_of_sample_strategy": best_strategy,
        "model_baseline_on_test_fold": baseline_model_metrics,
        "feature_importance": feature_importance_section,
        "locked_holdout_report": locked_holdout_report,
    }


def write_backtest_summary(payload: dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
