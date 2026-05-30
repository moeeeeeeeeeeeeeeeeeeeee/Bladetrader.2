"""Live paper-trading signals for upcoming earnings (no capital deployed)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from finhack.case4_features import (
    compute_news_features,
    direction_from_sign,
    predict_enhanced,
    resolve_db_path,
    sign_from_return,
)
from finhack.config import Settings, load_settings
from finhack.data.trading_universe import resolve_trading_universe
from finhack.market_data import get_close_series, get_upcoming_earnings_batch
from finhack.research.case4_backtest import _position_weight
from finhack.research.case4_trade_path import compute_swing_levels
from finhack.research.constants import DEFAULT_MIN_CONFIDENCE


def _load_best_strategy() -> tuple[str | None, float | None]:
    root = Path(__file__).resolve().parents[2]
    path = root / "data" / "case4_backtest_summary.json"
    if not path.exists():
        return None, None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        best = payload.get("best_out_of_sample_strategy")
        hit = None
        if isinstance(best, str):
            strategies = payload.get("strategies")
            if isinstance(strategies, dict) and best in strategies:
                metrics = strategies[best].get("metrics")
                if isinstance(metrics, dict):
                    raw = metrics.get("hit_rate")
                    if raw is not None:
                        hit = float(raw)
        return (best if isinstance(best, str) else None), hit
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None, None


def _pre_7d_return_pct(symbol: str, *, settings: Settings) -> float | None:
    end = datetime.now(timezone.utc).date()
    start = end - timedelta(days=45)
    close = get_close_series(symbol, start.isoformat(), end.isoformat(), settings=settings)
    if close.empty or len(close) < 8:
        return None
    i1 = len(close) - 1
    i0 = max(0, i1 - 7)
    p0 = float(close.iloc[i0])
    p1 = float(close.iloc[i1])
    if p0 == 0:
        return None
    return ((p1 - p0) / p0) * 100.0


@dataclass(slots=True)
class PaperEarningsSignal:
    symbol: str
    company_name: str
    sector: str
    earnings_utc: str
    days_to_earnings: int
    signal: int
    direction: str
    confidence: float
    position_weight: float
    baseline_pre_7d_return_pct: float
    sent_doc_count: int
    spillover_mentions_7d: int
    strategy: str
    actionable: bool
    entry_price: float | None
    stop_price: float | None
    target_price: float | None
    suggested_hedge: str | None
    rationale: str


@dataclass(slots=True)
class PaperSignalsBundle:
    generated_at_utc: str
    horizon_days: int
    min_confidence: float
    best_backtest_strategy: str | None
    historical_hit_rate: float | None
    universe_scanned: int
    upcoming_earnings_count: int
    actionable_count: int
    signals: list[PaperEarningsSignal]


def build_earnings_paper_signals(
    *,
    horizon_days: int = 14,
    universe_limit: int | None = None,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    settings: Settings | None = None,
) -> PaperSignalsBundle:
    cfg = settings or load_settings()
    db_path = resolve_db_path(cfg)
    universe = resolve_trading_universe(settings=cfg, limit=universe_limit)
    symbols = [u.symbol for u in universe]
    upcoming = get_upcoming_earnings_batch(symbols, horizon_days=horizon_days, settings=cfg)
    best_strategy, hist_hit = _load_best_strategy()
    strategy_name = best_strategy or "enhanced_heuristic"

    now = datetime.now(timezone.utc)
    signals: list[PaperEarningsSignal] = []

    for entry in universe:
        sym = entry.symbol
        earnings_dt = upcoming.get(sym)
        if earnings_dt is None:
            continue

        pre_ret = _pre_7d_return_pct(sym, settings=cfg)
        if pre_ret is None:
            continue

        news = compute_news_features(sym, earnings_dt, db_path=db_path, settings=cfg)
        enhanced_sign, enhanced_dir, confidence = predict_enhanced(pre_ret, news)
        signal = enhanced_sign
        if signal == 0:
            signal = sign_from_return(pre_ret, dead_zone=0.10)
            enhanced_dir = direction_from_sign(signal)

        weight = _position_weight(confidence)
        actionable = signal != 0 and confidence >= min_confidence

        entry_price = None
        stop_price = None
        target_price = None
        hedge = None
        close = get_close_series(
            sym,
            (now - timedelta(days=5)).date().isoformat(),
            now.date().isoformat(),
            settings=cfg,
        )
        if not close.empty:
            spot = float(close.iloc[-1])
            levels = compute_swing_levels(
                entry=spot,
                signal=signal,
                predicted_move_pct=pre_ret,
                symbol=sym,
            )
            if levels:
                entry_price = levels.entry_price
                stop_price = levels.stop_price
                target_price = levels.target_price
                hedge = levels.suggested_hedge

        days_out = max(0, (earnings_dt.date() - now.date()).days)
        rationale = (
            f"{strategy_name}: {enhanced_dir} into earnings in {days_out}d · "
            f"pre-7d {pre_ret:+.2f}% · docs {news['sent_doc_count']} · "
            f"spillover {news['spillover_mentions_7d']} · conf {confidence:.2f}"
        )

        signals.append(
            PaperEarningsSignal(
                symbol=sym,
                company_name=entry.name,
                sector=entry.sector,
                earnings_utc=earnings_dt.replace(microsecond=0).isoformat(),
                days_to_earnings=days_out,
                signal=signal,
                direction=enhanced_dir,
                confidence=round(confidence, 4),
                position_weight=weight,
                baseline_pre_7d_return_pct=round(pre_ret, 4),
                sent_doc_count=int(news["sent_doc_count"]),
                spillover_mentions_7d=int(news["spillover_mentions_7d"]),
                strategy=strategy_name,
                actionable=actionable,
                entry_price=entry_price,
                stop_price=stop_price,
                target_price=target_price,
                suggested_hedge=hedge,
                rationale=rationale,
            )
        )

    signals.sort(key=lambda s: (s.actionable, s.confidence, -s.days_to_earnings), reverse=True)
    actionable_count = sum(1 for s in signals if s.actionable)

    return PaperSignalsBundle(
        generated_at_utc=now.replace(microsecond=0).isoformat(),
        horizon_days=horizon_days,
        min_confidence=min_confidence,
        best_backtest_strategy=best_strategy,
        historical_hit_rate=hist_hit,
        universe_scanned=len(universe),
        upcoming_earnings_count=len(signals),
        actionable_count=actionable_count,
        signals=signals,
    )


def paper_signals_to_dict(bundle: PaperSignalsBundle) -> dict[str, Any]:
    return {
        **{k: v for k, v in asdict(bundle).items() if k != "signals"},
        "signals": [asdict(s) for s in bundle.signals],
    }
