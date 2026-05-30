"""Swing stop/target levels and daily path simulation for earnings trades."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Literal

import pandas as pd

from finhack.config import Settings, load_settings
from finhack.data.company_graph import SECTOR_HEDGE_CANDIDATES, SYMBOL_TO_COMPANY
from finhack.market_data import get_daily_volatility_pct, get_ohlc_series

ExitReason = Literal["stop", "target", "time", "no_data"]


@dataclass(slots=True)
class SwingLevels:
    direction: str
    entry_price: float
    stop_price: float
    target_price: float
    stop_pct: float
    target_pct: float
    leverage_or_hedge: str
    suggested_hedge: str | None


@dataclass(slots=True)
class PathResult:
    exit_price: float | None
    exit_reason: ExitReason
    holding_days: int
    return_pct: float | None
    won: bool
    stop_hit: bool
    target_hit: bool


def compute_swing_levels(
    *,
    entry: float,
    signal: int,
    predicted_move_pct: float,
    vol_pct: float | None = None,
    symbol: str | None = None,
) -> SwingLevels | None:
    if entry <= 0 or signal == 0:
        return None
    vol = vol_pct if vol_pct is not None and vol_pct > 0 else max(1.0, min(4.0, abs(predicted_move_pct) / 3.0))
    stop_pct = round(max(1.0, min(8.0, 0.45 * abs(predicted_move_pct) + 0.65 * vol)), 4)
    target_pct = round(max(stop_pct * 1.5, 0.55 * abs(predicted_move_pct)), 4)
    long = signal > 0
    if long:
        stop = round(entry * (1.0 - stop_pct / 100.0), 4)
        target = round(entry * (1.0 + target_pct / 100.0), 4)
    else:
        stop = round(entry * (1.0 + stop_pct / 100.0), 4)
        target = round(entry * (1.0 - target_pct / 100.0), 4)

    comp = SYMBOL_TO_COMPANY.get((symbol or "").upper())
    sector = comp.sector_bucket if comp else "AI Compute"
    hedges = SECTOR_HEDGE_CANDIDATES.get(sector, ("PSQ", "SH"))
    suggested = hedges[0] if hedges else None
    return SwingLevels(
        direction="long" if long else "short",
        entry_price=round(entry, 4),
        stop_price=stop,
        target_price=target,
        stop_pct=stop_pct,
        target_pct=target_pct,
        leverage_or_hedge="Leverage" if long else "Hedge",
        suggested_hedge=suggested if not long else None,
    )


def simulate_daily_path(
    *,
    signal: int,
    entry: float,
    stop: float,
    target: float,
    daily_bars: list[tuple[float, float, float]],
    max_days: int = 5,
) -> PathResult:
    """Walk post-entry daily bars (high, low, close). First touch of stop/target wins."""
    if signal == 0 or entry <= 0 or not daily_bars:
        return PathResult(
            exit_price=None,
            exit_reason="no_data",
            holding_days=0,
            return_pct=None,
            won=False,
            stop_hit=False,
            target_hit=False,
        )

    long = signal > 0
    for day_idx, (high, low, close) in enumerate(daily_bars[:max_days], start=1):
        if long:
            if low <= stop:
                ret = ((stop - entry) / entry) * 100.0
                return PathResult(
                    exit_price=stop,
                    exit_reason="stop",
                    holding_days=day_idx,
                    return_pct=round(ret, 4),
                    won=ret > 0,
                    stop_hit=True,
                    target_hit=False,
                )
            if high >= target:
                ret = ((target - entry) / entry) * 100.0
                return PathResult(
                    exit_price=target,
                    exit_reason="target",
                    holding_days=day_idx,
                    return_pct=round(ret, 4),
                    won=ret > 0,
                    stop_hit=False,
                    target_hit=True,
                )
        else:
            if high >= stop:
                ret = ((entry - stop) / entry) * 100.0
                return PathResult(
                    exit_price=stop,
                    exit_reason="stop",
                    holding_days=day_idx,
                    return_pct=round(ret, 4),
                    won=ret > 0,
                    stop_hit=True,
                    target_hit=False,
                )
            if low <= target:
                ret = ((entry - target) / entry) * 100.0
                return PathResult(
                    exit_price=target,
                    exit_reason="target",
                    holding_days=day_idx,
                    return_pct=round(ret, 4),
                    won=ret > 0,
                    stop_hit=False,
                    target_hit=True,
                )

    last_close = daily_bars[min(len(daily_bars), max_days) - 1][2]
    if long:
        ret = ((last_close - entry) / entry) * 100.0
    else:
        ret = ((entry - last_close) / entry) * 100.0
    return PathResult(
        exit_price=round(last_close, 4),
        exit_reason="time",
        holding_days=min(len(daily_bars), max_days),
        return_pct=round(ret, 4),
        won=ret > 0,
        stop_hit=False,
        target_hit=False,
    )


def _parse_event_dt(raw: str) -> datetime | None:
    txt = str(raw or "").strip()
    if not txt:
        return None
    if txt.endswith("Z"):
        txt = txt[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(txt)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def enrich_event_with_trade_path(
    row: dict[str, Any],
    *,
    settings: Settings | None = None,
    signal_key: str = "enhanced_pred_sign",
    fallback_signal_key: str = "baseline_pred_sign",
) -> dict[str, Any]:
    """Add stop/target levels and simulated path exit using daily OHLC after earnings."""
    cfg = settings or load_settings()
    symbol = str(row.get("symbol", "")).upper().strip()
    t_event = _parse_event_dt(str(row.get("t_event_utc", "")))
    if not symbol or t_event is None:
        return row

    signal = int(row.get(signal_key, 0) or 0)
    if signal == 0:
        signal = int(row.get(fallback_signal_key, 0) or 0)
    if signal == 0:
        return row

    start = (t_event - timedelta(days=5)).date().isoformat()
    end = (t_event + timedelta(days=20)).date().isoformat()
    ohlc = get_ohlc_series(symbol, start, end, settings=cfg, interval="1d")
    if ohlc.empty:
        return row

    idx = list(ohlc.index)
    event_i = None
    for i, ts in enumerate(idx):
        if ts.to_pydatetime().date() >= t_event.date():
            event_i = i
            break
    if event_i is None:
        return row

    entry = float(ohlc.iloc[event_i]["Close"])
    if entry <= 0:
        return row

    pred_move = float(row.get("baseline_pre_7d_return_pct", 0.0))
    try:
        vol = get_daily_volatility_pct(symbol, settings=cfg)
    except Exception:
        vol = None

    levels = compute_swing_levels(
        entry=entry,
        signal=signal,
        predicted_move_pct=pred_move,
        vol_pct=vol,
        symbol=symbol,
    )
    if levels is None:
        return row

    post_bars: list[tuple[float, float, float]] = []
    for j in range(event_i + 1, min(event_i + 6, len(ohlc))):
        bar = ohlc.iloc[j]
        post_bars.append((float(bar["High"]), float(bar["Low"]), float(bar["Close"])))

    path = simulate_daily_path(
        signal=signal,
        entry=entry,
        stop=levels.stop_price,
        target=levels.target_price,
        daily_bars=post_bars,
        max_days=5,
    )

    enriched = dict(row)
    enriched.update(
        {
            "trade_entry_price": levels.entry_price,
            "trade_stop_price": levels.stop_price,
            "trade_target_price": levels.target_price,
            "trade_stop_pct": levels.stop_pct,
            "trade_target_pct": levels.target_pct,
            "trade_direction": levels.direction,
            "leverage_or_hedge": levels.leverage_or_hedge,
            "suggested_hedge_symbol": levels.suggested_hedge,
            "path_exit_price": path.exit_price,
            "path_exit_reason": path.exit_reason,
            "path_holding_days": path.holding_days,
            "path_return_pct": path.return_pct,
            "path_won": path.won,
            "path_stop_hit": path.stop_hit,
            "path_target_hit": path.target_hit,
            "direction_correct": (
                path.return_pct is not None
                and ((signal > 0 and path.return_pct > 0) or (signal < 0 and path.return_pct > 0))
            ),
        }
    )
    return enriched
