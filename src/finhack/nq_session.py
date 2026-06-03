"""
NQ / MNQ session trading layer — QQQ is the traded proxy, mega-cap earnings are the signal.

We do NOT predict "QQQ earnings" (the ETF has none). The program:

1. Scans NQ-100 mega-cap constituents for upcoming/recent earnings + news sentiment.
2. Aggregates their directional signals into a single session bias (long/short MNQ).
3. Maps that bias onto QQQ price structure: prior-day liquidity, swing levels,
   fib retracements, and simple liquidity-sweep flags (ICT-style *concepts*,
   implemented as measurable rules — not PDF keyword matching).
4. Produces a session trade plan: entry (last close / next open proxy), stop, target,
   and optional Monte Carlo drawdown on historical session returns.

Honesty: upstream stock-direction accuracy is ~50% OOS; overlay hit rate on QQQ
sessions has not beaten the permutation null. This module formats decisions —
it does not claim a winning edge.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Literal

import numpy as np
import pandas as pd

from finhack.config import Settings, load_settings
from finhack.market_data import CHART_ALL_INTERVALS, CHART_INTRADAY_INTERVALS, get_ohlc_intraday, get_ohlc_series
from finhack.paper_signals import build_earnings_paper_signals, paper_signals_to_dict
from finhack.research.constants import (
    NQ_CAP_WEIGHTS,
    NQ_FUTURES_INSTRUMENTS,
    NQ_SIGNAL_SYMBOLS,
    OVERLAY_PRE_REGISTERED_MIN_CONFIDENCE,
)

SweepBias = Literal["bullish", "bearish", "none"]


@dataclass(slots=True)
class StructureLevels:
    prior_day_high: float | None
    prior_day_low: float | None
    swing_high: float | None
    swing_low: float | None
    fib_382: float | None
    fib_500: float | None
    fib_618: float | None
    atr_14_pct: float | None
    liquidity_sweep: SweepBias
    sweep_note: str


@dataclass(slots=True)
class SessionTradePlan:
    instrument: str
    proxy_symbol: str
    direction: str  # long | short | flat
    direction_sign: int
    entry_price: float | None
    stop_price: float | None
    target_price: float | None
    stop_pct: float | None
    target_pct: float | None
    risk_reward: float | None
    rationale: str
    structure: StructureLevels
    monte_carlo: dict[str, Any] | None = None


def _atr_pct(ohlc: pd.DataFrame, period: int = 14) -> float | None:
    if ohlc is None or len(ohlc) < period + 1:
        return None
    high = pd.to_numeric(ohlc["High"], errors="coerce")
    low = pd.to_numeric(ohlc["Low"], errors="coerce")
    close = pd.to_numeric(ohlc["Close"], errors="coerce")
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.rolling(period).mean().iloc[-1]
    last = float(close.iloc[-1])
    if last <= 0 or pd.isna(atr):
        return None
    return round(float(atr / last) * 100.0, 4)


def _swing_window(ohlc: pd.DataFrame, lookback: int = 5) -> tuple[float | None, float | None]:
    if ohlc is None or len(ohlc) < lookback:
        return None, None
    tail = ohlc.iloc[-lookback:]
    return float(tail["High"].max()), float(tail["Low"].min())


def _fib_levels(
    swing_low: float | None, swing_high: float | None
) -> tuple[float | None, float | None, float | None]:
    if swing_low is None or swing_high is None or swing_high <= swing_low:
        return None, None, None
    span = swing_high - swing_low
    # Retracement from swing high toward swing low (pullback in uptrend context).
    return (
        round(swing_high - 0.382 * span, 4),
        round(swing_high - 0.500 * span, 4),
        round(swing_high - 0.618 * span, 4),
    )


def _detect_liquidity_sweep(ohlc: pd.DataFrame) -> tuple[SweepBias, str]:
    """Simple PDH/PDL sweep: wick beyond prior day extreme, close back inside."""
    if ohlc is None or len(ohlc) < 2:
        return "none", "insufficient bars"
    last = ohlc.iloc[-1]
    prior = ohlc.iloc[-2]
    hi = float(last["High"])
    lo = float(last["Low"])
    cl = float(last["Close"])
    pdh = float(prior["High"])
    pdl = float(prior["Low"])
    if lo < pdl and cl > pdl:
        return "bullish", f"Low {lo:.2f} swept PDL {pdl:.2f}, closed {cl:.2f} above — buy-side liquidity grab"
    if hi > pdh and cl < pdh:
        return "bearish", f"High {hi:.2f} swept PDH {pdh:.2f}, closed {cl:.2f} below — sell-side liquidity grab"
    return "none", "No PDH/PDL sweep on latest bar"


def compute_structure_levels(
    ohlc: pd.DataFrame,
    *,
    swing_lookback: int = 5,
) -> StructureLevels:
    if ohlc is None or ohlc.empty:
        return StructureLevels(
            prior_day_high=None,
            prior_day_low=None,
            swing_high=None,
            swing_low=None,
            fib_382=None,
            fib_500=None,
            fib_618=None,
            atr_14_pct=None,
            liquidity_sweep="none",
            sweep_note="no OHLC",
        )
    ohlc = ohlc.sort_index()
    pdh = pdl = None
    if len(ohlc) >= 2:
        prior = ohlc.iloc[-2]
        pdh = float(prior["High"])
        pdl = float(prior["Low"])
    swing_hi, swing_lo = _swing_window(ohlc, swing_lookback)
    fib_382, fib_500, fib_618 = _fib_levels(swing_lo, swing_hi)
    sweep, sweep_note = _detect_liquidity_sweep(ohlc)
    return StructureLevels(
        prior_day_high=pdh,
        prior_day_low=pdl,
        swing_high=swing_hi,
        swing_low=swing_lo,
        fib_382=fib_382,
        fib_500=fib_500,
        fib_618=fib_618,
        atr_14_pct=_atr_pct(ohlc),
        liquidity_sweep=sweep,
        sweep_note=sweep_note,
    )


def _session_sl_tp(
    *,
    entry: float,
    direction_sign: int,
    atr_pct: float | None,
    structure: StructureLevels,
) -> tuple[float, float, float, float]:
    """Stop beyond nearest liquidity pool; target at 2R or opposing pool."""
    vol = atr_pct if atr_pct and atr_pct > 0 else 1.2
    stop_pct = round(max(0.35, min(2.5, 0.65 * vol)), 4)
    target_pct = round(stop_pct * 2.0, 4)

    if direction_sign > 0:
        stop = entry * (1.0 - stop_pct / 100.0)
        target = entry * (1.0 + target_pct / 100.0)
        if structure.prior_day_low and structure.prior_day_low < entry:
            stop = min(stop, structure.prior_day_low * 0.999)
        if structure.prior_day_high and structure.prior_day_high > entry:
            target = max(target, structure.prior_day_high)
    elif direction_sign < 0:
        stop = entry * (1.0 + stop_pct / 100.0)
        target = entry * (1.0 - target_pct / 100.0)
        if structure.prior_day_high and structure.prior_day_high > entry:
            stop = max(stop, structure.prior_day_high * 1.001)
        if structure.prior_day_low and structure.prior_day_low < entry:
            target = min(target, structure.prior_day_low)
    else:
        stop = target = entry
    stop_pct_eff = abs(entry - stop) / entry * 100.0 if entry else stop_pct
    target_pct_eff = abs(target - entry) / entry * 100.0 if entry else target_pct
    return round(stop, 4), round(target, 4), round(stop_pct_eff, 4), round(target_pct_eff, 4)


def _monte_carlo_session_risk(
    session_returns_pct: list[float],
    *,
    n_sims: int = 2000,
    seed: int = 42,
) -> dict[str, Any] | None:
    if len(session_returns_pct) < 20:
        return None
    rng = np.random.default_rng(seed)
    arr = np.asarray(session_returns_pct, dtype=float)
    n = len(arr)
    sims = 200
    max_dds: list[float] = []
    for _ in range(n_sims):
        path = arr[rng.integers(0, n, size=sims)]
        equity = np.cumprod(1.0 + path / 100.0)
        peak = np.maximum.accumulate(equity)
        dd = (equity - peak) / peak
        max_dds.append(float(dd.min()) * 100.0)
    max_dds_arr = np.asarray(max_dds)
    return {
        "n_simulations": n_sims,
        "sessions_per_path": sims,
        "historical_sessions": n,
        "median_max_drawdown_pct": round(float(np.median(max_dds_arr)), 2),
        "p95_max_drawdown_pct": round(float(np.percentile(max_dds_arr, 95)), 2),
        "note": "Bootstrap of historical QQQ session returns; not a forecast of edge.",
    }


def aggregate_nq_session_signal(
    *,
    horizon_days: int = 14,
    min_confidence: float = OVERLAY_PRE_REGISTERED_MIN_CONFIDENCE,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Pull mega-cap earnings signals and aggregate to one NQ session bias."""
    cfg = settings or load_settings()
    bundle = build_earnings_paper_signals(
        horizon_days=horizon_days,
        universe_limit=len(NQ_SIGNAL_SYMBOLS),
        min_confidence=0.0,
        settings=cfg,
        symbol_filter=list(NQ_SIGNAL_SYMBOLS),
    )
    payload = paper_signals_to_dict(bundle)
    signals = [
        s
        for s in payload.get("signals", [])
        if isinstance(s, dict) and int(s.get("signal", 0) or 0) != 0
    ]
    signed_sum = 0.0
    for s in signals:
        sym = str(s.get("symbol", "")).upper()
        sign = int(s.get("signal", 0) or 0)
        conf = float(s.get("confidence", 0.0) or 0.0)
        w = float(NQ_CAP_WEIGHTS.get(sym, 0.0))
        signed_sum += sign * conf * w
    if signed_sum > 0:
        direction, direction_sign = "long", 1
    elif signed_sum < 0:
        direction, direction_sign = "short", -1
    else:
        direction, direction_sign = "flat", 0
    avg_conf = abs(signed_sum) / max(1, len(signals)) if signals else 0.0
    actionable = direction_sign != 0 and avg_conf >= min_confidence
    return {
        "direction": direction,
        "direction_sign": direction_sign,
        "aggregated_signed_confidence": round(signed_sum, 4),
        "avg_confidence": round(avg_conf, 4),
        "n_contributors": len(signals),
        "actionable": actionable,
        "min_confidence_floor": min_confidence,
        "contributors": signals[:20],
        "aggregation": "cap_weight",
        "signal_universe": list(NQ_SIGNAL_SYMBOLS),
    }


def build_futures_chart_payload(
    instrument_key: str = "MNQ",
    *,
    days: int = 120,
    interval: str = "1d",
    horizon_days: int = 14,
    min_confidence: float = OVERLAY_PRE_REGISTERED_MIN_CONFIDENCE,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """OHLC + structure levels + session trade plan for the chart UI."""
    cfg = settings or load_settings()
    meta = NQ_FUTURES_INSTRUMENTS.get(instrument_key.upper())
    if meta is None:
        meta = NQ_FUTURES_INSTRUMENTS["MNQ"]
        instrument_key = "MNQ"

    proxy = str(meta["proxy_symbol"])
    end_dt = datetime.now(timezone.utc)
    iv = (interval or "1d").strip().lower()
    if iv not in CHART_ALL_INTERVALS:
        iv = "1d"
    window_note = ""
    ohlc_provider = ""

    # Structure levels (PDH/PDL, fib, ATR) always from daily bars — the levels
    # traders mark on a higher timeframe while executing on 5m/15m/etc.
    daily_start = end_dt - timedelta(days=max(days + 30, 90))
    daily = get_ohlc_series(
        proxy,
        start=daily_start.date().isoformat(),
        end=(end_dt + timedelta(days=1)).date().isoformat(),
        interval="1d",
        settings=cfg,
    )
    structure = compute_structure_levels(daily)

    if iv in CHART_INTRADAY_INTERVALS:
        ohlc, window_note, ohlc_provider = get_ohlc_intraday(
            proxy, iv, settings=cfg, days=days
        )
    else:
        start_dt = end_dt - timedelta(days=days + 30)
        ohlc = get_ohlc_series(
            proxy,
            start=start_dt.date().isoformat(),
            end=(end_dt + timedelta(days=1)).date().isoformat(),
            interval=iv,
            settings=cfg,
        )
        ohlc_provider = "eod/yfinance"
        window_note = f"{len(ohlc)} bars · {iv} · {days}d lookback"

    session_sig = aggregate_nq_session_signal(
        horizon_days=horizon_days,
        min_confidence=min_confidence,
        settings=cfg,
    )

    entry: float | None = None
    stop = target = None
    stop_pct = target_pct = None
    rr: float | None = None
    rationale = "No direction — flat or below confidence floor."

    if not ohlc.empty:
        entry = float(ohlc.iloc[-1]["Close"])
    direction_sign = int(session_sig.get("direction_sign", 0))
    if entry and direction_sign != 0 and session_sig.get("actionable"):
        stop, target, stop_pct, target_pct = _session_sl_tp(
            entry=entry,
            direction_sign=direction_sign,
            atr_pct=structure.atr_14_pct,
            structure=structure,
        )
        if stop_pct and stop_pct > 0:
            rr = round(target_pct / stop_pct, 2) if target_pct else None
        sweep_align = ""
        if structure.liquidity_sweep == "bullish" and direction_sign > 0:
            sweep_align = " Liquidity sweep aligns with long bias."
        elif structure.liquidity_sweep == "bearish" and direction_sign < 0:
            sweep_align = " Liquidity sweep aligns with short bias."
        rationale = (
            f"{session_sig['n_contributors']} NQ mega-cap earnings names → "
            f"{session_sig['direction'].upper()} MNQ (Σ conf {session_sig['aggregated_signed_confidence']:.3f})."
            f" Stop beyond PDL/PDH pool, target 2R.{sweep_align}"
        )
    elif direction_sign != 0:
        rationale = (
            f"Bias {session_sig['direction']} but avg confidence "
            f"{session_sig['avg_confidence']:.3f} < floor {min_confidence} — no trade plan."
        )

    session_returns: list[float] = []
    if len(daily) >= 2:
        for i in range(1, len(daily)):
            o = float(daily.iloc[i]["Open"])
            c = float(daily.iloc[i]["Close"])
            if o:
                session_returns.append((c - o) / o * 100.0)

    mc = _monte_carlo_session_risk(session_returns)

    is_intraday = iv in CHART_INTRADAY_INTERVALS
    bars: list[dict[str, Any]] = []
    if not ohlc.empty:
        for ts, row in ohlc.iterrows():
            ts_pd = pd.Timestamp(ts)
            if is_intraday:
                if ts_pd.tzinfo is None:
                    ts_pd = ts_pd.tz_localize("America/New_York")
                t_txt = ts_pd.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
            else:
                if ts_pd.tzinfo is not None:
                    ts_pd = ts_pd.tz_convert("America/New_York")
                t_txt = ts_pd.strftime("%Y-%m-%d")
            bars.append(
                {
                    "t_utc": t_txt,
                    "open": float(row["Open"]),
                    "high": float(row["High"]),
                    "low": float(row["Low"]),
                    "close": float(row["Close"]),
                }
            )

    plan = SessionTradePlan(
        instrument=instrument_key,
        proxy_symbol=proxy,
        direction=str(session_sig.get("direction", "flat")),
        direction_sign=direction_sign,
        entry_price=entry,
        stop_price=stop,
        target_price=target,
        stop_pct=stop_pct,
        target_pct=target_pct,
        risk_reward=rr,
        rationale=rationale,
        structure=structure,
        monte_carlo=mc,
    )

    level_lines: list[dict[str, Any]] = []
    for name, val in (
        ("PDH", structure.prior_day_high),
        ("PDL", structure.prior_day_low),
        ("Swing High", structure.swing_high),
        ("Swing Low", structure.swing_low),
        ("Fib 38.2%", structure.fib_382),
        ("Fib 50%", structure.fib_500),
        ("Fib 61.8%", structure.fib_618),
    ):
        if val is not None:
            level_lines.append({"label": name, "price": val})

    if stop is not None:
        level_lines.append({"label": "Stop", "price": stop, "kind": "stop"})
    if target is not None:
        level_lines.append({"label": "Target", "price": target, "kind": "target"})

    return {
        "instrument": instrument_key,
        "instrument_meta": meta,
        "proxy_symbol": proxy,
        "interval": iv,
        "is_intraday": is_intraday,
        "window_note": window_note,
        "ohlc_provider": ohlc_provider,
        "bars": bars,
        "levels": level_lines,
        "structure": asdict(structure),
        "session_signal": session_sig,
        "trade_plan": {
            "instrument": plan.instrument,
            "proxy_symbol": plan.proxy_symbol,
            "direction": plan.direction,
            "direction_sign": plan.direction_sign,
            "entry_price": plan.entry_price,
            "stop_price": plan.stop_price,
            "target_price": plan.target_price,
            "stop_pct": plan.stop_pct,
            "target_pct": plan.target_pct,
            "risk_reward": plan.risk_reward,
            "rationale": plan.rationale,
            "monte_carlo": plan.monte_carlo,
        },
        "available_instruments": [
            {"key": k, **v} for k, v in NQ_FUTURES_INSTRUMENTS.items()
        ],
        "honesty": (
            "Chart proxy is the cash ETF (QQQ/SPY). MNQ/MES are micro futures on "
            "the same index. Session signal comes from NQ mega-cap earnings names, "
            "not from ICT PDF rules. OOS stock-direction uplift is ~0; QQQ session "
            "overlay has not passed permutation tests — treat levels as structure "
            "context, not proof of edge."
        ),
    }
