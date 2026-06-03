"""
Cap-weighted NQ session backtest + Monte Carlo validation.

Replays historical mega-cap earnings events into QQQ open→close sessions
using two aggregation modes:

- ``equal_weight``: Σ sign × confidence  (legacy overlay)
- ``cap_weight``:   Σ sign × confidence × NQ index weight

Runs permutation-style Monte Carlo on session directional returns to estimate
whether observed hit rate / cumulative return is distinguishable from noise.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from finhack.config import load_settings
from finhack.market_data import get_ohlc_series
from finhack.research.constants import NQ_CAP_WEIGHTS, NQ_SIGNAL_SYMBOLS

AggMode = Literal["equal_weight", "cap_weight"]

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_VALIDATION = PROJECT_ROOT / "data" / "case4_earnings_validation.json"


@dataclass(slots=True)
class _EventSig:
    symbol: str
    t_event_utc: datetime
    sign: int
    confidence: float
    weight: float


@dataclass(slots=True)
class SessionRow:
    session_date: str
    direction: int
    agg_score: float
    n_events: int
    symbols: list[str]
    qqq_open: float
    qqq_close: float
    session_return_pct: float
    directional_return_pct: float
    hit: bool


def _parse_dt(raw: str) -> datetime | None:
    if not raw:
        return None
    txt = raw.strip()
    if txt.endswith("Z"):
        txt = txt[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(txt)
    except ValueError:
        return None
    if dt.tzinfo is None:
        from datetime import timezone as tz

        dt = dt.replace(tzinfo=tz.utc)
    return dt.astimezone(timezone.utc)


def _assign_session(event_dt: datetime, trading_dates: list[pd.Timestamp]) -> pd.Timestamp | None:
    event_day = pd.Timestamp(event_dt.date())
    for ts in trading_dates:
        if ts.normalize() > event_day:
            return ts.normalize()
    return None


def _load_megacap_signals(path: Path, *, min_confidence: float = 0.0) -> list[_EventSig]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    events = payload.get("events") or []
    allowed = {s.upper() for s in NQ_SIGNAL_SYMBOLS}
    out: list[_EventSig] = []
    for ev in events:
        if not isinstance(ev, dict):
            continue
        sym = str(ev.get("symbol", "")).upper()
        if sym not in allowed:
            continue
        sign = int(ev.get("enhanced_pred_sign", 0) or 0)
        if sign == 0:
            continue
        conf = float(ev.get("enhanced_confidence", 0.0) or 0.0)
        if conf < min_confidence:
            continue
        dt = _parse_dt(str(ev.get("t_event_utc", "")))
        if dt is None:
            continue
        w = float(NQ_CAP_WEIGHTS.get(sym, 0.0))
        out.append(_EventSig(symbol=sym, t_event_utc=dt, sign=sign, confidence=conf, weight=w))
    return out


def build_sessions(
    signals: list[_EventSig],
    ohlc: pd.DataFrame,
    *,
    mode: AggMode = "cap_weight",
) -> list[SessionRow]:
    if ohlc.empty or not signals:
        return []
    ohlc = ohlc.sort_index()
    trading_dates = sorted(pd.Timestamp(idx).normalize() for idx in ohlc.index)
    grouped: dict[pd.Timestamp, list[_EventSig]] = defaultdict(list)
    for sig in signals:
        sess = _assign_session(sig.t_event_utc, trading_dates)
        if sess is not None:
            grouped[sess].append(sig)

    rows: list[SessionRow] = []
    for session_ts in sorted(grouped):
        ts_key = session_ts
        if ts_key not in ohlc.index:
            idx = pd.Index(trading_dates).get_indexer([ts_key])
            if idx[0] < 0:
                continue
            ts_key = pd.Timestamp(ohlc.index[idx[0]])
        bar = ohlc.loc[ts_key]
        o = float(bar["Open"])
        c = float(bar["Close"])
        if o == 0 or not math.isfinite(o) or not math.isfinite(c):
            continue

        contribs = grouped[session_ts]
        if mode == "cap_weight":
            agg = sum(s.sign * s.confidence * s.weight for s in contribs)
        else:
            agg = sum(s.sign * s.confidence for s in contribs)
        if agg == 0:
            continue
        direction = 1 if agg > 0 else -1
        sess_ret = (c - o) / o * 100.0
        dir_ret = direction * sess_ret
        rows.append(
            SessionRow(
                session_date=session_ts.date().isoformat(),
                direction=direction,
                agg_score=round(agg, 4),
                n_events=len(contribs),
                symbols=sorted({s.symbol for s in contribs}),
                qqq_open=round(o, 4),
                qqq_close=round(c, 4),
                session_return_pct=round(sess_ret, 4),
                directional_return_pct=round(dir_ret, 4),
                hit=dir_ret > 0,
            )
        )
    return rows


def _metrics(rows: list[SessionRow], *, cost_bps: float = 5.0) -> dict[str, Any]:
    if not rows:
        return {"trades": 0}
    hits = sum(1 for r in rows if r.hit)
    rets = [r.directional_return_pct for r in rows]
    cum = 1.0
    for r in rets:
        cum *= 1.0 + r / 100.0
    cum_pct = (cum - 1.0) * 100.0
    mean_ret = float(np.mean(rets))
    std = float(np.std(rets, ddof=1)) if len(rets) > 1 else 0.0
    sharpe = (mean_ret / std * math.sqrt(252)) if std > 1e-9 else 0.0
    cost_per_trade = 2.0 * cost_bps / 100.0
    after_cost = [r - cost_per_trade for r in rets]
    eq = 1.0
    peak = 1.0
    max_dd = 0.0
    for r in after_cost:
        eq *= 1.0 + r / 100.0
        peak = max(peak, eq)
        max_dd = min(max_dd, (eq - peak) / peak * 100.0)
    return {
        "trades": len(rows),
        "hit_rate": round(hits / len(rows), 4),
        "mean_directional_return_pct": round(mean_ret, 4),
        "cumulative_return_pct": round(cum_pct, 4),
        "sharpe_annualized": round(sharpe, 4),
        "max_drawdown_pct": round(max_dd, 4),
        "after_cost_cumulative_pct": round((np.prod([1 + x / 100 for x in after_cost]) - 1) * 100, 4),
    }


def _monte_carlo(
    rows: list[SessionRow],
    *,
    n_sims: int = 3000,
    seed: int = 42,
) -> dict[str, Any]:
    if len(rows) < 15:
        return {"available": False, "reason": "fewer than 15 sessions"}
    rng = np.random.default_rng(seed)
    actual_hits = sum(1 for r in rows if r.hit) / len(rows)
    actual_cum = float(np.prod([1 + r.directional_return_pct / 100 for r in rows]) - 1)
    pool = np.array([r.session_return_pct for r in rows], dtype=float)
    n = len(pool)
    sim_hits: list[float] = []
    sim_cums: list[float] = []
    for _ in range(n_sims):
        sample_returns = pool[rng.integers(0, n, size=n)]
        dirs = rng.choice([-1, 1], size=n)
        dir_rets = dirs * sample_returns / 100.0
        sim_hits.append(float(np.mean(dir_rets > 0)))
        sim_cums.append(float(np.prod(1.0 + dir_rets) - 1.0))
    sim_hits_arr = np.asarray(sim_hits)
    sim_cums_arr = np.asarray(sim_cums)
    return {
        "available": True,
        "n_simulations": n_sims,
        "actual_hit_rate": round(actual_hits, 4),
        "actual_cumulative_return": round(actual_cum, 4),
        "p_value_hit_rate_greater": round(float(np.mean(sim_hits_arr >= actual_hits)), 4),
        "p_value_cum_return_greater": round(float(np.mean(sim_cums_arr >= actual_cum)), 4),
        "null_mean_hit_rate": round(float(np.mean(sim_hits_arr)), 4),
        "null_median_cum_return": round(float(np.median(sim_cums_arr)), 4),
    }


def run_nq_session_backtest(
    *,
    validation_path: Path | None = None,
    min_confidence: float = 0.0,
    cost_bps: float = 5.0,
    n_sims: int = 3000,
) -> dict[str, Any]:
    path = validation_path or DEFAULT_VALIDATION
    if not path.exists():
        raise FileNotFoundError(f"Missing validation file: {path.as_posix()}")

    signals = _load_megacap_signals(path, min_confidence=min_confidence)
    end = datetime.now(timezone.utc)
    start = end - pd.Timedelta(days=1100)
    ohlc = get_ohlc_series(
        "QQQ",
        start=start.date().isoformat(),
        end=(end + pd.Timedelta(days=1)).date().isoformat(),
        settings=load_settings(),
    )

    equal_rows = build_sessions(signals, ohlc, mode="equal_weight")
    cap_rows = build_sessions(signals, ohlc, mode="cap_weight")

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "validation_path": str(path.as_posix()),
        "min_confidence": min_confidence,
        "signal_events": len(signals),
        "universe": list(NQ_SIGNAL_SYMBOLS),
        "aggregation_modes": {
            "equal_weight": {
                "description": "Legacy: Σ sign × confidence",
                "metrics": _metrics(equal_rows, cost_bps=cost_bps),
                "monte_carlo": _monte_carlo(equal_rows, n_sims=n_sims),
                "recent_sessions": [asdict(r) for r in equal_rows[-25:]],
            },
            "cap_weight": {
                "description": "Index-weighted: Σ sign × confidence × NDX weight",
                "metrics": _metrics(cap_rows, cost_bps=cost_bps),
                "monte_carlo": _monte_carlo(cap_rows, n_sims=n_sims),
                "recent_sessions": [asdict(r) for r in cap_rows[-25:]],
            },
        },
        "recommended_mode": "cap_weight",
        "honesty": (
            "Neither mode has consistently beaten the Monte Carlo null on held-out "
            "data. Use this panel to compare aggregation methods, not as proof of edge."
        ),
    }
