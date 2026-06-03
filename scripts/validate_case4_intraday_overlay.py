"""
Earnings -> NQ (Nasdaq-100) session overlay validation.

Hypothesis:
    The per-event ``enhanced_pred_sign`` signal that drives the 5-trading-day
    post-earnings prediction can be aggregated into a same-session directional
    bet on the Nasdaq-100 index. If true, a TopStep-tradable instrument
    (MNQ futures) can be traded intraday based on the program's signal.

What this script does
---------------------
1. Loads per-event predictions from ``data/case4_earnings_validation.json``
   (each event has ``enhanced_pred_sign``, ``enhanced_confidence``, and
   ``t_event_utc``). No re-running of the upstream pipeline.
2. Maps each event to a single US trading session using a conservative
   "T+1 session" rule: the signal for an earnings event dated calendar day T
   is applied to the next US trading session strictly after T. This is the
   worst-case AMC assumption and guarantees no leakage: every feature input
   for the signal is from ``[T-7d, T]`` and the session opens strictly after
   that window.
3. For each session that has at least one signal, aggregates signals into a
   session direction:
       agg_score(session) = sum(sign_i * confidence_i)
   The session bet is long if agg_score > 0, short if < 0. Sessions with
   exactly zero aggregate score are skipped (no actionable signal).
4. Fetches QQQ daily OHLC (cash ETF tracking NDX -- same index MNQ tracks)
   and computes the session open->close return for each signal-bearing
   session.
5. Computes:
       - Hit rate (sign of agg_score matches sign of session return)
       - Mean directional return per session (signed)
       - Cumulative return (compounded, 1 contract weight per session)
       - Annualised Sharpe (sqrt(252) scaling)
       - Max drawdown of the cumulative return curve
       - Per-confidence-bucket breakdown
       - Permutation p-value (greater) under within-session-return shuffle
6. Writes ``data/case4_intraday_overlay_validation.json``.

Important honesty constraints
-----------------------------
- This study uses the production signal as-is. The upstream
  ``case4_earnings_validation.json`` is the same signal the existing
  ``paper_signals.py`` would have emitted on the day before each session.
- A coverage warning is emitted if fewer than 80 signal-bearing sessions
  are available. Below that threshold, hit-rate noise is large enough that
  the result must not be acted on.
- All metrics are computed BEFORE any cost. A separate "after_cost"
  block applies a fixed per-trade slippage assumption so the user sees
  both raw edge and net-of-friction edge.

Output: ``data/case4_intraday_overlay_validation.json``
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

load_dotenv(ROOT / ".env")

from finhack.config import load_settings  # noqa: E402
from finhack.market_data import get_ohlc_series  # noqa: E402
from finhack.research.constants import (  # noqa: E402
    OVERLAY_PRE_REGISTERED_MIN_CONFIDENCE,
)


PROXY_SYMBOL = "QQQ"
MIN_SESSIONS_FOR_CONFIDENCE = 80
DEFAULT_PERMUTATIONS = 1000
DEFAULT_SEED = 42
DEFAULT_COST_BPS_PER_SIDE = 1.0  # ~1bp per side ≈ 2bp round-trip on MNQ
DEFAULT_MIN_CONFIDENCE = OVERLAY_PRE_REGISTERED_MIN_CONFIDENCE

# NQ-100 names that historically drive >50% of index moves. Used as the
# default "NQ-relevant" subset when --nq-megacaps is passed.
NQ_MEGACAPS = [
    "NVDA", "MSFT", "AAPL", "AMZN", "META", "GOOGL", "GOOG", "AVGO",
    "TSLA", "NFLX", "ADBE", "AMD", "COST", "PEP", "CSCO", "CMCSA",
    "INTC", "TXN", "INTU", "QCOM", "AMGN", "AMAT", "BKNG", "ADI",
    "MU", "ISRG", "GILD", "REGN", "LRCX", "MDLZ",
]


@dataclass(slots=True)
class EventSignal:
    symbol: str
    t_event_utc: datetime
    enhanced_pred_sign: int
    enhanced_confidence: float
    sent_doc_count: int


@dataclass(slots=True)
class SessionRecord:
    session_date: str
    agg_signed_confidence: float
    direction: int
    n_contributing_events: int
    contributing_symbols: list[str]
    confidence_bucket: str
    qqq_open: float
    qqq_close: float
    session_return_pct: float
    directional_return_pct: float
    hit: bool


def _parse_event_dt(raw: str) -> datetime | None:
    txt = (raw or "").strip()
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


def load_event_signals(
    path: Path,
    *,
    require_news: bool,
    symbol_filter: set[str] | None,
    min_confidence: float,
) -> tuple[list[EventSignal], dict[str, Any]]:
    """Return (signals, upstream_uplift_summary) for the filtered subset.

    The upstream uplift summary measures baseline vs enhanced classification
    accuracy on the SAME subset after the symbol / news / confidence
    filters, so the overlay caller can see whether the upstream model
    already beats the price-only baseline on this slice. Events whose
    actual_sign or baseline_pred_sign is 0 are excluded from accuracy
    counts (matches the upstream scoring rule).
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    events = payload.get("events", [])
    signals: list[EventSignal] = []
    baseline_total = baseline_correct = 0
    enhanced_total = enhanced_correct = 0
    raw_kept = 0

    for row in events:
        sym = str(row.get("symbol", "")).upper()
        if symbol_filter is not None and sym not in symbol_filter:
            continue
        sent_docs = int(row.get("sent_doc_count", 0) or 0)
        if require_news and sent_docs <= 0:
            continue
        sign = row.get("enhanced_pred_sign")
        try:
            sign_int = int(sign) if sign is not None else 0
        except (TypeError, ValueError):
            sign_int = 0
        conf = row.get("enhanced_confidence")
        try:
            conf_f = float(conf) if conf is not None else 0.0
        except (TypeError, ValueError):
            conf_f = 0.0
        if conf_f < min_confidence and sign_int != 0:
            sign_int = 0  # treat as no signal but still count for upstream uplift

        raw_kept += 1

        actual = int(row.get("actual_sign", 0) or 0)
        base_sign = int(row.get("baseline_pred_sign", 0) or 0)
        if actual != 0 and base_sign != 0:
            baseline_total += 1
            if actual == base_sign:
                baseline_correct += 1
        if actual != 0 and sign_int != 0:
            enhanced_total += 1
            if actual == sign_int:
                enhanced_correct += 1

        if sign_int == 0:
            continue
        dt = _parse_event_dt(str(row.get("t_event_utc", "")))
        if dt is None:
            continue
        signals.append(
            EventSignal(
                symbol=sym,
                t_event_utc=dt,
                enhanced_pred_sign=sign_int,
                enhanced_confidence=conf_f,
                sent_doc_count=sent_docs,
            )
        )

    baseline_acc = (baseline_correct / baseline_total) if baseline_total else None
    enhanced_acc = (enhanced_correct / enhanced_total) if enhanced_total else None
    uplift_pp = (
        (enhanced_acc - baseline_acc) * 100.0
        if baseline_acc is not None and enhanced_acc is not None
        else None
    )
    summary = {
        "events_in_subset": raw_kept,
        "baseline": {
            "correct": baseline_correct,
            "total": baseline_total,
            "accuracy": round(baseline_acc, 4) if baseline_acc is not None else None,
        },
        "enhanced": {
            "correct": enhanced_correct,
            "total": enhanced_total,
            "accuracy": round(enhanced_acc, 4) if enhanced_acc is not None else None,
        },
        "uplift_pp": round(uplift_pp, 2) if uplift_pp is not None else None,
    }
    return signals, summary


def _assign_session(
    event_dt: datetime, *, trading_dates: list[pd.Timestamp]
) -> pd.Timestamp | None:
    """Return the next trading session strictly after the event calendar day.

    Conservative T+1 rule: even if an earnings release is BMO, this assigns
    its signal to the next session, guaranteeing the open of that session is
    strictly after the [T-7d, T] feature window.
    """
    event_day = pd.Timestamp(event_dt.date())
    for ts in trading_dates:
        if ts.normalize() > event_day:
            return ts.normalize()
    return None


def _confidence_bucket(conf: float) -> str:
    if conf < 0.55:
        return "0.50-0.55"
    if conf < 0.60:
        return "0.55-0.60"
    if conf < 0.70:
        return "0.60-0.70"
    if conf < 0.80:
        return "0.70-0.80"
    return ">=0.80"


def build_session_records(
    signals: list[EventSignal],
    ohlc: pd.DataFrame,
) -> list[SessionRecord]:
    if ohlc.empty:
        return []
    trading_dates = [pd.Timestamp(idx).normalize() for idx in ohlc.index]
    trading_dates.sort()

    grouped: dict[pd.Timestamp, list[EventSignal]] = defaultdict(list)
    for sig in signals:
        session_ts = _assign_session(sig.t_event_utc, trading_dates=trading_dates)
        if session_ts is None:
            continue
        grouped[session_ts].append(sig)

    records: list[SessionRecord] = []
    for session_ts in sorted(grouped):
        ts_key = session_ts
        if ts_key not in ohlc.index:
            # Index lookup using normalised timestamp
            mask = pd.Index(trading_dates).get_indexer([ts_key])
            if mask[0] < 0:
                continue
            ts_key = pd.Timestamp(ohlc.index[mask[0]])
        row = ohlc.loc[ts_key]
        qqq_open = float(row["Open"])
        qqq_close = float(row["Close"])
        if not math.isfinite(qqq_open) or not math.isfinite(qqq_close) or qqq_open == 0:
            continue

        contributing = grouped[session_ts]
        agg = sum(s.enhanced_pred_sign * s.enhanced_confidence for s in contributing)
        if agg == 0:
            continue
        direction = 1 if agg > 0 else -1
        sess_ret_pct = ((qqq_close - qqq_open) / qqq_open) * 100.0
        directional_ret_pct = direction * sess_ret_pct
        avg_conf = abs(agg) / max(1, len(contributing))
        records.append(
            SessionRecord(
                session_date=session_ts.date().isoformat(),
                agg_signed_confidence=round(agg, 4),
                direction=direction,
                n_contributing_events=len(contributing),
                contributing_symbols=sorted({s.symbol for s in contributing}),
                confidence_bucket=_confidence_bucket(avg_conf),
                qqq_open=round(qqq_open, 4),
                qqq_close=round(qqq_close, 4),
                session_return_pct=round(sess_ret_pct, 4),
                directional_return_pct=round(directional_ret_pct, 4),
                hit=directional_ret_pct > 0,
            )
        )
    return records


def _safe_round(value: float | None, digits: int) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    return round(float(value), digits)


def _compute_metrics(
    records: list[SessionRecord], *, cost_bps_per_side: float
) -> dict[str, Any]:
    if not records:
        return {
            "trades": 0,
            "hit_rate": None,
            "long_trades": 0,
            "short_trades": 0,
            "mean_session_return_long_pct": None,
            "mean_session_return_short_pct": None,
            "directional_return_pct_mean": None,
            "directional_return_pct_cum": None,
            "sharpe_annualized": None,
            "max_drawdown_pct": None,
            "after_cost": None,
        }
    rets = np.array([r.directional_return_pct for r in records], dtype=float)
    hits = sum(1 for r in records if r.hit)
    long_trades = [r for r in records if r.direction > 0]
    short_trades = [r for r in records if r.direction < 0]

    growth = np.cumprod(1.0 + rets / 100.0)
    cum_return_pct = float((growth[-1] - 1.0) * 100.0)
    peaks = np.maximum.accumulate(growth)
    drawdowns = (growth - peaks) / peaks
    max_dd_pct = float(drawdowns.min() * 100.0) if drawdowns.size else 0.0

    mean_ret = float(rets.mean())
    std_ret = float(rets.std(ddof=1)) if rets.size > 1 else 0.0
    sharpe = (mean_ret / std_ret * math.sqrt(252.0)) if std_ret > 0 else None

    round_trip_pct = (cost_bps_per_side * 2.0) / 100.0
    rets_after = rets - round_trip_pct
    growth_after = np.cumprod(1.0 + rets_after / 100.0)
    cum_after_pct = float((growth_after[-1] - 1.0) * 100.0)
    mean_after = float(rets_after.mean())
    std_after = float(rets_after.std(ddof=1)) if rets_after.size > 1 else 0.0
    sharpe_after = (
        mean_after / std_after * math.sqrt(252.0) if std_after > 0 else None
    )
    hits_after = int(np.sum(rets_after > 0))

    def _mean_pct(items: list[SessionRecord]) -> float | None:
        if not items:
            return None
        return float(np.mean([i.session_return_pct for i in items]))

    return {
        "trades": len(records),
        "hit_rate": _safe_round(hits / len(records), 4),
        "long_trades": len(long_trades),
        "short_trades": len(short_trades),
        "mean_session_return_long_pct": _safe_round(_mean_pct(long_trades), 4),
        "mean_session_return_short_pct": _safe_round(_mean_pct(short_trades), 4),
        "directional_return_pct_mean": _safe_round(mean_ret, 4),
        "directional_return_pct_cum": _safe_round(cum_return_pct, 4),
        "sharpe_annualized": _safe_round(sharpe, 4),
        "max_drawdown_pct": _safe_round(max_dd_pct, 4),
        "after_cost": {
            "round_trip_cost_pct": round(round_trip_pct, 4),
            "hit_rate": _safe_round(hits_after / len(records), 4),
            "directional_return_pct_mean": _safe_round(mean_after, 4),
            "directional_return_pct_cum": _safe_round(cum_after_pct, 4),
            "sharpe_annualized": _safe_round(sharpe_after, 4),
        },
    }


def _by_confidence_bucket(records: list[SessionRecord]) -> list[dict[str, Any]]:
    buckets: dict[str, list[SessionRecord]] = defaultdict(list)
    for r in records:
        buckets[r.confidence_bucket].append(r)
    order = ["0.50-0.55", "0.55-0.60", "0.60-0.70", "0.70-0.80", ">=0.80"]
    out: list[dict[str, Any]] = []
    for key in order:
        items = buckets.get(key, [])
        if not items:
            out.append({"bucket": key, "trades": 0, "hit_rate": None,
                        "directional_return_pct_mean": None})
            continue
        hits = sum(1 for r in items if r.hit)
        mean_ret = float(np.mean([r.directional_return_pct for r in items]))
        out.append({
            "bucket": key,
            "trades": len(items),
            "hit_rate": round(hits / len(items), 4),
            "directional_return_pct_mean": round(mean_ret, 4),
        })
    return out


def _permutation_pvalue(
    records: list[SessionRecord],
    *,
    n_perm: int,
    seed: int,
) -> dict[str, Any]:
    if not records:
        return {"iterations": 0, "p_value": None, "null_mean_hit_rate": None,
                "null_p95_hit_rate": None, "null_mean_cum_return_pct": None}
    rng = np.random.default_rng(seed)
    directions = np.array([r.direction for r in records], dtype=int)
    session_rets = np.array([r.session_return_pct for r in records], dtype=float)
    observed_hits = int(np.sum((directions * session_rets) > 0))
    observed_hit_rate = observed_hits / len(records)

    null_hits = np.empty(n_perm, dtype=float)
    null_cum = np.empty(n_perm, dtype=float)
    for i in range(n_perm):
        shuffled = rng.permutation(session_rets)
        directional = directions * shuffled
        null_hits[i] = float(np.sum(directional > 0) / len(records))
        growth = np.cumprod(1.0 + directional / 100.0)
        null_cum[i] = float((growth[-1] - 1.0) * 100.0)
    p_hits = float((1 + int(np.sum(null_hits >= observed_hit_rate))) / (n_perm + 1))
    observed_cum_growth = float(np.cumprod(1.0 + (directions * session_rets) / 100.0)[-1] - 1.0) * 100.0
    p_cum = float((1 + int(np.sum(null_cum >= observed_cum_growth))) / (n_perm + 1))

    return {
        "iterations": n_perm,
        "observed_hit_rate": round(observed_hit_rate, 4),
        "p_value_hit_rate_greater": round(p_hits, 4),
        "p_value_cum_return_greater": round(p_cum, 4),
        "null_mean_hit_rate": round(float(null_hits.mean()), 4),
        "null_p95_hit_rate": round(float(np.percentile(null_hits, 95)), 4),
        "null_mean_cum_return_pct": round(float(null_cum.mean()), 4),
    }


def _build_coverage_warning(records: list[SessionRecord]) -> str | None:
    if len(records) >= MIN_SESSIONS_FOR_CONFIDENCE:
        return None
    return (
        f"Only {len(records)} signal-bearing sessions. Below the "
        f"{MIN_SESSIONS_FOR_CONFIDENCE}-session threshold; results are "
        "noisy and must not be acted on. Acquire more history (longer "
        "validation window) before drawing conclusions."
    )


def run_overlay_validation(
    *,
    validation_path: Path,
    out_path: Path,
    require_news: bool,
    n_permutations: int,
    seed: int,
    cost_bps_per_side: float,
    symbol_filter: set[str] | None,
    symbol_filter_label: str,
    min_confidence: float,
) -> dict[str, Any]:
    settings = load_settings()
    signals, upstream_uplift = load_event_signals(
        validation_path,
        require_news=require_news,
        symbol_filter=symbol_filter,
        min_confidence=min_confidence,
    )
    if not signals:
        raise SystemExit(
            "No usable signals after filters (non-zero enhanced_pred_sign + filters)."
        )

    earliest = min(s.t_event_utc for s in signals).date() - timedelta(days=5)
    latest = max(s.t_event_utc for s in signals).date() + timedelta(days=10)
    ohlc = get_ohlc_series(
        PROXY_SYMBOL,
        earliest.isoformat(),
        latest.isoformat(),
        settings=settings,
        interval="1d",
    )
    if ohlc.empty:
        raise SystemExit(
            f"Could not fetch {PROXY_SYMBOL} OHLC for {earliest}..{latest}. "
            "Check EODHD_API_KEY in .env."
        )
    # Ensure index is sorted Timestamps; market_data already returns sorted DatetimeIndex.
    ohlc = ohlc.sort_index()

    records = build_session_records(signals, ohlc)
    metrics = _compute_metrics(records, cost_bps_per_side=cost_bps_per_side)
    perm = _permutation_pvalue(records, n_perm=n_permutations, seed=seed)
    coverage_warning = _build_coverage_warning(records)
    bucket_breakdown = _by_confidence_bucket(records)

    payload: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "objective": (
            "Test whether per-earnings enhanced_pred_sign signals translate into a "
            "same-session directional bet on the Nasdaq-100 (proxied by QQQ; same "
            "underlying index MNQ futures track)."
        ),
        "data_source": str(validation_path.resolve()),
        "proxy_instrument": {
            "symbol": PROXY_SYMBOL,
            "rationale": "QQQ is the cash ETF on NDX; MNQ/NQ futures track NDX. "
                         "Daily OPEN->CLOSE return is a faithful proxy for an "
                         "intraday MNQ session bet under TopStep's 4PM-ET flat rule.",
        },
        "methodology": {
            "session_assignment": "Conservative T+1: signal for earnings event "
                                   "on calendar day T applied to the NEXT US "
                                   "trading session strictly after T. No "
                                   "intraday data needed; no leakage.",
            "aggregation": "agg_score(session) = sum_i sign_i * confidence_i "
                            "over events mapped to that session. Direction = "
                            "sign(agg_score). Sessions with agg_score = 0 skipped.",
            "signal_filter": (
                "require_news=true" if require_news else "all non-zero enhanced_pred_sign"
            ),
            "session_return": "QQQ Open -> QQQ Close, same trading day.",
            "leakage_guards": [
                "Signals built from documents in [T-7d, T] only (upstream).",
                "Signal applied to session strictly after T (this script).",
                "No intraday news touches the bet; news-window rule never violated.",
            ],
        },
        "parameters": {
            "require_news": require_news,
            "symbol_filter": symbol_filter_label,
            "min_confidence": min_confidence,
            "n_permutations": n_permutations,
            "seed": seed,
            "cost_bps_per_side": cost_bps_per_side,
            "min_sessions_for_confidence": MIN_SESSIONS_FOR_CONFIDENCE,
        },
        "upstream_uplift_on_subset": upstream_uplift,
        "coverage": {
            "raw_signals_loaded": len(signals),
            "sessions_with_signal": len(records),
            "date_range": (
                [records[0].session_date, records[-1].session_date]
                if records else None
            ),
            "unique_contributing_symbols": len({
                s for r in records for s in r.contributing_symbols
            }),
            "coverage_warning": coverage_warning,
        },
        "results": metrics,
        "by_confidence_bucket": bucket_breakdown,
        "permutation": perm,
        "honesty_notes": [
            "The upstream validation file reports enhanced_accuracy == "
            "baseline_accuracy on the full 929-event universe; the enhanced "
            "model currently does not beat the price-only baseline in "
            "aggregate. If this overlay shows edge, it is because the "
            "subset of events with non-zero signal AND non-trivial "
            "confidence is informative -- not because the model wins "
            "across all events.",
            "QQQ is a cash proxy. Real MNQ fills, slippage, and overnight "
            "settlement differences will degrade these numbers further; "
            "the after_cost block is a minimal correction, not a real fill "
            "simulator.",
            "Conservative T+1 throws away any same-day-BMO edge. A tighter "
            "session assignment using ET hour-of-day could be added later.",
        ],
        "sessions": [
            {
                "session_date": r.session_date,
                "direction": r.direction,
                "agg_signed_confidence": r.agg_signed_confidence,
                "n_contributing_events": r.n_contributing_events,
                "contributing_symbols": r.contributing_symbols,
                "confidence_bucket": r.confidence_bucket,
                "qqq_open": r.qqq_open,
                "qqq_close": r.qqq_close,
                "session_return_pct": r.session_return_pct,
                "directional_return_pct": r.directional_return_pct,
                "hit": r.hit,
            }
            for r in records
        ],
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate the earnings -> NQ (QQQ) session overlay."
    )
    parser.add_argument(
        "--validation",
        type=Path,
        default=ROOT / "data" / "case4_earnings_validation.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "data" / "case4_intraday_overlay_validation.json",
    )
    parser.add_argument(
        "--require-news",
        action="store_true",
        help="Only use signals from events that had >=1 news document in "
             "[T-7d, T] (filters out pure-momentum signals).",
    )
    parser.add_argument(
        "--nq-megacaps",
        action="store_true",
        help="Restrict to the NQ-100 mega-cap subset (NVDA, MSFT, AAPL, ...).",
    )
    parser.add_argument(
        "--symbols",
        type=str,
        default=None,
        help="Comma-separated symbol list to filter signals (overrides --nq-megacaps).",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=DEFAULT_MIN_CONFIDENCE,
        help=(
            "Drop signals whose enhanced_confidence is below this floor. "
            f"Defaults to the pre-registered overlay cutoff "
            f"({OVERLAY_PRE_REGISTERED_MIN_CONFIDENCE}). Lowering this for "
            "new data without an out-of-sample rerun is research malpractice."
        ),
    )
    parser.add_argument("--permutations", type=int, default=DEFAULT_PERMUTATIONS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--cost-bps-per-side",
        type=float,
        default=DEFAULT_COST_BPS_PER_SIDE,
    )
    args = parser.parse_args()

    if not args.validation.exists():
        raise SystemExit(
            f"Validation file not found: {args.validation}. "
            "Run scripts/validate_case4_earnings.py first."
        )

    if args.symbols:
        symbol_set = {s.strip().upper() for s in args.symbols.split(",") if s.strip()}
        symbol_label = f"custom({len(symbol_set)})"
    elif args.nq_megacaps:
        symbol_set = set(NQ_MEGACAPS)
        symbol_label = "nq_megacaps"
    else:
        symbol_set = None
        symbol_label = "all"

    payload = run_overlay_validation(
        validation_path=args.validation,
        out_path=args.out,
        require_news=args.require_news,
        n_permutations=args.permutations,
        seed=args.seed,
        cost_bps_per_side=args.cost_bps_per_side,
        symbol_filter=symbol_set,
        symbol_filter_label=symbol_label,
        min_confidence=args.min_confidence,
    )

    coverage = payload["coverage"]
    results = payload["results"]
    perm = payload["permutation"]
    uplift = payload["upstream_uplift_on_subset"]
    print(f"Wrote: {args.out.as_posix()}")
    print(json.dumps({
        "filter": {
            "symbols": symbol_label,
            "require_news": args.require_news,
            "min_confidence": args.min_confidence,
        },
        "upstream_subset_baseline_accuracy": uplift["baseline"]["accuracy"],
        "upstream_subset_enhanced_accuracy": uplift["enhanced"]["accuracy"],
        "upstream_subset_uplift_pp": uplift["uplift_pp"],
        "sessions_with_signal": coverage["sessions_with_signal"],
        "date_range": coverage["date_range"],
        "coverage_warning": coverage["coverage_warning"],
        "hit_rate": results["hit_rate"],
        "directional_return_pct_cum": results["directional_return_pct_cum"],
        "sharpe_annualized": results["sharpe_annualized"],
        "max_drawdown_pct": results["max_drawdown_pct"],
        "after_cost_hit_rate": (results.get("after_cost") or {}).get("hit_rate"),
        "after_cost_cum_return_pct": (results.get("after_cost") or {}).get(
            "directional_return_pct_cum"
        ),
        "permutation_p_hit_rate_greater": perm.get("p_value_hit_rate_greater"),
        "permutation_p_cum_return_greater": perm.get("p_value_cum_return_greater"),
        "null_mean_hit_rate": perm.get("null_mean_hit_rate"),
    }, indent=2))


if __name__ == "__main__":
    main()
