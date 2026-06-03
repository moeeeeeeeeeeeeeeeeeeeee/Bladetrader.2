"""
Session-watch composition: live upcoming-session aggregation + cached overlay
metrics + a small TopStep risk-state calculator.

This module is the data layer behind the ``/api/session_watch`` HTTP
endpoint. It does not execute any trades and does not write any state to
disk. Every output is a deterministic function of:

- the cached overlay validation JSON files in ``data/``,
- the current upcoming-earnings forward predictions from
  :mod:`finhack.paper_signals`, and
- optional TopStep-account inputs (account size, current balance,
  peak end-of-day balance, today P&L) supplied by the caller.

Honesty contract
----------------
The overlay metrics block exposes hit rate, after-cost cumulative return,
Sharpe, max drawdown, permutation p-value, and a coverage warning. The
endpoint surfaces all of those — including the no-edge findings — so the
UI cannot accidentally present the overlay as a working trading edge when
the upstream evidence does not support that.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from finhack.config import Settings, load_settings
from finhack.paper_signals import (
    build_earnings_paper_signals,
    paper_signals_to_dict,
)
from finhack.research.constants import (
    NQ_CAP_WEIGHTS,
    NQ_SIGNAL_SYMBOLS,
    OVERLAY_PRE_REGISTERED_MIN_CONFIDENCE,
)
from finhack.market_data import get_ohlc_series


PROJECT_ROOT = Path(__file__).resolve().parents[2]


# Recognised overlay variants and the JSON files they emit. The endpoint
# defaults to the variant most likely to be used as a tradeable signal
# (high-confidence-only, full universe), but will gracefully fall back if
# that file is absent.
OVERLAY_VARIANTS: list[tuple[str, str, str]] = [
    (
        "all_signals_min_confidence_0.80",
        "case4_overlay_hiconf.json",
        "Full universe, signals with confidence >= 0.80, conservative T+1 session map.",
    ),
    (
        "news_attributed",
        "case4_intraday_overlay_validation_news_only.json",
        "Signals from events with >=1 news document in [T-7d, T].",
    ),
    (
        "all_signals",
        "case4_intraday_overlay_validation.json",
        "All non-zero enhanced_pred_sign signals (no filters).",
    ),
    (
        "nq_megacaps",
        "case4_overlay_megacap.json",
        "Restricted to NQ-100 mega-cap subset.",
    ),
    (
        "nq_megacaps_min_confidence_0.80",
        "case4_overlay_megacap_hiconf.json",
        "NQ mega-caps with confidence >= 0.80.",
    ),
]


# --- TopStep account presets -------------------------------------------------

TOPSTEP_PRESETS: dict[str, dict[str, float]] = {
    "50K": {"starting_balance": 50_000.0, "trailing_dd": 2_000.0,
            "profit_target": 3_000.0},
    "100K": {"starting_balance": 100_000.0, "trailing_dd": 3_000.0,
             "profit_target": 6_000.0},
    "150K": {"starting_balance": 150_000.0, "trailing_dd": 4_500.0,
             "profit_target": 9_000.0},
}


@dataclass(slots=True)
class TopstepState:
    account_label: str
    starting_balance: float
    trailing_drawdown: float
    profit_target: float
    current_balance: float
    peak_eod_balance: float
    todays_pnl: float
    trailing_floor: float
    buffer_remaining: float
    distance_to_target: float
    is_locked_floor: bool
    kill_switch_active: bool
    kill_switch_reason: str | None


def compute_topstep_state(
    *,
    account_label: str,
    current_balance: float,
    peak_eod_balance: float | None,
    todays_pnl: float,
) -> TopstepState:
    """Compute the trailing MLL floor and a kill-switch flag for a TopStep account.

    The trailing floor follows ``peak_eod_balance - trailing_dd`` while the
    peak is below the starting balance, and locks at ``starting_balance``
    once the peak reaches it. Kill switch fires when the running balance is
    within $200 of the floor — this is a research display kill-switch, not
    a real execution hook.
    """
    preset = TOPSTEP_PRESETS.get(account_label, TOPSTEP_PRESETS["50K"])
    start = float(preset["starting_balance"])
    trail = float(preset["trailing_dd"])
    target = float(preset["profit_target"])
    peak = float(peak_eod_balance) if peak_eod_balance is not None else start

    # Floor is the maximum of starting-balance-minus-trail (locked) and
    # peak-minus-trail (trailing), capped at the starting balance.
    locked = peak >= start
    if locked:
        floor = start
    else:
        floor = peak - trail
    buffer_remaining = float(current_balance) - floor
    distance_to_target = (start + target) - float(current_balance)

    kill_reason: str | None = None
    kill = False
    if buffer_remaining <= 0:
        kill = True
        kill_reason = "Trailing MLL breached — account would be ineligible."
    elif buffer_remaining < 200.0:
        kill = True
        kill_reason = (
            f"Within ${buffer_remaining:.0f} of trailing MLL. Stop trading until "
            "the next session."
        )

    return TopstepState(
        account_label=account_label,
        starting_balance=start,
        trailing_drawdown=trail,
        profit_target=target,
        current_balance=round(float(current_balance), 2),
        peak_eod_balance=round(peak, 2),
        todays_pnl=round(float(todays_pnl), 2),
        trailing_floor=round(floor, 2),
        buffer_remaining=round(buffer_remaining, 2),
        distance_to_target=round(distance_to_target, 2),
        is_locked_floor=locked,
        kill_switch_active=kill,
        kill_switch_reason=kill_reason,
    )


# --- Overlay summary loader --------------------------------------------------


@dataclass(slots=True)
class OverlaySummary:
    variant_key: str
    variant_file: str
    variant_description: str
    generated_at_utc: str | None
    parameters: dict[str, Any]
    coverage: dict[str, Any]
    results: dict[str, Any]
    permutation: dict[str, Any]
    upstream_uplift_on_subset: dict[str, Any]
    by_confidence_bucket: list[dict[str, Any]] = field(default_factory=list)
    equity_curve: list[dict[str, Any]] = field(default_factory=list)
    recent_sessions: list[dict[str, Any]] = field(default_factory=list)
    honesty_notes: list[str] = field(default_factory=list)


def _build_equity_curve(sessions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compound daily directional returns into a $1 -> $X curve."""
    growth = 1.0
    out: list[dict[str, Any]] = []
    for row in sessions:
        ret = float(row.get("directional_return_pct", 0.0) or 0.0)
        growth = growth * (1.0 + ret / 100.0)
        out.append(
            {
                "session_date": row.get("session_date"),
                "directional_return_pct": ret,
                "equity": round(growth, 6),
                "drawdown_pct": None,  # filled below
            }
        )
    peak = 1.0
    for row in out:
        peak = max(peak, row["equity"])
        row["drawdown_pct"] = round(((row["equity"] - peak) / peak) * 100.0, 4)
    return out


def load_overlay_summary(variant_key: str | None = None) -> OverlaySummary:
    """Load an overlay validation JSON, choosing the first variant that exists.

    Raises FileNotFoundError if no variant file exists yet.
    """
    candidates = OVERLAY_VARIANTS
    if variant_key is not None:
        candidates = [v for v in candidates if v[0] == variant_key] + [
            v for v in OVERLAY_VARIANTS if v[0] != variant_key
        ]
    for key, filename, desc in candidates:
        path = PROJECT_ROOT / "data" / filename
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        sessions = payload.get("sessions") or []
        equity_curve = _build_equity_curve(sessions)
        recent = sessions[-30:] if len(sessions) > 30 else sessions
        return OverlaySummary(
            variant_key=key,
            variant_file=filename,
            variant_description=desc,
            generated_at_utc=payload.get("generated_at_utc"),
            parameters=payload.get("parameters") or {},
            coverage=payload.get("coverage") or {},
            results=payload.get("results") or {},
            permutation=payload.get("permutation") or {},
            upstream_uplift_on_subset=payload.get("upstream_uplift_on_subset") or {},
            by_confidence_bucket=payload.get("by_confidence_bucket") or [],
            equity_curve=equity_curve,
            recent_sessions=recent,
            honesty_notes=payload.get("honesty_notes") or [],
        )
    raise FileNotFoundError(
        "No overlay validation JSON found in data/. "
        "Run scripts/validate_case4_intraday_overlay.py first."
    )


# --- Upcoming-session aggregation --------------------------------------------


@dataclass(slots=True)
class UpcomingSessionSignal:
    target_session_date: str
    aggregated_signed_confidence: float
    direction: str  # "long" | "short" | "flat"
    direction_sign: int
    n_contributing_events: int
    contributors: list[dict[str, Any]]
    actionable_above_threshold: bool


_QQQ_SESSION_DATES: list[date] | None = None


def _load_qqq_trading_dates(settings: Settings | None = None) -> list[date]:
    """Cached QQQ session calendar for conservative T+1 assignment."""
    global _QQQ_SESSION_DATES
    if _QQQ_SESSION_DATES is not None:
        return _QQQ_SESSION_DATES
    cfg = settings or load_settings()
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=800)
    ohlc = get_ohlc_series(
        "QQQ",
        start=start.date().isoformat(),
        end=(end + timedelta(days=30)).date().isoformat(),
        settings=cfg,
    )
    if ohlc.empty:
        _QQQ_SESSION_DATES = []
        return _QQQ_SESSION_DATES
    _QQQ_SESSION_DATES = sorted(
        {pd.Timestamp(idx).date() for idx in ohlc.index}  # type: ignore[name-defined]
    )
    return _QQQ_SESSION_DATES


def _session_date_for_event(event_iso: str, *, settings: Settings | None = None) -> str:
    """Next QQQ trading session strictly after the earnings calendar day."""
    try:
        if event_iso.endswith("Z"):
            event_iso = event_iso[:-1] + "+00:00"
        dt = datetime.fromisoformat(event_iso)
    except ValueError:
        return ""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    event_day = dt.astimezone(timezone.utc).date()
    for sess in _load_qqq_trading_dates(settings):
        if sess > event_day:
            return sess.isoformat()
    return ""


def aggregate_upcoming_sessions(
    *,
    horizon_days: int,
    min_confidence_floor: float,
    universe_limit: int | None,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Pull forward predictions and aggregate them by target session date.

    Returns a payload with one entry per future session that has at least
    one earnings event within the requested horizon. Each entry includes
    the aggregated signed-confidence and the list of contributing symbols
    so the UI can show "what is driving today's signal".
    """
    cfg = settings or load_settings()
    bundle = build_earnings_paper_signals(
        horizon_days=horizon_days,
        universe_limit=universe_limit,
        min_confidence=0.0,  # we filter here, after grouping
        symbol_filter=list(NQ_SIGNAL_SYMBOLS),
        settings=cfg,
    )
    payload = paper_signals_to_dict(bundle)

    grouped: dict[str, list[dict[str, Any]]] = {}
    for sig in payload.get("signals", []):
        if not isinstance(sig, dict):
            continue
        if int(sig.get("signal", 0) or 0) == 0:
            continue
        session_iso = _session_date_for_event(str(sig.get("earnings_utc", "")), settings=cfg)
        if not session_iso:
            continue
        grouped.setdefault(session_iso, []).append(sig)

    sessions: list[UpcomingSessionSignal] = []
    for session_iso in sorted(grouped):
        contribs = grouped[session_iso]
        signed_sum = 0.0
        for s in contribs:
            sym = str(s.get("symbol", "")).upper()
            signed_sum += (
                int(s.get("signal", 0) or 0)
                * float(s.get("confidence", 0.0) or 0.0)
                * float(NQ_CAP_WEIGHTS.get(sym, 0.0))
            )
        if signed_sum > 0:
            direction = "long"
            direction_sign = 1
        elif signed_sum < 0:
            direction = "short"
            direction_sign = -1
        else:
            direction = "flat"
            direction_sign = 0
        avg_conf = abs(signed_sum) / max(1, len(contribs))
        actionable = direction_sign != 0 and avg_conf >= min_confidence_floor
        contributor_rows = [
            {
                "symbol": s.get("symbol"),
                "company_name": s.get("company_name"),
                "sector": s.get("sector"),
                "earnings_utc": s.get("earnings_utc"),
                "signal": int(s.get("signal", 0) or 0),
                "direction": s.get("direction"),
                "confidence": float(s.get("confidence", 0.0) or 0.0),
                "baseline_pre_7d_return_pct": s.get("baseline_pre_7d_return_pct"),
                "sent_doc_count": s.get("sent_doc_count"),
                "spillover_mentions_7d": s.get("spillover_mentions_7d"),
                "rationale": s.get("rationale"),
            }
            for s in contribs
        ]
        sessions.append(
            UpcomingSessionSignal(
                target_session_date=session_iso,
                aggregated_signed_confidence=round(signed_sum, 4),
                direction=direction,
                direction_sign=direction_sign,
                n_contributing_events=len(contribs),
                contributors=contributor_rows,
                actionable_above_threshold=actionable,
            )
        )
    return {
        "generated_at_utc": payload.get("generated_at_utc"),
        "horizon_days": horizon_days,
        "universe_scanned": payload.get("universe_scanned"),
        "raw_upcoming_count": payload.get("upcoming_earnings_count"),
        "min_confidence_floor": min_confidence_floor,
        "sessions": [
            {
                "target_session_date": s.target_session_date,
                "aggregated_signed_confidence": s.aggregated_signed_confidence,
                "direction": s.direction,
                "direction_sign": s.direction_sign,
                "n_contributing_events": s.n_contributing_events,
                "actionable_above_threshold": s.actionable_above_threshold,
                "contributors": s.contributors,
            }
            for s in sessions
        ],
    }


# --- Composer ----------------------------------------------------------------


def build_session_watch_payload(
    *,
    variant_key: str | None,
    horizon_days: int,
    min_confidence_floor: float = OVERLAY_PRE_REGISTERED_MIN_CONFIDENCE,
    universe_limit: int | None = None,
    account_label: str,
    current_balance: float,
    peak_eod_balance: float | None,
    todays_pnl: float,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Compose the full session-watch payload: overlay + upcoming + risk state."""
    overlay = load_overlay_summary(variant_key=variant_key)
    upcoming = aggregate_upcoming_sessions(
        horizon_days=horizon_days,
        min_confidence_floor=min_confidence_floor,
        universe_limit=universe_limit,
        settings=settings,
    )
    risk = compute_topstep_state(
        account_label=account_label,
        current_balance=current_balance,
        peak_eod_balance=peak_eod_balance,
        todays_pnl=todays_pnl,
    )
    available_variants = [
        {
            "key": key,
            "file": filename,
            "description": desc,
            "available": (PROJECT_ROOT / "data" / filename).exists(),
        }
        for key, filename, desc in OVERLAY_VARIANTS
    ]
    return {
        "generated_at_utc": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat(),
        "research_disclaimer": (
            "Research overlay, not a trade-execution system. The cached "
            "overlay below was validated on historical sessions only. "
            "Read the coverage_warning and permutation p-value before "
            "treating any signal as actionable."
        ),
        "available_overlay_variants": available_variants,
        "overlay": {
            "variant_key": overlay.variant_key,
            "variant_file": overlay.variant_file,
            "variant_description": overlay.variant_description,
            "generated_at_utc": overlay.generated_at_utc,
            "parameters": overlay.parameters,
            "coverage": overlay.coverage,
            "results": overlay.results,
            "permutation": overlay.permutation,
            "upstream_uplift_on_subset": overlay.upstream_uplift_on_subset,
            "by_confidence_bucket": overlay.by_confidence_bucket,
            "equity_curve": overlay.equity_curve,
            "recent_sessions": overlay.recent_sessions,
            "honesty_notes": overlay.honesty_notes,
        },
        "upcoming": upcoming,
        "topstep": {
            "account_label": risk.account_label,
            "starting_balance": risk.starting_balance,
            "trailing_drawdown": risk.trailing_drawdown,
            "profit_target": risk.profit_target,
            "current_balance": risk.current_balance,
            "peak_eod_balance": risk.peak_eod_balance,
            "todays_pnl": risk.todays_pnl,
            "trailing_floor": risk.trailing_floor,
            "buffer_remaining": risk.buffer_remaining,
            "distance_to_target": risk.distance_to_target,
            "is_locked_floor": risk.is_locked_floor,
            "kill_switch_active": risk.kill_switch_active,
            "kill_switch_reason": risk.kill_switch_reason,
            "presets_available": list(TOPSTEP_PRESETS.keys()),
        },
    }
