"""Market-derived features added at validation time.

These features depend on price series and cross-symbol earnings timing, so
they are computed once during ``validate_case4_earnings`` and persisted on
each event row in the output JSON. ``case4_dataset.event_features`` then
just reads them back.

Every feature is constrained to ``[..., T]`` — nothing post-event leaks in.
"""

from __future__ import annotations

import bisect
import logging
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd

from finhack.config import Settings, load_settings
from finhack.data.company_graph import SPILLOVER_MAP, SYMBOL_TO_COMPANY
from finhack.market_data import get_close_series

logger = logging.getLogger(__name__)


def _parse_event_dt(raw: str | None) -> datetime | None:
    if not raw:
        return None
    txt = str(raw).strip()
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


def _index_of(close_idx: list[pd.Timestamp], target: datetime) -> int | None:
    """Find the latest bar at or before ``target.date()``."""
    if not close_idx:
        return None
    target_d = target.date()
    pos = bisect.bisect_right(close_idx, pd.Timestamp(target_d, tz="UTC")) - 1
    if pos < 0:
        # bisect against naive timestamps when index isn't UTC-aware.
        for i, ts in enumerate(close_idx):
            if ts.to_pydatetime().date() > target_d:
                return i - 1
        return len(close_idx) - 1
    if pos >= len(close_idx):
        return len(close_idx) - 1
    return pos


def _index_of_or_after(
    close_idx: list[pd.Timestamp], target: datetime
) -> int | None:
    if not close_idx:
        return None
    target_d = target.date()
    for i, ts in enumerate(close_idx):
        if ts.to_pydatetime().date() >= target_d:
            return i
    return None


def _return_pct(close: pd.Series, i0: int, i1: int) -> float | None:
    if i0 < 0 or i1 < 0 or i0 >= len(close) or i1 >= len(close):
        return None
    p0 = float(close.iloc[i0])
    p1 = float(close.iloc[i1])
    if p0 == 0:
        return None
    return ((p1 - p0) / p0) * 100.0


def _realized_vol_pct(close: pd.Series, end_i: int, *, window: int) -> float | None:
    start_i = max(0, end_i - window)
    if end_i - start_i < 5:
        return None
    sub = close.iloc[start_i : end_i + 1].pct_change().dropna()
    if sub.empty:
        return None
    return float(sub.std(ddof=1) * 100.0)


def compute_per_symbol_features(
    close: pd.Series,
    events: list[datetime],
) -> dict[datetime, dict[str, Any]]:
    """Compute per-event features that only need the symbol's own price series.

    Returns a dict keyed by event datetime.
    """
    out: dict[datetime, dict[str, Any]] = {}
    if close.empty or not events:
        return out
    # Ensure the index is timezone-aware so comparisons are consistent.
    if close.index.tz is None:
        close = close.copy()
        close.index = close.index.tz_localize("UTC")
    idx_list = list(close.index)
    sorted_events = sorted(events)
    for pos, t_event in enumerate(sorted_events):
        event_i = _index_of_or_after(idx_list, t_event)
        if event_i is None:
            continue
        # Use the bar immediately before T for "pre" returns to keep them
        # strictly inside [T-Nd, T-1].
        pre_end_i = max(0, event_i - 1)
        pre_30d_ret = _return_pct(close, max(0, pre_end_i - 30), pre_end_i)
        pre_60d_ret = _return_pct(close, max(0, pre_end_i - 60), pre_end_i)
        pre_30d_vol = _realized_vol_pct(close, pre_end_i, window=30)

        prior_drift = None
        if pos > 0:
            prior_t = sorted_events[pos - 1]
            prior_i = _index_of_or_after(idx_list, prior_t)
            if prior_i is not None and prior_i + 5 < len(close):
                prior_drift = _return_pct(close, prior_i, prior_i + 5)

        out[t_event] = {
            "pre_30d_return_pct": round(pre_30d_ret, 4) if pre_30d_ret is not None else 0.0,
            "pre_60d_return_pct": round(pre_60d_ret, 4) if pre_60d_ret is not None else 0.0,
            "pre_30d_vol_pct": round(pre_30d_vol, 4) if pre_30d_vol is not None else 0.0,
            "prior_post_earnings_5d_return_pct": (
                round(prior_drift, 4) if prior_drift is not None else 0.0
            ),
        }
    return out


def _sector_peers(symbol: str) -> list[str]:
    """Return symbols in the same sector bucket as ``symbol`` (excluding it)."""
    company = SYMBOL_TO_COMPANY.get((symbol or "").strip().upper())
    if company is None:
        return []
    bucket = company.sector_bucket
    return [
        c.symbol
        for c in SYMBOL_TO_COMPANY.values()
        if c.sector_bucket == bucket and c.symbol != company.symbol
    ]


def _close_5d_return(close: pd.Series, t_event: datetime) -> float | None:
    if close.empty:
        return None
    if close.index.tz is None:
        close = close.copy()
        close.index = close.index.tz_localize("UTC")
    idx_list = list(close.index)
    end_i = _index_of_or_after(idx_list, t_event)
    if end_i is None:
        return None
    end_i = min(end_i, len(close) - 1)
    start_i = max(0, end_i - 5)
    return _return_pct(close, start_i, end_i)


def compute_cross_symbol_features(
    events_by_symbol: dict[str, list[datetime]],
    earnings_by_symbol: dict[str, list[datetime]],
    *,
    settings: Settings | None = None,
    progress: bool = False,
) -> dict[tuple[str, datetime], dict[str, Any]]:
    """Compute features that need data from related symbols.

    - ``sector_cohort_5d_return_pct``: mean pre-event 5d return of sector peers
      ending at T (the same window used as the price baseline).
    - ``neighbor_earnings_count_30d``: number of graph-neighbor (or sector
      peer) symbols whose own earnings date falls in [T-30d, T].

    Returns a dict keyed by ``(symbol, t_event)``.
    """
    cfg = settings or load_settings()
    out: dict[tuple[str, datetime], dict[str, Any]] = {}
    if not events_by_symbol:
        return out

    # Pre-fetch close series for every symbol that appears as an event holder
    # OR as a sector peer of one. We keep a single cache to avoid double work.
    needed: set[str] = set(events_by_symbol.keys())
    for sym in events_by_symbol:
        for peer in _sector_peers(sym):
            needed.add(peer)
        for peer in SPILLOVER_MAP.get(sym, []):
            needed.add(peer)

    earliest = min(
        min(dts) for dts in events_by_symbol.values() if dts
    ) if any(events_by_symbol.values()) else datetime.now(timezone.utc)
    latest = max(
        max(dts) for dts in events_by_symbol.values() if dts
    ) if any(events_by_symbol.values()) else datetime.now(timezone.utc)
    start = (earliest - timedelta(days=20)).date().isoformat()
    end = (latest + timedelta(days=2)).date().isoformat()

    cache: dict[str, pd.Series] = {}
    for i, sym in enumerate(sorted(needed), start=1):
        if progress and (i % 50 == 0 or i == len(needed)):
            print(
                f"[market-features] fetched cohort closes for {i}/{len(needed)} symbols",
                flush=True,
            )
        try:
            cache[sym] = get_close_series(sym, start=start, end=end, settings=cfg)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Cohort close fetch failed for %s: %s", sym, exc)
            cache[sym] = pd.Series(dtype=float)

    # Build per-symbol sorted earnings dates for neighbor counting.
    earnings_sorted: dict[str, list[datetime]] = {
        sym: sorted(dates) for sym, dates in earnings_by_symbol.items() if dates
    }

    for sym, events in events_by_symbol.items():
        peers = _sector_peers(sym)
        graph_neighbors = SPILLOVER_MAP.get(sym, [])
        neighbor_universe = list({*peers, *graph_neighbors}) or peers
        for t_event in events:
            cohort_returns: list[float] = []
            for peer in peers:
                series = cache.get(peer)
                if series is None or series.empty:
                    continue
                ret = _close_5d_return(series, t_event)
                if ret is not None:
                    cohort_returns.append(ret)
            cohort_mean = (
                round(sum(cohort_returns) / len(cohort_returns), 4)
                if cohort_returns
                else 0.0
            )

            window_start = t_event - timedelta(days=30)
            neighbor_count = 0
            for peer in neighbor_universe:
                peer_dates = earnings_sorted.get(peer, [])
                for d in peer_dates:
                    if window_start <= d <= t_event:
                        neighbor_count += 1
                        break  # count peer at most once per event window

            out[(sym, t_event)] = {
                "sector_cohort_5d_return_pct": cohort_mean,
                "neighbor_earnings_count_30d": int(neighbor_count),
            }
    return out


def enrich_events_in_place(
    events: list[dict[str, Any]],
    *,
    earnings_by_symbol: dict[str, list[datetime]] | None = None,
    settings: Settings | None = None,
    progress: bool = False,
) -> dict[str, Any]:
    """Add market-derived features to ``events`` in place.

    ``events`` is the list of evaluated rows produced by
    ``validate_case4_earnings`` (each has ``symbol`` and ``t_event_utc``).
    Returns a small summary dict for logging.
    """
    if not events:
        return {"events_enriched": 0}

    # Group events by symbol and parse timestamps.
    events_by_symbol: dict[str, list[datetime]] = defaultdict(list)
    parsed: list[tuple[datetime, dict[str, Any]]] = []
    for row in events:
        sym = str(row.get("symbol", "")).upper().strip()
        t = _parse_event_dt(row.get("t_event_utc"))
        if not sym or t is None:
            parsed.append((datetime.min.replace(tzinfo=timezone.utc), row))
            continue
        events_by_symbol[sym].append(t)
        parsed.append((t, row))

    # Per-symbol price-only features.
    cfg = settings or load_settings()
    per_event: dict[tuple[str, datetime], dict[str, Any]] = {}
    total_syms = len(events_by_symbol)
    for i, (sym, dts) in enumerate(events_by_symbol.items(), start=1):
        if progress and (i % 25 == 0 or i == total_syms):
            print(
                f"[market-features] per-symbol features {i}/{total_syms}",
                flush=True,
            )
        sorted_dts = sorted(dts)
        start = (sorted_dts[0] - timedelta(days=120)).date().isoformat()
        end = (sorted_dts[-1] + timedelta(days=2)).date().isoformat()
        try:
            close = get_close_series(sym, start=start, end=end, settings=cfg)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Close fetch failed for %s: %s", sym, exc)
            close = pd.Series(dtype=float)
        feats = compute_per_symbol_features(close, sorted_dts)
        for t, vals in feats.items():
            per_event[(sym, t)] = vals

    # Cross-symbol features (sector cohort + neighbor count).
    cross = compute_cross_symbol_features(
        events_by_symbol,
        earnings_by_symbol or events_by_symbol,
        settings=cfg,
        progress=progress,
    )
    for key, vals in cross.items():
        merged = per_event.get(key, {})
        merged.update(vals)
        per_event[key] = merged

    enriched = 0
    for row in events:
        sym = str(row.get("symbol", "")).upper().strip()
        t = _parse_event_dt(row.get("t_event_utc"))
        if not sym or t is None:
            continue
        feats = per_event.get((sym, t))
        if not feats:
            continue
        for k, v in feats.items():
            row.setdefault(k, v)
        enriched += 1

    return {
        "events_enriched": enriched,
        "symbols_with_events": len(events_by_symbol),
    }
