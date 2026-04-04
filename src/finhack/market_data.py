"""Market data adapter (Yahoo + EODHD) for validation scripts."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import httpx
import pandas as pd
import yfinance as yf

from finhack.config import MarketDataProvider, Settings, load_settings

EODHD_BASE_URL = "https://eodhd.com/api"


def _symbol_to_eodhd(symbol: str) -> str:
    clean = (symbol or "").strip().upper()
    if "." in clean:
        return clean
    return f"{clean}.US"


def _parse_dt(raw: str | datetime | None) -> datetime | None:
    if raw is None:
        return None
    if isinstance(raw, datetime):
        dt = raw
    else:
        txt = str(raw).strip()
        if not txt:
            return None
        if txt.endswith("Z"):
            txt = txt[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(txt)
        except ValueError:
            try:
                dt = datetime.strptime(txt[:10], "%Y-%m-%d")
            except ValueError:
                return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _yahoo_close_series(symbol: str, start: str, end: str) -> pd.Series:
    df = yf.Ticker(symbol).history(start=start, end=end, auto_adjust=False)
    if df.empty or "Close" not in df.columns:
        return pd.Series(dtype=float)
    return df["Close"].dropna()


def _eodhd_close_series(symbol: str, start: str, end: str, api_key: str) -> pd.Series:
    eodhd_symbol = _symbol_to_eodhd(symbol)
    params = {
        "api_token": api_key,
        "fmt": "json",
        "period": "d",
        "from": start,
        "to": end,
    }
    with httpx.Client(timeout=30.0) as client:
        res = client.get(f"{EODHD_BASE_URL}/eod/{eodhd_symbol}", params=params)
        res.raise_for_status()
        payload = res.json()
    if not isinstance(payload, list) or not payload:
        return pd.Series(dtype=float)
    frame = pd.DataFrame(payload)
    if "date" not in frame.columns:
        return pd.Series(dtype=float)
    value_col = "adjusted_close" if "adjusted_close" in frame.columns else "close"
    if value_col not in frame.columns:
        return pd.Series(dtype=float)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame[value_col] = pd.to_numeric(frame[value_col], errors="coerce")
    frame = frame.dropna(subset=["date", value_col]).sort_values("date")
    if frame.empty:
        return pd.Series(dtype=float)
    out = pd.Series(frame[value_col].values, index=frame["date"])
    out.name = "Close"
    return out


def _eodhd_earnings_events(
    symbol: str, limit: int, recent_days: int, api_key: str
) -> list[datetime]:
    now = datetime.now(timezone.utc)
    from_dt = now - timedelta(days=max(30, recent_days))
    params = {
        "api_token": api_key,
        "fmt": "json",
        "symbols": _symbol_to_eodhd(symbol),
        "from": from_dt.date().isoformat(),
        "to": now.date().isoformat(),
    }
    with httpx.Client(timeout=30.0) as client:
        res = client.get(f"{EODHD_BASE_URL}/calendar/earnings", params=params)
        res.raise_for_status()
        payload = res.json()

    rows: list[dict[str, Any]] = []
    if isinstance(payload, dict):
        maybe_rows = payload.get("earnings")
        if isinstance(maybe_rows, list):
            rows = [r for r in maybe_rows if isinstance(r, dict)]
    elif isinstance(payload, list):
        rows = [r for r in payload if isinstance(r, dict)]

    out: list[datetime] = []
    for row in rows:
        if str(row.get("code", "")).upper() not in {
            symbol.upper(),
            _symbol_to_eodhd(symbol),
        }:
            continue
        dt = _parse_dt(row.get("date") or row.get("report_date"))
        if dt is None:
            continue
        if from_dt <= dt < now:
            out.append(dt)
    out = sorted(set(out), reverse=True)
    return sorted(out[: max(1, limit)])


def get_close_series(
    symbol: str, start: str, end: str, *, settings: Settings | None = None
) -> pd.Series:
    cfg = settings or load_settings()
    if cfg.market_data_provider == MarketDataProvider.EODHD and cfg.eodhd_api_key:
        try:
            out = _eodhd_close_series(symbol, start, end, cfg.eodhd_api_key)
            if not out.empty:
                return out
        except Exception:
            pass
    return _yahoo_close_series(symbol, start, end)


def get_earnings_events(
    symbol: str,
    limit: int = 8,
    recent_days: int = 365,
    *,
    settings: Settings | None = None,
) -> list[datetime]:
    cfg = settings or load_settings()
    if cfg.market_data_provider == MarketDataProvider.EODHD and cfg.eodhd_api_key:
        try:
            out = _eodhd_earnings_events(symbol, limit, recent_days, cfg.eodhd_api_key)
            if out:
                return out
        except Exception:
            pass

    # Yahoo fallback for resilience in dev.
    try:
        t = yf.Ticker(symbol)
        df = t.get_earnings_dates(limit=limit)
    except Exception:
        return []
    if df is None or df.empty:
        return []
    out: list[datetime] = []
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=max(30, recent_days))
    for idx in df.index:
        dt = _parse_dt(idx.to_pydatetime() if hasattr(idx, "to_pydatetime") else idx)
        if dt is None:
            continue
        if cutoff <= dt < now:
            out.append(dt)
    return sorted(out)
