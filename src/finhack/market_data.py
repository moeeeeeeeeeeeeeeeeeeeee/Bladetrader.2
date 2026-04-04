"""Market data access for validation and Case 4 dashboard."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx
import pandas as pd
import yfinance as yf

from finhack.config import MarketDataProvider, Settings, load_settings
from finhack.data.company_graph import CASE4_SYMBOLS, SPILLOVER_MAP, SYMBOL_TO_COMPANY

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


@dataclass(slots=True)
class MarketPoint:
    symbol: str
    price: float | None
    previous_close: float | None
    change: float | None
    change_percent: float | None
    as_of_utc: str
    source: str
    impacted_symbols: list[str]


@dataclass(slots=True)
class MarketSymbol:
    symbol: str
    company_name: str
    source: str


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _yahoo_close_series(symbol: str, start: str, end: str) -> pd.Series:
    df = yf.Ticker(symbol).history(start=start, end=end, auto_adjust=False)
    if df.empty or "Close" not in df.columns:
        return pd.Series(dtype=float)
    return df["Close"].dropna()


def _eodhd_close_series(symbol: str, start: str, end: str, api_key: str) -> pd.Series:
    params = {
        "api_token": api_key,
        "fmt": "json",
        "period": "d",
        "from": start,
        "to": end,
    }
    with httpx.Client(timeout=30.0) as client:
        res = client.get(f"{EODHD_BASE_URL}/eod/{_symbol_to_eodhd(symbol)}", params=params)
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
        if str(row.get("code", "")).upper() not in {symbol.upper(), _symbol_to_eodhd(symbol)}:
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


def _point_from_eodhd(symbol: str, payload: dict[str, Any]) -> MarketPoint:
    price = _to_float(payload.get("close"))
    prev = _to_float(payload.get("previousClose"))
    change = _to_float(payload.get("change"))
    pct = _to_float(payload.get("change_p"))
    if change is None and price is not None and prev is not None:
        change = price - prev
    if pct is None and change is not None and prev not in (None, 0):
        pct = (change / prev) * 100.0
    return MarketPoint(
        symbol=symbol,
        price=price,
        previous_close=prev,
        change=change,
        change_percent=pct,
        as_of_utc=_now_iso(),
        source="eodhd",
        impacted_symbols=SPILLOVER_MAP.get(symbol, []),
    )


def _point_from_yahoo(symbol: str) -> MarketPoint:
    data = yf.Ticker(symbol).history(period="5d", interval="1d", auto_adjust=False)
    if data.empty:
        return MarketPoint(
            symbol=symbol,
            price=None,
            previous_close=None,
            change=None,
            change_percent=None,
            as_of_utc=_now_iso(),
            source="yahoo",
            impacted_symbols=SPILLOVER_MAP.get(symbol, []),
        )
    close = data["Close"].dropna()
    price = _to_float(close.iloc[-1]) if len(close) >= 1 else None
    prev = _to_float(close.iloc[-2]) if len(close) >= 2 else None
    change = (price - prev) if (price is not None and prev is not None) else None
    pct = ((change / prev) * 100.0) if (change is not None and prev not in (None, 0)) else None
    return MarketPoint(
        symbol=symbol,
        price=price,
        previous_close=prev,
        change=change,
        change_percent=pct,
        as_of_utc=_now_iso(),
        source="yahoo",
        impacted_symbols=SPILLOVER_MAP.get(symbol, []),
    )


def _fetch_eodhd_realtime(symbol: str, api_key: str) -> dict[str, Any]:
    eodhd_symbol = _symbol_to_eodhd(symbol)
    with httpx.Client(timeout=20.0) as client:
        resp = client.get(
            f"{EODHD_BASE_URL}/real-time/{eodhd_symbol}",
            params={"api_token": api_key, "fmt": "json"},
        )
        resp.raise_for_status()
        payload = resp.json()
    if not isinstance(payload, dict):
        raise ValueError(f"Unexpected EODHD payload for {symbol}")
    if payload.get("code"):
        raise ValueError(f"EODHD error for {symbol}: {payload.get('message') or payload.get('code')}")
    return payload


def _fetch_eodhd_us_symbols(api_key: str, limit: int) -> list[MarketSymbol]:
    with httpx.Client(timeout=45.0) as client:
        resp = client.get(
            f"{EODHD_BASE_URL}/exchange-symbol-list/US",
            params={"api_token": api_key, "fmt": "json"},
        )
        resp.raise_for_status()
        payload = resp.json()
    if not isinstance(payload, list):
        return []

    out: list[MarketSymbol] = []
    for row in payload:
        if not isinstance(row, dict):
            continue
        code = str(row.get("Code") or "").strip().upper()
        name = str(row.get("Name") or "").strip()
        if not code or len(code) > 12:
            continue
        stock_type = str(row.get("Type") or "").strip().lower()
        if stock_type and stock_type not in {"common stock", "etf", "fund"}:
            continue
        out.append(
            MarketSymbol(
                symbol=code,
                company_name=name or code,
                source="eodhd",
            )
        )
        if len(out) >= limit:
            break
    return out


def get_market_symbols(limit: int = 1500, settings: Settings | None = None) -> tuple[str, list[MarketSymbol]]:
    cfg = settings or load_settings()
    safe_limit = max(50, min(limit, 5000))
    provider = cfg.market_data_provider.value
    if provider == "eodhd" and cfg.eodhd_api_key:
        try:
            rows = _fetch_eodhd_us_symbols(cfg.eodhd_api_key, safe_limit)
            if rows:
                return provider, rows
        except Exception:
            pass

    fallback = sorted(CASE4_SYMBOLS)[:safe_limit]
    rows = [
        MarketSymbol(
            symbol=symbol,
            company_name=SYMBOL_TO_COMPANY.get(symbol).name if SYMBOL_TO_COMPANY.get(symbol) else symbol,
            source="case4",
        )
        for symbol in fallback
    ]
    return provider, rows


def get_case4_market_points(settings: Settings | None = None) -> tuple[str, list[MarketPoint]]:
    cfg = settings or load_settings()
    provider = cfg.market_data_provider.value
    points: list[MarketPoint] = []
    for symbol in CASE4_SYMBOLS:
        if provider == "eodhd" and cfg.eodhd_api_key:
            try:
                payload = _fetch_eodhd_realtime(symbol, cfg.eodhd_api_key)
                points.append(_point_from_eodhd(symbol, payload))
                continue
            except Exception:
                pass
        points.append(_point_from_yahoo(symbol))
    return provider, points
