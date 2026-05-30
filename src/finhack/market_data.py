"""Market data access for validation and Case 4 dashboard.

EODHD is the only supported provider for prices, quotes, and earnings dates.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from threading import Lock
from typing import Any

import httpx
import pandas as pd

from finhack.config import MarketDataProvider, Settings, load_settings
from finhack.data.company_graph import CASE4_SYMBOLS, SPILLOVER_MAP, SYMBOL_TO_COMPANY

logger = logging.getLogger(__name__)

EODHD_BASE_URL = "https://eodhd.com/api"
_EODHD_EARNINGS_FORBIDDEN = False
_CHART_INTRADAY_MAX_BARS = 450

_EODHD_INTRADAY_MAP: dict[str, tuple[str, int, str]] = {
    "1m": ("1m", 7, "Up to 7d, 1-minute bars (EODHD)"),
    "2m": ("5m", 30, "Up to 30d, 5-minute bars (EODHD, 2m mapped)"),
    "5m": ("5m", 60, "Up to 60d, 5-minute bars (EODHD)"),
    "15m": ("5m", 60, "Up to 60d, 5-minute bars (EODHD, 15m mapped)"),
    "30m": ("5m", 90, "Up to 90d, 5-minute bars (EODHD, 30m mapped)"),
    "1h": ("1h", 180, "Up to 180d, 1-hour bars (EODHD)"),
}

_http_client: httpx.Client | None = None
_cache: dict[str, tuple[float, Any]] = {}
_cache_lock = Lock()


def _symbol_to_eodhd(symbol: str) -> str:
    clean = (symbol or "").strip().upper()
    if "." in clean:
        return clean
    return f"{clean}.US"


def _symbol_from_eodhd(code: str) -> str:
    clean = (code or "").strip().upper()
    if clean.endswith(".US"):
        return clean[:-3]
    return clean.split(".", 1)[0]


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


def _get_http_client() -> httpx.Client:
    global _http_client
    if _http_client is None or _http_client.is_closed:
        _http_client = httpx.Client(timeout=30.0)
    return _http_client


def _cache_get(key: str, ttl_seconds: float) -> Any | None:
    if ttl_seconds <= 0:
        return None
    with _cache_lock:
        entry = _cache.get(key)
        if entry is None:
            return None
        ts, val = entry
        if time.monotonic() - ts > ttl_seconds:
            del _cache[key]
            return None
        return val


def _cache_set(key: str, val: Any) -> None:
    with _cache_lock:
        _cache[key] = (time.monotonic(), val)


def _eodhd_enabled(cfg: Settings) -> bool:
    return cfg.market_data_provider == MarketDataProvider.EODHD and bool(
        (cfg.eodhd_api_key or "").strip()
    )


def _note_eodhd_earnings_forbidden(exc: Exception) -> None:
    global _EODHD_EARNINGS_FORBIDDEN
    if _EODHD_EARNINGS_FORBIDDEN:
        return
    if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 403:
        _EODHD_EARNINGS_FORBIDDEN = True
        logger.warning(
            "EODHD earnings calendar returned 403; earnings dates disabled "
            "(your EODHD plan does not include the calendar endpoint)."
        )


def _eodhd_earnings_enabled(cfg: Settings) -> bool:
    return _eodhd_enabled(cfg) and not _EODHD_EARNINGS_FORBIDDEN


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


def _eodhd_get(path: str, params: dict[str, Any]) -> Any:
    client = _get_http_client()
    res = client.get(f"{EODHD_BASE_URL}{path}", params=params)
    res.raise_for_status()
    return res.json()


def _eodhd_close_series(symbol: str, start: str, end: str, api_key: str) -> pd.Series:
    params = {
        "api_token": api_key,
        "fmt": "json",
        "period": "d",
        "from": start,
        "to": end,
    }
    payload = _eodhd_get(f"/eod/{_symbol_to_eodhd(symbol)}", params)
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


def _eodhd_ohlc_df(
    symbol: str, start: str, end: str, api_key: str, *, period: str = "d"
) -> pd.DataFrame:
    params = {
        "api_token": api_key,
        "fmt": "json",
        "period": period,
        "from": start,
        "to": end,
    }
    payload = _eodhd_get(f"/eod/{_symbol_to_eodhd(symbol)}", params)
    if not isinstance(payload, list) or not payload:
        return pd.DataFrame()
    frame = pd.DataFrame(payload)
    if "date" not in frame.columns:
        return pd.DataFrame()
    need = ("open", "high", "low")
    if not all(c in frame.columns for c in need):
        return pd.DataFrame()
    close_col = "adjusted_close" if "adjusted_close" in frame.columns else "close"
    if close_col not in frame.columns:
        return pd.DataFrame()
    dated = pd.to_datetime(frame["date"], errors="coerce")
    o = pd.to_numeric(frame["open"], errors="coerce")
    hi = pd.to_numeric(frame["high"], errors="coerce")
    lo = pd.to_numeric(frame["low"], errors="coerce")
    cl = pd.to_numeric(frame[close_col], errors="coerce")
    built = pd.DataFrame({"date": dated, "Open": o, "High": hi, "Low": lo, "Close": cl})
    built = built.dropna(subset=["date", "Open", "High", "Low", "Close"]).sort_values("date")
    if built.empty:
        return pd.DataFrame()
    built = built.set_index("date")
    return built[["Open", "High", "Low", "Close"]]


def _parse_eodhd_earnings_rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        maybe_rows = payload.get("earnings")
        if isinstance(maybe_rows, list):
            return [r for r in maybe_rows if isinstance(r, dict)]
    if isinstance(payload, list):
        return [r for r in payload if isinstance(r, dict)]
    return []


def _earnings_rows_to_events(
    rows: list[dict[str, Any]],
    *,
    symbol: str | None,
    limit: int,
    recent_days: int,
) -> list[datetime]:
    now = datetime.now(timezone.utc)
    from_dt = now - timedelta(days=max(30, recent_days))
    allowed = None
    if symbol:
        allowed = {symbol.upper(), _symbol_to_eodhd(symbol)}
    out: list[datetime] = []
    for row in rows:
        code = str(row.get("code", "")).upper()
        if allowed is not None and code not in allowed:
            continue
        dt = _parse_dt(row.get("date") or row.get("report_date"))
        if dt is None:
            continue
        if from_dt <= dt < now:
            out.append(dt)
    out = sorted(set(out), reverse=True)
    return sorted(out[: max(1, limit)])


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
    payload = _eodhd_get("/calendar/earnings", params)
    rows = _parse_eodhd_earnings_rows(payload)
    return _earnings_rows_to_events(rows, symbol=symbol, limit=limit, recent_days=recent_days)


def _eodhd_earnings_events_batch(
    symbols: list[str], limit: int, recent_days: int, api_key: str
) -> dict[str, list[datetime]]:
    clean = sorted({(s or "").strip().upper() for s in symbols if (s or "").strip()})
    if not clean:
        return {}
    now = datetime.now(timezone.utc)
    from_dt = now - timedelta(days=max(30, recent_days))
    params = {
        "api_token": api_key,
        "fmt": "json",
        "symbols": ",".join(_symbol_to_eodhd(s) for s in clean),
        "from": from_dt.date().isoformat(),
        "to": now.date().isoformat(),
    }
    payload = _eodhd_get("/calendar/earnings", params)
    rows = _parse_eodhd_earnings_rows(payload)
    grouped: dict[str, list[datetime]] = {s: [] for s in clean}
    for row in rows:
        code = str(row.get("code", "")).upper()
        sym = _symbol_from_eodhd(code)
        if sym not in grouped:
            continue
        dt = _parse_dt(row.get("date") or row.get("report_date"))
        if dt is None:
            continue
        if from_dt <= dt < now:
            grouped[sym].append(dt)
    out: dict[str, list[datetime]] = {}
    for sym, events in grouped.items():
        deduped = sorted(set(events), reverse=True)
        out[sym] = sorted(deduped[: max(1, limit)])
    return out


def get_close_series(
    symbol: str, start: str, end: str, *, settings: Settings | None = None
) -> pd.Series:
    cfg = settings or load_settings()
    cache_key = f"close:{symbol}:{start}:{end}:{cfg.market_data_provider.value}"
    cached = _cache_get(cache_key, cfg.market_data_cache_ttl_seconds)
    if cached is not None:
        return cached.copy()

    if not _eodhd_enabled(cfg):
        return pd.Series(dtype=float)

    try:
        out = _eodhd_close_series(symbol, start, end, cfg.eodhd_api_key or "")
    except Exception as exc:
        logger.warning("EODHD close series failed for %s: %s", symbol, exc)
        return pd.Series(dtype=float)
    _cache_set(cache_key, out)
    return out


_OHLC_EOD_INTERVALS = frozenset({"1d", "1wk", "1mo"})
_EODHD_OHLC_PERIOD = {"1d": "d", "1wk": "w", "1mo": "m"}


def get_ohlc_series(
    symbol: str,
    start: str,
    end: str,
    *,
    settings: Settings | None = None,
    interval: str = "1d",
) -> pd.DataFrame:
    cfg = settings or load_settings()
    iv = (interval or "1d").strip().lower()
    if iv not in _OHLC_EOD_INTERVALS:
        iv = "1d"
    cache_key = f"ohlc:{symbol}:{start}:{end}:{iv}:{cfg.market_data_provider.value}"
    cached = _cache_get(cache_key, cfg.market_data_cache_ttl_seconds)
    if cached is not None:
        return cached.copy()

    if not _eodhd_enabled(cfg):
        return pd.DataFrame()

    eodhd_period = _EODHD_OHLC_PERIOD[iv]
    try:
        out = _eodhd_ohlc_df(
            symbol, start, end, cfg.eodhd_api_key or "", period=eodhd_period
        )
    except Exception as exc:
        logger.warning("EODHD OHLC failed for %s: %s", symbol, exc)
        return pd.DataFrame()
    _cache_set(cache_key, out)
    return out


def ohlc_frame_to_point_rows(
    frame: pd.DataFrame,
    *,
    intraday: bool = False,
) -> list[tuple[str, float, float, float, float]]:
    """Serialize an OHLC dataframe (DatetimeIndex) to API history rows."""
    out: list[tuple[str, float, float, float, float]] = []
    if frame is None or frame.empty:
        return out
    for idx, row in frame.iterrows():
        ts = pd.Timestamp(idx)
        if intraday:
            if ts.tzinfo is None:
                ts = ts.tz_localize("America/New_York")
            dt_txt = ts.tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
        else:
            if ts.tzinfo is not None:
                # Keep the exchange/session calendar date (avoid UTC day shifts).
                ts = ts.tz_convert("America/New_York")
            dt_txt = ts.strftime("%Y-%m-%d")
        out.append(
            (
                dt_txt,
                float(row["Open"]),
                float(row["High"]),
                float(row["Low"]),
                float(row["Close"]),
            )
        )
    return out


def _eodhd_ohlc_intraday_df(
    symbol: str, interval: str, api_key: str, max_bars: int = _CHART_INTRADAY_MAX_BARS
) -> tuple[pd.DataFrame, str]:
    clean = (symbol or "").strip().upper()
    iv = (interval or "").strip().lower()
    mapped = _EODHD_INTRADAY_MAP.get(iv)
    if not clean or mapped is None:
        return pd.DataFrame(), ""
    eodhd_interval, lookback_days, note = mapped
    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=lookback_days)
    params = {
        "api_token": api_key,
        "fmt": "json",
        "interval": eodhd_interval,
        "from": int(start_dt.timestamp()),
        "to": int(end_dt.timestamp()),
    }
    payload = _eodhd_get(f"/intraday/{_symbol_to_eodhd(clean)}", params)
    if not isinstance(payload, list) or not payload:
        return pd.DataFrame(), note
    frame = pd.DataFrame(payload)
    dt_col = "datetime" if "datetime" in frame.columns else "date"
    if dt_col not in frame.columns:
        return pd.DataFrame(), note
    for col in ("open", "high", "low", "close"):
        if col not in frame.columns:
            return pd.DataFrame(), note
    frame[dt_col] = pd.to_datetime(frame[dt_col], errors="coerce", utc=True)
    built = pd.DataFrame(
        {
            "date": frame[dt_col],
            "Open": pd.to_numeric(frame["open"], errors="coerce"),
            "High": pd.to_numeric(frame["high"], errors="coerce"),
            "Low": pd.to_numeric(frame["low"], errors="coerce"),
            "Close": pd.to_numeric(frame["close"], errors="coerce"),
        }
    )
    built = built.dropna(subset=["date", "Open", "High", "Low", "Close"]).sort_values("date")
    if built.empty:
        return pd.DataFrame(), note
    built = built.set_index("date")[["Open", "High", "Low", "Close"]]
    if len(built) > max_bars:
        built = built.tail(max_bars)
    return built, note


def get_ohlc_intraday(
    symbol: str,
    interval: str,
    *,
    settings: Settings | None = None,
    max_bars: int = _CHART_INTRADAY_MAX_BARS,
) -> tuple[pd.DataFrame, str, str]:
    """Intraday OHLC (EODHD only). Returns (df, note, provider)."""
    cfg = settings or load_settings()
    iv = (interval or "").strip().lower()
    cache_key = f"intraday:{symbol}:{iv}:{cfg.market_data_provider.value}"
    cached = _cache_get(cache_key, cfg.market_data_live_cache_ttl_seconds)
    if cached is not None:
        frame, note, provider = cached
        return frame.copy(), note, provider

    if not _eodhd_enabled(cfg):
        return pd.DataFrame(), "", "eodhd-intraday"

    try:
        frame, note = _eodhd_ohlc_intraday_df(
            symbol, iv, cfg.eodhd_api_key or "", max_bars=max_bars
        )
    except Exception as exc:
        logger.warning("EODHD intraday failed for %s (%s): %s", symbol, iv, exc)
        return pd.DataFrame(), "", "eodhd-intraday"

    _cache_set(cache_key, (frame, note, "eodhd-intraday"))
    return frame, note, "eodhd-intraday"


def get_earnings_events(
    symbol: str,
    limit: int = 8,
    recent_days: int = 365,
    *,
    settings: Settings | None = None,
) -> list[datetime]:
    cfg = settings or load_settings()
    cache_key = f"earnings:{symbol}:{limit}:{recent_days}:{cfg.market_data_provider.value}"
    cached = _cache_get(cache_key, cfg.market_data_cache_ttl_seconds)
    if cached is not None:
        return list(cached)

    if not _eodhd_earnings_enabled(cfg):
        return []

    try:
        out = _eodhd_earnings_events(symbol, limit, recent_days, cfg.eodhd_api_key or "")
    except Exception as exc:
        _note_eodhd_earnings_forbidden(exc)
        logger.warning("EODHD earnings failed for %s: %s", symbol, exc)
        return []

    _cache_set(cache_key, out)
    return out


def get_earnings_events_batch(
    symbols: list[str],
    limit: int = 8,
    recent_days: int = 365,
    *,
    settings: Settings | None = None,
    chunk_size: int = 40,
) -> dict[str, list[datetime]]:
    """Fetch earnings dates for many symbols (EODHD batched)."""
    cfg = settings or load_settings()
    clean = sorted({(s or "").strip().upper() for s in symbols if (s or "").strip()})
    if not clean:
        return {}

    cache_key = (
        f"earnings-batch:{','.join(clean)}:{limit}:{recent_days}:{cfg.market_data_provider.value}"
    )
    cached = _cache_get(cache_key, cfg.market_data_cache_ttl_seconds)
    if cached is not None:
        return dict(cached)

    if not _eodhd_earnings_enabled(cfg):
        return {s: [] for s in clean}

    merged: dict[str, list[datetime]] = {s: [] for s in clean}
    step = max(1, chunk_size)
    try:
        for i in range(0, len(clean), step):
            chunk = clean[i : i + step]
            part = _eodhd_earnings_events_batch(
                chunk, limit, recent_days, cfg.eodhd_api_key or ""
            )
            for sym, events in part.items():
                if sym in merged:
                    merged[sym] = events
    except Exception as exc:
        _note_eodhd_earnings_forbidden(exc)
        logger.warning("EODHD batch earnings failed: %s", exc)
        return {s: [] for s in clean}

    _cache_set(cache_key, merged)
    return merged


def _earnings_rows_to_upcoming(
    rows: list[dict[str, Any]],
    *,
    symbol: str | None,
    horizon_days: int,
) -> list[datetime]:
    now = datetime.now(timezone.utc)
    horizon = now + timedelta(days=max(1, horizon_days))
    allowed = None
    if symbol:
        allowed = {symbol.upper(), _symbol_to_eodhd(symbol)}
    out: list[datetime] = []
    for row in rows:
        code = str(row.get("code", "")).upper()
        if allowed is not None and code not in allowed:
            continue
        dt = _parse_dt(row.get("date") or row.get("report_date"))
        if dt is None:
            continue
        if now <= dt <= horizon:
            out.append(dt)
    return sorted(set(out))


def get_upcoming_earnings_batch(
    symbols: list[str],
    *,
    horizon_days: int = 14,
    settings: Settings | None = None,
    chunk_size: int = 40,
) -> dict[str, datetime]:
    """Next earnings date per symbol within horizon (EODHD)."""
    cfg = settings or load_settings()
    clean = sorted({(s or "").strip().upper() for s in symbols if (s or "").strip()})
    if not clean:
        return {}

    if not _eodhd_earnings_enabled(cfg):
        return {}

    now = datetime.now(timezone.utc)
    horizon = now + timedelta(days=max(1, horizon_days))
    out: dict[str, datetime] = {}

    step = max(1, chunk_size)
    for i in range(0, len(clean), step):
        chunk = clean[i : i + step]
        params = {
            "api_token": cfg.eodhd_api_key or "",
            "fmt": "json",
            "symbols": ",".join(_symbol_to_eodhd(s) for s in chunk),
            "from": now.date().isoformat(),
            "to": horizon.date().isoformat(),
        }
        try:
            payload = _eodhd_get("/calendar/earnings", params)
            rows = _parse_eodhd_earnings_rows(payload)
            grouped: dict[str, list[datetime]] = {s: [] for s in chunk}
            for row in rows:
                code = str(row.get("code", "")).upper()
                sym = _symbol_from_eodhd(code)
                if sym not in grouped:
                    continue
                dt = _parse_dt(row.get("date") or row.get("report_date"))
                if dt and now <= dt <= horizon:
                    grouped[sym].append(dt)
            for sym, dts in grouped.items():
                if dts:
                    out[sym] = min(dts)
        except Exception as exc:
            _note_eodhd_earnings_forbidden(exc)
            logger.warning("EODHD upcoming earnings chunk failed: %s", exc)
            break

    return out


def _point_from_eodhd(symbol: str, payload: dict[str, Any]) -> MarketPoint:
    price = _to_float(payload.get("close"))
    prev = _to_float(payload.get("previousClose"))
    change = _to_float(payload.get("change"))
    pct = _to_float(payload.get("change_p"))
    if change is None and price is not None and prev is not None:
        change = price - prev
    if pct is None and change is not None and prev not in (None, 0):
        pct = (change / prev) * 100.0
    ts = payload.get("timestamp")
    as_of = _now_iso()
    if ts is not None:
        try:
            as_of = datetime.fromtimestamp(int(ts), tz=timezone.utc).replace(microsecond=0).isoformat()
        except (TypeError, ValueError, OSError):
            pass
    return MarketPoint(
        symbol=symbol,
        price=price,
        previous_close=prev,
        change=change,
        change_percent=pct,
        as_of_utc=as_of,
        source="eodhd",
        impacted_symbols=SPILLOVER_MAP.get(symbol, []),
    )


def _empty_point(symbol: str, *, source: str) -> MarketPoint:
    return MarketPoint(
        symbol=symbol,
        price=None,
        previous_close=None,
        change=None,
        change_percent=None,
        as_of_utc=_now_iso(),
        source=source,
        impacted_symbols=SPILLOVER_MAP.get(symbol, []),
    )


def _fetch_eodhd_realtime_batch(symbols: list[str], api_key: str) -> dict[str, dict[str, Any]]:
    clean = [(s or "").strip().upper() for s in symbols if (s or "").strip()]
    if not clean:
        return {}
    eodhd_symbols = [_symbol_to_eodhd(s) for s in clean]
    primary = eodhd_symbols[0]
    params: dict[str, Any] = {"api_token": api_key, "fmt": "json"}
    if len(eodhd_symbols) > 1:
        params["s"] = ",".join(eodhd_symbols[1:])
    payload = _eodhd_get(f"/real-time/{primary}", params)
    rows: list[dict[str, Any]]
    if isinstance(payload, list):
        rows = [r for r in payload if isinstance(r, dict)]
    elif isinstance(payload, dict):
        rows = [payload]
    else:
        raise ValueError("Unexpected EODHD realtime batch payload")

    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        code = str(row.get("code") or row.get("symbol") or "").upper()
        if not code:
            continue
        sym = _symbol_from_eodhd(code)
        if row.get("code") and str(row.get("code")).upper() not in {"", "NA"}:
            out[sym] = row
    return out


def _fetch_eodhd_us_symbols(api_key: str, limit: int) -> list[MarketSymbol]:
    payload = _eodhd_get("/exchange-symbol-list/US", {"api_token": api_key, "fmt": "json"})
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
        if stock_type and stock_type not in {"common stock"}:
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
    cache_key = f"symbols:{safe_limit}:{provider}"
    cached = _cache_get(cache_key, cfg.market_data_symbols_cache_ttl_seconds)
    if cached is not None:
        return provider, list(cached)

    if _eodhd_enabled(cfg):
        try:
            rows = _fetch_eodhd_us_symbols(cfg.eodhd_api_key or "", safe_limit)
            if rows:
                _cache_set(cache_key, rows)
                return provider, rows
        except Exception as exc:
            logger.warning("EODHD symbol catalog failed: %s", exc)

    fallback = sorted(CASE4_SYMBOLS)[:safe_limit]
    rows = [
        MarketSymbol(
            symbol=symbol,
            company_name=SYMBOL_TO_COMPANY.get(symbol).name if SYMBOL_TO_COMPANY.get(symbol) else symbol,
            source="case4",
        )
        for symbol in fallback
    ]
    _cache_set(cache_key, rows)
    return provider, rows


def get_case4_market_points(settings: Settings | None = None) -> tuple[str, list[MarketPoint]]:
    cfg = settings or load_settings()
    provider = cfg.market_data_provider.value
    cache_key = f"case4-live:{provider}"
    cached = _cache_get(cache_key, cfg.market_data_live_cache_ttl_seconds)
    if cached is not None:
        return provider, list(cached)

    points: list[MarketPoint] = []
    if not _eodhd_enabled(cfg):
        points = [_empty_point(symbol, source="eodhd-missing-key") for symbol in CASE4_SYMBOLS]
        _cache_set(cache_key, points)
        return provider, points

    try:
        batch = _fetch_eodhd_realtime_batch(list(CASE4_SYMBOLS), cfg.eodhd_api_key or "")
        for symbol in CASE4_SYMBOLS:
            payload = batch.get(symbol)
            if payload:
                points.append(_point_from_eodhd(symbol, payload))
            else:
                points.append(_empty_point(symbol, source="eodhd-missing"))
    except Exception as exc:
        logger.warning("EODHD batch realtime failed: %s", exc)
        points = [_empty_point(symbol, source="eodhd-error") for symbol in CASE4_SYMBOLS]

    _cache_set(cache_key, points)
    return provider, points


def _daily_vol_pct(closes: pd.Series) -> float:
    if closes is None or len(closes) < 5:
        return 2.0
    rets = closes.pct_change().dropna()
    if rets.empty:
        return 2.0
    return float(rets.std(ddof=1) * 100.0)


def get_daily_volatility_pct(
    symbol: str,
    *,
    lookback_days: int = 60,
    settings: Settings | None = None,
) -> float:
    cfg = settings or load_settings()
    end = datetime.now(timezone.utc).date()
    start = end - timedelta(days=max(30, lookback_days + 10))
    series = get_close_series(symbol, start.isoformat(), end.isoformat(), settings=cfg)
    return round(_daily_vol_pct(series.tail(max(5, lookback_days))), 4)


def _corr_series(a: pd.Series, b: pd.Series) -> float:
    if a.empty or b.empty:
        return 0.0
    joined = pd.concat([a, b], axis=1, join="inner").dropna()
    if len(joined) < 10:
        return 0.0
    x = joined.iloc[:, 0]
    y = joined.iloc[:, 1]
    if float(x.std()) == 0 or float(y.std()) == 0:
        return 0.0
    return float(x.corr(y))


def get_return_correlation(
    symbol_a: str,
    symbol_b: str,
    *,
    lookback_days: int = 252,
    settings: Settings | None = None,
) -> float:
    cfg = settings or load_settings()
    end = datetime.now(timezone.utc).date()
    start = end - timedelta(days=max(90, lookback_days + 30))
    a = get_close_series(symbol_a, start.isoformat(), end.isoformat(), settings=cfg).pct_change().dropna()
    b = get_close_series(symbol_b, start.isoformat(), end.isoformat(), settings=cfg).pct_change().dropna()
    return round(_corr_series(a.tail(lookback_days), b.tail(lookback_days)), 4)


def get_upcoming_earnings_date(
    symbol: str,
    *,
    horizon_days: int = 14,
    settings: Settings | None = None,
) -> datetime | None:
    """Return next earnings date within horizon if known (EODHD)."""
    cfg = settings or load_settings()
    batch = get_upcoming_earnings_batch(
        [symbol], horizon_days=horizon_days, settings=cfg
    )
    return batch.get((symbol or "").strip().upper())
