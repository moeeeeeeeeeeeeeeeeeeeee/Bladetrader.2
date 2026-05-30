"""Ad-hoc: run the case4 forward prediction for an explicit ticker list.

Steps per symbol:
  1. Resolve company name via EODHD /search.
  2. (optional --fetch-news) Pull EODHD /news for [today-30d, today] into SQLite.
  3. Scan the just-fetched press releases for an actual earnings date.
     Fall back to --t YYYY-MM-DD (assumed BMO, 13:30 UTC) if nothing found.
  4. Build [T-7d, T] sentiment + spillover features (with the resolved company
     name so short / common-word tickers like HERE / TOUR don't false-positive).
  5. Run the enhanced predictor + swing levels heuristic.

Usage:
    py scripts/predict_for_tickers.py --fetch-news --t 2026-06-05 ABM GIII HERE HURC VIRC TOUR QSG
"""

from __future__ import annotations

import json
import re
import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
sys.path.insert(0, str(ROOT / "src"))

import httpx

from finhack.case4_features import (
    clear_doc_cache,
    compute_news_features,
    predict_enhanced,
    resolve_db_path,
)
from finhack.config import load_settings
from finhack.eodhd_news import (
    EODHD_NEWS_URL,
    ensure_document_schema,
    fetch_eodhd_news_all,
    insert_eodhd_article,
)
from finhack.market_data import (
    EODHD_BASE_URL,
    get_close_series,
    get_daily_volatility_pct,
    get_upcoming_earnings_batch,
)
from finhack.research.case4_trade_path import compute_swing_levels


# ---------- helpers ---------------------------------------------------------


_MONTHS = (
    "january february march april may june july august september october "
    "november december"
).split()
_MONTH_IDX = {m: i + 1 for i, m in enumerate(_MONTHS)}
_DATE_RE = re.compile(
    r"\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+(\d{1,2}),?\s+(\d{4})\b",
    re.IGNORECASE,
)
_CONTEXT_RE = re.compile(
    r"(conference call|earnings (?:call|release|report)|webcast|results conference|quarter(?:ly)? (?:financial )?results|fiscal (?:first|second|third|fourth) quarter)",
    re.IGNORECASE,
)


def search_company_name(api_key: str, symbol: str) -> str | None:
    try:
        r = httpx.get(
            f"{EODHD_BASE_URL}/search/{symbol}",
            params={"api_token": api_key, "fmt": "json"},
            timeout=20,
        )
        r.raise_for_status()
        rows = r.json()
        if not isinstance(rows, list):
            return None
        for row in rows:
            if not isinstance(row, dict):
                continue
            code = str(row.get("Code") or "").upper()
            ex = str(row.get("Exchange") or "").upper()
            if code == symbol.upper() and ex in ("US", "NASDAQ", "NYSE"):
                name = str(row.get("Name") or "").strip()
                return name or None
        # fallback: first row
        first = rows[0]
        if isinstance(first, dict):
            name = str(first.get("Name") or "").strip()
            return name or None
    except Exception:
        return None
    return None


def fetch_and_store_news(
    symbol: str, db_path: Path, api_key: str, days_back: int = 30
) -> int:
    """Pull EODHD news for symbol over [today-days_back, today] into SQLite. Returns inserted count."""
    today = datetime.now(timezone.utc).date()
    start = today - timedelta(days=days_back)
    try:
        articles = fetch_eodhd_news_all(
            symbol,
            start.isoformat(),
            today.isoformat(),
            api_key,
            max_articles=500,
            page_size=100,
        )
    except Exception as exc:
        print(f"  [news] {symbol}: fetch failed: {exc}", file=sys.stderr)
        return 0
    if not articles:
        return 0
    ensure_document_schema(db_path)
    inserted = 0
    conn = sqlite3.connect(str(db_path))
    try:
        for art in articles:
            if insert_eodhd_article(
                conn, art, symbol=symbol, query=f"eodhd:{symbol}:{start}:{today}"
            ):
                inserted += 1
        conn.commit()
    finally:
        conn.close()
    return inserted


def extract_earnings_date(
    db_path: Path, symbol: str, name_lower: str
) -> tuple[datetime, str] | None:
    """Look for press-release language ("Conference Call ... on <date>") in recent news.

    Returns (T as UTC datetime at 13:30 BMO, source URL) or None.
    Only considers documents whose title/body mention the ticker (case-sensitive)
    or the resolved company name (case-insensitive, word-boundary).
    """
    sym = symbol.upper()
    sym_re = re.compile(rf"\b{re.escape(sym)}\b")
    name_re = re.compile(rf"\b{re.escape(name_lower)}\b") if name_lower else None
    today = datetime.now(timezone.utc).date()
    horizon = today + timedelta(days=60)

    if not db_path.exists():
        return None
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT title, body, url, source_domain, published_at
            FROM document
            ORDER BY COALESCE(published_at, fetched_at) DESC
            LIMIT 5000
            """
        ).fetchall()
    finally:
        conn.close()

    for row in rows:
        title = str(row["title"] or "")
        body = str(row["body"] or "")
        text = f"{title} {body}"
        if not sym_re.search(text) and (name_re is None or not name_re.search(text.lower())):
            continue
        if not _CONTEXT_RE.search(text):
            continue
        for m in _DATE_RE.finditer(text):
            month = _MONTH_IDX.get(m.group(1).lower())
            day = int(m.group(2))
            year = int(m.group(3))
            try:
                d = datetime(year, month, day, 13, 30, tzinfo=timezone.utc)
            except ValueError:
                continue
            if today <= d.date() <= horizon:
                return d, str(row["url"] or "")
    return None


def pre_7d_return_pct(symbol: str, settings) -> float | None:
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


# ---------- main ------------------------------------------------------------


def run(symbols: list[str], assumed_t: datetime | None, fetch_news: bool) -> dict:
    cfg = load_settings()
    api_key = (cfg.eodhd_api_key or "").strip()
    db_path = resolve_db_path(cfg)
    upcoming = get_upcoming_earnings_batch(symbols, horizon_days=30, settings=cfg)

    name_map: dict[str, str] = {}
    if api_key:
        for sym in symbols:
            nm = search_company_name(api_key, sym)
            if nm:
                name_map[sym.upper()] = nm

    news_stats: dict[str, int] = {}
    if fetch_news and api_key:
        print("[news] pulling EODHD per-symbol news (last 30d)...", flush=True)
        for sym in symbols:
            n = fetch_and_store_news(sym, db_path, api_key, days_back=30)
            news_stats[sym.upper()] = n
            print(f"  [news] {sym}: inserted {n}", flush=True)
        clear_doc_cache()

    now = datetime.now(timezone.utc)
    rows: list[dict] = []

    for sym in symbols:
        sym = sym.upper().strip()
        name = name_map.get(sym, "")
        name_lower = name.lower()

        t_event = upcoming.get(sym)
        t_source = "provider"
        t_url: str | None = None
        if t_event is None:
            extracted = extract_earnings_date(db_path, sym, name_lower)
            if extracted is not None:
                t_event, t_url = extracted
                t_source = "press_release"
        if t_event is None and assumed_t is not None:
            t_event = assumed_t
            t_source = "user_assumed"

        pre_ret = pre_7d_return_pct(sym, cfg)

        if t_event is None:
            rows.append({
                "symbol": sym,
                "company": name or None,
                "error": "no earnings date from provider, press release, or --t",
            })
            continue
        if pre_ret is None:
            rows.append({
                "symbol": sym,
                "company": name or None,
                "earnings_utc": t_event.isoformat(),
                "earnings_source": t_source,
                "error": "insufficient recent close data (delisted/illiquid?)",
            })
            continue

        news = compute_news_features(
            sym,
            t_event,
            db_path=db_path,
            settings=cfg,
            name_overrides={sym: name} if name else None,
        )
        sign, direction, confidence = predict_enhanced(pre_ret, news)

        close = get_close_series(
            sym,
            (now - timedelta(days=5)).date().isoformat(),
            now.date().isoformat(),
            settings=cfg,
        )
        spot = float(close.iloc[-1]) if not close.empty else None

        try:
            vol = get_daily_volatility_pct(sym, settings=cfg)
        except Exception:
            vol = None

        levels = None
        if spot is not None and sign != 0:
            lv = compute_swing_levels(
                entry=spot,
                signal=sign,
                predicted_move_pct=pre_ret,
                vol_pct=vol,
                symbol=sym,
            )
            if lv:
                levels = {
                    "direction": lv.direction,
                    "entry": lv.entry_price,
                    "stop": lv.stop_price,
                    "target": lv.target_price,
                    "stop_pct": lv.stop_pct,
                    "target_pct": lv.target_pct,
                    "suggested_hedge": lv.suggested_hedge,
                }

        days_out = max(0, (t_event.date() - now.date()).days)
        rows.append({
            "symbol": sym,
            "company": name or None,
            "earnings_utc": t_event.isoformat(),
            "earnings_source": t_source,
            "earnings_source_url": t_url,
            "days_to_earnings": days_out,
            "spot": spot,
            "pre_7d_return_pct": round(pre_ret, 3),
            "daily_vol_pct": round(vol, 3) if vol else None,
            "direction": direction,
            "signal": sign,
            "confidence": confidence,
            "news": {
                "direct_mentions_7d": news["enhanced_direct_mentions"],
                "spillover_mentions_7d": news["enhanced_spillover_mentions"],
                "mean_sentiment_score": round(float(news["sent_mean_score"]), 3),
                "spillover_weighted_score": float(news["spillover_weighted_score_7d"]),
                "exposure_score": float(news["enhanced_exposure_score"]),
                "top_drivers": news["top_drivers"],
            },
            "swing_levels": levels,
        })

    return {
        "generated_at_utc": now.isoformat(),
        "news_pulled_per_symbol": news_stats or None,
        "results": rows,
    }


def main() -> None:
    args = sys.argv[1:]
    assumed: datetime | None = None
    fetch_news = False
    syms: list[str] = []
    i = 0
    while i < len(args):
        tok = args[i]
        if tok in ("--t", "-t") and i + 1 < len(args):
            try:
                d = datetime.strptime(args[i + 1], "%Y-%m-%d")
            except ValueError:
                print(f"Bad --t value (expected YYYY-MM-DD): {args[i + 1]}", file=sys.stderr)
                sys.exit(2)
            assumed = d.replace(hour=13, minute=30, tzinfo=timezone.utc)
            i += 2
            continue
        if tok == "--fetch-news":
            fetch_news = True
            i += 1
            continue
        syms.append(tok)
        i += 1
    if not syms:
        print("Pass one or more ticker symbols.", file=sys.stderr)
        sys.exit(2)
    print(json.dumps(run(syms, assumed, fetch_news), indent=2))


if __name__ == "__main__":
    main()
