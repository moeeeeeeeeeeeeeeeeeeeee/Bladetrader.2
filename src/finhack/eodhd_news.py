"""EODHD news fetch + SQLite storage for live ingest and Case 4 validation."""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from finhack.case4_features import resolve_db_path
from finhack.config import Settings, load_settings
from finhack.market_data import EODHD_BASE_URL, _get_http_client, _symbol_to_eodhd
from finhack.text_encoder import lexicon_score

logger = logging.getLogger(__name__)

EODHD_NEWS_URL = f"{EODHD_BASE_URL}/news"


def _extract_domain(url: str) -> str:
    host = (urlparse(url).netloc or "").lower()
    return host[4:] if host.startswith("www.") else host


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ensure_document_schema(db_path: Path | str) -> None:
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS document (
                doc_id TEXT PRIMARY KEY,
                url TEXT NOT NULL UNIQUE,
                published_at TEXT,
                fetched_at TEXT NOT NULL,
                source TEXT NOT NULL,
                source_url TEXT,
                source_domain TEXT NOT NULL,
                title TEXT,
                body TEXT,
                keyword_hits TEXT NOT NULL,
                relevance_score REAL NOT NULL,
                query TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_document_published_at
            ON document (published_at)
            """
        )
        conn.commit()
    finally:
        conn.close()


def fetch_eodhd_news(
    symbol: str,
    from_date: str,
    to_date: str,
    api_key: str,
    *,
    limit: int = 100,
    offset: int = 0,
) -> list[dict[str, Any]]:
    params: dict[str, Any] = {
        "api_token": api_key,
        "fmt": "json",
        "s": _symbol_to_eodhd(symbol),
        "from": from_date,
        "to": to_date,
        "limit": max(1, min(limit, 1000)),
        "offset": max(0, offset),
    }
    client = _get_http_client()
    res = client.get(EODHD_NEWS_URL, params=params)
    res.raise_for_status()
    payload = res.json()
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    return []


def fetch_eodhd_news_all(
    symbol: str,
    from_date: str,
    to_date: str,
    api_key: str,
    *,
    max_articles: int = 500,
    page_size: int = 100,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    offset = 0
    while len(out) < max_articles:
        page = fetch_eodhd_news(
            symbol,
            from_date,
            to_date,
            api_key,
            limit=page_size,
            offset=offset,
        )
        if not page:
            break
        out.extend(page)
        if len(page) < page_size:
            break
        offset += page_size
    return out[:max_articles]


def fetch_eodhd_live_for_symbols(
    symbols: list[str],
    from_dt: datetime,
    to_dt: datetime,
    api_key: str,
    *,
    max_articles_per_symbol: int = 40,
) -> list[tuple[str, dict[str, Any]]]:
    """Fetch recent ticker news for many symbols. Returns (symbol, article) pairs."""
    from_date = from_dt.astimezone(timezone.utc).date().isoformat()
    to_date = to_dt.astimezone(timezone.utc).date().isoformat()
    clean = sorted({(s or "").strip().upper() for s in symbols if (s or "").strip()})
    out: list[tuple[str, dict[str, Any]]] = []
    for symbol in clean:
        try:
            articles = fetch_eodhd_news_all(
                symbol,
                from_date,
                to_date,
                api_key,
                max_articles=max_articles_per_symbol,
                page_size=min(100, max_articles_per_symbol),
            )
        except Exception as exc:
            logger.warning("EODHD live news failed for %s: %s", symbol, exc)
            continue
        for article in articles:
            out.append((symbol, article))
    return out


def _article_relevance(article: dict[str, Any], title: str, body: str) -> float:
    sentiment = article.get("sentiment")
    polarity = 0.0
    if isinstance(sentiment, dict):
        try:
            polarity = float(sentiment.get("polarity") or 0.0)
        except (TypeError, ValueError):
            polarity = 0.0
    return round(5.0 + abs(polarity) * 5.0 + max(0.0, lexicon_score(f"{title} {body}")), 2)


def _article_keyword_hits(article: dict[str, Any]) -> list[str]:
    hits: list[str] = []
    for key in ("tags", "symbols"):
        raw = article.get(key)
        if isinstance(raw, list):
            hits.extend(str(x).strip() for x in raw if str(x).strip())
    return sorted(set(hits))


def insert_eodhd_article(
    conn: sqlite3.Connection,
    article: dict[str, Any],
    *,
    symbol: str,
    query: str,
) -> bool:
    url = str(article.get("link") or "").strip()
    if not url:
        return False
    title = str(article.get("title") or "").strip()
    content = str(article.get("content") or "").strip()
    body = content or title
    if not title and not body:
        return False

    doc_id = hashlib.sha256(url.encode("utf-8")).hexdigest()[:24]
    published_at = str(article.get("date") or "").strip() or None
    source = str(article.get("source") or "eodhd").strip() or "eodhd"
    cur = conn.execute(
        """
        INSERT OR IGNORE INTO document (
            doc_id, url, published_at, fetched_at, source, source_url, source_domain,
            title, body, keyword_hits, relevance_score, query
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            doc_id,
            url,
            published_at,
            _utc_now_iso(),
            source,
            url,
            _extract_domain(url),
            title,
            body,
            json.dumps(_article_keyword_hits(article)),
            _article_relevance(article, title, body),
            query,
        ),
    )
    return cur.rowcount > 0


def _symbol_news_window(
    events: list[datetime],
    *,
    pre_days: int = 7,
) -> tuple[str, str] | None:
    if not events:
        return None
    start = min(events) - timedelta(days=pre_days)
    end = max(events)
    return start.date().isoformat(), end.date().isoformat()


def backfill_eodhd_news_for_earnings(
    earnings_by_symbol: dict[str, list[datetime]],
    *,
    settings: Settings | None = None,
    db_path: Path | None = None,
    max_articles_per_symbol: int = 500,
) -> dict[str, Any]:
    """Fetch EODHD news for earnings windows and store in SQLite."""
    cfg = settings or load_settings()
    api_key = (cfg.eodhd_api_key or "").strip()
    if not api_key:
        return {
            "ok": False,
            "reason": "missing_eodhd_api_key",
            "symbols_processed": 0,
            "articles_fetched": 0,
            "articles_inserted": 0,
            "articles_skipped": 0,
            "symbol_windows": [],
        }

    db = db_path or resolve_db_path(cfg)
    db.parent.mkdir(parents=True, exist_ok=True)
    ensure_document_schema(db)

    fetched_total = 0
    inserted_total = 0
    skipped_total = 0
    symbols_processed = 0
    symbol_windows: list[dict[str, Any]] = []

    conn = sqlite3.connect(str(db))
    try:
        for symbol, events in sorted(earnings_by_symbol.items()):
            clean = (symbol or "").strip().upper()
            if not clean or not events:
                continue
            window = _symbol_news_window(events)
            if window is None:
                continue
            from_date, to_date = window
            try:
                articles = fetch_eodhd_news_all(
                    clean,
                    from_date,
                    to_date,
                    api_key,
                    max_articles=max_articles_per_symbol,
                )
            except Exception as exc:
                logger.warning("EODHD news fetch failed for %s: %s", clean, exc)
                symbol_windows.append(
                    {
                        "symbol": clean,
                        "from": from_date,
                        "to": to_date,
                        "events": len(events),
                        "fetched": 0,
                        "inserted": 0,
                        "error": str(exc),
                    }
                )
                continue

            inserted = 0
            for article in articles:
                query = f"eodhd:{clean}:{from_date}:{to_date}"
                if insert_eodhd_article(conn, article, symbol=clean, query=query):
                    inserted += 1
            conn.commit()

            fetched_total += len(articles)
            inserted_total += inserted
            skipped_total += max(0, len(articles) - inserted)
            symbols_processed += 1
            symbol_windows.append(
                {
                    "symbol": clean,
                    "from": from_date,
                    "to": to_date,
                    "events": len(events),
                    "fetched": len(articles),
                    "inserted": inserted,
                }
            )
    finally:
        conn.close()

    return {
        "ok": True,
        "reason": None,
        "symbols_processed": symbols_processed,
        "articles_fetched": fetched_total,
        "articles_inserted": inserted_total,
        "articles_skipped": skipped_total,
        "symbol_windows": symbol_windows,
    }
