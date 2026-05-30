"""Shared news population helpers for scripts and API."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

from finhack.case4_features import resolve_db_path
from finhack.config import Settings, load_settings
from finhack.data.trading_universe import trading_symbols
from finhack.eodhd_news import ensure_document_schema, fetch_eodhd_live_for_symbols, insert_eodhd_article


def document_stats(db_path: Path | None = None) -> dict[str, int]:
    db = db_path or resolve_db_path(load_settings())
    if not db.exists():
        return {"total": 0, "last_7_days": 0}
    conn = sqlite3.connect(str(db))
    try:
        total = conn.execute("SELECT COUNT(*) FROM document").fetchone()[0]
        recent = conn.execute(
            """
            SELECT COUNT(*) FROM document
            WHERE COALESCE(published_at, fetched_at) >= datetime('now', '-7 days')
            """
        ).fetchone()[0]
        return {"total": int(total), "last_7_days": int(recent)}
    finally:
        conn.close()


def ingest_eodhd_symbol_news(
    symbols: list[str],
    *,
    days_back: int,
    api_key: str,
    db_path: Path | None = None,
) -> dict[str, int]:
    db = db_path or resolve_db_path(load_settings())
    ensure_document_schema(db)
    now = datetime.now(timezone.utc)
    from_dt = now - timedelta(days=max(7, days_back))
    pairs = fetch_eodhd_live_for_symbols(
        symbols,
        from_dt,
        now,
        api_key,
        max_articles_per_symbol=60,
    )
    conn = sqlite3.connect(str(db))
    inserted = 0
    try:
        for symbol, article in pairs:
            query = f"populate:eodhd:{symbol}"
            if insert_eodhd_article(conn, article, symbol=symbol, query=query):
                inserted += 1
        conn.commit()
    finally:
        conn.close()
    return {"fetched": len(pairs), "inserted": inserted}


def universe_symbols_for_news(settings: Settings | None = None, limit: int | None = None) -> list[str]:
    return trading_symbols(settings=settings, limit=limit)
