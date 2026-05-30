"""
Populate SQLite news store from all configured sources.

  py -3 scripts/populate_news.py
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

load_dotenv(ROOT / ".env")

from finhack.agents.news_intake_agent import NewsIntakeAgent
from finhack.case4_features import resolve_db_path
from finhack.config import load_settings
from finhack.eodhd_news import backfill_eodhd_news_for_earnings
from finhack.market_data import get_earnings_events_batch
from finhack.news_populate import document_stats, ingest_eodhd_symbol_news, universe_symbols_for_news
from finhack.research.constants import EARNINGS_BATCH_CHUNK, EARNINGS_LIMIT_PER_SYMBOL, EARNINGS_RECENT_DAYS


def main() -> None:
    parser = argparse.ArgumentParser(description="Populate news documents for sentiment features")
    parser.add_argument("--days-back", type=int, default=None)
    parser.add_argument("--skip-backfill", action="store_true")
    parser.add_argument("--skip-earnings-eodhd", action="store_true")
    parser.add_argument("--eodhd-symbol-limit", type=int, default=200)
    parser.add_argument("--universe-limit", type=int, default=None)
    args = parser.parse_args()

    settings = load_settings()
    db_path = resolve_db_path(settings)
    before = document_stats(db_path)

    has_eodhd = bool((settings.eodhd_api_key or "").strip())
    has_gnews = bool((settings.gnews_api_key or "").strip())
    print(f"API keys: EODHD={'yes' if has_eodhd else 'no'} GNEWS={'yes' if has_gnews else 'no'}")
    print(f"Documents before: {before}")

    news = NewsIntakeAgent()
    backfill_days = args.days_back if args.days_back is not None else settings.news_backfill_days
    symbols = universe_symbols_for_news(settings=settings, limit=args.universe_limit)

    if args.skip_backfill:
        backfill = news.run_ingest(
            max_queries=14,
            max_per_query=30,
            hours_back=24 * 30,
            require_gnews=False,
            require_primary_api=False,
            enable_gdelt=True,
            enable_rss_fallback=True,
        )
    else:
        print(f"Historical backfill ({backfill_days} days)...", flush=True)
        try:
            backfill = news.run_historical_backfill(
                days_back=backfill_days,
                chunk_days=14,
                max_queries=14,
                max_per_query=40,
                max_pages=2,
                require_gnews=False,
                require_primary_api=False,
                enable_gdelt=True,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"Backfill error: {exc}", flush=True)
            backfill = news.run_ingest(
                max_queries=10,
                max_per_query=25,
                hours_back=24 * 60,
                require_gnews=False,
                require_primary_api=False,
                enable_gdelt=True,
                enable_rss_fallback=True,
            )

    print(f"Backfill inserted={backfill.inserted_documents}")

    ingest = news.run_ingest(
        max_queries=14,
        max_per_query=30,
        hours_back=24 * 30,
        require_gnews=False,
        require_primary_api=False,
        enable_gdelt=True,
        enable_rss_fallback=True,
        enable_eodhd=True,
    )
    print(f"Top-up inserted={ingest.inserted_documents}")

    eodhd_symbol_stats: dict[str, int] = {}
    if has_eodhd:
        cap = max(10, min(args.eodhd_symbol_limit, len(symbols)))
        eodhd_symbol_stats = ingest_eodhd_symbol_news(
            symbols[:cap],
            days_back=min(365, backfill_days),
            api_key=settings.eodhd_api_key or "",
            db_path=db_path,
        )
        print(f"EODHD symbol news: {eodhd_symbol_stats}")

    earnings_news = None
    if has_eodhd and not args.skip_earnings_eodhd:
        earnings_by_symbol = get_earnings_events_batch(
            symbols,
            limit=EARNINGS_LIMIT_PER_SYMBOL,
            recent_days=EARNINGS_RECENT_DAYS,
            settings=settings,
            chunk_size=EARNINGS_BATCH_CHUNK,
        )
        earnings_news = backfill_eodhd_news_for_earnings(
            earnings_by_symbol,
            settings=settings,
            db_path=db_path,
        )
        print(f"Earnings-window news inserted={earnings_news.get('articles_inserted')}")

    after = document_stats(db_path)
    summary = {
        "documents_before": before,
        "documents_after": after,
        "backfill": asdict(backfill),
        "ingest": asdict(ingest),
        "eodhd_symbol_news": eodhd_symbol_stats,
        "eodhd_earnings_news": earnings_news,
    }
    out_path = ROOT / "data" / "news_populate_summary.json"
    out_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"Documents after: {after}")
    print(f"Wrote {out_path.as_posix()}")


if __name__ == "__main__":
    main()
