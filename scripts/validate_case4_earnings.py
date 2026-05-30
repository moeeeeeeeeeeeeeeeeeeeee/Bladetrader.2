"""
Case 4 earnings validation — live data only.

- Universe: core mapped stocks + EODHD US common-stock catalog (see TRADING_UNIVERSE_LIMIT)
- Ingest/backfill news via GNews, GDELT, RSS, and EODHD
- Baseline + sentiment/spillover features from [T-7d, T]
- Label: 5-trading-day post-earnings return (+ optional stop/target path)
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

load_dotenv(ROOT / ".env")

from finhack.agents.news_intake_agent import NewsIntakeAgent
from finhack.case4_features import apply_enhanced_fields, sign_from_return
from finhack.config import MarketDataProvider, load_settings
from finhack.data.trading_universe import resolve_trading_universe
from finhack.eodhd_news import backfill_eodhd_news_for_earnings
from finhack.market_data import get_close_series, get_earnings_events_batch
from finhack.research.case4_trade_path import enrich_event_with_trade_path
from finhack.research.constants import (
    EARNINGS_BATCH_CHUNK,
    EARNINGS_LIMIT_PER_SYMBOL,
    EARNINGS_RECENT_DAYS,
    NEWS_INGEST_HOURS_BACK,
)
from finhack.research.market_features import enrich_events_in_place
from finhack.text_encoder import (
    FINBERT_MODEL,
    ensure_finbert_scores,
    ensure_lexicon_scores,
)


def compute_return_pct(series, i0: int, i1: int) -> float | None:
    if i0 < 0 or i1 < 0 or i0 >= len(series) or i1 >= len(series):
        return None
    p0 = float(series.iloc[i0])
    p1 = float(series.iloc[i1])
    if p0 == 0:
        return None
    return ((p1 - p0) / p0) * 100.0


def _evaluate_event_with_close(
    symbol: str,
    t_event: datetime,
    close: Any,
    idx: list[Any],
    *,
    price_source: str,
) -> dict[str, Any] | None:
    event_i = None
    for i, ts in enumerate(idx):
        if ts.to_pydatetime().date() >= t_event.date():
            event_i = i
            break
    if event_i is None:
        return None

    pre_start_i = max(0, event_i - 7)
    pre_end_i = max(0, event_i - 1)
    pre_ret = compute_return_pct(close, pre_start_i, pre_end_i)
    if pre_ret is None:
        return None
    baseline_pred = sign_from_return(pre_ret, dead_zone=0.10)

    post_i = event_i + 5
    post_ret = compute_return_pct(close, event_i, post_i)
    if post_ret is None:
        return None
    actual = sign_from_return(post_ret)

    return {
        "symbol": symbol,
        "t_event_utc": t_event.isoformat(),
        "actual_5d_return_pct": round(post_ret, 4),
        "actual_sign": actual,
        "baseline_pre_7d_return_pct": round(pre_ret, 4),
        "baseline_pred_sign": baseline_pred,
        "price_source": price_source,
    }


def evaluate_events_for_symbol(
    symbol: str,
    events: list[datetime],
    *,
    settings=None,
    price_source: str = "unknown",
    db_path: Path | None = None,
    include_trade_path: bool = True,
) -> list[dict[str, Any]]:
    if not events:
        return []
    cfg = settings or load_settings()
    sorted_events = sorted(events)
    # 120d back covers pre_60d_return + safety; 20d forward covers post-event 5d.
    start = (sorted_events[0] - timedelta(days=120)).date().isoformat()
    end = (sorted_events[-1] + timedelta(days=20)).date().isoformat()
    close = get_close_series(symbol, start=start, end=end, settings=cfg)
    if close.empty:
        return []
    idx = list(close.index)

    out: list[dict[str, Any]] = []
    for t_event in sorted_events:
        row = _evaluate_event_with_close(
            symbol, t_event, close, idx, price_source=price_source
        )
        if not row:
            continue
        enriched = apply_enhanced_fields(row, db_path=db_path, settings=cfg)
        if include_trade_path:
            enriched = enrich_event_with_trade_path(enriched, settings=cfg)
        out.append(enriched)
    return out


def accuracy(rows: list[dict[str, Any]], pred_key: str) -> tuple[int, int, float | None]:
    total = 0
    correct = 0
    for row in rows:
        pred = int(row.get(pred_key, 0))
        actual = int(row.get("actual_sign", 0))
        if pred == 0 or actual == 0:
            continue
        total += 1
        if pred == actual:
            correct += 1
    if total == 0:
        return correct, total, None
    return correct, total, correct / total


def _resolve_db_path(database_url: str) -> Path:
    raw = database_url.replace("sqlite:///", "", 1) if database_url.startswith("sqlite:///") else database_url
    p = Path(raw)
    return p if p.is_absolute() else Path.cwd() / p


def main() -> None:
    parser = argparse.ArgumentParser(description="Live earnings validator (full universe)")
    parser.add_argument("--skip-eodhd-news", action="store_true")
    parser.add_argument("--skip-news-backfill", action="store_true", help="Skip long historical news backfill")
    parser.add_argument("--skip-news-ingest", action="store_true", help="Skip all news ingest (use existing SQLite docs)")
    parser.add_argument("--skip-trade-path", action="store_true", help="Skip stop/target path simulation")
    parser.add_argument("--recent-days", type=int, default=EARNINGS_RECENT_DAYS)
    parser.add_argument("--earnings-limit", type=int, default=EARNINGS_LIMIT_PER_SYMBOL)
    parser.add_argument(
        "--universe-limit",
        type=int,
        default=None,
        help="Max symbols (default: TRADING_UNIVERSE_LIMIT from .env, usually 500)",
    )
    args = parser.parse_args()

    settings = load_settings()
    news = NewsIntakeAgent()
    db_path = _resolve_db_path(settings.database_url)
    universe = resolve_trading_universe(settings=settings, limit=args.universe_limit)
    symbols = [u.symbol for u in universe]

    if not (settings.eodhd_api_key or "").strip():
        print(
            "Warning: EODHD_API_KEY not set — earnings/prices unavailable; "
            "set the key in .env to fetch market data.",
            flush=True,
        )
    if not (settings.gnews_api_key or "").strip():
        print("Warning: GNEWS_API_KEY not set — using EODHD/GDELT/RSS for news.", flush=True)

    print(f"Universe: {len(universe)} symbols (core + EODHD catalog).", flush=True)

    backfill_warning: str | None = None
    backfill_days = settings.news_backfill_days
    query_budget = min(20, max(10, len(symbols) // 25))

    if args.skip_news_ingest:
        print("Skipping all news ingest (--skip-news-ingest); using existing SQLite documents.", flush=True)
        from finhack.agents.news_intake_agent import NewsIngestResult, _empty_source_counts

        empty = NewsIngestResult(
            queries_used=[],
            fetched_articles=0,
            inserted_documents=0,
            skipped_documents=0,
            documents=[],
            transport="skipped",
            primary_api_enforced=False,
            source_counts=_empty_source_counts(),
        )
        backfill = empty
        ingest = empty
    elif args.skip_news_backfill:
        print("Skipping historical news backfill (--skip-news-backfill).", flush=True)
        backfill = news.run_ingest(
            max_queries=min(8, query_budget),
            max_per_query=15,
            hours_back=24 * 14,
            trusted_sources_only=False,
            require_gnews=False,
            require_primary_api=False,
            enable_gdelt=True,
            enable_rss_fallback=True,
        )
    else:
        print(f"Ingesting news history ({backfill_days} days back)...", flush=True)
        try:
            backfill = news.run_historical_backfill(
                days_back=backfill_days,
                chunk_days=21,
                max_queries=query_budget,
                max_per_query=50,
                max_pages=2,
                trusted_sources_only=False,
                require_gnews=False,
                require_primary_api=True,
                enable_gdelt=True,
            )
        except Exception as exc:  # noqa: BLE001
            backfill_warning = str(exc)
            backfill = news.run_ingest(
                max_queries=query_budget,
                max_per_query=25,
                hours_back=min(NEWS_INGEST_HOURS_BACK, 24 * 90),
                trusted_sources_only=False,
                require_gnews=False,
                require_primary_api=False,
                enable_gdelt=True,
                enable_rss_fallback=True,
            )

    print(
        f"Backfill: inserted={backfill.inserted_documents} transport={backfill.transport}",
        flush=True,
    )

    if args.skip_news_ingest or args.skip_news_backfill:
        ingest = backfill
    else:
        ingest = news.run_ingest(
            max_queries=min(8, query_budget),
            max_per_query=15,
            hours_back=24 * 14,
            trusted_sources_only=False,
            require_gnews=False,
            require_primary_api=bool((settings.eodhd_api_key or settings.gnews_api_key or "").strip()),
            enable_gdelt=True,
            enable_rss_fallback=True,
        )
        print(f"Top-up ingest: inserted={ingest.inserted_documents}", flush=True)

    price_source = settings.market_data_provider.value
    if (
        settings.market_data_provider == MarketDataProvider.EODHD
        and not (settings.eodhd_api_key or "").strip()
    ):
        price_source = "eodhd-missing-key"

    print("Fetching earnings calendar (batched)...", flush=True)
    earnings_by_symbol = get_earnings_events_batch(
        symbols,
        limit=args.earnings_limit,
        recent_days=args.recent_days,
        settings=settings,
        chunk_size=EARNINGS_BATCH_CHUNK,
    )
    total_events = sum(len(v) for v in earnings_by_symbol.values())
    symbols_with_events = sum(1 for v in earnings_by_symbol.values() if v)
    print(
        f"Earnings: {total_events} events across {symbols_with_events} symbols ({args.recent_days}d).",
        flush=True,
    )

    eodhd_news_backfill: dict[str, Any] | None = None
    if not args.skip_eodhd_news and (settings.eodhd_api_key or "").strip():
        print("Backfilling EODHD news for earnings windows...", flush=True)
        eodhd_news_backfill = backfill_eodhd_news_for_earnings(
            earnings_by_symbol,
            settings=settings,
            db_path=db_path,
        )
    elif args.skip_eodhd_news:
        eodhd_news_backfill = {"ok": False, "reason": "skipped_by_flag"}
    else:
        eodhd_news_backfill = {"ok": False, "reason": "missing_eodhd_api_key"}

    # Top up document_score cache (lexicon always, FinBERT if available).
    encoder_summary: dict[str, Any] = {"db_path": str(db_path.as_posix())}
    try:
        lex_added = ensure_lexicon_scores(db_path)
        finbert_added, finbert_model_used = ensure_finbert_scores(
            db_path, progress=True
        )
        encoder_summary.update(
            {
                "lexicon_rows_added": lex_added,
                "finbert_rows_added": finbert_added,
                "finbert_active": finbert_model_used == FINBERT_MODEL,
                "finbert_model_used": finbert_model_used,
            }
        )
        print(
            f"Encoder cache: lexicon +{lex_added}, "
            f"finbert +{finbert_added} ({finbert_model_used}).",
            flush=True,
        )
    except Exception as exc:  # noqa: BLE001
        encoder_summary.update({"error": str(exc)})
        print(f"Encoder scoring step failed: {exc}", flush=True)

    include_path = not args.skip_trade_path and len(universe) <= 150
    if not include_path and not args.skip_trade_path:
        print("Skipping trade-path simulation (universe > 150). Use --skip-trade-path to silence.", flush=True)

    rows: list[dict[str, Any]] = []
    per_symbol_count: dict[str, int] = {}
    eval_targets = [u for u in universe if earnings_by_symbol.get(u.symbol)]

    for idx, entry in enumerate(eval_targets, start=1):
        symbol = entry.symbol
        events = earnings_by_symbol.get(symbol, [])
        if not events:
            continue
        if idx % 25 == 0 or idx == len(eval_targets):
            print(f"  Progress {idx}/{len(eval_targets)} symbols...", flush=True)
        for row in evaluate_events_for_symbol(
            symbol,
            events,
            settings=settings,
            price_source=price_source,
            db_path=db_path,
            include_trade_path=include_path,
        ):
            rows.append(row)
            per_symbol_count[symbol] = per_symbol_count.get(symbol, 0) + 1

    print("Enriching events with market-derived features...", flush=True)
    enrichment_summary = enrich_events_in_place(
        rows,
        earnings_by_symbol=earnings_by_symbol,
        settings=settings,
        progress=True,
    )
    print(
        f"Enriched {enrichment_summary.get('events_enriched', 0)} events "
        f"with cohort + drift features.",
        flush=True,
    )

    baseline_correct, baseline_total, baseline_acc = accuracy(rows, "baseline_pred_sign")
    enhanced_correct, enhanced_total, enhanced_acc = accuracy(rows, "enhanced_pred_sign")

    enhanced_rows = [r for r in rows if r.get("enhanced_documents_considered", 0) > 0]
    enhanced_cov = len(enhanced_rows) / len(rows) if rows else None
    spillover_rows = [r for r in enhanced_rows if r.get("enhanced_spillover_mentions", 0) > 0]
    spillover_cov = len(spillover_rows) / len(enhanced_rows) if enhanced_rows else None
    finbert_rows = [r for r in rows if r.get("sent_finbert_doc_count", 0) > 0]
    finbert_cov = len(finbert_rows) / len(rows) if rows else None

    per_sym_coverage: dict[str, dict[str, int]] = {}
    for r in rows:
        sym = str(r.get("symbol", "")).upper()
        bucket = per_sym_coverage.setdefault(
            sym, {"events": 0, "events_with_news": 0}
        )
        bucket["events"] += 1
        if int(r.get("sent_doc_count", 0) or 0) > 0:
            bucket["events_with_news"] += 1
    low_cov: list[dict[str, Any]] = []
    for sym, bucket in per_sym_coverage.items():
        n = bucket["events"]
        with_news = bucket["events_with_news"]
        if n >= 4 and with_news / n < 0.25:
            low_cov.append(
                {
                    "symbol": sym,
                    "events": n,
                    "events_with_news": with_news,
                    "coverage_pct": round(with_news / n, 4),
                }
            )
    low_cov.sort(key=lambda r: (r["coverage_pct"], -r["events"]))

    path_rows = [r for r in rows if r.get("path_return_pct") is not None]
    path_wins = sum(1 for r in path_rows if r.get("path_won"))
    path_hit = path_wins / len(path_rows) if path_rows else None
    path_metrics = {
        "events_with_path": len(path_rows),
        "path_win_rate": round(path_hit, 4) if path_hit is not None else None,
        "stop_hit_rate": round(sum(1 for r in path_rows if r.get("path_stop_hit")) / len(path_rows), 4)
        if path_rows
        else None,
        "target_hit_rate": round(sum(1 for r in path_rows if r.get("path_target_hit")) / len(path_rows), 4)
        if path_rows
        else None,
        "direction_accuracy_with_stops": round(
            sum(1 for r in path_rows if r.get("direction_correct")) / len(path_rows), 4
        )
        if path_rows
        else None,
        "mean_path_return_pct": round(
            sum(float(r["path_return_pct"]) for r in path_rows) / len(path_rows), 4
        )
        if path_rows
        else None,
    }

    summary = {
        "run_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "live",
        "market_data_provider": settings.market_data_provider.value,
        "price_source": price_source,
        "eodhd_news_backfill": eodhd_news_backfill,
        "encoder": encoder_summary,
        "market_features": enrichment_summary,
        "backfill": asdict(backfill),
        "backfill_warning": backfill_warning,
        "ingest": asdict(ingest),
        "history_days": args.recent_days,
        "earnings_limit_per_symbol": args.earnings_limit,
        "trading_universe_limit": args.universe_limit or settings.trading_universe_limit,
        "stock_universe_size": len(universe),
        "symbols_evaluated": len(eval_targets),
        "earnings_events_evaluated": len(rows),
        "symbols_with_events": len(per_symbol_count),
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
        "uplift_vs_baseline_pp": round((enhanced_acc - baseline_acc) * 100.0, 2)
        if enhanced_acc is not None and baseline_acc is not None
        else None,
        "enhanced_feature_coverage": round(enhanced_cov, 4) if enhanced_cov is not None else None,
        "spillover_feature_coverage_within_enhanced": round(spillover_cov, 4)
        if spillover_cov is not None
        else None,
        "finbert_feature_coverage": round(finbert_cov, 4) if finbert_cov is not None else None,
        "low_news_coverage_symbols": low_cov[:25],
        "trade_path_metrics": path_metrics,
        "events": rows,
    }

    out_path = Path("data") / "case4_earnings_validation.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote: {out_path.as_posix()}")
    print(
        json.dumps(
            {
                "universe": summary["stock_universe_size"],
                "symbols_evaluated": summary["symbols_evaluated"],
                "events": summary["earnings_events_evaluated"],
                "baseline_accuracy": summary["baseline"]["accuracy"],
                "enhanced_accuracy": summary["enhanced"]["accuracy"],
                "enhanced_feature_coverage": summary["enhanced_feature_coverage"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
