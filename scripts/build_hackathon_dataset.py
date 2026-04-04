"""
Build a reusable local Case 4 dataset snapshot for hackathon iteration.

Goal:
- Pull as much AI-market news as practical (with graceful fallbacks)
- Persist to SQLite via Agent 1
- Export a stable JSONL snapshot so repeated model tests need no API calls
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from finhack.agents.news_intake_agent import NewsIngestResult, NewsIntakeAgent
from finhack.config import load_settings

load_dotenv()


PROFILES: dict[str, dict[str, int]] = {
    "quick": {
        "backfill_days": 60,
        "backfill_chunk_days": 30,
        "backfill_queries": 4,
        "backfill_per_query": 15,
        "backfill_pages": 1,
        "topup_hours": 24 * 30,
        "topup_queries": 4,
        "topup_per_query": 10,
    },
    "standard": {
        "backfill_days": 180,
        "backfill_chunk_days": 21,
        "backfill_queries": 8,
        "backfill_per_query": 20,
        "backfill_pages": 1,
        "topup_hours": 24 * 90,
        "topup_queries": 8,
        "topup_per_query": 15,
    },
    "deep": {
        "backfill_days": 365,
        "backfill_chunk_days": 14,
        "backfill_queries": 10,
        "backfill_per_query": 25,
        "backfill_pages": 2,
        "topup_hours": 24 * 180,
        "topup_queries": 10,
        "topup_per_query": 20,
    },
}


def _parse_dt(raw: str | None) -> datetime | None:
    if not raw:
        return None
    text = str(raw).strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        try:
            dt = parsedate_to_datetime(text)
        except Exception:  # noqa: BLE001
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _resolve_db_path(database_url: str) -> Path:
    raw = database_url
    if raw.startswith("sqlite:///"):
        raw = raw.replace("sqlite:///", "", 1)
    p = Path(raw)
    return p if p.is_absolute() else Path.cwd() / p


def _load_documents(db_path: Path) -> list[dict[str, Any]]:
    if not db_path.exists():
        return []
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT
                doc_id, url, published_at, fetched_at, source, source_url, source_domain,
                title, body, keyword_hits, relevance_score, query
            FROM document
            ORDER BY COALESCE(published_at, fetched_at) DESC
            """
        ).fetchall()
    finally:
        conn.close()
    docs: list[dict[str, Any]] = []
    for r in rows:
        hits_raw = r["keyword_hits"]
        try:
            hits = json.loads(hits_raw) if hits_raw else []
            if not isinstance(hits, list):
                hits = []
        except Exception:  # noqa: BLE001
            hits = []
        docs.append(
            {
                "doc_id": r["doc_id"],
                "url": r["url"],
                "published_at": r["published_at"],
                "fetched_at": r["fetched_at"],
                "source": r["source"],
                "source_url": r["source_url"],
                "source_domain": r["source_domain"],
                "title": r["title"] or "",
                "body": r["body"] or "",
                "keyword_hits": hits,
                "relevance_score": float(r["relevance_score"] or 0.0),
                "query": r["query"] or "",
            }
        )
    return docs


def _export_snapshot(docs: list[dict[str, Any]], out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_dir / "case4_dataset_snapshot.jsonl"
    summary_path = out_dir / "case4_dataset_summary.json"

    with jsonl_path.open("w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=True) + "\n")

    domains = Counter(d.get("source_domain", "") for d in docs if d.get("source_domain"))
    top_domains = domains.most_common(15)
    timestamps = [
        _parse_dt(d.get("published_at")) or _parse_dt(d.get("fetched_at")) for d in docs
    ]
    timestamps = [t for t in timestamps if t is not None]
    min_ts = min(timestamps).isoformat() if timestamps else None
    max_ts = max(timestamps).isoformat() if timestamps else None
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "document_count": len(docs),
        "date_range": {"min": min_ts, "max": max_ts},
        "top_source_domains": top_domains,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return jsonl_path, summary_path


def _safe_run_ingest(agent: NewsIntakeAgent, **kwargs: Any) -> tuple[NewsIngestResult | None, str | None]:
    try:
        return agent.run_ingest(**kwargs), None
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)


def _safe_run_backfill(
    agent: NewsIntakeAgent, **kwargs: Any
) -> tuple[NewsIngestResult | None, str | None]:
    try:
        return agent.run_historical_backfill(**kwargs), None
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build local hackathon dataset snapshot")
    parser.add_argument("--profile", choices=tuple(PROFILES.keys()), default="quick")
    parser.add_argument("--skip-ingest", action="store_true")
    parser.add_argument("--trusted-sources-only", action="store_true")
    args = parser.parse_args()

    cfg = PROFILES[args.profile]
    settings = load_settings()
    agent = NewsIntakeAgent(settings=settings)
    run_log: dict[str, Any] = {
        "profile": args.profile,
        "skip_ingest": args.skip_ingest,
        "trusted_sources_only": args.trusted_sources_only,
        "steps": [],
    }

    if not args.skip_ingest:
        backfill_result, backfill_error = _safe_run_backfill(
            agent,
            days_back=cfg["backfill_days"],
            chunk_days=cfg["backfill_chunk_days"],
            max_queries=cfg["backfill_queries"],
            max_per_query=cfg["backfill_per_query"],
            max_pages=cfg["backfill_pages"],
            trusted_sources_only=args.trusted_sources_only,
            require_gnews=False,
            require_primary_api=True,
            enable_gdelt=True,
        )
        run_log["steps"].append(
            {
                "name": "backfill",
                "ok": backfill_result is not None,
                "error": backfill_error,
                "result": asdict(backfill_result) if backfill_result else None,
            }
        )

        topup_result, topup_error = _safe_run_ingest(
            agent,
            max_queries=cfg["topup_queries"],
            max_per_query=cfg["topup_per_query"],
            hours_back=cfg["topup_hours"],
            trusted_sources_only=args.trusted_sources_only,
            require_gnews=False,
            require_primary_api=True,
            enable_gdelt=True,
            enable_rss_fallback=True,
        )
        run_log["steps"].append(
            {
                "name": "topup_ingest",
                "ok": topup_result is not None,
                "error": topup_error,
                "result": asdict(topup_result) if topup_result else None,
            }
        )

    db_path = _resolve_db_path(settings.database_url)
    docs = _load_documents(db_path)
    data_dir = Path("data")
    jsonl_path, summary_path = _export_snapshot(docs, data_dir)

    run_log["database_path"] = str(db_path)
    run_log["snapshot_jsonl"] = str(jsonl_path)
    run_log["snapshot_summary"] = str(summary_path)
    run_log["document_count"] = len(docs)
    run_log_path = data_dir / "case4_dataset_runlog.json"
    run_log_path.write_text(json.dumps(run_log, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "profile": args.profile,
                "document_count": len(docs),
                "snapshot_jsonl": str(jsonl_path),
                "snapshot_summary": str(summary_path),
                "runlog": str(run_log_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

