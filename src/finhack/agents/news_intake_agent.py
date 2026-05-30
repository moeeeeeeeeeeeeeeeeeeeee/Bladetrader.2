"""
Agent 1: ingest AI-market web events from GNews and store documents.

This module intentionally focuses on internet document ingestion only.
It does not fetch OHLC market data (see market data providers instead).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse
from xml.etree import ElementTree as ET

import httpx

from finhack.config import Settings, load_settings
from finhack.data.company_graph import SYMBOL_TO_COMPANY
from finhack.data.trading_universe import trading_symbols
from finhack.eodhd_news import _article_keyword_hits, _article_relevance, fetch_eodhd_live_for_symbols

GNEWS_SEARCH_URL = "https://gnews.io/api/v4/search"
GDELT_DOC_API_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

# Extracted from user-provided keyword doc and grouped for relevance checks.
AI_CORE_TERMS: tuple[str, ...] = (
    "artificial intelligence",
    "ai",
    "machine learning",
    "deep learning",
    "neural network",
    "large language model",
    "llm",
    "generative ai",
    "foundation model",
    "multimodal ai",
    "openai",
    "nvidia",
    "microsoft ai",
    "google ai",
    "deepmind",
    "anthropic",
    "meta ai",
    "xai",
    "ai startup",
    "ai partnership",
)

MARKET_IMPACT_TERMS: tuple[str, ...] = (
    "market",
    "markets",
    "stock market",
    "stocks",
    "stock",
    "shares",
    "equity market",
    "market reaction",
    "investor",
    "investors",
    "earnings",
    "revenue",
    "guidance",
    "forecast",
    "nasdaq",
    "s&p 500",
    "sp500",
    "valuation",
    "ai earnings",
    "ai guidance",
    "ai outlook",
    "ai growth",
    "ai boom",
    "ai slowdown",
    "ai disruption",
    "ai valuation",
    "ai spending",
    "data center",
    "hyperscaler",
    "cloud capacity",
    "compute capacity",
    "high performance computing",
    "hpc",
    "gpu demand",
    "semiconductor demand",
    "chip shortage",
    "ai accelerators",
    "tpu",
    "asic",
    "energy demand",
    "electricity usage",
    "grid capacity",
    "cloud spending",
    "cloud growth",
    "aws growth",
    "azure growth",
    "google cloud growth",
    "capital expenditure",
    "capex",
    "infrastructure investment",
    "data center investment",
    "server investment",
    "hardware spending",
    "technology investment",
    "supply chain disruption",
    "component shortages",
    "memory demand",
    "dram demand",
    "hbm memory",
    "regulatory framework",
    "ai regulation",
    "ai policy",
    "ai legislation",
    "government ai",
    "ai compliance",
    "ai governance",
    "workforce automation",
    "labor displacement",
    "future of work",
    "productivity gains",
    "economic transformation",
)

# Query expansion seeds so the agent discovers more candidate sources.
QUERY_SEEDS: tuple[str, ...] = (
    "artificial intelligence stock market impact",
    "generative ai earnings guidance",
    "ai regulation impact on technology stocks",
    "gpu demand semiconductor stocks",
    "hyperscaler ai capex spending",
    "data center expansion ai demand",
    "cloud infrastructure ai investment",
    "ai policy and market reaction",
    "ai startup funding public markets",
    "enterprise ai adoption revenue growth",
    "ai chips supply chain and stocks",
    "power demand from ai data centers",
    "hbm memory demand ai boom",
    "labor automation ai productivity market",
)

# Trusted domains extracted from the user-provided reliable-source document.
TRUSTED_SOURCE_DOMAINS: tuple[str, ...] = (
    "aws.amazon.com",
    "developer.nvidia.com",
    "blog.google",
    "openai.com",
    "huggingface.co",
    "finance.yahoo.com",
    "marketwatch.com",
    "cnbc.com",
    "seekingalpha.com",
    "benzinga.com",
    "sec.gov",
    "federalreserve.gov",
    "whitehouse.gov",
    "ec.europa.eu",
    "gov.uk",
    "arxiv.org",
    "technologyreview.com",
    "spectrum.ieee.org",
    "brookings.edu",
    "reuters.com",
    "apnews.com",
    "bbc.co.uk",
    "bbc.com",
    "aljazeera.com",
    "npr.org",
    "usatoday.com",
    "pbs.org",
    "theverge.com",
    "arstechnica.com",
    "techcrunch.com",
    "wired.com",
)

# Trusted RSS/Atom feeds (from user-provided reliable sources).
TRUSTED_RSS_FEEDS: tuple[str, ...] = (
    "https://aws.amazon.com/blogs/machine-learning/feed/",
    "https://developer.nvidia.com/blog/feed/",
    "https://blog.google/technology/ai/rss/",
    "https://openai.com/blog/rss.xml",
    "https://huggingface.co/blog/feed.xml",
    "https://feeds.marketwatch.com/marketwatch/topstories/",
    "https://www.cnbc.com/id/100003114/device/rss/rss.html",
    "https://seekingalpha.com/feed.xml",
    "https://www.sec.gov/news/pressreleases.rss",
    "https://www.federalreserve.gov/feeds/press_all.xml",
    "https://www.whitehouse.gov/briefing-room/feed/",
    "https://arxiv.org/rss/cs.LG",
    "https://www.technologyreview.com/feed/",
    "https://feeds.feedburner.com/reuters/worldNews",
    "https://apnews.com/apf-topnews?utm_source=rss",
    "https://feeds.bbci.co.uk/news/technology/rss.xml",
    "https://www.aljazeera.com/xml/rss/all.xml",
    "https://feeds.npr.org/1001/rss.xml",
    "https://www.theverge.com/rss/index.xml",
    "https://techcrunch.com/feed/",
    "https://www.wired.com/feed/rss",
)

@dataclass(slots=True)
class NewsDocument:
    doc_id: str
    url: str
    published_at: str | None
    fetched_at: str
    source: str
    source_url: str | None
    source_domain: str
    title: str
    body: str
    keyword_hits: list[str]
    relevance_score: float
    query: str


@dataclass(slots=True)
class NewsIngestResult:
    queries_used: list[str]
    fetched_articles: int
    inserted_documents: int
    skipped_documents: int
    documents: list[NewsDocument]
    transport: str
    primary_api_enforced: bool
    source_counts: dict[str, int]


def _empty_source_counts() -> dict[str, int]:
    return {"gnews": 0, "gdelt": 0, "rss": 0, "eodhd": 0}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _compact_text(value: str | None) -> str:
    if not value:
        return ""
    return re.sub(r"\s+", " ", value).strip()


def _canonicalize_url(raw_url: str) -> str:
    parsed = urlparse(raw_url.strip())
    keep_query = []
    for k, v in parse_qsl(parsed.query, keep_blank_values=True):
        lk = k.lower()
        if lk.startswith("utm_"):
            continue
        if lk in {"fbclid", "gclid", "igshid"}:
            continue
        keep_query.append((k, v))
    canonical = parsed._replace(
        scheme=(parsed.scheme or "https").lower(),
        netloc=parsed.netloc.lower(),
        query=urlencode(keep_query, doseq=True),
        fragment="",
    )
    return urlunparse(canonical)


def _extract_domain(raw_url: str) -> str:
    host = (urlparse(raw_url).netloc or "").lower()
    if host.startswith("www."):
        host = host[4:]
    return host


def _is_trusted_source_domain(domain: str) -> bool:
    if not domain:
        return False
    for trusted in TRUSTED_SOURCE_DOMAINS:
        if domain == trusted or domain.endswith(f".{trusted}"):
            return True
    return False


def _compile_term_patterns(terms: tuple[str, ...]) -> tuple[tuple[str, re.Pattern[str]], ...]:
    """
    Pre-compile keyword regexes once at import time.

    _match_terms previously rebuilt and compiled a pattern per term on every
    document; ingestion calls this for two large tuples per article.
    """
    out: list[tuple[str, re.Pattern[str]]] = []
    for term in terms:
        escaped = re.escape(term.lower()).replace(r"\ ", r"\s+")
        out.append((term, re.compile(rf"(?<!\w){escaped}(?!\w)")))
    return tuple(out)


# Module-level patterns: same matching behavior as dynamic _match_terms, lower per-doc cost.
_AI_CORE_TERM_PATTERNS = _compile_term_patterns(AI_CORE_TERMS)
_MARKET_IMPACT_TERM_PATTERNS = _compile_term_patterns(MARKET_IMPACT_TERMS)


def _match_terms(text: str, patterns: tuple[tuple[str, re.Pattern[str]], ...]) -> list[str]:
    hay = text.lower()
    hits: list[str] = []
    for term, pattern in patterns:
        if pattern.search(hay):
            hits.append(term)
    return hits


def _build_query_plan(max_queries: int) -> list[str]:
    if max_queries <= 0:
        return []
    return list(QUERY_SEEDS[:max_queries])


class NewsIntakeAgent:
    """
    Agent 1 implementation for AI-market web document intake.

    Data shape follows the `document` schema in FUNCTIONAL_SPECIFICATION.md.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or load_settings()
        self._db_path = self._resolve_db_path()
        self._last_gdelt_request_at: float | None = None
        self._ensure_db()

    def _resolve_db_path(self) -> str:
        db_path = self.settings.database_url
        if db_path.startswith("sqlite:///"):
            db_path = db_path.replace("sqlite:///", "", 1)
        if os.path.isabs(db_path):
            out = db_path
        else:
            out = os.path.abspath(db_path.replace("\\", "/"))
        parent = os.path.dirname(out) or "."
        os.makedirs(parent, exist_ok=True)
        return out

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_db(self) -> None:
        with self._connect() as conn:
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
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_document_source
                ON document (source)
                """
            )
            columns = conn.execute("PRAGMA table_info(document)").fetchall()
            col_names = {str(c[1]) for c in columns}
            if "source_domain" not in col_names:
                conn.execute(
                    "ALTER TABLE document ADD COLUMN source_domain TEXT NOT NULL DEFAULT ''"
                )

    def run_ingest(
        self,
        *,
        max_queries: int = 8,
        max_per_query: int = 10,
        hours_back: int = 24 * 7,
        trusted_sources_only: bool | None = None,
        require_gnews: bool | None = None,
        require_primary_api: bool | None = None,
        enable_gdelt: bool | None = None,
        enable_rss_fallback: bool | None = None,
        enable_eodhd: bool | None = None,
    ) -> NewsIngestResult:
        queries = _build_query_plan(max_queries)
        if not queries:
            return NewsIngestResult(
                queries_used=[],
                fetched_articles=0,
                inserted_documents=0,
                skipped_documents=0,
                documents=[],
                transport="none",
                primary_api_enforced=False,
                source_counts=_empty_source_counts(),
            )

        fetched_count = 0
        inserted_count = 0
        skipped_count = 0
        inserted_docs: list[NewsDocument] = []
        dedupe_urls: set[str] = set()
        from_ts = (datetime.now(timezone.utc) - timedelta(hours=hours_back)).replace(
            microsecond=0
        )
        enforce_trusted = (
            self.settings.news_trusted_sources_only
            if trusted_sources_only is None
            else trusted_sources_only
        )
        enforce_gnews = (
            self.settings.news_require_gnews
            if require_gnews is None
            else require_gnews
        )
        enforce_primary_api = (
            self.settings.news_require_primary_api
            if require_primary_api is None
            else require_primary_api
        )
        use_gdelt = self.settings.news_enable_gdelt if enable_gdelt is None else enable_gdelt
        use_rss_fallback = (
            self.settings.news_enable_rss_fallback
            if enable_rss_fallback is None
            else enable_rss_fallback
        )
        use_eodhd = self.settings.news_enable_eodhd if enable_eodhd is None else enable_eodhd

        if enforce_gnews and not self.settings.gnews_api_key:
            raise ValueError(
                "NEWS_REQUIRE_GNEWS=true but GNEWS_API_KEY is missing. "
                "Set GNEWS_API_KEY in .env for mandatory primary API ingest."
            )

        source_counts = _empty_source_counts()
        transports_used: set[str] = set()

        if use_eodhd and (self.settings.eodhd_api_key or "").strip():
            eodhd_fetched, eodhd_inserted, eodhd_skipped, eodhd_docs = self._ingest_eodhd_live(
                from_ts=from_ts,
                dedupe_urls=dedupe_urls,
                trusted_sources_only=enforce_trusted,
                max_per_symbol=max(5, min(max_per_query * 2, 60)),
            )
            fetched_count += eodhd_fetched
            inserted_count += eodhd_inserted
            skipped_count += eodhd_skipped
            inserted_docs.extend(eodhd_docs)
            source_counts["eodhd"] += eodhd_fetched
            if eodhd_fetched:
                transports_used.add("eodhd")

        with httpx.Client(timeout=30.0) as client:
            for q in queries:
                articles: list[dict[str, Any]] = []
                # Primary APIs first (mandatory ingestion layer).
                if self.settings.gnews_api_key:
                    gnews_articles = self._fetch_gnews_articles(
                        client=client,
                        query=q,
                        max_per_query=max_per_query,
                        from_iso=from_ts.isoformat(),
                    )
                    articles.extend(gnews_articles)
                    source_counts["gnews"] += len(gnews_articles)
                    if gnews_articles:
                        transports_used.add("gnews")

                if use_gdelt:
                    gdelt_articles = self._fetch_gdelt_articles(
                        client=client,
                        query=q,
                        max_per_query=max_per_query,
                        from_iso=from_ts.isoformat(),
                    )
                    articles.extend(gdelt_articles)
                    source_counts["gdelt"] += len(gdelt_articles)
                    if gdelt_articles:
                        transports_used.add("gdelt")

                # Secondary layer: smaller links / RSS expansion.
                if use_rss_fallback:
                    rss_articles = self._fetch_rss_articles(
                        client=client,
                        query=q,
                        max_per_query=max_per_query,
                        from_iso=from_ts.isoformat(),
                    )
                    articles.extend(rss_articles)
                    source_counts["rss"] += len(rss_articles)
                    if rss_articles:
                        transports_used.add("rss")
                fetched_count += len(articles)
                for raw in articles:
                    candidate = self._to_document(
                        raw,
                        query=q,
                        trusted_sources_only=enforce_trusted,
                    )
                    if not candidate:
                        skipped_count += 1
                        continue
                    if candidate.url in dedupe_urls:
                        skipped_count += 1
                        continue
                    dedupe_urls.add(candidate.url)
                    if self._insert_document(candidate):
                        inserted_count += 1
                        inserted_docs.append(candidate)
                    else:
                        skipped_count += 1

        if enforce_gnews and source_counts["gnews"] == 0:
            raise ValueError(
                "Primary GNews ingest returned zero articles. "
                "Your API quota may be exhausted or the query window returned no data."
            )
        if enforce_primary_api and (
            source_counts["gnews"] + source_counts["gdelt"] + source_counts["eodhd"]
        ) == 0:
            raise ValueError(
                "Primary API ingest returned zero articles (GNews + GDELT + EODHD). "
                "Check provider quotas or query windows."
            )

        return NewsIngestResult(
            queries_used=queries,
            fetched_articles=fetched_count,
            inserted_documents=inserted_count,
            skipped_documents=skipped_count,
            documents=inserted_docs,
            transport="+".join(sorted(transports_used)) if transports_used else "none",
            primary_api_enforced=enforce_primary_api,
            source_counts=source_counts,
        )

    def run_historical_backfill(
        self,
        *,
        days_back: int = 365,
        chunk_days: int = 7,
        max_queries: int = 14,
        max_per_query: int = 50,
        max_pages: int = 2,
        trusted_sources_only: bool | None = None,
        require_gnews: bool | None = None,
        require_primary_api: bool | None = None,
        enable_gdelt: bool | None = None,
    ) -> NewsIngestResult:
        queries = _build_query_plan(max_queries)
        if not queries:
            return NewsIngestResult(
                queries_used=[],
                fetched_articles=0,
                inserted_documents=0,
                skipped_documents=0,
                documents=[],
                transport="none",
                primary_api_enforced=True,
                source_counts=_empty_source_counts(),
            )

        enforce_trusted = (
            self.settings.news_trusted_sources_only
            if trusted_sources_only is None
            else trusted_sources_only
        )
        enforce_gnews = (
            self.settings.news_require_gnews
            if require_gnews is None
            else require_gnews
        )
        enforce_primary_api = (
            self.settings.news_require_primary_api
            if require_primary_api is None
            else require_primary_api
        )
        use_gdelt = self.settings.news_enable_gdelt if enable_gdelt is None else enable_gdelt
        if enforce_gnews and not self.settings.gnews_api_key:
            raise ValueError(
                "NEWS_REQUIRE_GNEWS=true but GNEWS_API_KEY is missing. "
                "Set GNEWS_API_KEY in .env for mandatory primary API ingest."
            )
        if not self.settings.gnews_api_key and not use_gdelt:
            raise ValueError("Historical backfill requires GNEWS_API_KEY or NEWS_ENABLE_GDELT=true")

        now = datetime.now(timezone.utc).replace(microsecond=0)
        start = now - timedelta(days=max(1, days_back))
        slice_days = max(1, chunk_days)
        pages = max(1, min(max_pages, 10))

        fetched_count = 0
        inserted_count = 0
        skipped_count = 0
        inserted_docs: list[NewsDocument] = []
        dedupe_urls: set[str] = set()
        source_counts = _empty_source_counts()

        with httpx.Client(timeout=30.0) as client:
            cursor = start
            while cursor < now:
                chunk_end = min(now, cursor + timedelta(days=slice_days))
                from_iso = cursor.isoformat()
                to_iso = chunk_end.isoformat()
                for q in queries:
                    if self.settings.gnews_api_key:
                        for page in range(1, pages + 1):
                            articles = self._fetch_gnews_articles(
                                client=client,
                                query=q,
                                max_per_query=max_per_query,
                                from_iso=from_iso,
                                to_iso=to_iso,
                                page=page,
                            )
                            time.sleep(0.15)
                            if not articles:
                                break
                            fetched_count += len(articles)
                            source_counts["gnews"] += len(articles)
                            for raw in articles:
                                candidate = self._to_document(
                                    raw,
                                    query=q,
                                    trusted_sources_only=enforce_trusted,
                                )
                                if not candidate:
                                    skipped_count += 1
                                    continue
                                if candidate.url in dedupe_urls:
                                    skipped_count += 1
                                    continue
                                dedupe_urls.add(candidate.url)
                                if self._insert_document(candidate):
                                    inserted_count += 1
                                    inserted_docs.append(candidate)
                                else:
                                    skipped_count += 1
                            if len(articles) < max_per_query:
                                break
                    if use_gdelt:
                        gdelt_rows = self._fetch_gdelt_articles(
                            client=client,
                            query=q,
                            max_per_query=max_per_query,
                            from_iso=from_iso,
                            to_iso=to_iso,
                        )
                        fetched_count += len(gdelt_rows)
                        source_counts["gdelt"] += len(gdelt_rows)
                        for raw in gdelt_rows:
                            candidate = self._to_document(
                                raw,
                                query=q,
                                trusted_sources_only=enforce_trusted,
                            )
                            if not candidate:
                                skipped_count += 1
                                continue
                            if candidate.url in dedupe_urls:
                                skipped_count += 1
                                continue
                            dedupe_urls.add(candidate.url)
                            if self._insert_document(candidate):
                                inserted_count += 1
                                inserted_docs.append(candidate)
                            else:
                                skipped_count += 1
                cursor = chunk_end
                print(
                    f"  backfill chunk through {chunk_end.date()} "
                    f"(inserted={inserted_count}, fetched={fetched_count})",
                    flush=True,
                )

        if enforce_gnews and source_counts["gnews"] == 0:
            raise ValueError(
                "Historical backfill fetched zero GNews articles. "
                "Check GNews quota/plan or reduce backfill scope."
            )
        if enforce_primary_api and (source_counts["gnews"] + source_counts["gdelt"]) == 0:
            raise ValueError(
                "Historical backfill fetched zero primary API articles (GNews + GDELT)."
            )

        return NewsIngestResult(
            queries_used=queries,
            fetched_articles=fetched_count,
            inserted_documents=inserted_count,
            skipped_documents=skipped_count,
            documents=inserted_docs,
            transport="gnews+gdelt_backfill",
            primary_api_enforced=enforce_primary_api,
            source_counts=source_counts,
        )

    def _parse_datetime(self, raw: str | None) -> datetime | None:
        text = _compact_text(raw)
        if not text:
            return None
        normalized = text
        if normalized.endswith("Z"):
            normalized = normalized[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(normalized)
        except ValueError:
            try:
                dt = parsedate_to_datetime(text)
            except Exception:  # noqa: BLE001
                return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)

    def _fetch_rss_articles(
        self,
        *,
        client: httpx.Client,
        query: str,
        max_per_query: int,
        from_iso: str,
    ) -> list[dict[str, Any]]:
        from_dt = self._parse_datetime(from_iso)
        if from_dt is None:
            from_dt = datetime.now(timezone.utc) - timedelta(days=7)
        out: list[dict[str, Any]] = []
        for feed_url in TRUSTED_RSS_FEEDS:
            if len(out) >= max_per_query * 3:
                break
            try:
                response = client.get(feed_url)
                response.raise_for_status()
                xml_text = response.text
                rows = self._parse_feed_xml(xml_text, feed_url=feed_url)
            except Exception:  # noqa: BLE001
                continue
            for row in rows:
                pub_dt = self._parse_datetime(str(row.get("publishedAt", "")))
                if pub_dt is not None and pub_dt < from_dt:
                    continue
                out.append(row)
                if len(out) >= max_per_query * 3:
                    break
        return out

    def _parse_feed_xml(self, xml_text: str, *, feed_url: str) -> list[dict[str, Any]]:
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError:
            return []
        items: list[dict[str, Any]] = []
        # RSS
        for item in root.findall(".//item"):
            title = _compact_text(item.findtext("title"))
            link = _compact_text(item.findtext("link"))
            desc = _compact_text(item.findtext("description"))
            content = _compact_text(item.findtext("{http://purl.org/rss/1.0/modules/content/}encoded"))
            pub = _compact_text(item.findtext("pubDate"))
            if link:
                items.append(
                    {
                        "title": title,
                        "url": link,
                        "description": desc,
                        "content": content,
                        "publishedAt": pub,
                        "source": {"name": _extract_domain(feed_url), "url": feed_url},
                    }
                )
        # Atom
        if not items:
            for entry in root.findall(".//{http://www.w3.org/2005/Atom}entry"):
                title = _compact_text(entry.findtext("{http://www.w3.org/2005/Atom}title"))
                summary = _compact_text(entry.findtext("{http://www.w3.org/2005/Atom}summary"))
                content = _compact_text(entry.findtext("{http://www.w3.org/2005/Atom}content"))
                updated = _compact_text(entry.findtext("{http://www.w3.org/2005/Atom}updated"))
                link = ""
                for link_node in entry.findall("{http://www.w3.org/2005/Atom}link"):
                    href = _compact_text(link_node.attrib.get("href"))
                    if href:
                        link = href
                        break
                if link:
                    items.append(
                        {
                            "title": title,
                            "url": link,
                            "description": summary,
                            "content": content,
                            "publishedAt": updated,
                            "source": {"name": _extract_domain(feed_url), "url": feed_url},
                        }
                    )
        return items

    def _to_gdelt_datetime(self, iso_text: str) -> str:
        dt = self._parse_datetime(iso_text) or datetime.now(timezone.utc)
        return dt.strftime("%Y%m%d%H%M%S")

    def _fetch_gdelt_articles(
        self,
        *,
        client: httpx.Client,
        query: str,
        max_per_query: int,
        from_iso: str,
        to_iso: str | None = None,
    ) -> list[dict[str, Any]]:
        params = {
            "query": query,
            "mode": "ArtList",
            "format": "json",
            "sort": "DateDesc",
            "maxrecords": max(1, min(max_per_query, 250)),
            "startdatetime": self._to_gdelt_datetime(from_iso),
        }
        if to_iso:
            params["enddatetime"] = self._to_gdelt_datetime(to_iso)
        # GDELT free endpoint expects low request cadence (~1 request / 5s).
        min_gap_seconds = 5.2
        backoff_seconds = [5.5, 8.0, 12.0]
        for i in range(len(backoff_seconds) + 1):
            try:
                if self._last_gdelt_request_at is not None:
                    elapsed = time.time() - self._last_gdelt_request_at
                    if elapsed < min_gap_seconds:
                        time.sleep(min_gap_seconds - elapsed)
                response = client.get(GDELT_DOC_API_URL, params=params)
                self._last_gdelt_request_at = time.time()
                response.raise_for_status()
                payload = response.json()
                rows = payload.get("articles", [])
                if not isinstance(rows, list):
                    return []
                out: list[dict[str, Any]] = []
                for row in rows:
                    if not isinstance(row, dict):
                        continue
                    seen_raw = _compact_text(str(row.get("seendate", "")))
                    seen_iso = seen_raw
                    if len(seen_raw) == 14 and seen_raw.isdigit():
                        try:
                            seen_iso = datetime.strptime(
                                seen_raw, "%Y%m%d%H%M%S"
                            ).replace(tzinfo=timezone.utc).isoformat()
                        except ValueError:
                            seen_iso = seen_raw
                    out.append(
                        {
                            "title": _compact_text(str(row.get("title", ""))),
                            "url": _compact_text(str(row.get("url", ""))),
                            "description": _compact_text(f"{row.get('title', '')} {query}"),
                            "content": _compact_text(f"{row.get('title', '')} {query}"),
                            "publishedAt": seen_iso,
                            "source": {
                                "name": _compact_text(str(row.get("domain", ""))) or "gdelt",
                                "url": None,
                            },
                        }
                    )
                return out
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                if status in {429, 500, 502, 503, 504} and i < len(backoff_seconds):
                    time.sleep(backoff_seconds[i])
                    continue
                return []
            except Exception:  # noqa: BLE001
                if i < len(backoff_seconds):
                    time.sleep(backoff_seconds[i])
                    continue
                return []
        return []

    def _fetch_gnews_articles(
        self,
        *,
        client: httpx.Client,
        query: str,
        max_per_query: int,
        from_iso: str,
        to_iso: str | None = None,
        page: int = 1,
    ) -> list[dict[str, Any]]:
        params = {
            "q": query,
            "lang": "en",
            "max": max(1, min(max_per_query, 50)),
            "expand": "content",
            "from": from_iso,
            "page": max(1, page),
            "apikey": self.settings.gnews_api_key,
        }
        if to_iso:
            params["to"] = to_iso
        backoff_seconds = [1.0, 2.0, 4.0]
        for i in range(len(backoff_seconds) + 1):
            try:
                response = client.get(GNEWS_SEARCH_URL, params=params)
                response.raise_for_status()
                payload = response.json()
                rows = payload.get("articles", [])
                if not isinstance(rows, list):
                    return []
                return [r for r in rows if isinstance(r, dict)]
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                # Rate limit / transient errors: retry with backoff.
                if status in {429, 500, 502, 503, 504} and i < len(backoff_seconds):
                    time.sleep(backoff_seconds[i])
                    continue
                return []
            except Exception:  # noqa: BLE001
                if i < len(backoff_seconds):
                    time.sleep(backoff_seconds[i])
                    continue
                return []
        return []

    def _ingest_eodhd_live(
        self,
        *,
        from_ts: datetime,
        dedupe_urls: set[str],
        trusted_sources_only: bool,
        max_per_symbol: int = 40,
    ) -> tuple[int, int, int, list[NewsDocument]]:
        api_key = (self.settings.eodhd_api_key or "").strip()
        if not api_key:
            return 0, 0, 0, []

        now = datetime.now(timezone.utc)
        pairs = fetch_eodhd_live_for_symbols(
            trading_symbols(settings=self.settings, limit=200),
            from_ts,
            now,
            api_key,
            max_articles_per_symbol=max_per_symbol,
        )
        fetched = len(pairs)
        inserted = 0
        skipped = 0
        docs: list[NewsDocument] = []
        for symbol, raw in pairs:
            candidate = self._to_document_eodhd(
                raw,
                symbol=symbol,
                trusted_sources_only=trusted_sources_only,
            )
            if not candidate:
                skipped += 1
                continue
            if candidate.url in dedupe_urls:
                skipped += 1
                continue
            dedupe_urls.add(candidate.url)
            if self._insert_document(candidate):
                inserted += 1
                docs.append(candidate)
            else:
                skipped += 1
        return fetched, inserted, skipped, docs

    def _to_document_eodhd(
        self,
        article: dict[str, Any],
        *,
        symbol: str,
        trusted_sources_only: bool,
    ) -> NewsDocument | None:
        raw_url = _compact_text(str(article.get("link") or article.get("url") or ""))
        if not raw_url:
            return None
        canonical_url = _canonicalize_url(raw_url)
        source_domain = _extract_domain(canonical_url)
        if trusted_sources_only and not _is_trusted_source_domain(source_domain):
            return None

        title = _compact_text(str(article.get("title") or ""))
        content = _compact_text(str(article.get("content") or ""))
        body = content or title
        combined = _compact_text(" ".join(x for x in [title, body] if x))
        if not combined:
            return None

        clean = (symbol or "").strip().upper()
        company = SYMBOL_TO_COMPANY.get(clean)
        company_name = company.name.lower() if company else clean.lower()
        low = combined.lower()
        ai_hits = _match_terms(combined, _AI_CORE_TERM_PATTERNS)
        market_hits = _match_terms(combined, _MARKET_IMPACT_TERM_PATTERNS)
        tag_hits = _article_keyword_hits(article)
        has_ticker_context = (
            clean.lower() in low
            or company_name in low
            or any(clean in hit.upper() for hit in tag_hits)
        )
        if not (ai_hits or market_hits or has_ticker_context):
            return None

        keyword_hits = sorted(set(ai_hits + market_hits + tag_hits))
        relevance_score = _article_relevance(article, title, body)
        if relevance_score < 3.0 and not has_ticker_context:
            return None

        source_name = _compact_text(str(article.get("source") or "eodhd")) or "eodhd"
        published_at = _compact_text(str(article.get("date") or "")) or None
        doc_id = hashlib.sha256(canonical_url.encode("utf-8")).hexdigest()[:24]
        query = f"eodhd:{clean}"

        return NewsDocument(
            doc_id=doc_id,
            url=canonical_url,
            published_at=published_at,
            fetched_at=_utc_now_iso(),
            source=source_name,
            source_url=canonical_url,
            source_domain=source_domain,
            title=title,
            body=body,
            keyword_hits=keyword_hits,
            relevance_score=relevance_score,
            query=query,
        )

    def _to_document(
        self,
        article: dict[str, Any],
        *,
        query: str,
        trusted_sources_only: bool,
    ) -> NewsDocument | None:
        raw_url = _compact_text(str(article.get("url", "")))
        if not raw_url:
            return None
        canonical_url = _canonicalize_url(raw_url)
        source_domain = _extract_domain(canonical_url)
        if trusted_sources_only and not _is_trusted_source_domain(source_domain):
            return None

        title = _compact_text(str(article.get("title", "")))
        description = _compact_text(str(article.get("description", "")))
        content = _compact_text(str(article.get("content", "")))
        body = _compact_text(" ".join(x for x in [description, content] if x))
        combined = _compact_text(" ".join(x for x in [title, body] if x))
        if not combined:
            return None

        ai_hits = _match_terms(combined, _AI_CORE_TERM_PATTERNS)
        market_hits = _match_terms(combined, _MARKET_IMPACT_TERM_PATTERNS)
        if not ai_hits or not market_hits:
            return None

        keyword_hits = sorted(set(ai_hits + market_hits))
        relevance_score = round((len(ai_hits) * 2.0) + (len(market_hits) * 1.5), 2)
        if relevance_score < 3.5:
            return None

        source_obj = article.get("source", {})
        source_name = "unknown"
        source_url = None
        if isinstance(source_obj, dict):
            source_name = _compact_text(str(source_obj.get("name", ""))) or "unknown"
            source_url = _compact_text(str(source_obj.get("url", ""))) or None

        published_at = _compact_text(str(article.get("publishedAt", ""))) or None
        fetched_at = _utc_now_iso()
        doc_id = hashlib.sha256(canonical_url.encode("utf-8")).hexdigest()[:24]

        return NewsDocument(
            doc_id=doc_id,
            url=canonical_url,
            published_at=published_at,
            fetched_at=fetched_at,
            source=source_name,
            source_url=source_url,
            source_domain=source_domain,
            title=title,
            body=body,
            keyword_hits=keyword_hits,
            relevance_score=relevance_score,
            query=query,
        )

    def _insert_document(self, doc: NewsDocument) -> bool:
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT OR IGNORE INTO document (
                    doc_id, url, published_at, fetched_at, source, source_url, source_domain,
                    title, body, keyword_hits, relevance_score, query
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    doc.doc_id,
                    doc.url,
                    doc.published_at,
                    doc.fetched_at,
                    doc.source,
                    doc.source_url,
                    doc.source_domain,
                    doc.title,
                    doc.body,
                    json.dumps(doc.keyword_hits),
                    doc.relevance_score,
                    doc.query,
                ),
            )
            return cur.rowcount > 0

    def list_documents(self, *, limit: int = 50) -> list[NewsDocument]:
        safe_limit = max(1, min(limit, 250))
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    doc_id, url, published_at, fetched_at, source, source_url, source_domain,
                    title, body, keyword_hits, relevance_score, query
                FROM document
                ORDER BY COALESCE(published_at, fetched_at) DESC
                LIMIT ?
                """,
                (safe_limit,),
            ).fetchall()
        out: list[NewsDocument] = []
        for row in rows:
            raw_hits = row["keyword_hits"]
            hits: list[str]
            try:
                parsed = json.loads(raw_hits)
                hits = parsed if isinstance(parsed, list) else []
            except Exception:  # noqa: BLE001 - resilient read for mixed rows
                hits = []
            out.append(
                NewsDocument(
                    doc_id=row["doc_id"],
                    url=row["url"],
                    published_at=row["published_at"],
                    fetched_at=row["fetched_at"],
                    source=row["source"],
                    source_url=row["source_url"],
                    source_domain=row["source_domain"] or _extract_domain(row["url"]),
                    title=row["title"] or "",
                    body=row["body"] or "",
                    keyword_hits=hits,
                    relevance_score=float(row["relevance_score"]),
                    query=row["query"] or "",
                )
            )
        return out

