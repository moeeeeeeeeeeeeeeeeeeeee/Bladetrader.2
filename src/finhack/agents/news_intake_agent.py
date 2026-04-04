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
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse
from xml.etree import ElementTree as ET

import httpx

from finhack.config import Settings, load_settings

GNEWS_SEARCH_URL = "https://gnews.io/api/v4/search"

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

YFINANCE_NEWS_TICKERS: tuple[str, ...] = (
    "NVDA",
    "MSFT",
    "GOOGL",
    "AMZN",
    "META",
    "AAPL",
    "AMD",
    "AVGO",
    "TSM",
    "ASML",
    "QCOM",
    "INTC",
    "ANET",
    "SMCI",
    "PLTR",
    "SNOW",
    "ORCL",
    "CRM",
    "PANW",
    "CRWD",
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


def _match_terms(text: str, terms: tuple[str, ...]) -> list[str]:
    hay = text.lower()
    hits: list[str] = []
    for term in terms:
        escaped = re.escape(term.lower()).replace(r"\ ", r"\s+")
        pattern = rf"(?<!\w){escaped}(?!\w)"
        if re.search(pattern, hay):
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

        transport = "gnews" if self.settings.gnews_api_key else "rss_fallback"
        with httpx.Client(timeout=30.0) as client:
            for q in queries:
                if self.settings.gnews_api_key:
                    articles = self._fetch_gnews_articles(
                        client=client,
                        query=q,
                        max_per_query=max_per_query,
                        from_iso=from_ts.isoformat(),
                    )
                else:
                    rss_articles = self._fetch_rss_articles(
                        client=client,
                        query=q,
                        max_per_query=max_per_query,
                        from_iso=from_ts.isoformat(),
                    )
                    yf_articles = self._fetch_yfinance_news_articles(
                        query=q,
                        max_per_query=max_per_query,
                        from_iso=from_ts.isoformat(),
                    )
                    articles = rss_articles + yf_articles
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

        return NewsIngestResult(
            queries_used=queries,
            fetched_articles=fetched_count,
            inserted_documents=inserted_count,
            skipped_documents=skipped_count,
            documents=inserted_docs,
            transport=transport,
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

    def _fetch_yfinance_news_articles(
        self,
        *,
        query: str,
        max_per_query: int,
        from_iso: str,
    ) -> list[dict[str, Any]]:
        _ = query
        from_dt = self._parse_datetime(from_iso)
        if from_dt is None:
            from_dt = datetime.now(timezone.utc) - timedelta(days=60)
        out: list[dict[str, Any]] = []
        try:
            import yfinance as yf
        except Exception:  # noqa: BLE001
            return out

        for symbol in YFINANCE_NEWS_TICKERS:
            if len(out) >= max_per_query * 3:
                break
            try:
                rows = yf.Ticker(symbol).news or []
            except Exception:  # noqa: BLE001
                continue
            for row in rows:
                if not isinstance(row, dict):
                    continue
                title = _compact_text(str(row.get("title", "")))
                summary = _compact_text(str(row.get("summary", "")))
                link = _compact_text(str(row.get("link", "")))
                provider = _compact_text(str(row.get("publisher", ""))) or "yfinance"
                ts = row.get("providerPublishTime")
                published_at = None
                if isinstance(ts, (int, float)):
                    published_at = (
                        datetime.fromtimestamp(float(ts), tz=timezone.utc)
                        .replace(microsecond=0)
                        .isoformat()
                    )
                pub_dt = self._parse_datetime(published_at)
                if pub_dt is not None and pub_dt < from_dt:
                    continue
                if not link:
                    continue
                out.append(
                    {
                        "title": title,
                        "url": link,
                        "description": summary,
                        "content": summary,
                        "publishedAt": published_at,
                        "source": {"name": provider, "url": None},
                    }
                )
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

    def _fetch_gnews_articles(
        self,
        *,
        client: httpx.Client,
        query: str,
        max_per_query: int,
        from_iso: str,
    ) -> list[dict[str, Any]]:
        params = {
            "q": query,
            "lang": "en",
            "max": max(1, min(max_per_query, 50)),
            "expand": "content",
            "from": from_iso,
            "apikey": self.settings.gnews_api_key,
        }
        response = client.get(GNEWS_SEARCH_URL, params=params)
        response.raise_for_status()
        payload = response.json()
        rows = payload.get("articles", [])
        if not isinstance(rows, list):
            return []
        return [r for r in rows if isinstance(r, dict)]

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

        ai_hits = _match_terms(combined, AI_CORE_TERMS)
        market_hits = _match_terms(combined, MARKET_IMPACT_TERMS)
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

