"""Annual-report intelligence dataset for AI leaders and spillover graph."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import math
import os
import sqlite3
from typing import Any

import httpx

from finhack.config import Settings, load_settings
from finhack.market_data import get_close_series, get_earnings_events

EODHD_NEWS_URL = "https://eodhd.com/api/news"

POSITIVE_TERMS: tuple[str, ...] = (
    "beat",
    "beats",
    "growth",
    "strong",
    "surge",
    "upside",
    "upgrade",
    "record",
    "demand",
    "expansion",
    "partnership",
)
NEGATIVE_TERMS: tuple[str, ...] = (
    "miss",
    "misses",
    "weak",
    "decline",
    "downgrade",
    "slowdown",
    "lawsuit",
    "investigation",
    "ban",
    "restriction",
    "headwind",
)
AI_TERMS: tuple[str, ...] = (
    "ai",
    "artificial intelligence",
    "generative ai",
    "llm",
    "foundation model",
    "gpu",
    "inference",
    "training",
    "data center",
    "automation",
)


@dataclass(frozen=True)
class CompanyDef:
    symbol: str
    name: str
    ai_category: str
    role: str


@dataclass(frozen=True)
class SpilloverEdge:
    source_symbol: str
    target_symbol: str
    edge_weight: float
    rationale: str


PRIMARY_AI_COMPANIES: tuple[CompanyDef, ...] = (
    CompanyDef("NVDA", "NVIDIA", "AI Compute", "ComputeSupplier"),
    CompanyDef("AMD", "AMD", "AI Compute", "ComputeSupplier"),
    CompanyDef("INTC", "Intel", "AI Compute", "ComputeSupplier"),
    CompanyDef("MSFT", "Microsoft", "Cloud & Hyperscalers", "CloudPlatform"),
    CompanyDef("AMZN", "Amazon", "Cloud & Hyperscalers", "CloudPlatform"),
    CompanyDef("GOOGL", "Alphabet", "AI Models & Platforms", "ModelPlatform"),
    CompanyDef("META", "Meta", "AI-Driven Consumer Platforms", "ConsumerDistributor"),
    CompanyDef("SNOW", "Snowflake", "Data & Analytics Layer", "DataEnabler"),
    CompanyDef("PLTR", "Palantir", "Data & Analytics Layer", "DataEnabler"),
    CompanyDef("CRM", "Salesforce", "Enterprise AI Applications", "EnterpriseApp"),
    CompanyDef("ORCL", "Oracle", "Enterprise AI Applications", "EnterpriseApp"),
    CompanyDef("ADBE", "Adobe", "Enterprise AI Applications", "EnterpriseApp"),
    CompanyDef("NOW", "ServiceNow", "Enterprise AI Applications", "EnterpriseApp"),
    CompanyDef("AVGO", "Broadcom", "AI Compute", "InfrastructureSupplier"),
)

SPILLOVER_COMPANIES: tuple[CompanyDef, ...] = (
    CompanyDef("EQIX", "Equinix", "AI Physical Infrastructure", "DataCenter"),
    CompanyDef("NEE", "NextEra Energy", "AI Physical Infrastructure", "PowerUtility"),
    CompanyDef("VRT", "Vertiv", "AI Physical Infrastructure", "CoolingPower"),
    CompanyDef("TSLA", "Tesla", "AI-Enabled Industries", "AutoAI"),
    CompanyDef("JPM", "JPMorgan Chase", "AI-Enabled Industries", "FinanceAI"),
    CompanyDef("LMT", "Lockheed Martin", "AI-Enabled Industries", "DefenseAI"),
)

SPILLOVER_EDGES: tuple[SpilloverEdge, ...] = (
    SpilloverEdge("NVDA", "MSFT", 0.90, "GPU supply impacts cloud capacity"),
    SpilloverEdge("NVDA", "AMZN", 0.85, "GPU supply impacts cloud capacity"),
    SpilloverEdge("NVDA", "EQIX", 0.72, "Compute demand drives colocation usage"),
    SpilloverEdge("NVDA", "NEE", 0.60, "Compute demand increases power load"),
    SpilloverEdge("AMD", "MSFT", 0.75, "Alternative accelerator supply impacts cloud"),
    SpilloverEdge("AMD", "AMZN", 0.70, "Alternative accelerator supply impacts cloud"),
    SpilloverEdge("MSFT", "CRM", 0.65, "Enterprise AI adoption spillover"),
    SpilloverEdge("GOOGL", "META", 0.55, "Model/platform competition spillover"),
    SpilloverEdge("AMZN", "VRT", 0.58, "Infra scaling impacts cooling/power vendors"),
    SpilloverEdge("PLTR", "JPM", 0.50, "AI analytics adoption spillover to finance"),
    SpilloverEdge("MSFT", "LMT", 0.45, "Government AI contracts spillover"),
    SpilloverEdge("GOOGL", "TSLA", 0.40, "AI model/platform progress spillover"),
)


def _normalize_text(text: str | None) -> str:
    return " ".join((text or "").lower().split())


def _match_count(text: str, terms: tuple[str, ...]) -> int:
    return sum(1 for term in terms if term in text)


def _sentiment_score(title: str, content: str) -> tuple[float, int]:
    text = _normalize_text(f"{title} {content}")
    pos = _match_count(text, POSITIVE_TERMS)
    neg = _match_count(text, NEGATIVE_TERMS)
    ai_hits = _match_count(text, AI_TERMS)
    raw = (pos - neg) + (0.35 * ai_hits)
    score = math.tanh(raw / 4.5)
    return round(score, 6), ai_hits


def _safe_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


class AnnualIntelligenceBuilder:
    """Build/store annual-report sentiment + spillover dataset."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or load_settings()
        self._db_path = self._resolve_db_path()
        self._ensure_tables()

    def _resolve_db_path(self) -> str:
        db_path = self.settings.database_url
        if db_path.startswith("sqlite:///"):
            db_path = db_path.replace("sqlite:///", "", 1)
        if os.path.isabs(db_path):
            return db_path
        return os.path.abspath(db_path.replace("\\", "/"))

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure_tables(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS annual_company_universe (
                    symbol TEXT PRIMARY KEY,
                    company_name TEXT NOT NULL,
                    ai_category TEXT NOT NULL,
                    role TEXT NOT NULL,
                    is_primary INTEGER NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS annual_event (
                    event_id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    event_year INTEGER NOT NULL,
                    anchor_date TEXT NOT NULL,
                    pre_window_start TEXT NOT NULL,
                    pre_window_end TEXT NOT NULL,
                    post_window_end TEXT NOT NULL,
                    pre_return_pct REAL,
                    post_return_pct REAL,
                    target_sign INTEGER,
                    news_count INTEGER NOT NULL,
                    mean_sentiment REAL NOT NULL,
                    ai_term_hits INTEGER NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS annual_event_news (
                    row_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    published_at TEXT,
                    source TEXT,
                    title TEXT NOT NULL,
                    url TEXT,
                    sentiment_score REAL NOT NULL,
                    ai_term_hits INTEGER NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS annual_spillover_effect (
                    row_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT NOT NULL,
                    source_symbol TEXT NOT NULL,
                    target_symbol TEXT NOT NULL,
                    edge_weight REAL NOT NULL,
                    spillover_score REAL NOT NULL,
                    rationale TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_annual_event_symbol_year
                ON annual_event(symbol, event_year DESC)
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_annual_spillover_event
                ON annual_spillover_effect(event_id)
                """
            )

    def _upsert_company_universe(self) -> None:
        rows = [(c.symbol, c.name, c.ai_category, c.role, 1) for c in PRIMARY_AI_COMPANIES]
        rows.extend((c.symbol, c.name, c.ai_category, c.role, 0) for c in SPILLOVER_COMPANIES)
        with self._connect() as conn:
            conn.executemany(
                """
                INSERT INTO annual_company_universe (
                    symbol, company_name, ai_category, role, is_primary
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(symbol) DO UPDATE SET
                    company_name=excluded.company_name,
                    ai_category=excluded.ai_category,
                    role=excluded.role,
                    is_primary=excluded.is_primary
                """,
                rows,
            )

    def _annual_anchors_for_symbol(self, symbol: str, years_back: int) -> list[datetime]:
        events = get_earnings_events(symbol, limit=max(8, years_back * 5), recent_days=365 * years_back)
        by_year: dict[int, datetime] = {}
        for dt in events:
            y = dt.year
            prev = by_year.get(y)
            if prev is None or dt > prev:
                by_year[y] = dt
        selected_years = sorted(by_year.keys(), reverse=True)[:years_back]
        return sorted(by_year[y] for y in selected_years)

    def _fetch_eodhd_news(
        self, symbol: str, from_date: str, to_date: str, limit: int
    ) -> list[dict[str, Any]]:
        if not self.settings.eodhd_api_key:
            return []
        params = {
            "api_token": self.settings.eodhd_api_key,
            "fmt": "json",
            "s": f"{symbol}.US",
            "from": from_date,
            "to": to_date,
            "limit": max(1, min(limit, 100)),
            "offset": 0,
        }
        with httpx.Client(timeout=30.0) as client:
            res = client.get(EODHD_NEWS_URL, params=params)
            res.raise_for_status()
            payload = res.json()
        if isinstance(payload, list):
            return [r for r in payload if isinstance(r, dict)]
        return []

    def _compute_returns(
        self, symbol: str, pre_start: datetime, anchor: datetime, post_end: datetime
    ) -> tuple[float | None, float | None, int]:
        closes = get_close_series(
            symbol,
            start=(pre_start - timedelta(days=7)).date().isoformat(),
            end=(post_end + timedelta(days=7)).date().isoformat(),
            settings=self.settings,
        )
        if closes.empty:
            return None, None, 0
        idx = list(closes.index)
        if not idx:
            return None, None, 0

        def first_idx_on_or_after(dt: datetime) -> int | None:
            d = dt.date()
            for i, ts in enumerate(idx):
                if ts.to_pydatetime().date() >= d:
                    return i
            return None

        pre_i0 = first_idx_on_or_after(pre_start)
        anch_i = first_idx_on_or_after(anchor)
        post_i = first_idx_on_or_after(post_end)
        if pre_i0 is None or anch_i is None or post_i is None:
            return None, None, 0

        p0 = _safe_float(closes.iloc[pre_i0])
        pa = _safe_float(closes.iloc[anch_i])
        p5 = _safe_float(closes.iloc[post_i])
        if p0 in {None, 0.0} or pa in {None, 0.0} or p5 in {None, 0.0}:
            return None, None, 0
        pre_ret = ((pa - p0) / p0) * 100.0
        post_ret = ((p5 - pa) / pa) * 100.0
        target = 1 if post_ret > 0 else -1 if post_ret < 0 else 0
        return round(pre_ret, 4), round(post_ret, 4), target

    def build(
        self,
        *,
        years_back: int = 5,
        pre_days: int = 30,
        post_days: int = 20,
        max_news_per_event: int = 25,
    ) -> dict[str, Any]:
        now_iso = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        self._upsert_company_universe()

        events_written = 0
        news_written = 0
        spillover_written = 0
        symbols_processed = 0

        with self._connect() as conn:
            for company in PRIMARY_AI_COMPANIES:
                anchors = self._annual_anchors_for_symbol(company.symbol, years_back)
                if not anchors:
                    continue
                symbols_processed += 1
                for anchor in anchors:
                    pre_start = anchor - timedelta(days=pre_days)
                    post_end = anchor + timedelta(days=post_days)
                    event_id = f"{company.symbol}_{anchor.date().isoformat()}"
                    pre_ret, post_ret, target = self._compute_returns(
                        company.symbol, pre_start, anchor, post_end
                    )

                    news_rows = self._fetch_eodhd_news(
                        company.symbol,
                        from_date=pre_start.date().isoformat(),
                        to_date=anchor.date().isoformat(),
                        limit=max_news_per_event,
                    )
                    event_news: list[tuple[Any, ...]] = []
                    sentiment_values: list[float] = []
                    ai_hits_total = 0
                    for row in news_rows:
                        title = str(row.get("title", "")).strip()
                        content = str(row.get("content", "")).strip()
                        score, ai_hits = _sentiment_score(title, content)
                        sentiment_values.append(score)
                        ai_hits_total += ai_hits
                        event_news.append(
                            (
                                event_id,
                                company.symbol,
                                str(row.get("date", "")).strip() or None,
                                str(row.get("source", "")).strip() or None,
                                title,
                                str(row.get("link", "")).strip() or None,
                                score,
                                ai_hits,
                            )
                        )
                    mean_sent = round(
                        (sum(sentiment_values) / len(sentiment_values)) if sentiment_values else 0.0,
                        6,
                    )

                    conn.execute(
                        """
                        INSERT INTO annual_event (
                            event_id, symbol, event_year, anchor_date, pre_window_start,
                            pre_window_end, post_window_end, pre_return_pct, post_return_pct,
                            target_sign, news_count, mean_sentiment, ai_term_hits, created_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(event_id) DO UPDATE SET
                            pre_return_pct=excluded.pre_return_pct,
                            post_return_pct=excluded.post_return_pct,
                            target_sign=excluded.target_sign,
                            news_count=excluded.news_count,
                            mean_sentiment=excluded.mean_sentiment,
                            ai_term_hits=excluded.ai_term_hits,
                            created_at=excluded.created_at
                        """,
                        (
                            event_id,
                            company.symbol,
                            anchor.year,
                            anchor.date().isoformat(),
                            pre_start.date().isoformat(),
                            anchor.date().isoformat(),
                            post_end.date().isoformat(),
                            pre_ret,
                            post_ret,
                            target,
                            len(event_news),
                            mean_sent,
                            ai_hits_total,
                            now_iso,
                        ),
                    )
                    events_written += 1

                    conn.execute("DELETE FROM annual_event_news WHERE event_id = ?", (event_id,))
                    if event_news:
                        conn.executemany(
                            """
                            INSERT INTO annual_event_news (
                                event_id, symbol, published_at, source, title, url,
                                sentiment_score, ai_term_hits
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            event_news,
                        )
                        news_written += len(event_news)

                    conn.execute("DELETE FROM annual_spillover_effect WHERE event_id = ?", (event_id,))
                    for edge in SPILLOVER_EDGES:
                        if edge.source_symbol != company.symbol:
                            continue
                        spill = round(mean_sent * edge.edge_weight, 6)
                        conn.execute(
                            """
                            INSERT INTO annual_spillover_effect (
                                event_id, source_symbol, target_symbol, edge_weight,
                                spillover_score, rationale
                            ) VALUES (?, ?, ?, ?, ?, ?)
                            """,
                            (
                                event_id,
                                edge.source_symbol,
                                edge.target_symbol,
                                edge.edge_weight,
                                spill,
                                edge.rationale,
                            ),
                        )
                        spillover_written += 1

        return {
            "run_at_utc": now_iso,
            "years_back": years_back,
            "pre_days": pre_days,
            "post_days": post_days,
            "max_news_per_event": max_news_per_event,
            "symbols_processed": symbols_processed,
            "events_written": events_written,
            "news_written": news_written,
            "spillover_written": spillover_written,
            "provider": self.settings.market_data_provider.value,
            "news_provider": "eodhd_news",
        }

    def list_events(self, *, limit: int = 100) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    event_id, symbol, event_year, anchor_date, pre_window_start, pre_window_end,
                    post_window_end, pre_return_pct, post_return_pct, target_sign,
                    news_count, mean_sentiment, ai_term_hits, created_at
                FROM annual_event
                ORDER BY anchor_date DESC, symbol ASC
                LIMIT ?
                """,
                (max(1, min(limit, 1000)),),
            ).fetchall()
        return [dict(r) for r in rows]

    def list_spillovers(self, *, event_id: str | None = None, limit: int = 200) -> list[dict[str, Any]]:
        with self._connect() as conn:
            if event_id:
                rows = conn.execute(
                    """
                    SELECT
                        event_id, source_symbol, target_symbol, edge_weight,
                        spillover_score, rationale
                    FROM annual_spillover_effect
                    WHERE event_id = ?
                    ORDER BY ABS(spillover_score) DESC
                    LIMIT ?
                    """,
                    (event_id, max(1, min(limit, 2000))),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT
                        event_id, source_symbol, target_symbol, edge_weight,
                        spillover_score, rationale
                    FROM annual_spillover_effect
                    ORDER BY row_id DESC
                    LIMIT ?
                    """,
                    (max(1, min(limit, 2000)),),
                ).fetchall()
        return [dict(r) for r in rows]
