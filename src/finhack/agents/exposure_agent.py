"""
Agent 2: determine stock exposure to AI-market events from Agent 1 documents.

This module reads saved `document` rows and estimates:
- direct vs spillover exposure for a stock
- likely direction (bullish / bearish / mixed)
- confidence and key supporting documents
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from finhack.config import Settings, load_settings
from finhack.data.ticker_aliases import get_default_aliases


POSITIVE_IMPACT_TERMS: tuple[str, ...] = (
    "growth",
    "beat",
    "upgrade",
    "surge",
    "expansion",
    "demand",
    "investment",
    "partnership",
    "strong guidance",
    "productivity gains",
    "record revenue",
)

NEGATIVE_IMPACT_TERMS: tuple[str, ...] = (
    "slowdown",
    "downgrade",
    "shortage",
    "delay",
    "ban",
    "regulatory pressure",
    "compliance risk",
    "investigation",
    "miss",
    "weak guidance",
    "supply constraint",
    "disruption",
)

THEME_KEYWORDS: dict[str, tuple[str, ...]] = {
    "chips_compute": (
        "gpu demand",
        "semiconductor demand",
        "ai accelerators",
        "tpu",
        "asic",
        "hbm memory",
        "chip shortage",
        "compute capacity",
        "high performance computing",
    ),
    "cloud_infra": (
        "cloud spending",
        "cloud growth",
        "aws growth",
        "azure growth",
        "google cloud growth",
        "data center",
        "hyperscaler",
        "server investment",
        "capital expenditure",
        "capex",
    ),
    "policy_regulation": (
        "ai regulation",
        "ai policy",
        "ai legislation",
        "regulatory framework",
        "ai governance",
        "ai compliance",
        "government ai",
    ),
    "enterprise_adoption": (
        "ai adoption",
        "enterprise ai",
        "automation",
        "ai productivity",
        "ai revenue",
        "ai demand",
        "ai monetization",
        "productivity gains",
    ),
    "energy_power": (
        "energy demand",
        "electricity usage",
        "grid capacity",
        "power infrastructure",
        "data center energy",
    ),
    "labor_macro": (
        "workforce automation",
        "labor displacement",
        "future of work",
        "economic transformation",
        "digital economy",
    ),
}

SECTOR_THEME_MAP: dict[str, tuple[str, ...]] = {
    "semiconductors": ("chips_compute",),
    "hardware": ("chips_compute", "cloud_infra"),
    "cloud": ("cloud_infra",),
    "software": ("enterprise_adoption", "cloud_infra"),
    "platforms": ("enterprise_adoption", "cloud_infra"),
    "internet": ("enterprise_adoption",),
    "energy": ("energy_power",),
    "utilities": ("energy_power",),
    "financials": ("policy_regulation", "enterprise_adoption"),
    "fintech": ("policy_regulation", "enterprise_adoption"),
    "industrials": ("enterprise_adoption", "labor_macro"),
}


@dataclass(slots=True)
class StockProfile:
    symbol: str
    company_name: str | None = None
    sector: str | None = None
    aliases: list[str] | None = None
    ai_themes: list[str] | None = None


@dataclass(slots=True)
class ExposureDriver:
    doc_id: str
    title: str
    url: str
    source: str
    published_at: str | None
    impact_direction: str
    impact_score: float
    direct_mention: bool
    matched_themes: list[str]
    why: str


@dataclass(slots=True)
class ExposureAnalysis:
    symbol: str
    looked_back_hours: int
    documents_considered: int
    direct_mentions: int
    spillover_mentions: int
    exposure_score: float
    direct_exposure_score: float
    spillover_exposure_score: float
    impact_direction: str
    confidence: float
    matched_themes: list[str]
    top_drivers: list[ExposureDriver]


def _normalize_text(value: str | None) -> str:
    if not value:
        return ""
    return re.sub(r"\s+", " ", value).strip().lower()


def _parse_iso_or_none(value: str | None) -> datetime | None:
    if not value:
        return None
    raw = value.strip()
    if not raw:
        return None
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _bounded(value: float, lo: float, hi: float) -> float:
    return max(lo, min(value, hi))


class ExposureAgent:
    """Agent 2 implementation using Agent 1 ingested document corpus."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or load_settings()
        self._db_path = self._resolve_db_path()
        self._custom_aliases = self._load_custom_aliases()

    def _resolve_db_path(self) -> str:
        db_path = self.settings.database_url
        if db_path.startswith("sqlite:///"):
            db_path = db_path.replace("sqlite:///", "", 1)
        if os.path.isabs(db_path):
            out = db_path
        else:
            out = os.path.abspath(db_path.replace("\\", "/"))
        return out

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _load_custom_aliases(self) -> dict[str, list[str]]:
        path = self.settings.ticker_aliases_path
        if not path:
            return {}
        resolved = path if os.path.isabs(path) else os.path.abspath(path.replace("\\", "/"))
        if not os.path.exists(resolved):
            return {}
        try:
            with open(resolved, encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:  # noqa: BLE001
            return {}
        if not isinstance(payload, dict):
            return {}
        out: dict[str, list[str]] = {}
        for key, value in payload.items():
            symbol = str(key).strip().upper()
            if not symbol:
                continue
            if isinstance(value, list):
                aliases = [str(v).strip().lower() for v in value if str(v).strip()]
            else:
                aliases = []
            if aliases:
                out[symbol] = aliases
        return out

    def _profile_terms(self, profile: StockProfile) -> list[str]:
        terms = [profile.symbol.lower()]
        default_aliases = get_default_aliases(profile.symbol)
        custom_aliases = self._custom_aliases.get(profile.symbol.upper(), [])
        terms.extend(default_aliases)
        terms.extend(custom_aliases)
        if profile.company_name:
            terms.append(profile.company_name.lower())
            for token in re.split(r"[\s\-/(),.&]+", profile.company_name.lower()):
                if len(token) >= 4:
                    terms.append(token)
        if profile.aliases:
            for alias in profile.aliases:
                alias_n = alias.strip().lower()
                if alias_n:
                    terms.append(alias_n)
        return sorted(set(t for t in terms if t))

    def _derive_themes(self, profile: StockProfile) -> set[str]:
        themes = set()
        if profile.ai_themes:
            themes.update(t.strip().lower() for t in profile.ai_themes if t.strip())
        if profile.sector:
            sector = profile.sector.strip().lower()
            for key, mapped in SECTOR_THEME_MAP.items():
                if key in sector:
                    themes.update(mapped)
        return themes

    def _keyword_hits_from_row(self, raw: str | None) -> list[str]:
        if not raw:
            return []
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return [str(v).strip().lower() for v in parsed if str(v).strip()]
        except Exception:  # noqa: BLE001
            return []
        return []

    def _impact_direction(self, text: str) -> tuple[str, float]:
        pos = 0
        neg = 0
        for term in POSITIVE_IMPACT_TERMS:
            if re.search(rf"(?<!\w){re.escape(term)}(?!\w)", text):
                pos += 1
        for term in NEGATIVE_IMPACT_TERMS:
            if re.search(rf"(?<!\w){re.escape(term)}(?!\w)", text):
                neg += 1
        if pos == 0 and neg == 0:
            return "mixed", 0.0
        net = pos - neg
        if net > 0:
            return "bullish", float(net)
        if net < 0:
            return "bearish", float(abs(net))
        return "mixed", float(pos)

    def _fetch_documents(self, *, hours_back: int, limit: int) -> list[sqlite3.Row]:
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(hours=hours_back)
        rows: list[sqlite3.Row] = []
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
                (max(1, min(limit, 1000)),),
            ).fetchall()
        out = []
        for row in rows:
            dt = _parse_iso_or_none(row["published_at"]) or _parse_iso_or_none(
                row["fetched_at"]
            )
            if dt is None or dt >= cutoff:
                out.append(row)
        return out

    def _fetch_documents_between(
        self,
        *,
        start_at: datetime,
        end_at: datetime,
        limit: int,
    ) -> list[sqlite3.Row]:
        rows: list[sqlite3.Row] = []
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
                (max(1, min(limit, 2000)),),
            ).fetchall()
        out = []
        for row in rows:
            dt = _parse_iso_or_none(row["published_at"]) or _parse_iso_or_none(
                row["fetched_at"]
            )
            if dt is None:
                continue
            if start_at <= dt <= end_at:
                out.append(row)
        return out

    def analyze_stock_exposure(
        self,
        profile: StockProfile,
        *,
        hours_back: int = 24 * 14,
        max_documents: int = 250,
        top_k: int = 8,
    ) -> ExposureAnalysis:
        profile_terms = self._profile_terms(profile)
        target_themes = self._derive_themes(profile)
        docs = self._fetch_documents(hours_back=hours_back, limit=max_documents)

        considered = 0
        direct_mentions = 0
        spillover_mentions = 0
        direct_score = 0.0
        spillover_score = 0.0
        bullish = 0.0
        bearish = 0.0
        theme_counter: dict[str, int] = {}
        drivers: list[ExposureDriver] = []

        for row in docs:
            title = row["title"] or ""
            body = row["body"] or ""
            text = _normalize_text(f"{title} {body}")
            if not text:
                continue

            matched_terms = [t for t in profile_terms if re.search(rf"(?<!\w){re.escape(t)}(?!\w)", text)]
            direct = len(matched_terms) > 0

            keyword_hits = self._keyword_hits_from_row(row["keyword_hits"])
            matched_doc_themes: list[str] = []
            for theme, kws in THEME_KEYWORDS.items():
                for kw in kws:
                    if kw in keyword_hits or re.search(rf"(?<!\w){re.escape(kw)}(?!\w)", text):
                        matched_doc_themes.append(theme)
                        break
            matched_doc_themes = sorted(set(matched_doc_themes))
            theme_overlap = set(matched_doc_themes).intersection(target_themes)

            # Drop unrelated docs to keep signal clean.
            if not direct and not theme_overlap:
                continue

            considered += 1
            base = float(row["relevance_score"] or 0.0)
            recency_weight = 1.0
            event_dt = _parse_iso_or_none(row["published_at"]) or _parse_iso_or_none(
                row["fetched_at"]
            )
            if event_dt:
                age_hours = max(
                    0.0,
                    (datetime.now(timezone.utc) - event_dt).total_seconds() / 3600.0,
                )
                recency_weight = 1.6 if age_hours < 24 else 1.25 if age_hours < 72 else 1.0

            exposure_points = base * recency_weight
            if direct:
                direct_mentions += 1
                direct_score += exposure_points * 1.6
            elif theme_overlap:
                spillover_mentions += 1
                spillover_score += exposure_points * 1.1

            direction, direction_strength = self._impact_direction(text)
            if direction == "bullish":
                bullish += direction_strength + (0.8 if direct else 0.3)
            elif direction == "bearish":
                bearish += direction_strength + (0.8 if direct else 0.3)
            else:
                bullish += 0.2
                bearish += 0.2

            for th in matched_doc_themes:
                theme_counter[th] = theme_counter.get(th, 0) + 1

            why_parts = []
            if direct:
                why_parts.append("direct company mention")
            if theme_overlap:
                why_parts.append(f"theme overlap: {', '.join(sorted(theme_overlap))}")
            if direction != "mixed":
                why_parts.append(f"{direction} wording in article")
            why = "; ".join(why_parts) if why_parts else "AI-event relevance match"

            impact_score = round(exposure_points * (1.4 if direct else 1.0), 2)
            drivers.append(
                ExposureDriver(
                    doc_id=row["doc_id"],
                    title=title,
                    url=row["url"],
                    source=row["source"] or "unknown",
                    published_at=row["published_at"],
                    impact_direction=direction,
                    impact_score=impact_score,
                    direct_mention=direct,
                    matched_themes=matched_doc_themes,
                    why=why,
                )
            )

        total_score = direct_score + spillover_score
        exposure_score = _bounded(round(total_score, 2), 0.0, 100.0)
        direct_exposure = _bounded(round(direct_score, 2), 0.0, 100.0)
        spillover_exposure = _bounded(round(spillover_score, 2), 0.0, 100.0)

        if bullish == 0 and bearish == 0:
            impact_direction = "mixed"
        elif bullish > bearish * 1.2:
            impact_direction = "bullish"
        elif bearish > bullish * 1.2:
            impact_direction = "bearish"
        else:
            impact_direction = "mixed"

        signal_strength = abs(bullish - bearish)
        coverage = min(1.0, considered / 12.0)
        confidence = _bounded(round((signal_strength * 8.0 + coverage * 35.0), 2), 5.0, 99.0)

        top_themes = sorted(theme_counter.items(), key=lambda x: x[1], reverse=True)
        matched_themes = [name for name, _ in top_themes[:6]]
        top_drivers = sorted(drivers, key=lambda d: d.impact_score, reverse=True)[: max(1, min(top_k, 15))]

        return ExposureAnalysis(
            symbol=profile.symbol.upper(),
            looked_back_hours=hours_back,
            documents_considered=considered,
            direct_mentions=direct_mentions,
            spillover_mentions=spillover_mentions,
            exposure_score=exposure_score,
            direct_exposure_score=direct_exposure,
            spillover_exposure_score=spillover_exposure,
            impact_direction=impact_direction,
            confidence=confidence,
            matched_themes=matched_themes,
            top_drivers=top_drivers,
        )

    def analyze_stock_exposure_at(
        self,
        profile: StockProfile,
        *,
        anchor_at: datetime,
        lookback_days: int = 7,
        max_documents: int = 500,
        top_k: int = 8,
    ) -> ExposureAnalysis:
        anchor = anchor_at.astimezone(timezone.utc)
        start = anchor - timedelta(days=max(1, lookback_days))
        profile_terms = self._profile_terms(profile)
        target_themes = self._derive_themes(profile)
        docs = self._fetch_documents_between(
            start_at=start,
            end_at=anchor,
            limit=max_documents,
        )

        considered = 0
        direct_mentions = 0
        spillover_mentions = 0
        direct_score = 0.0
        spillover_score = 0.0
        bullish = 0.0
        bearish = 0.0
        theme_counter: dict[str, int] = {}
        drivers: list[ExposureDriver] = []

        for row in docs:
            title = row["title"] or ""
            body = row["body"] or ""
            text = _normalize_text(f"{title} {body}")
            if not text:
                continue

            matched_terms = [
                t
                for t in profile_terms
                if re.search(rf"(?<!\w){re.escape(t)}(?!\w)", text)
            ]
            direct = len(matched_terms) > 0

            keyword_hits = self._keyword_hits_from_row(row["keyword_hits"])
            matched_doc_themes: list[str] = []
            for theme, kws in THEME_KEYWORDS.items():
                for kw in kws:
                    if kw in keyword_hits or re.search(rf"(?<!\w){re.escape(kw)}(?!\w)", text):
                        matched_doc_themes.append(theme)
                        break
            matched_doc_themes = sorted(set(matched_doc_themes))
            theme_overlap = set(matched_doc_themes).intersection(target_themes)

            if not direct and not theme_overlap:
                continue

            considered += 1
            base = float(row["relevance_score"] or 0.0)
            event_dt = _parse_iso_or_none(row["published_at"]) or _parse_iso_or_none(
                row["fetched_at"]
            )
            recency_weight = 1.0
            if event_dt:
                age_hours = max(0.0, (anchor - event_dt).total_seconds() / 3600.0)
                recency_weight = 1.4 if age_hours < 24 else 1.15 if age_hours < 72 else 1.0

            exposure_points = base * recency_weight
            if direct:
                direct_mentions += 1
                direct_score += exposure_points * 1.6
            elif theme_overlap:
                spillover_mentions += 1
                spillover_score += exposure_points * 1.1

            direction, direction_strength = self._impact_direction(text)
            if direction == "bullish":
                bullish += direction_strength + (0.8 if direct else 0.3)
            elif direction == "bearish":
                bearish += direction_strength + (0.8 if direct else 0.3)
            else:
                bullish += 0.2
                bearish += 0.2

            for th in matched_doc_themes:
                theme_counter[th] = theme_counter.get(th, 0) + 1

            why_parts = []
            if direct:
                why_parts.append("direct company mention")
            if theme_overlap:
                why_parts.append(f"theme overlap: {', '.join(sorted(theme_overlap))}")
            if direction != "mixed":
                why_parts.append(f"{direction} wording in article")
            why = "; ".join(why_parts) if why_parts else "AI-event relevance match"

            impact_score = round(exposure_points * (1.4 if direct else 1.0), 2)
            drivers.append(
                ExposureDriver(
                    doc_id=row["doc_id"],
                    title=title,
                    url=row["url"],
                    source=row["source"] or "unknown",
                    published_at=row["published_at"],
                    impact_direction=direction,
                    impact_score=impact_score,
                    direct_mention=direct,
                    matched_themes=matched_doc_themes,
                    why=why,
                )
            )

        total_score = direct_score + spillover_score
        exposure_score = _bounded(round(total_score, 2), 0.0, 100.0)
        direct_exposure = _bounded(round(direct_score, 2), 0.0, 100.0)
        spillover_exposure = _bounded(round(spillover_score, 2), 0.0, 100.0)

        if bullish == 0 and bearish == 0:
            impact_direction = "mixed"
        elif bullish > bearish * 1.2:
            impact_direction = "bullish"
        elif bearish > bullish * 1.2:
            impact_direction = "bearish"
        else:
            impact_direction = "mixed"

        signal_strength = abs(bullish - bearish)
        coverage = min(1.0, considered / 12.0)
        confidence = _bounded(
            round((signal_strength * 8.0 + coverage * 35.0), 2),
            5.0,
            99.0,
        )

        top_themes = sorted(theme_counter.items(), key=lambda x: x[1], reverse=True)
        matched_themes = [name for name, _ in top_themes[:6]]
        top_drivers = sorted(drivers, key=lambda d: d.impact_score, reverse=True)[: max(1, min(top_k, 15))]

        return ExposureAnalysis(
            symbol=profile.symbol.upper(),
            looked_back_hours=lookback_days * 24,
            documents_considered=considered,
            direct_mentions=direct_mentions,
            spillover_mentions=spillover_mentions,
            exposure_score=exposure_score,
            direct_exposure_score=direct_exposure,
            spillover_exposure_score=spillover_exposure,
            impact_direction=impact_direction,
            confidence=confidence,
            matched_themes=matched_themes,
            top_drivers=top_drivers,
        )

