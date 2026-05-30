"""Agent 3/4: sector intelligence over 5y price + AI news patterns."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
import math
import sqlite3
from typing import Any

from finhack.config import Settings, load_settings
from finhack.data.company_graph import CASE4_SYMBOLS, SECTOR_BUCKETS, SPILLOVER_MAP, SYMBOL_TO_COMPANY
from finhack.market_data import get_close_series


SECTOR_KEYWORDS: dict[str, tuple[str, ...]] = {
    "AI Compute": ("gpu", "accelerator", "chip", "semiconductor", "cuda"),
    "Cloud & Hyperscalers": ("cloud", "hyperscaler", "azure", "aws", "gcp"),
    "AI Models & Platforms": ("model", "llm", "inference", "openai", "anthropic"),
    "Data & Analytics Layer": ("data", "analytics", "vector", "pipeline", "warehouse"),
    "Enterprise AI Applications": ("enterprise", "crm", "workflow", "copilot", "saas"),
    "AI-Driven Consumer Platforms": ("consumer", "ads", "social", "search", "assistant"),
    "AI Physical Infrastructure": ("server", "network", "datacenter", "fabric", "hbm"),
    "AI-Enabled Industries": ("healthcare", "finance", "manufacturing", "robotics", "automation"),
}

POSITIVE_TERMS = (
    "beats", "growth", "upgrade", "record", "expands", "strong", "surge", "bullish"
)
NEGATIVE_TERMS = (
    "miss", "cut", "downgrade", "weak", "decline", "lawsuit", "delay", "bearish"
)


@dataclass(slots=True)
class CompanyImpact:
    symbol: str
    company_name: str
    role: str
    current_price: float | None
    predicted_direction: str
    predicted_move_pct: float
    correlation_to_sector: float
    leverage_or_hedge: str
    rationale: str
    connected_to: str | None


@dataclass(slots=True)
class SectorPrediction:
    sector: str
    horizon_days: int
    predicted_sector_move_pct: float
    confidence: float
    metric_a_news_pressure: float
    metric_b_correlation_strength: float
    metric_c_content_impact: float
    metric_d_network_spillover: float
    news_articles_7d: int
    top_owned: list[CompanyImpact]
    movers_not_owned: list[CompanyImpact]
    generated_at_utc: str


@dataclass(slots=True)
class UniversePrediction:
    horizon_days: int
    confidence: float
    news_articles_7d: int
    stocks: list[CompanyImpact]
    sector_summary: list[dict[str, Any]]
    generated_at_utc: str


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _sign(value: float, dead_zone: float = 0.05) -> int:
    if value > dead_zone:
        return 1
    if value < -dead_zone:
        return -1
    return 0


def _direction(value: float) -> str:
    s = _sign(value)
    if s > 0:
        return "up"
    if s < 0:
        return "down"
    return "flat"


def _corr(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2 or len(xs) != len(ys):
        return 0.0
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    deny = math.sqrt(sum((y - my) ** 2 for y in ys))
    den = denx * deny
    if den == 0:
        return 0.0
    return num / den


class SectorIntelligenceAgent:
    """Pattern engine for sector-level 5-7 day movement from Agent 1 news."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or load_settings()
        self._db_path = self._resolve_db_path(self.settings.database_url)

    def _resolve_db_path(self, database_url: str) -> str:
        raw = database_url
        if raw.startswith("sqlite:///"):
            raw = raw.replace("sqlite:///", "", 1)
        return raw

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _load_recent_docs(self, days_back: int) -> list[dict[str, Any]]:
        from_dt = datetime.now(timezone.utc) - timedelta(days=max(1, days_back))
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT title, body, published_at, fetched_at, source, url
                FROM document
                WHERE COALESCE(published_at, fetched_at) >= ?
                ORDER BY COALESCE(published_at, fetched_at) DESC
                LIMIT 10000
                """,
                (from_dt.isoformat(),),
            ).fetchall()
        out: list[dict[str, Any]] = []
        for row in rows:
            out.append(
                {
                    "title": str(row["title"] or ""),
                    "body": str(row["body"] or ""),
                    "published_at": str(row["published_at"] or row["fetched_at"] or ""),
                    "source": str(row["source"] or ""),
                    "url": str(row["url"] or ""),
                }
            )
        return out

    def _price_history(self, symbol: str, years: int = 5) -> list[float]:
        to_dt = datetime.now(timezone.utc).date()
        from_dt = to_dt - timedelta(days=max(365, years * 366))
        series = get_close_series(
            symbol,
            start=from_dt.isoformat(),
            end=to_dt.isoformat(),
            settings=self.settings,
        )
        if series.empty:
            return []
        return [float(v) for v in series.dropna().tolist()]

    def _daily_returns(self, prices: list[float]) -> list[float]:
        if len(prices) < 2:
            return []
        out: list[float] = []
        for i in range(1, len(prices)):
            prev = prices[i - 1]
            cur = prices[i]
            if prev == 0:
                continue
            out.append(((cur - prev) / prev) * 100.0)
        return out

    def _score_doc(self, doc: dict[str, Any], sector: str, symbol: str) -> tuple[float, bool]:
        text = f"{doc.get('title','')} {doc.get('body','')}".lower()
        sym_hit = symbol.lower() in text
        comp = SYMBOL_TO_COMPANY.get(symbol)
        name_hit = (comp.name.lower() in text) if comp else False
        sector_terms = SECTOR_KEYWORDS.get(sector, ())
        sector_hits = sum(1 for kw in sector_terms if kw in text)
        pos_hits = sum(1 for kw in POSITIVE_TERMS if kw in text)
        neg_hits = sum(1 for kw in NEGATIVE_TERMS if kw in text)
        sentiment = float(pos_hits - neg_hits)
        score = (1.2 * sector_hits) + (1.0 if (sym_hit or name_hit) else 0.0) + sentiment
        return score, (sym_hit or name_hit)

    def predict_sector(
        self,
        *,
        sector: str,
        owned_symbols: list[str] | None = None,
        horizon_days: int = 5,
    ) -> SectorPrediction:
        safe_sector = sector if sector in SECTOR_BUCKETS else SECTOR_BUCKETS[0]
        owned = {s.strip().upper() for s in (owned_symbols or []) if s.strip()}
        if not owned:
            owned = {"NVDA", "MSFT", "AMZN"}

        docs_5y = self._load_recent_docs(days_back=365 * 5)
        docs_7d = self._load_recent_docs(days_back=7)

        company_rows: list[CompanyImpact] = []
        baseline_returns: list[float] = []
        article_hits_7d = 0

        sector_symbols = [c.symbol for c in SYMBOL_TO_COMPANY.values() if c.sector_bucket == safe_sector]
        if not sector_symbols:
            sector_symbols = list(CASE4_SYMBOLS[:3])

        sector_ret_pool: list[float] = []
        for symbol in sector_symbols:
            sector_ret_pool.extend(self._daily_returns(self._price_history(symbol, years=5)))
        sector_mean = sum(sector_ret_pool) / len(sector_ret_pool) if sector_ret_pool else 0.0

        for symbol in CASE4_SYMBOLS:
            prices = self._price_history(symbol, years=5)
            daily = self._daily_returns(prices)
            if daily:
                baseline_returns.append(sum(daily[-7:]) / max(1, len(daily[-7:])))

            hist_scores = [self._score_doc(d, safe_sector, symbol)[0] for d in docs_5y]
            recent_scores: list[float] = []
            recent_mentions = 0
            for d in docs_7d:
                s, mention = self._score_doc(d, safe_sector, symbol)
                if mention:
                    recent_mentions += 1
                if s != 0.0:
                    recent_scores.append(s)
            article_hits_7d += recent_mentions

            score_a = (sum(recent_scores) / len(recent_scores)) if recent_scores else 0.0
            hist_effect = (sum(hist_scores) / len(hist_scores)) if hist_scores else 0.0
            corr = _corr(daily[-252:], sector_ret_pool[-252:][: len(daily[-252:])]) if daily and sector_ret_pool else 0.0
            network = len(SPILLOVER_MAP.get(symbol, [])) / 4.0
            pred_move = max(-12.0, min(12.0, (0.45 * score_a) + (0.35 * hist_effect) + (1.1 * corr)))
            direction = _direction(pred_move)
            relation = "Leverage" if pred_move >= 0 else "Hedge"
            connected = SPILLOVER_MAP.get(symbol, [None])[0]
            if connected is None:
                connected = sector_symbols[0] if sector_symbols else None

            latest_price = prices[-1] if prices else None
            rationale = (
                f"Metric A/C news pressure={score_a:.2f}, Metric B corr={corr:.2f}, "
                f"Metric D network={network:.2f}."
            )
            company_rows.append(
                CompanyImpact(
                    symbol=symbol,
                    company_name=SYMBOL_TO_COMPANY[symbol].name,
                    role="node",
                    current_price=latest_price,
                    predicted_direction=direction,
                    predicted_move_pct=round(pred_move, 2),
                    correlation_to_sector=round(corr, 3),
                    leverage_or_hedge=relation,
                    rationale=rationale,
                    connected_to=connected,
                )
            )

        company_rows.sort(key=lambda r: abs(r.predicted_move_pct), reverse=True)
        top_owned = [r for r in company_rows if r.symbol in owned][:3]
        movers_not_owned = [r for r in company_rows if r.symbol not in owned][:5]

        metric_a = round(sum(abs(r.predicted_move_pct) for r in company_rows[:5]) / 5.0, 2) if company_rows else 0.0
        metric_b = round(sum(abs(r.correlation_to_sector) for r in company_rows[:5]) / 5.0, 3) if company_rows else 0.0
        metric_c = round(
            sum(max(0.0, r.predicted_move_pct) for r in company_rows[:5]) / 5.0, 2
        ) if company_rows else 0.0
        metric_d = round(sum(len(SPILLOVER_MAP.get(r.symbol, [])) for r in company_rows[:5]) / 5.0, 2) if company_rows else 0.0

        sector_move = round(
            max(-12.0, min(12.0, (0.55 * metric_a) + (2.0 * sector_mean) + (0.75 * metric_b))),
            2,
        )
        confidence = round(
            max(5.0, min(95.0, 30.0 + (min(40, article_hits_7d) * 1.1) + (metric_b * 20.0))),
            1,
        )

        return SectorPrediction(
            sector=safe_sector,
            horizon_days=max(5, min(horizon_days, 7)),
            predicted_sector_move_pct=sector_move,
            confidence=confidence,
            metric_a_news_pressure=metric_a,
            metric_b_correlation_strength=metric_b,
            metric_c_content_impact=metric_c,
            metric_d_network_spillover=metric_d,
            news_articles_7d=article_hits_7d,
            top_owned=top_owned,
            movers_not_owned=movers_not_owned,
            generated_at_utc=_now_iso(),
        )

    def predict_case4_universe(self, *, horizon_days: int = 5) -> UniversePrediction:
        """Run a full Case-4 stock pass using each symbol's native sector context."""
        docs_5y = self._load_recent_docs(days_back=365 * 5)
        docs_7d = self._load_recent_docs(days_back=7)

        price_cache: dict[str, list[float]] = {}
        daily_cache: dict[str, list[float]] = {}
        for symbol in CASE4_SYMBOLS:
            prices = self._price_history(symbol, years=5)
            price_cache[symbol] = prices
            daily_cache[symbol] = self._daily_returns(prices)

        sector_ret_cache: dict[str, list[float]] = {}
        for sector in SECTOR_BUCKETS:
            sector_symbols = [
                c.symbol for c in SYMBOL_TO_COMPANY.values() if c.sector_bucket == sector
            ]
            ret_pool: list[float] = []
            for symbol in sector_symbols:
                ret_pool.extend(daily_cache.get(symbol, []))
            sector_ret_cache[sector] = ret_pool

        rows: list[CompanyImpact] = []
        articles_7d = 0
        sector_rollup: dict[str, list[float]] = {s: [] for s in SECTOR_BUCKETS}

        for symbol in CASE4_SYMBOLS:
            comp = SYMBOL_TO_COMPANY[symbol]
            sector = comp.sector_bucket
            daily = daily_cache.get(symbol, [])
            sector_pool = sector_ret_cache.get(sector, [])

            hist_scores = [self._score_doc(d, sector, symbol)[0] for d in docs_5y]
            recent_scores: list[float] = []
            recent_mentions = 0
            for doc in docs_7d:
                score, mention = self._score_doc(doc, sector, symbol)
                if mention:
                    recent_mentions += 1
                if score != 0.0:
                    recent_scores.append(score)
            articles_7d += recent_mentions

            score_a = (sum(recent_scores) / len(recent_scores)) if recent_scores else 0.0
            hist_effect = (sum(hist_scores) / len(hist_scores)) if hist_scores else 0.0
            corr = (
                _corr(daily[-252:], sector_pool[-252:][: len(daily[-252:])])
                if daily and sector_pool
                else 0.0
            )
            network = len(SPILLOVER_MAP.get(symbol, [])) / 4.0
            pred_move = max(
                -12.0,
                min(12.0, (0.45 * score_a) + (0.35 * hist_effect) + (1.1 * corr)),
            )
            sector_rollup[sector].append(pred_move)
            direction = _direction(pred_move)
            relation = "Leverage" if pred_move >= 0 else "Hedge"
            connected = SPILLOVER_MAP.get(symbol, [None])[0]

            rows.append(
                CompanyImpact(
                    symbol=symbol,
                    company_name=comp.name,
                    role="node",
                    current_price=(price_cache.get(symbol, [])[-1] if price_cache.get(symbol) else None),
                    predicted_direction=direction,
                    predicted_move_pct=round(pred_move, 2),
                    correlation_to_sector=round(corr, 3),
                    leverage_or_hedge=relation,
                    rationale=(
                        f"Sector={sector}; Metric A/C={score_a:.2f}, "
                        f"Metric B corr={corr:.2f}, Metric D network={network:.2f}."
                    ),
                    connected_to=connected,
                )
            )

        rows.sort(key=lambda r: abs(r.predicted_move_pct), reverse=True)
        top_moves = [abs(r.predicted_move_pct) for r in rows[:5]]
        top_corr = [abs(r.correlation_to_sector) for r in rows[:5]]
        confidence = round(
            max(
                5.0,
                min(
                    95.0,
                    30.0
                    + (min(40, articles_7d) * 1.1)
                    + ((sum(top_corr) / max(1, len(top_corr))) * 20.0),
                ),
            ),
            1,
        )

        summary: list[dict[str, Any]] = []
        for sector in SECTOR_BUCKETS:
            values = sector_rollup.get(sector, [])
            if not values:
                continue
            summary.append(
                {
                    "sector": sector,
                    "predicted_move_pct": round(sum(values) / len(values), 2),
                    "stock_count": len(values),
                }
            )
        summary.sort(key=lambda x: abs(float(x["predicted_move_pct"])), reverse=True)

        return UniversePrediction(
            horizon_days=max(5, min(horizon_days, 7)),
            confidence=confidence if top_moves else 0.0,
            news_articles_7d=articles_7d,
            stocks=rows,
            sector_summary=summary,
            generated_at_utc=_now_iso(),
        )
