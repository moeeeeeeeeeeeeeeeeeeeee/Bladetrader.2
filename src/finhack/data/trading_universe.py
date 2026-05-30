"""Resolve the live trading/research symbol universe (core + EODHD catalog)."""

from __future__ import annotations

from dataclasses import dataclass

from finhack.config import Settings, load_settings
from finhack.data.company_graph import CASE4_COMPANIES, SYMBOL_TO_COMPANY
from finhack.market_data import get_market_symbols


@dataclass(frozen=True, slots=True)
class UniverseEntry:
    symbol: str
    name: str
    sector: str
    source: str


def resolve_trading_universe(
    *,
    settings: Settings | None = None,
    limit: int | None = None,
    common_stock_only: bool = True,
) -> list[UniverseEntry]:
    """
    Build symbol list for research/trading:
    1. Prioritize mapped core companies (sectors + spillover graph).
    2. Fill remaining slots from EODHD US common-stock catalog.
    """
    cfg = settings or load_settings()
    cap = limit if limit is not None else cfg.trading_universe_limit
    cap = max(10, min(cap, 5000))

    out: list[UniverseEntry] = []
    seen: set[str] = set()

    for company in CASE4_COMPANIES:
        if company.symbol in seen:
            continue
        seen.add(company.symbol)
        out.append(
            UniverseEntry(
                symbol=company.symbol,
                name=company.name,
                sector=company.sector_bucket,
                source="core",
            )
        )

    if len(out) >= cap:
        return out[:cap]

    fetch_limit = min(5000, max(cap * 3, 500))
    _, catalog = get_market_symbols(limit=fetch_limit, settings=cfg)
    for row in catalog:
        sym = (row.symbol or "").strip().upper()
        if not sym or sym in seen:
            continue
        if len(sym) > 6:
            continue
        seen.add(sym)
        comp = SYMBOL_TO_COMPANY.get(sym)
        out.append(
            UniverseEntry(
                symbol=sym,
                name=comp.name if comp else (row.company_name or sym),
                sector=comp.sector_bucket if comp else "Other AI",
                source=row.source or "catalog",
            )
        )
        if len(out) >= cap:
            break

    return out


def trading_symbols(
    *,
    settings: Settings | None = None,
    limit: int | None = None,
) -> list[str]:
    return [e.symbol for e in resolve_trading_universe(settings=settings, limit=limit)]
