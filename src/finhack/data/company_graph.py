"""Core company/sector mapping used by market + sector intelligence agents."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Company:
    symbol: str
    name: str
    sector_bucket: str


SECTOR_BUCKETS: tuple[str, ...] = (
    "AI Compute",
    "Cloud & Hyperscalers",
    "AI Models & Platforms",
    "Data & Analytics Layer",
    "Enterprise AI Applications",
    "AI-Driven Consumer Platforms",
    "AI Physical Infrastructure",
    "AI-Enabled Industries",
)


CASE4_COMPANIES: tuple[Company, ...] = (
    Company("NVDA", "NVIDIA", "AI Compute"),
    Company("MSFT", "Microsoft", "Cloud & Hyperscalers"),
    Company("GOOGL", "Alphabet", "AI Models & Platforms"),
    Company("AMZN", "Amazon", "Cloud & Hyperscalers"),
    Company("META", "Meta Platforms", "AI-Driven Consumer Platforms"),
    Company("AMD", "Advanced Micro Devices", "AI Compute"),
    Company("AVGO", "Broadcom", "AI Physical Infrastructure"),
    Company("TSM", "Taiwan Semiconductor", "AI Physical Infrastructure"),
    Company("ASML", "ASML Holding", "AI Physical Infrastructure"),
    Company("ANET", "Arista Networks", "AI Physical Infrastructure"),
    Company("SMCI", "Super Micro Computer", "AI Physical Infrastructure"),
    Company("PLTR", "Palantir", "Data & Analytics Layer"),
    Company("ORCL", "Oracle", "Enterprise AI Applications"),
    Company("CRM", "Salesforce", "Enterprise AI Applications"),
)


CASE4_SYMBOLS: tuple[str, ...] = tuple(c.symbol for c in CASE4_COMPANIES)

SYMBOL_TO_COMPANY: dict[str, Company] = {c.symbol: c for c in CASE4_COMPANIES}

SPILLOVER_MAP: dict[str, list[str]] = {
    "NVDA": ["AMD", "AVGO", "TSM", "SMCI"],
    "MSFT": ["AMZN", "GOOGL", "ORCL", "CRM"],
    "GOOGL": ["MSFT", "AMZN", "META"],
    "AMZN": ["MSFT", "GOOGL", "ORCL"],
    "META": ["GOOGL", "MSFT", "AMZN"],
    "AMD": ["NVDA", "AVGO", "TSM", "ASML"],
    "AVGO": ["NVDA", "AMD", "ANET", "TSM"],
    "TSM": ["NVDA", "AMD", "ASML", "AVGO"],
    "ASML": ["TSM", "NVDA", "AMD"],
    "ANET": ["MSFT", "NVDA", "AVGO"],
    "SMCI": ["NVDA", "AMD", "ANET"],
    "PLTR": ["MSFT", "GOOGL", "AMZN"],
    "ORCL": ["MSFT", "AMZN", "CRM"],
    "CRM": ["MSFT", "ORCL", "AMZN"],
}
