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
    "Cybersecurity",
)


CASE4_COMPANIES: tuple[Company, ...] = (
    # ── Original Case-4 core (14) ──
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
    # ── Expanded AI universe (22) ──
    Company("AAPL", "Apple", "AI-Driven Consumer Platforms"),
    Company("INTC", "Intel", "AI Compute"),
    Company("MU", "Micron Technology", "AI Compute"),
    Company("MRVL", "Marvell Technology", "AI Compute"),
    Company("QCOM", "Qualcomm", "AI Compute"),
    Company("SNOW", "Snowflake", "Data & Analytics Layer"),
    Company("MDB", "MongoDB", "Data & Analytics Layer"),
    Company("ADBE", "Adobe", "Enterprise AI Applications"),
    Company("NOW", "ServiceNow", "Enterprise AI Applications"),
    Company("TEAM", "Atlassian", "Enterprise AI Applications"),
    Company("NET", "Cloudflare", "Cloud & Hyperscalers"),
    Company("DELL", "Dell Technologies", "AI Physical Infrastructure"),
    Company("SNPS", "Synopsys", "AI Physical Infrastructure"),
    Company("CDNS", "Cadence Design Systems", "AI Physical Infrastructure"),
    Company("CRWD", "CrowdStrike", "Cybersecurity"),
    Company("PANW", "Palo Alto Networks", "Cybersecurity"),
    Company("EQIX", "Equinix", "AI Physical Infrastructure"),
    Company("NEE", "NextEra Energy", "AI-Enabled Industries"),
    Company("TSLA", "Tesla", "AI-Enabled Industries"),
    Company("JPM", "JPMorgan Chase", "AI-Enabled Industries"),
    Company("IBM", "IBM", "Enterprise AI Applications"),
    Company("CSCO", "Cisco Systems", "AI Physical Infrastructure"),
)


CASE4_SYMBOLS: tuple[str, ...] = tuple(c.symbol for c in CASE4_COMPANIES)

SYMBOL_TO_COMPANY: dict[str, Company] = {c.symbol: c for c in CASE4_COMPANIES}

# Directed spillover peers (used for news mention counting + network UI).
SPILLOVER_MAP: dict[str, list[str]] = {
    "NVDA": ["AMD", "AVGO", "TSM", "SMCI", "MU", "MRVL"],
    "MSFT": ["AMZN", "GOOGL", "ORCL", "CRM", "NOW", "NET"],
    "GOOGL": ["MSFT", "AMZN", "META", "AAPL"],
    "AMZN": ["MSFT", "GOOGL", "ORCL", "CRM", "NET"],
    "META": ["GOOGL", "MSFT", "AMZN", "AAPL"],
    "AMD": ["NVDA", "AVGO", "TSM", "ASML", "INTC", "MU"],
    "AVGO": ["NVDA", "AMD", "ANET", "TSM", "MRVL"],
    "TSM": ["NVDA", "AMD", "ASML", "AVGO"],
    "ASML": ["TSM", "NVDA", "AMD", "SNPS", "CDNS"],
    "ANET": ["MSFT", "NVDA", "AVGO", "CSCO", "NET"],
    "SMCI": ["NVDA", "AMD", "ANET", "DELL"],
    "PLTR": ["MSFT", "GOOGL", "AMZN", "SNOW", "CRM"],
    "ORCL": ["MSFT", "AMZN", "CRM", "NOW", "IBM"],
    "CRM": ["MSFT", "ORCL", "AMZN", "NOW", "TEAM"],
    "AAPL": ["GOOGL", "META", "QCOM", "TSM"],
    "INTC": ["AMD", "NVDA", "TSM", "MU"],
    "MU": ["NVDA", "AMD", "TSM", "AVGO"],
    "MRVL": ["NVDA", "AVGO", "AMD", "QCOM"],
    "QCOM": ["AAPL", "NVDA", "AVGO", "MRVL"],
    "SNOW": ["PLTR", "CRM", "MSFT", "MDB"],
    "MDB": ["SNOW", "CRM", "AMZN", "NOW"],
    "ADBE": ["CRM", "NOW", "MSFT", "ORCL"],
    "NOW": ["CRM", "MSFT", "ORCL", "TEAM"],
    "TEAM": ["CRM", "NOW", "MSFT"],
    "NET": ["AMZN", "MSFT", "CRWD", "PANW"],
    "DELL": ["SMCI", "NVDA", "AMD", "ANET"],
    "SNPS": ["ASML", "NVDA", "AMD", "CDNS"],
    "CDNS": ["SNPS", "ASML", "NVDA"],
    "CRWD": ["PANW", "NET", "MSFT", "GOOGL"],
    "PANW": ["CRWD", "NET", "CSCO", "MSFT"],
    "EQIX": ["NVDA", "MSFT", "AMZN", "NEE"],
    "NEE": ["NVDA", "EQIX", "TSM"],
    "TSLA": ["NVDA", "AMD", "QCOM"],
    "JPM": ["PLTR", "MSFT", "GOOGL"],
    "IBM": ["ORCL", "MSFT", "CRM", "NOW"],
    "CSCO": ["ANET", "PANW", "MSFT", "NET"],
}

# Weighted spillover edges for feature scoring (source → target, weight 0–1).
SPILLOVER_WEIGHTS: dict[str, list[tuple[str, float]]] = {
    "NVDA": [("AMD", 0.85), ("TSM", 0.80), ("AVGO", 0.72), ("SMCI", 0.65), ("MU", 0.60)],
    "MSFT": [("AMZN", 0.90), ("GOOGL", 0.88), ("CRM", 0.65), ("ORCL", 0.62), ("NOW", 0.58)],
    "AMD": [("NVDA", 0.75), ("TSM", 0.70), ("INTC", 0.55), ("MU", 0.50)],
    "CRWD": [("PANW", 0.78), ("NET", 0.55), ("MSFT", 0.50)],
    "PLTR": [("MSFT", 0.70), ("SNOW", 0.55), ("JPM", 0.50)],
    "EQIX": [("NVDA", 0.72), ("MSFT", 0.60), ("NEE", 0.60)],
}

# Sector-level inverse ETF / hedge candidates (liquid US listings).
SECTOR_HEDGE_CANDIDATES: dict[str, tuple[str, ...]] = {
    "AI Compute": ("SOXS", "PSQ", "SH"),
    "Cloud & Hyperscalers": ("PSQ", "SQQQ", "SH"),
    "AI Models & Platforms": ("PSQ", "SQQQ", "SH"),
    "Data & Analytics Layer": ("PSQ", "SH", "XLK"),
    "Enterprise AI Applications": ("PSQ", "SH", "XLK"),
    "AI-Driven Consumer Platforms": ("PSQ", "SQQQ", "SH"),
    "AI Physical Infrastructure": ("SOXS", "PSQ", "SH"),
    "AI-Enabled Industries": ("PSQ", "SH", "IWM"),
    "Cybersecurity": ("HACK", "CIBR", "PSQ"),
}
