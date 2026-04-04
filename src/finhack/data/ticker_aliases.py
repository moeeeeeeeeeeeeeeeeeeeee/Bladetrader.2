"""Ticker-to-alias mappings for better document entity matching."""

from __future__ import annotations

from typing import Final

# Keep this focused on AI-relevant names first; expand as portfolio grows.
TICKER_ALIASES: Final[dict[str, tuple[str, ...]]] = {
    "NVDA": ("nvidia", "nvidia corporation", "geforce", "cuda"),
    "MSFT": ("microsoft", "microsoft corporation", "azure"),
    "GOOGL": ("alphabet", "google", "google llc", "google cloud"),
    "GOOG": ("alphabet", "google", "google llc", "google cloud"),
    "AMZN": ("amazon", "amazon.com", "aws", "amazon web services"),
    "META": ("meta", "meta platforms", "facebook", "instagram", "whatsapp"),
    "AAPL": ("apple", "apple inc"),
    "TSM": ("tsmc", "taiwan semiconductor", "taiwan semiconductor manufacturing"),
    "ASML": ("asml", "asml holding"),
    "AMD": ("amd", "advanced micro devices"),
    "AVGO": ("broadcom", "broadcom inc"),
    "QCOM": ("qualcomm", "qualcomm inc"),
    "INTC": ("intel", "intel corporation"),
    "MU": ("micron", "micron technology"),
    "ANET": ("arista", "arista networks"),
    "SNOW": ("snowflake", "snowflake inc"),
    "NOW": ("servicenow", "service now"),
    "PLTR": ("palantir", "palantir technologies"),
    "CRM": ("salesforce", "salesforce inc"),
    "ORCL": ("oracle", "oracle corporation"),
    "IBM": ("ibm", "international business machines"),
    "ADBE": ("adobe", "adobe inc"),
    "PANW": ("palo alto networks", "palo alto"),
    "CRWD": ("crowdstrike", "crowdstrike holdings"),
    "SMCI": ("super micro computer", "supermicro", "smci"),
    "DELL": ("dell", "dell technologies"),
    "TSLA": ("tesla", "tesla inc"),
    "ARM": ("arm holdings", "arm"),
}


def get_default_aliases(symbol: str) -> list[str]:
    """Return built-in aliases for a ticker symbol."""
    return list(TICKER_ALIASES.get(symbol.upper(), ()))

