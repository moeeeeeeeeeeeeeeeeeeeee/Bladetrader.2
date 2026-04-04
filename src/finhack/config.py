"""Environment-driven settings (Yahoo vs EODHD, paths)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum


class MarketDataProvider(str, Enum):
    YAHOO = "yahoo"
    EODHD = "eodhd"


def _get_env(key: str, default: str | None = None) -> str | None:
    val = os.getenv(key)
    if val is None or val.strip() == "":
        return default
    return val.strip()


def _get_env_bool(key: str, default: bool) -> bool:
    val = _get_env(key)
    if val is None:
        return default
    normalized = val.strip().lower()
    return normalized in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    market_data_provider: MarketDataProvider
    eodhd_api_key: str | None
    data_dir: str
    gnews_api_key: str | None
    database_url: str
    news_trusted_sources_only: bool
    news_require_gnews: bool
    news_require_primary_api: bool
    news_enable_gdelt: bool
    news_enable_rss_fallback: bool
    ticker_aliases_path: str | None

    @classmethod
    def load(cls) -> Settings:
        raw = (_get_env("MARKET_DATA_PROVIDER", "yahoo") or "yahoo").lower()
        try:
            provider = MarketDataProvider(raw)
        except ValueError:
            provider = MarketDataProvider.YAHOO
        return cls(
            market_data_provider=provider,
            eodhd_api_key=_get_env("EODHD_API_KEY"),
            data_dir=_get_env("DATA_DIR", "data") or "data",
            gnews_api_key=_get_env("GNEWS_API_KEY"),
            database_url=_get_env("DATABASE_URL", "data/finhack.db") or "data/finhack.db",
            news_trusted_sources_only=_get_env_bool("NEWS_TRUSTED_SOURCES_ONLY", True),
            news_require_gnews=_get_env_bool("NEWS_REQUIRE_GNEWS", True),
            news_require_primary_api=_get_env_bool("NEWS_REQUIRE_PRIMARY_API", True),
            news_enable_gdelt=_get_env_bool("NEWS_ENABLE_GDELT", True),
            news_enable_rss_fallback=_get_env_bool("NEWS_ENABLE_RSS_FALLBACK", True),
            ticker_aliases_path=_get_env("TICKER_ALIASES_PATH"),
        )


def load_settings() -> Settings:
    """Call after optional load_dotenv() from main or scripts."""
    return Settings.load()
