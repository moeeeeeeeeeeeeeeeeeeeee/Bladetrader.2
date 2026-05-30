"""Environment-driven settings (EODHD market data, paths, news flags)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum


class MarketDataProvider(str, Enum):
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


def _normalize_database_url(raw: str) -> str:
    val = (raw or "").strip()
    if not val:
        return "data/finhack.db"
    # Keep sqlite URL/file paths; reject unsupported remote DB URLs for now.
    if val.startswith("sqlite:///") or val.startswith("sqlite://"):
        return val
    if "://" in val:
        return "data/finhack.db"
    return val


def _get_env_int(key: str, default: int) -> int:
    val = _get_env(key)
    if val is None:
        return default
    try:
        return max(0, int(val))
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    market_data_provider: MarketDataProvider
    eodhd_api_key: str | None
    market_data_cache_ttl_seconds: int
    market_data_live_cache_ttl_seconds: int
    market_data_symbols_cache_ttl_seconds: int
    data_dir: str
    gnews_api_key: str | None
    database_url: str
    news_trusted_sources_only: bool
    news_require_gnews: bool
    news_require_primary_api: bool
    news_enable_gdelt: bool
    news_enable_rss_fallback: bool
    news_enable_eodhd: bool
    trading_universe_limit: int
    news_backfill_days: int

    @classmethod
    def load(cls) -> Settings:
        # EODHD is the only supported provider. The variable is kept so that
        # any legacy MARKET_DATA_PROVIDER value in old .env files is harmless.
        return cls(
            market_data_provider=MarketDataProvider.EODHD,
            eodhd_api_key=_get_env("EODHD_API_KEY"),
            market_data_cache_ttl_seconds=_get_env_int("MARKET_DATA_CACHE_TTL_SECONDS", 300),
            market_data_live_cache_ttl_seconds=_get_env_int("MARKET_DATA_LIVE_CACHE_TTL_SECONDS", 15),
            market_data_symbols_cache_ttl_seconds=_get_env_int(
                "MARKET_DATA_SYMBOLS_CACHE_TTL_SECONDS", 3600
            ),
            data_dir=_get_env("DATA_DIR", "data") or "data",
            gnews_api_key=_get_env("GNEWS_API_KEY"),
            database_url=_normalize_database_url(
                _get_env("DATABASE_URL", "data/finhack.db") or "data/finhack.db"
            ),
            news_trusted_sources_only=_get_env_bool("NEWS_TRUSTED_SOURCES_ONLY", True),
            news_require_gnews=_get_env_bool("NEWS_REQUIRE_GNEWS", False),
            news_require_primary_api=_get_env_bool("NEWS_REQUIRE_PRIMARY_API", True),
            news_enable_gdelt=_get_env_bool("NEWS_ENABLE_GDELT", True),
            news_enable_rss_fallback=_get_env_bool("NEWS_ENABLE_RSS_FALLBACK", True),
            news_enable_eodhd=_get_env_bool("NEWS_ENABLE_EODHD", True),
            trading_universe_limit=_get_env_int("TRADING_UNIVERSE_LIMIT", 500),
            news_backfill_days=_get_env_int("NEWS_BACKFILL_DAYS", 730),
        )


def load_settings() -> Settings:
    """Call after optional load_dotenv() from main or scripts."""
    return Settings.load()
