"""Research utilities: dataset building, walk-forward splits, and backtesting."""

from finhack.research.case4_backtest import run_full_backtest
from finhack.research.constants import (
    EARNINGS_LIMIT_PER_SYMBOL,
    EARNINGS_RECENT_DAYS,
    NEWS_BACKFILL_DAYS,
    NEWS_INGEST_HOURS_BACK,
)

__all__ = [
    "EARNINGS_LIMIT_PER_SYMBOL",
    "EARNINGS_RECENT_DAYS",
    "NEWS_BACKFILL_DAYS",
    "NEWS_INGEST_HOURS_BACK",
    "run_full_backtest",
]
