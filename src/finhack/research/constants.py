"""Shared research pipeline defaults (live data, maximum practical history)."""

from __future__ import annotations

# ~3 years of earnings events per symbol (4 quarterly reports × ~3 years).
EARNINGS_RECENT_DAYS = 1095
EARNINGS_LIMIT_PER_SYMBOL = 24

# Default news backfill window (override with NEWS_BACKFILL_DAYS in .env).
NEWS_BACKFILL_DAYS = 730
NEWS_INGEST_HOURS_BACK = 24 * 730

# EODHD earnings API batch size (URL length safety).
EARNINGS_BATCH_CHUNK = 40

# Backtest defaults
DEFAULT_TRAIN_RATIO = 0.7
DEFAULT_ROUND_TRIP_COST_BPS = 10.0
DEFAULT_MIN_CONFIDENCE = 0.15
DEFAULT_MODEL_MIN_CONFIDENCE = 0.50
DEFAULT_MAX_POSITION_WEIGHT = 1.0
DEFAULT_MIN_POSITION_WEIGHT = 0.25
WALK_FORWARD_MIN_TRAIN_EVENTS = 48
WALK_FORWARD_TEST_CHUNK = 12

DEFAULT_TOP_K_LONG_PER_FOLD = 1
DEFAULT_TOP_K_SHORT_PER_FOLD = 1
DEFAULT_TARGET_TRADES_PER_MONTH = 3.0
DEFAULT_HOLDOUT_DAYS = 90
DEFAULT_PERMUTATION_IMPORTANCE_REPEATS = 20
DEFAULT_PERMUTATION_IMPORTANCE_SEED = 42

# --- Pre-registered overlay cutoff -------------------------------------------
# After running multiple confidence cutoffs on the historical overlay
# (data/case4_overlay_*.json), 0.80 was the lowest threshold that produced
# both (a) >= 80 signal-bearing sessions and (b) a hit rate noticeably above
# the permutation null. To avoid p-hacking this is now the pre-registered
# cutoff for any forward (live or new-data) overlay claims. Lowering this
# value for a new data sample without an out-of-sample rerun is research
# malpractice; raising it is allowed but reduces the sample.
OVERLAY_PRE_REGISTERED_MIN_CONFIDENCE = 0.80

# --- NQ / futures focus -------------------------------------------------------
# We trade MNQ/NQ (Nasdaq-100). QQQ is the OHLC proxy. Signals come ONLY from
# these mega-cap constituents — not the full 500-symbol catalog.
NQ_SIGNAL_SYMBOLS: tuple[str, ...] = (
    "NVDA", "MSFT", "AAPL", "AMZN", "META", "GOOGL", "GOOG", "AVGO",
    "TSLA", "NFLX", "ADBE", "AMD", "COST", "PEP", "CSCO", "CMCSA",
    "INTC", "TXN", "INTU", "QCOM", "AMGN", "AMAT", "BKNG", "ADI",
    "MU", "ISRG", "GILD", "REGN", "LRCX", "MDLZ",
)

# Approximate NDX index weights (Apr 2026 ballpark). Normalized over
# NQ_SIGNAL_SYMBOLS in nq_session_backtest when aggregating.
NQ_CAP_WEIGHTS: dict[str, float] = {
    "NVDA": 0.090,
    "MSFT": 0.085,
    "AAPL": 0.075,
    "AMZN": 0.055,
    "META": 0.045,
    "GOOGL": 0.030,
    "GOOG": 0.025,
    "AVGO": 0.048,
    "TSLA": 0.038,
    "NFLX": 0.028,
    "ADBE": 0.022,
    "AMD": 0.025,
    "COST": 0.024,
    "PEP": 0.018,
    "CSCO": 0.016,
    "CMCSA": 0.014,
    "INTC": 0.012,
    "TXN": 0.015,
    "INTU": 0.020,
    "QCOM": 0.018,
    "AMGN": 0.014,
    "AMAT": 0.016,
    "BKNG": 0.015,
    "ADI": 0.012,
    "MU": 0.014,
    "ISRG": 0.013,
    "GILD": 0.012,
    "REGN": 0.011,
    "LRCX": 0.013,
    "MDLZ": 0.010,
}

# Chart / TopStep instrument presets. ``proxy_symbol`` is what we fetch OHLC for.
NQ_FUTURES_INSTRUMENTS: dict[str, dict[str, str | float]] = {
    "MNQ": {
        "label": "Micro E-mini Nasdaq-100",
        "proxy_symbol": "QQQ",
        "index": "NDX",
        "point_value_usd": 2.0,
    },
    "NQ": {
        "label": "E-mini Nasdaq-100",
        "proxy_symbol": "QQQ",
        "index": "NDX",
        "point_value_usd": 20.0,
    },
    "QQQ": {
        "label": "Invesco QQQ Trust (chart proxy)",
        "proxy_symbol": "QQQ",
        "index": "NDX",
        "point_value_usd": 1.0,
    },
    "MES": {
        "label": "Micro E-mini S&P 500",
        "proxy_symbol": "SPY",
        "index": "SPX",
        "point_value_usd": 5.0,
    },
    "ES": {
        "label": "E-mini S&P 500",
        "proxy_symbol": "SPY",
        "index": "SPX",
        "point_value_usd": 50.0,
    },
}
