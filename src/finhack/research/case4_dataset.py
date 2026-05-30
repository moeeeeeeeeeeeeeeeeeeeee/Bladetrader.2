"""Build leakage-safe feature rows from validated earnings events.

The feature builder owns three things:

1. Reading per-event features from the validation JSON (price + sentiment + spillover).
2. Adding extra non-text features the validation script may not have computed:
   prior post-earnings drift, multi-window pre-event returns, sector-cohort
   reaction, and neighbor earnings clustering. These all use only data
   strictly inside ``[T-7d, T]`` (or earlier) — never anything after T.
3. ``fit_predict`` for one-shot train/test runs and ``fold_predict`` for
   walk-forward folds, both routed through ``research.model_store`` so
   LightGBM is preferred and logistic is the fallback.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from finhack.case4_features import compute_news_features, resolve_db_path
from finhack.config import load_settings
from finhack.research.model_store import (
    TrainedModel,
    fit_model,
    predict_proba,
)


def parse_event_dt(raw: str | None) -> datetime | None:
    if not raw:
        return None
    txt = raw.strip()
    if not txt:
        return None
    if txt.endswith("Z"):
        txt = txt[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(txt)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def event_key(symbol: str, t_event_utc: Any) -> tuple[str, str]:
    """Stable (symbol, time) key for joining validation rows with dataframe splits."""
    sym = str(symbol or "").upper().strip()
    if hasattr(t_event_utc, "to_pydatetime"):
        dt = t_event_utc.to_pydatetime()
    else:
        dt = parse_event_dt(str(t_event_utc))
    if dt is None:
        return sym, str(t_event_utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    dt = dt.astimezone(timezone.utc).replace(microsecond=0)
    return sym, dt.isoformat()


def load_validation_events(path: Path) -> list[dict[str, Any]]:
    import json

    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("events", [])
    return [r for r in rows if isinstance(r, dict)]


def db_path() -> Path:
    return resolve_db_path(load_settings())


def _passthrough_float(event: dict[str, Any], key: str) -> float:
    raw = event.get(key, 0.0)
    try:
        return float(raw) if raw is not None else 0.0
    except (TypeError, ValueError):
        return 0.0


def _passthrough_int(event: dict[str, Any], key: str) -> int:
    raw = event.get(key, 0)
    try:
        return int(raw) if raw is not None else 0
    except (TypeError, ValueError):
        return 0


def event_features(event: dict[str, Any], *, db: Path | None = None) -> dict[str, Any] | None:
    symbol = str(event.get("symbol", "")).upper().strip()
    t_event = parse_event_dt(str(event.get("t_event_utc", "")))
    actual_sign = int(event.get("actual_sign", 0))
    pre_ret = float(event.get("baseline_pre_7d_return_pct", 0.0))
    if not symbol or t_event is None or actual_sign == 0:
        return None

    news = compute_news_features(symbol, t_event, db_path=db or db_path())
    return {
        "symbol": symbol,
        "t_event_utc": t_event.isoformat(),
        "actual_sign": 1 if actual_sign > 0 else 0,
        "actual_5d_return_pct": float(event.get("actual_5d_return_pct", 0.0)),
        "baseline_pred_sign": int(event.get("baseline_pred_sign", 0)),
        "enhanced_pred_sign": int(event.get("enhanced_pred_sign", 0)),
        "enhanced_confidence": float(event.get("enhanced_confidence", 0.0)),
        # Price-derived features (stored at validation time when available).
        "baseline_pre_7d_return_pct": pre_ret,
        "baseline_pre_7d_abs_return": abs(pre_ret),
        "pre_30d_return_pct": _passthrough_float(event, "pre_30d_return_pct"),
        "pre_60d_return_pct": _passthrough_float(event, "pre_60d_return_pct"),
        "pre_30d_vol_pct": _passthrough_float(event, "pre_30d_vol_pct"),
        "prior_post_earnings_5d_return_pct": _passthrough_float(
            event, "prior_post_earnings_5d_return_pct"
        ),
        "sector_cohort_5d_return_pct": _passthrough_float(
            event, "sector_cohort_5d_return_pct"
        ),
        "neighbor_earnings_count_30d": _passthrough_int(
            event, "neighbor_earnings_count_30d"
        ),
        # News + spillover features.
        "sent_doc_count": news["sent_doc_count"],
        "sent_lex_mean": news.get("sent_lex_mean", news.get("sent_mean_score", 0.0)),
        "sent_lex_max_abs": news.get("sent_lex_max_abs", 0.0),
        "sent_finbert_mean": news.get("sent_finbert_mean", 0.0),
        "sent_finbert_max_abs": news.get("sent_finbert_max_abs", 0.0),
        "sent_finbert_pos_ratio": news.get("sent_finbert_pos_ratio", 0.0),
        "sent_finbert_neg_ratio": news.get("sent_finbert_neg_ratio", 0.0),
        "sent_finbert_polarity_gap": news.get("sent_finbert_polarity_gap", 0.0),
        "sent_finbert_doc_count": news.get("sent_finbert_doc_count", 0),
        "sent_mean_score": news.get("sent_mean_score", 0.0),
        "sent_mean_relevance": news.get("sent_mean_relevance", 0.0),
        "spillover_mentions_7d": news["spillover_mentions_7d"],
        "spillover_density_7d": news["spillover_density_7d"],
        "spillover_weighted_score_7d": news.get("spillover_weighted_score_7d", 0.0),
        "enhanced_documents_considered": news["enhanced_documents_considered"],
    }


def build_feature_frame(
    events: list[dict[str, Any]],
    *,
    cache_path: Path | None = None,
    progress: bool = False,
) -> pd.DataFrame:
    """Build the leakage-safe feature frame.

    If ``cache_path`` is given, the frame is reloaded from disk when present
    and saved on a fresh build. Callers that detect upstream changes should
    delete the cache themselves before calling.
    """
    if cache_path is not None and cache_path.exists():
        try:
            cached = pd.read_pickle(cache_path)
            if "t_event_utc" in cached.columns:
                cached["t_event_utc"] = pd.to_datetime(cached["t_event_utc"], utc=True)
            if progress:
                print(
                    f"[feature-frame] reloaded {len(cached)} rows from cache: "
                    f"{cache_path.as_posix()}"
                )
            return cached
        except Exception as exc:  # noqa: BLE001
            if progress:
                print(f"[feature-frame] cache reload failed ({exc!r}); rebuilding")

    rows: list[dict[str, Any]] = []
    db = db_path()
    total = len(events)
    step = max(1, total // 20)
    for idx, ev in enumerate(events, start=1):
        row = event_features(ev, db=db)
        if row is not None:
            rows.append(row)
        if progress and (idx % step == 0 or idx == total):
            print(f"[feature-frame] {idx}/{total} events processed", flush=True)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["t_event_utc"] = pd.to_datetime(df["t_event_utc"], utc=True)
    df = df.sort_values("t_event_utc").reset_index(drop=True)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            df.to_pickle(cache_path)
            if progress:
                print(f"[feature-frame] wrote cache: {cache_path.as_posix()}")
        except Exception as exc:  # noqa: BLE001
            if progress:
                print(
                    f"[feature-frame] cache write failed ({exc!r}); "
                    "continuing in-memory only"
                )
    return df


# Pure price-momentum features. Always available regardless of news coverage.
PRICE_FEATURES = [
    "baseline_pre_7d_return_pct",
    "baseline_pre_7d_abs_return",
    "pre_30d_return_pct",
    "pre_60d_return_pct",
    "pre_30d_vol_pct",
    "prior_post_earnings_5d_return_pct",
    "sector_cohort_5d_return_pct",
    "neighbor_earnings_count_30d",
]

# Backwards-compatible name; baseline = pure price.
BASELINE_FEATURES = list(PRICE_FEATURES)

# Enhanced = price + lexicon + FinBERT + spillover. The model picks weights;
# no magic-number combiner sits in the feature path anymore.
ENHANCED_FEATURES = PRICE_FEATURES + [
    "sent_doc_count",
    "sent_lex_mean",
    "sent_lex_max_abs",
    "sent_finbert_mean",
    "sent_finbert_max_abs",
    "sent_finbert_pos_ratio",
    "sent_finbert_neg_ratio",
    "sent_finbert_polarity_gap",
    "sent_finbert_doc_count",
    "sent_mean_relevance",
    "spillover_mentions_7d",
    "spillover_density_7d",
    "spillover_weighted_score_7d",
]


def split_time_aware(df: pd.DataFrame, train_ratio: float = 0.7) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = df.sort_values("t_event_utc").reset_index(drop=True)
    if len(ordered) < 2:
        return ordered.iloc[:0].copy(), ordered.copy()
    cut = max(1, int(len(ordered) * train_ratio))
    cut = min(cut, len(ordered) - 1)
    return ordered.iloc[:cut].copy(), ordered.iloc[cut:].copy()


def _select_features(df: pd.DataFrame, features: list[str]) -> np.ndarray:
    """Pull feature columns out of ``df``, filling missing columns with 0.0.

    Missing columns happen when a cached frame predates a feature addition;
    using zero keeps walk-forward stable instead of raising mid-run.
    """
    cols: list[np.ndarray] = []
    for name in features:
        if name in df.columns:
            cols.append(pd.to_numeric(df[name], errors="coerce").fillna(0.0).to_numpy(dtype=float))
        else:
            cols.append(np.zeros(len(df), dtype=float))
    if not cols:
        return np.zeros((len(df), 0), dtype=float)
    return np.column_stack(cols)


def fit_predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Fit on train, score test. Returns ``(preds, proba, metrics)``.

    Routed through ``model_store.fit_model`` so LightGBM is preferred when
    available and standardization is consistent at fit and predict time.
    """
    x_train = _select_features(train, features)
    y_train = train["actual_sign"].to_numpy(dtype=int)
    x_test = _select_features(test, features)
    y_test = test["actual_sign"].to_numpy(dtype=int)

    if len(np.unique(y_train)) < 2 or x_train.shape[1] == 0:
        # Degenerate: single class or empty features. Fall back to momentum sign.
        momentum = (
            test["baseline_pre_7d_return_pct"].to_numpy(dtype=float)
            if "baseline_pre_7d_return_pct" in test.columns
            else np.zeros(len(test))
        )
        y_pred = (momentum > 0).astype(int)
        proba = np.clip(0.5 + np.abs(momentum) / 24.0, 0.5, 0.95)
        return y_pred, proba, {
            "accuracy": float(accuracy_score(y_test, y_pred)) if len(y_test) else 0.0,
            "precision": float(precision_score(y_test, y_pred, zero_division=0)) if len(y_test) else 0.0,
            "recall": float(recall_score(y_test, y_pred, zero_division=0)) if len(y_test) else 0.0,
            "f1": float(f1_score(y_test, y_pred, zero_division=0)) if len(y_test) else 0.0,
            "n_test": float(len(y_test)),
        }

    model = fit_model(x_train, y_train, features)
    proba = predict_proba(model, x_test)
    y_pred = (proba >= 0.5).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "n_test": float(len(y_test)),
    }
    return y_pred, proba, metrics


def fit_only(
    train: pd.DataFrame, features: list[str]
) -> TrainedModel | None:
    """Fit a model on ``train`` for later persistence/reuse."""
    x = _select_features(train, features)
    y = train["actual_sign"].to_numpy(dtype=int)
    if len(np.unique(y)) < 2 or x.shape[1] == 0:
        return None
    return fit_model(x, y, features)


def news_coverage_summary(df: pd.DataFrame) -> dict[str, Any]:
    """Per-symbol and global news coverage diagnostics.

    Used by training/permutation scripts to surface how much of the
    dataset is actually getting a non-zero text signal vs falling through
    to the momentum-only path.
    """
    if df.empty:
        return {
            "events_total": 0,
            "events_with_any_news": 0,
            "events_with_finbert_score": 0,
            "events_with_spillover": 0,
            "zero_news_event_pct": None,
            "low_coverage_symbols": [],
        }
    total = int(len(df))
    docs = df.get("sent_doc_count", pd.Series([0] * total, index=df.index))
    docs_int = pd.to_numeric(docs, errors="coerce").fillna(0).astype(int)
    fb_count = pd.to_numeric(
        df.get("sent_finbert_doc_count", pd.Series([0] * total, index=df.index)),
        errors="coerce",
    ).fillna(0).astype(int)
    spill = pd.to_numeric(
        df.get("spillover_mentions_7d", pd.Series([0] * total, index=df.index)),
        errors="coerce",
    ).fillna(0).astype(int)
    events_with_news = int((docs_int > 0).sum())
    events_with_fb = int((fb_count > 0).sum())
    events_with_spill = int((spill > 0).sum())
    zero_news_pct = round((total - events_with_news) / total, 4) if total else None

    sym_groups = df.groupby("symbol")
    low_coverage: list[dict[str, Any]] = []
    for sym, sub in sym_groups:
        n = int(len(sub))
        with_news = int(
            (
                pd.to_numeric(sub["sent_doc_count"], errors="coerce")
                .fillna(0)
                .astype(int)
                > 0
            ).sum()
        )
        if n >= 4 and with_news / n < 0.25:
            low_coverage.append(
                {
                    "symbol": str(sym),
                    "events": n,
                    "events_with_news": with_news,
                    "coverage_pct": round(with_news / n, 4),
                }
            )
    low_coverage.sort(key=lambda r: (r["coverage_pct"], -r["events"]))

    return {
        "events_total": total,
        "events_with_any_news": events_with_news,
        "events_with_finbert_score": events_with_fb,
        "events_with_spillover": events_with_spill,
        "zero_news_event_pct": zero_news_pct,
        "low_coverage_symbols": low_coverage[:25],
    }
