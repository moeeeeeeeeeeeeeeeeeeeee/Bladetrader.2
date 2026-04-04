"""
Prototype model runner for Case-4 deliverables.

Builds a leakage-safe event dataset anchored on earnings date T:
- Baseline features from [T-7d, T] (price-only proxy from validation JSON)
- Sentiment features from documents in [T-7d, T]
- Spillover feature from connected symbols in company graph

Trains and compares:
- baseline logistic model
- sentiment-enhanced logistic model

Outputs:
- data/case4_model_comparison.json
- data/case4_visuals/*.png
"""

from __future__ import annotations

import json
import math
import sqlite3
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from finhack.config import load_settings
from finhack.data.company_graph import SPILLOVER_MAP, SYMBOL_TO_COMPANY

POSITIVE_TERMS = (
    "beat",
    "growth",
    "strong",
    "surge",
    "upgrade",
    "bullish",
    "record",
    "expansion",
)
NEGATIVE_TERMS = (
    "miss",
    "downgrade",
    "weak",
    "decline",
    "bearish",
    "lawsuit",
    "delay",
    "cut",
)


@dataclass
class ModelMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    n_test: int


def _parse_dt(raw: str | None) -> datetime | None:
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


def _load_events(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("events", [])
    return [r for r in rows if isinstance(r, dict)]


def _db_path() -> Path:
    raw = load_settings().database_url
    if raw.startswith("sqlite:///"):
        raw = raw.replace("sqlite:///", "", 1)
    p = Path(raw)
    return p if p.is_absolute() else Path.cwd() / p


def _fetch_docs_between(start: datetime, end: datetime) -> list[dict[str, Any]]:
    db = _db_path()
    if not db.exists():
        return []
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT title, body, published_at, fetched_at, source_domain, relevance_score
            FROM document
            WHERE COALESCE(published_at, fetched_at) >= ?
              AND COALESCE(published_at, fetched_at) <= ?
            ORDER BY COALESCE(published_at, fetched_at) ASC
            """,
            (start.isoformat(), end.isoformat()),
        ).fetchall()
    finally:
        conn.close()
    out: list[dict[str, Any]] = []
    for r in rows:
        out.append(
            {
                "title": str(r["title"] or ""),
                "body": str(r["body"] or ""),
                "published_at": str(r["published_at"] or r["fetched_at"] or ""),
                "source_domain": str(r["source_domain"] or ""),
                "relevance_score": float(r["relevance_score"] or 0.0),
            }
        )
    return out


def _text_score(text: str) -> float:
    low = text.lower()
    pos = sum(low.count(k) for k in POSITIVE_TERMS)
    neg = sum(low.count(k) for k in NEGATIVE_TERMS)
    return float(pos - neg)


def _event_features(event: dict[str, Any]) -> dict[str, Any] | None:
    symbol = str(event.get("symbol", "")).upper().strip()
    t_event = _parse_dt(str(event.get("t_event_utc", "")))
    actual_sign = int(event.get("actual_sign", 0))
    pre_ret = float(event.get("baseline_pre_7d_return_pct", 0.0))
    if not symbol or t_event is None or actual_sign == 0:
        return None

    start = t_event - timedelta(days=7)
    docs = _fetch_docs_between(start, t_event)
    company_name = SYMBOL_TO_COMPANY.get(symbol).name.lower() if symbol in SYMBOL_TO_COMPANY else symbol.lower()
    spill_symbols = SPILLOVER_MAP.get(symbol, [])

    sym_docs = 0
    sentiment_sum = 0.0
    relevance_sum = 0.0
    spill_mentions = 0
    for d in docs:
        text = f"{d['title']} {d['body']}"
        low = text.lower()
        has_symbol = symbol.lower() in low or company_name in low
        if has_symbol:
            sym_docs += 1
            sentiment_sum += _text_score(text)
            relevance_sum += float(d.get("relevance_score", 0.0))
        spill_mentions += sum(1 for s in spill_symbols if s.lower() in low)

    mean_sent = sentiment_sum / sym_docs if sym_docs else 0.0
    mean_rel = relevance_sum / sym_docs if sym_docs else 0.0
    spillover_density = spill_mentions / max(1, len(docs))

    return {
        "symbol": symbol,
        "t_event_utc": t_event.isoformat(),
        "actual_sign": 1 if actual_sign > 0 else 0,  # binary for classifier
        "baseline_pre_7d_return_pct": pre_ret,
        "baseline_pre_7d_abs_return": abs(pre_ret),
        "sent_doc_count": sym_docs,
        "sent_mean_score": mean_sent,
        "sent_mean_relevance": mean_rel,
        "spillover_mentions_7d": spill_mentions,
        "spillover_density_7d": spillover_density,
    }


def _split_time_aware(df: pd.DataFrame, train_ratio: float = 0.7) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = df.sort_values("t_event_utc").reset_index(drop=True)
    cut = max(1, int(len(ordered) * train_ratio))
    cut = min(cut, len(ordered) - 1)
    return ordered.iloc[:cut].copy(), ordered.iloc[cut:].copy()


def _fit_eval(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
) -> tuple[ModelMetrics, np.ndarray]:
    x_train = train[features].to_numpy(dtype=float)
    y_train = train["actual_sign"].to_numpy(dtype=int)
    x_test = test[features].to_numpy(dtype=float)
    y_test = test["actual_sign"].to_numpy(dtype=int)

    if len(np.unique(y_train)) < 2:
        # Degenerate fallback: use sign of baseline return proxy.
        y_pred = (test["baseline_pre_7d_return_pct"].to_numpy(dtype=float) > 0).astype(int)
    else:
        clf = LogisticRegression(max_iter=2000, class_weight="balanced")
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)

    metrics = ModelMetrics(
        accuracy=float(accuracy_score(y_test, y_pred)),
        precision=float(precision_score(y_test, y_pred, zero_division=0)),
        recall=float(recall_score(y_test, y_pred, zero_division=0)),
        f1=float(f1_score(y_test, y_pred, zero_division=0)),
        n_test=int(len(y_test)),
    )
    return metrics, y_pred


def _plot_outputs(df: pd.DataFrame, out_dir: Path, baseline: ModelMetrics, enhanced: ModelMetrics) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Sentiment trend over event time
    trend = df.sort_values("t_event_utc")
    plt.figure(figsize=(9, 4))
    plt.plot(trend["t_event_utc"], trend["sent_mean_score"], marker="o")
    plt.xticks(rotation=45, ha="right")
    plt.title("Company-Level Sentiment Score by Earnings Event")
    plt.ylabel("Sentiment score (pos-neg)")
    plt.tight_layout()
    plt.savefig(out_dir / "sentiment_trend.png", dpi=140)
    plt.close()

    # 2) Sentiment vs return proxy
    plt.figure(figsize=(6, 5))
    plt.scatter(df["sent_mean_score"], df["baseline_pre_7d_return_pct"], alpha=0.75)
    plt.title("Sentiment vs Baseline Pre-Earnings Return")
    plt.xlabel("Mean sentiment score (7d window)")
    plt.ylabel("Pre-7d return (%)")
    plt.tight_layout()
    plt.savefig(out_dir / "sentiment_vs_return.png", dpi=140)
    plt.close()

    # 3) Spillover feature distribution
    top = (
        df.groupby("symbol", as_index=False)["spillover_mentions_7d"]
        .mean()
        .sort_values("spillover_mentions_7d", ascending=False)
        .head(10)
    )
    plt.figure(figsize=(8, 4))
    plt.bar(top["symbol"], top["spillover_mentions_7d"])
    plt.title("Average Spillover Mentions (7d pre-earnings)")
    plt.ylabel("mentions")
    plt.tight_layout()
    plt.savefig(out_dir / "spillover_relationships.png", dpi=140)
    plt.close()

    # 4) Model comparison
    labels = ["Accuracy", "Precision", "Recall", "F1"]
    baseline_vals = [baseline.accuracy, baseline.precision, baseline.recall, baseline.f1]
    enhanced_vals = [enhanced.accuracy, enhanced.precision, enhanced.recall, enhanced.f1]
    x = np.arange(len(labels))
    width = 0.35
    plt.figure(figsize=(7, 4))
    plt.bar(x - width / 2, baseline_vals, width, label="Baseline")
    plt.bar(x + width / 2, enhanced_vals, width, label="Sentiment-Enhanced")
    plt.xticks(x, labels)
    plt.ylim(0, 1)
    plt.title("Model Comparison (Time-Aware Test Split)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "model_comparison.png", dpi=140)
    plt.close()


def main() -> None:
    root = ROOT
    events_path = root / "data" / "case4_earnings_validation.json"
    if not events_path.exists():
        raise SystemExit("Missing data/case4_earnings_validation.json")

    events = _load_events(events_path)
    feature_rows = []
    for ev in events:
        row = _event_features(ev)
        if row is not None:
            feature_rows.append(row)
    if len(feature_rows) < 10:
        raise SystemExit("Not enough labeled events for prototype modeling.")

    df = pd.DataFrame(feature_rows)
    train, test = _split_time_aware(df, train_ratio=0.7)

    baseline_features = [
        "baseline_pre_7d_return_pct",
        "baseline_pre_7d_abs_return",
    ]
    enhanced_features = baseline_features + [
        "sent_doc_count",
        "sent_mean_score",
        "sent_mean_relevance",
        "spillover_mentions_7d",
        "spillover_density_7d",
    ]

    baseline_metrics, baseline_pred = _fit_eval(train, test, baseline_features)
    enhanced_metrics, enhanced_pred = _fit_eval(train, test, enhanced_features)

    uplift = enhanced_metrics.accuracy - baseline_metrics.accuracy
    visuals_dir = root / "data" / "case4_visuals"
    _plot_outputs(df, visuals_dir, baseline_metrics, enhanced_metrics)

    out = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "objective": "Predict five-day post-earnings direction and compare baseline vs sentiment-enhanced model.",
        "dataset": {
            "events_total": int(len(df)),
            "train_events": int(len(train)),
            "test_events": int(len(test)),
            "symbols": sorted(df["symbol"].unique().tolist()),
            "features_baseline": baseline_features,
            "features_enhanced": enhanced_features,
        },
        "leakage_controls": {
            "window_policy": "Features only from 7-day window ending at earnings date T.",
            "target_policy": "Target is direction over the 5 trading days after T.",
            "validation_policy": "Chronological split by event date; no random shuffling.",
        },
        "results": {
            "baseline": asdict(baseline_metrics),
            "sentiment_enhanced": asdict(enhanced_metrics),
            "accuracy_uplift_pp": round(uplift * 100.0, 2),
        },
        "test_predictions_preview": [
            {
                "symbol": str(test.iloc[i]["symbol"]),
                "t_event_utc": str(test.iloc[i]["t_event_utc"]),
                "actual": int(test.iloc[i]["actual_sign"]),
                "baseline_pred": int(baseline_pred[i]),
                "enhanced_pred": int(enhanced_pred[i]),
            }
            for i in range(min(20, len(test)))
        ],
        "visualizations": {
            "sentiment_trend": "data/case4_visuals/sentiment_trend.png",
            "sentiment_vs_return": "data/case4_visuals/sentiment_vs_return.png",
            "spillover_relationships": "data/case4_visuals/spillover_relationships.png",
            "model_comparison": "data/case4_visuals/model_comparison.png",
        },
    }

    out_path = root / "data" / "case4_model_comparison.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote {out_path.as_posix()}")
    print(json.dumps(out["results"], indent=2))


if __name__ == "__main__":
    main()

