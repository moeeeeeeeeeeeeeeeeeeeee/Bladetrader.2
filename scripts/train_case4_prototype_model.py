"""Train and compare baseline vs sentiment-enhanced models on validated events.

Outputs:
- ``data/case4_model_comparison.json`` — accuracy/precision/recall/F1 for
  baseline (price-only) vs enhanced (price + lexicon + FinBERT + spillover).
- ``data/models/case4_enhanced.pkl`` — persisted enhanced model used at
  inference time by ``case4_features.predict_enhanced`` and
  ``paper_signals.build_earnings_paper_signals``.

Routes through ``research.model_store`` so LightGBM is preferred when
installed; logistic regression is the deterministic fallback.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

load_dotenv(ROOT / ".env")

from finhack.research.case4_dataset import (
    BASELINE_FEATURES,
    ENHANCED_FEATURES,
    build_feature_frame,
    fit_predict,
    fit_only,
    load_validation_events,
    news_coverage_summary,
    split_time_aware,
)
from finhack.research.constants import DEFAULT_TRAIN_RATIO
from finhack.research.model_store import DEFAULT_ARTIFACT_PATH, save_trained_model
from finhack.text_encoder import (
    FINBERT_MODEL,
    LEXICON_MODEL,
    ensure_finbert_scores,
    ensure_lexicon_scores,
)


@dataclass
class ModelMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    n_test: int


def _metrics_from_dict(raw: dict[str, float]) -> ModelMetrics:
    return ModelMetrics(
        accuracy=float(raw["accuracy"]),
        precision=float(raw["precision"]),
        recall=float(raw["recall"]),
        f1=float(raw["f1"]),
        n_test=int(raw["n_test"]),
    )


def _ensure_doc_scores(db_url: str) -> dict[str, object]:
    """Top up cached document_score rows; idempotent.

    Lexicon is always populated. FinBERT is attempted; on failure the
    function returns a "fallback" marker so the summary makes the
    encoder situation explicit instead of silently using only lexicon.
    """
    raw = (
        db_url.replace("sqlite:///", "", 1)
        if db_url.startswith("sqlite:///")
        else db_url
    )
    db = Path(raw)
    if not db.is_absolute():
        db = ROOT / raw
    info: dict[str, object] = {"db_path": str(db.as_posix())}
    if not db.exists():
        info["status"] = "no_db"
        return info
    info["lexicon_rows_added"] = ensure_lexicon_scores(db)
    written, model_used = ensure_finbert_scores(db, progress=True)
    info["finbert_rows_added"] = written
    info["finbert_model_used"] = model_used
    info["finbert_active"] = model_used == FINBERT_MODEL
    if model_used != FINBERT_MODEL:
        info["finbert_fallback_reason"] = (
            "FinBERT unavailable (transformers/torch not installed); "
            "lexicon scores are used in its place."
        )
    return info


def main() -> None:
    events_path = ROOT / "data" / "case4_earnings_validation.json"
    if not events_path.exists():
        raise SystemExit(
            "Missing data/case4_earnings_validation.json — "
            "run scripts/validate_case4_earnings.py"
        )

    from finhack.config import load_settings

    settings = load_settings()
    encoder_info = _ensure_doc_scores(settings.database_url)

    events = load_validation_events(events_path)
    df = build_feature_frame(events)
    if len(df) < 10:
        raise SystemExit("Not enough labeled events for modeling.")

    train, test = split_time_aware(df, train_ratio=DEFAULT_TRAIN_RATIO)

    baseline_pred, _, baseline_metrics_raw = fit_predict(train, test, BASELINE_FEATURES)
    enhanced_pred, _enhanced_proba, enhanced_metrics_raw = fit_predict(
        train, test, ENHANCED_FEATURES
    )
    baseline_metrics = _metrics_from_dict(baseline_metrics_raw)
    enhanced_metrics = _metrics_from_dict(enhanced_metrics_raw)
    uplift = enhanced_metrics.accuracy - baseline_metrics.accuracy

    # Persist a model trained on the *full* labeled history (train + test) so
    # the artifact reflects every event we have at the time of training. This
    # is the artifact loaded by paper_signals and predict_enhanced.
    full_model = fit_only(df, ENHANCED_FEATURES)
    artifact_path: Path | None = None
    if full_model is not None:
        full_model.extras = {
            "trained_at_utc": datetime.now(timezone.utc).isoformat(),
            "train_ratio": DEFAULT_TRAIN_RATIO,
            "events_total": int(len(df)),
            "encoder_info": encoder_info,
        }
        artifact_path = save_trained_model(full_model)

    test_predictions = []
    for i in range(len(test)):
        row = test.iloc[i]
        sign = int(enhanced_pred[i])
        signal = 1 if sign > 0 else -1 if sign < 0 else 0
        actual_ret = float(row["actual_5d_return_pct"])
        trade_ret = signal * actual_ret if signal else None
        test_predictions.append(
            {
                "symbol": str(row["symbol"]),
                "t_event_utc": str(row["t_event_utc"]),
                "actual": int(row["actual_sign"]),
                "baseline_pred": int(baseline_pred[i]),
                "enhanced_pred": sign,
                "actual_5d_return_pct": round(actual_ret, 4),
                "model_trade_return_pct": round(trade_ret, 4) if trade_ret is not None else None,
            }
        )

    coverage = news_coverage_summary(df)

    out = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "live",
        "objective": (
            "Predict five-day post-earnings direction; compare baseline (price-only) "
            "vs enhanced (price + lexicon + FinBERT + spillover)."
        ),
        "encoder": encoder_info,
        "model_artifact": str(artifact_path.as_posix()) if artifact_path else None,
        "model_backend": full_model.backend if full_model else None,
        "dataset": {
            "events_total": int(len(df)),
            "train_events": int(len(train)),
            "test_events": int(len(test)),
            "symbols": sorted(df["symbol"].unique().tolist()),
            "features_baseline": BASELINE_FEATURES,
            "features_enhanced": ENHANCED_FEATURES,
            "news_coverage": coverage,
        },
        "leakage_controls": {
            "window_policy": "Features only from 7-day window ending at earnings date T.",
            "target_policy": "Target is direction over the 5 trading days after T.",
            "validation_policy": "Chronological split by event date; no random shuffling.",
            "encoder_policy": (
                "FinBERT scores are computed offline and cached per-document; "
                "no model parameters cross the time-aware split."
            ),
        },
        "results": {
            "baseline": asdict(baseline_metrics),
            "sentiment_enhanced": asdict(enhanced_metrics),
            "accuracy_uplift_pp": round(uplift * 100.0, 2),
        },
        "test_predictions": test_predictions,
    }

    out_path = ROOT / "data" / "case4_model_comparison.json"
    out_path.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(f"Wrote {out_path.as_posix()}")
    print(json.dumps(out["results"], indent=2))
    if artifact_path:
        print(f"Saved model artifact: {artifact_path.as_posix()}")
    print(
        json.dumps(
            {
                "encoder": encoder_info.get("finbert_model_used", LEXICON_MODEL),
                "events_total": coverage["events_total"],
                "events_with_any_news": coverage["events_with_any_news"],
                "events_with_finbert_score": coverage["events_with_finbert_score"],
                "zero_news_event_pct": coverage["zero_news_event_pct"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
