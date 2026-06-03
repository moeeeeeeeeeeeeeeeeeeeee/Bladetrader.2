"""
Baseline vs enhanced uplift on the NQ-100 mega-cap subset only.

Reads ``data/case4_earnings_validation.json``, filters events to the
mega-cap names that drive Nasdaq-100 movement, re-fits the baseline
(price-only) and enhanced (price + lexicon + FinBERT + spillover)
classifiers on a time-aware split, and writes
``data/case4_nq_megacap_uplift.json`` with the result.

This script does NOT persist a new model artifact — it leaves
``data/models/case4_enhanced.pkl`` (the production one) alone.

Why this is interesting
-----------------------
On the full 500-symbol universe the enhanced model gets the same accuracy
as the baseline. The hypothesis worth testing is whether the signal is
better on the names that actually drive NQ — those are the only events
that matter for translating into a same-session NQ direction bet.
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

from finhack.research.case4_dataset import (  # noqa: E402
    BASELINE_FEATURES,
    ENHANCED_FEATURES,
    build_feature_frame,
    fit_predict,
    load_validation_events,
    news_coverage_summary,
    split_time_aware,
)
from finhack.research.constants import DEFAULT_TRAIN_RATIO  # noqa: E402

# Same mega-cap list used by the overlay validation script. Keep these two
# in sync. If you change one, change the other.
NQ_MEGACAPS = [
    "NVDA", "MSFT", "AAPL", "AMZN", "META", "GOOGL", "GOOG", "AVGO",
    "TSLA", "NFLX", "ADBE", "AMD", "COST", "PEP", "CSCO", "CMCSA",
    "INTC", "TXN", "INTU", "QCOM", "AMGN", "AMAT", "BKNG", "ADI",
    "MU", "ISRG", "GILD", "REGN", "LRCX", "MDLZ",
]


@dataclass
class SplitMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    n_test: int


def _to_metrics(raw: dict[str, float]) -> SplitMetrics:
    return SplitMetrics(
        accuracy=float(raw["accuracy"]),
        precision=float(raw["precision"]),
        recall=float(raw["recall"]),
        f1=float(raw["f1"]),
        n_test=int(raw["n_test"]),
    )


def main() -> None:
    events_path = ROOT / "data" / "case4_earnings_validation.json"
    if not events_path.exists():
        raise SystemExit(
            "Missing data/case4_earnings_validation.json — "
            "run scripts/validate_case4_earnings.py first."
        )

    events = load_validation_events(events_path)
    mega_set = {s.upper() for s in NQ_MEGACAPS}
    filtered = [e for e in events if str(e.get("symbol", "")).upper() in mega_set]
    if not filtered:
        raise SystemExit(
            "No mega-cap events found in validation file. Confirm that "
            "the validation universe includes the NQ mega-cap names."
        )

    df_full = build_feature_frame(events)
    df_subset = build_feature_frame(filtered)
    if len(df_subset) < 10:
        raise SystemExit(
            f"Only {len(df_subset)} labeled mega-cap events — not enough "
            "to fit a model. Re-run validation with the mega-cap names "
            "covered in the universe."
        )

    train, test = split_time_aware(df_subset, train_ratio=DEFAULT_TRAIN_RATIO)
    if test.empty or train.empty:
        raise SystemExit(
            f"Time-aware split produced an empty fold (train={len(train)}, "
            f"test={len(test)}). The mega-cap subset is too small for the "
            f"default {DEFAULT_TRAIN_RATIO} train ratio."
        )

    _, _, baseline_raw = fit_predict(train, test, BASELINE_FEATURES)
    _, _, enhanced_raw = fit_predict(train, test, ENHANCED_FEATURES)
    baseline = _to_metrics(baseline_raw)
    enhanced = _to_metrics(enhanced_raw)
    uplift_pp = (enhanced.accuracy - baseline.accuracy) * 100.0

    full_train, full_test = split_time_aware(df_full, train_ratio=DEFAULT_TRAIN_RATIO)
    _, _, full_base = fit_predict(full_train, full_test, BASELINE_FEATURES)
    _, _, full_enh = fit_predict(full_train, full_test, ENHANCED_FEATURES)
    full_uplift_pp = (full_enh["accuracy"] - full_base["accuracy"]) * 100.0

    coverage_subset = news_coverage_summary(df_subset)
    coverage_full = news_coverage_summary(df_full)

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "objective": (
            "Test whether the enhanced model beats the price-only baseline "
            "on the NQ-100 mega-cap subset, where translating to a same-"
            "session MNQ direction bet is most defensible."
        ),
        "subset": {
            "name": "nq_megacaps",
            "symbols": sorted(NQ_MEGACAPS),
            "events_in_universe": int(len(df_subset)),
            "events_covered_full_universe": int(len(df_full)),
            "train_events": int(len(train)),
            "test_events": int(len(test)),
            "news_coverage_subset": coverage_subset,
        },
        "results_subset": {
            "baseline": asdict(baseline),
            "sentiment_enhanced": asdict(enhanced),
            "accuracy_uplift_pp": round(uplift_pp, 2),
        },
        "results_full_universe_for_reference": {
            "baseline_accuracy": round(full_base["accuracy"], 4),
            "enhanced_accuracy": round(full_enh["accuracy"], 4),
            "accuracy_uplift_pp": round(full_uplift_pp, 2),
            "events_total": int(len(df_full)),
            "news_coverage": coverage_full,
        },
        "interpretation": (
            "If the subset uplift is positive while the full-universe uplift "
            "is ~0, the signal lives in the NQ-relevant names but is diluted "
            "across the long tail of the catalog. That is the configuration "
            "that makes the overlay actionable. If subset uplift is also ~0, "
            "the enhanced features add no real edge over price momentum "
            "anywhere yet."
        ),
    }

    out_path = ROOT / "data" / "case4_nq_megacap_uplift.json"
    out_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"Wrote {out_path.as_posix()}")
    print(json.dumps({
        "subset_baseline_acc": baseline.accuracy,
        "subset_enhanced_acc": enhanced.accuracy,
        "subset_uplift_pp": payload["results_subset"]["accuracy_uplift_pp"],
        "subset_events": int(len(df_subset)),
        "subset_test_events": int(len(test)),
        "full_baseline_acc": payload["results_full_universe_for_reference"]["baseline_accuracy"],
        "full_enhanced_acc": payload["results_full_universe_for_reference"]["enhanced_accuracy"],
        "full_uplift_pp": payload["results_full_universe_for_reference"]["accuracy_uplift_pp"],
    }, indent=2))


if __name__ == "__main__":
    main()
