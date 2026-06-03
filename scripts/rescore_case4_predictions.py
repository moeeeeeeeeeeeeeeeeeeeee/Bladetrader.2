"""
In-place rescorer for ``data/case4_earnings_validation.json``.

Reads each event, recomputes its news features against the current
document DB, scores it with the currently-saved enhanced model
(``data/models/case4_enhanced.pkl``), and writes the updated JSON back.

Use this after retraining the model (or changing the feature pipeline)
to refresh the stored ``enhanced_pred_sign`` / ``enhanced_confidence`` /
``earnings_kw_*`` fields without paying the full network cost of
``scripts/validate_case4_earnings.py``.

NOTE: This intentionally skips the trade-path simulation block. Those
fields stay at whatever the validator wrote last; the rescorer is for
news-feature + model-prediction refresh only.

IN-SAMPLE WARNING
-----------------
``scripts/train_case4_prototype_model.py`` persists the final artifact
from a fit on the **full** labeled history (train + test). Rescoring
every event with that artifact therefore makes the train-fold events
in-sample for the persisted model. Any "accuracy on the full universe"
computed from the rescored events overstates the true OOS edge.

The honest OOS uplift lives in ``data/case4_model_comparison.json``
under ``results.accuracy_uplift_pp``, which is computed on the held-out
chronological test fold only. Treat the rescored predictions as a
**display layer** (for the UI's contributing-events tables and the
overlay validation's coverage), not as an OOS performance claim.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

load_dotenv(ROOT / ".env")

from finhack.case4_features import apply_enhanced_fields  # noqa: E402
from finhack.config import load_settings  # noqa: E402


def _resolve_db_path(database_url: str) -> Path:
    raw = (
        database_url.replace("sqlite:///", "", 1)
        if database_url.startswith("sqlite:///")
        else database_url
    )
    p = Path(raw)
    return p if p.is_absolute() else ROOT / raw


def main() -> None:
    target = ROOT / "data" / "case4_earnings_validation.json"
    if not target.exists():
        raise SystemExit(
            f"Missing {target.as_posix()} — run scripts/validate_case4_earnings.py first."
        )

    settings = load_settings()
    db_path = _resolve_db_path(settings.database_url)
    if not db_path.exists():
        raise SystemExit(
            f"Document DB missing at {db_path.as_posix()} — nothing to rescore against."
        )

    payload = json.loads(target.read_text(encoding="utf-8"))
    events = payload.get("events") if isinstance(payload, dict) else None
    if not isinstance(events, list):
        raise SystemExit("Validation payload has no 'events' list.")

    total = len(events)
    if total == 0:
        raise SystemExit("Validation payload has zero events.")

    print(f"Rescoring {total} events against {db_path.as_posix()}...", flush=True)
    flip_count = 0
    confidence_delta_sum = 0.0
    confidence_delta_abs_sum = 0.0
    step = max(1, total // 20)
    rescored: list[dict] = []
    for idx, ev in enumerate(events, start=1):
        if not isinstance(ev, dict):
            rescored.append(ev)
            continue
        prior_sign = int(ev.get("enhanced_pred_sign", 0))
        prior_conf = float(ev.get("enhanced_confidence", 0.0))
        enriched = apply_enhanced_fields(ev, db_path=db_path, settings=settings)
        new_sign = int(enriched.get("enhanced_pred_sign", 0))
        new_conf = float(enriched.get("enhanced_confidence", 0.0))
        if new_sign != prior_sign:
            flip_count += 1
        confidence_delta_sum += new_conf - prior_conf
        confidence_delta_abs_sum += abs(new_conf - prior_conf)
        rescored.append(enriched)
        if idx % step == 0 or idx == total:
            print(f"  {idx}/{total} rescored", flush=True)

    payload["events"] = rescored
    payload["rescored_at_utc"] = datetime.now(timezone.utc).isoformat()
    payload["rescored_summary"] = {
        "events": total,
        "sign_flips_vs_prior": flip_count,
        "mean_confidence_delta": round(confidence_delta_sum / total, 6),
        "mean_abs_confidence_delta": round(confidence_delta_abs_sum / total, 6),
        "source": "scripts/rescore_case4_predictions.py",
    }

    target.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print()
    print(json.dumps(payload["rescored_summary"], indent=2))
    print(f"Wrote {target.as_posix()}")


if __name__ == "__main__":
    main()
