"""Tiny end-to-end smoke for the refactored Case 4 pipeline.

Run from the repo root:
    py scripts/smoke_test_case4.py
"""

from __future__ import annotations

import sqlite3
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from finhack.research.case4_dataset import (  # noqa: E402
    BASELINE_FEATURES,
    ENHANCED_FEATURES,
    fit_only,
    fit_predict,
    news_coverage_summary,
    split_time_aware,
)
from finhack.research.model_store import (  # noqa: E402
    DEFAULT_ARTIFACT_PATH,
    load_trained_model,
    predict_proba,
    predict_with_saved_model,
    save_trained_model,
)
from finhack.text_encoder import (  # noqa: E402
    FINBERT_MODEL,
    LEXICON_MODEL,
    ensure_finbert_scores,
    ensure_lexicon_scores,
    lexicon_score,
    load_doc_scores,
)


def make_synthetic_frame(n: int = 80, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base_dt = datetime(2024, 1, 1, tzinfo=timezone.utc)
    rows = []
    for i in range(n):
        pre7 = float(rng.normal(0.0, 4.0))
        pre30 = float(rng.normal(0.0, 8.0))
        finbert_signal = float(rng.normal(0.0, 0.4))
        # Construct a label that has a real linear-ish relationship with
        # the features, so the trained model should beat the baseline.
        score = (
            0.45 * np.sign(pre7)
            + 0.35 * np.sign(finbert_signal)
            + rng.normal(0.0, 0.5)
        )
        sign = 1 if score > 0 else 0
        rows.append(
            {
                "symbol": f"SYM{i % 5}",
                "t_event_utc": base_dt + timedelta(days=i * 3),
                "actual_sign": sign,
                "actual_5d_return_pct": float(rng.normal(0.0, 5.0)),
                "baseline_pred_sign": 1 if pre7 > 0 else -1,
                "enhanced_pred_sign": 0,
                "enhanced_confidence": 0.0,
                "baseline_pre_7d_return_pct": pre7,
                "baseline_pre_7d_abs_return": abs(pre7),
                "pre_30d_return_pct": pre30,
                "pre_60d_return_pct": float(rng.normal(0.0, 12.0)),
                "pre_30d_vol_pct": float(abs(rng.normal(2.0, 1.0))),
                "prior_post_earnings_5d_return_pct": float(rng.normal(0.0, 5.0)),
                "sector_cohort_5d_return_pct": float(rng.normal(0.0, 3.0)),
                "neighbor_earnings_count_30d": int(rng.integers(0, 4)),
                "sent_doc_count": int(rng.integers(0, 6)),
                "sent_lex_mean": float(rng.normal(0.0, 0.3)),
                "sent_lex_max_abs": float(abs(rng.normal(0.0, 0.5))),
                "sent_finbert_mean": finbert_signal,
                "sent_finbert_max_abs": abs(finbert_signal),
                "sent_finbert_pos_ratio": float(rng.uniform(0.0, 1.0)),
                "sent_finbert_neg_ratio": float(rng.uniform(0.0, 1.0)),
                "sent_finbert_polarity_gap": finbert_signal,
                "sent_finbert_doc_count": int(rng.integers(0, 6)),
                "sent_mean_score": 0.0,
                "sent_mean_relevance": float(rng.uniform(0.0, 1.0)),
                "spillover_mentions_7d": int(rng.integers(0, 5)),
                "spillover_density_7d": float(rng.uniform(0.0, 0.5)),
                "spillover_weighted_score_7d": float(rng.uniform(0.0, 2.0)),
                "enhanced_documents_considered": int(rng.integers(0, 12)),
            }
        )
    df = pd.DataFrame(rows)
    df = df.sort_values("t_event_utc").reset_index(drop=True)
    return df


def check_text_encoder() -> None:
    print("[smoke] text_encoder lexicon score:", lexicon_score("strong beat record growth"))
    assert lexicon_score("strong beat record growth") > 0
    assert lexicon_score("miss downgrade lawsuit cut") < 0
    assert lexicon_score("") == 0.0

    # Build a minimal SQLite DB with the document table + a couple of rows
    # and exercise the cache helpers without a real corpus.
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "smoke.db"
        conn = sqlite3.connect(str(db))
        conn.execute(
            """
            CREATE TABLE document (
                doc_id TEXT PRIMARY KEY,
                url TEXT NOT NULL UNIQUE,
                published_at TEXT,
                fetched_at TEXT NOT NULL,
                source TEXT NOT NULL,
                source_url TEXT,
                source_domain TEXT NOT NULL,
                title TEXT,
                body TEXT,
                keyword_hits TEXT NOT NULL,
                relevance_score REAL NOT NULL,
                query TEXT NOT NULL
            )
            """
        )
        conn.executemany(
            "INSERT INTO document (doc_id, url, published_at, fetched_at, source, "
            "source_url, source_domain, title, body, keyword_hits, relevance_score, query) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    "d1",
                    "https://x/1",
                    None,
                    datetime.now(timezone.utc).isoformat(),
                    "test",
                    None,
                    "test.com",
                    "Strong beat",
                    "earnings beat estimates with record growth",
                    "[]",
                    1.0,
                    "q",
                ),
                (
                    "d2",
                    "https://x/2",
                    None,
                    datetime.now(timezone.utc).isoformat(),
                    "test",
                    None,
                    "test.com",
                    "Lawsuit warning",
                    "company warning lawsuit and downgrade",
                    "[]",
                    1.0,
                    "q",
                ),
            ],
        )
        conn.commit()
        conn.close()

        added = ensure_lexicon_scores(db)
        assert added == 2, f"expected 2 lexicon scores, got {added}"
        scores = load_doc_scores(db, model_name=LEXICON_MODEL)
        assert "d1" in scores and scores["d1"].score > 0, scores
        assert "d2" in scores and scores["d2"].score < 0, scores

        # FinBERT should fall back to lexicon because transformers isn't
        # installed in this environment.
        written, model_used = ensure_finbert_scores(db)
        assert model_used in (LEXICON_MODEL, FINBERT_MODEL)
        print(
            f"[smoke] encoder wrote {written} rows ({model_used}); "
            "this is the lexicon-fallback path when FinBERT is unavailable"
        )


def check_model_store(df: pd.DataFrame) -> None:
    train, test = split_time_aware(df, train_ratio=0.7)
    base_pred, _, base_metrics = fit_predict(train, test, BASELINE_FEATURES)
    enh_pred, enh_proba, enh_metrics = fit_predict(train, test, ENHANCED_FEATURES)
    print(
        f"[smoke] baseline acc={base_metrics['accuracy']:.3f} "
        f"enhanced acc={enh_metrics['accuracy']:.3f} "
        f"(uplift {enh_metrics['accuracy'] - base_metrics['accuracy']:+.3f})"
    )
    assert 0.0 <= enh_metrics["accuracy"] <= 1.0
    assert enh_proba.shape == (len(test),)

    full = fit_only(df, ENHANCED_FEATURES)
    assert full is not None, "fit_only returned None on synthetic data"
    print(f"[smoke] backend={full.backend} train_events={full.train_events}")
    artifact = DEFAULT_ARTIFACT_PATH.with_name("case4_enhanced_smoke.pkl")
    save_trained_model(full, artifact)
    loaded = load_trained_model(artifact)
    assert loaded is not None
    proba = predict_proba(loaded, np.zeros((1, len(loaded.feature_names))))
    assert proba.shape == (1,) and 0.0 <= float(proba[0]) <= 1.0
    print(f"[smoke] persisted+reloaded model proba on zero-vector: {float(proba[0]):.3f}")

    # round-trip through the public predict_with_saved_model API the same
    # way paper_signals will at runtime.
    feats = {name: 0.0 for name in loaded.feature_names}
    sign, conf = predict_with_saved_model(
        pre_ret=2.5, features=feats, artifact_path=artifact
    )
    print(f"[smoke] predict_with_saved_model: sign={sign} confidence={conf:.3f}")
    artifact.unlink(missing_ok=True)
    artifact.with_suffix(".meta.json").unlink(missing_ok=True)


def check_coverage(df: pd.DataFrame) -> None:
    summary = news_coverage_summary(df)
    print(
        f"[smoke] coverage: events={summary['events_total']} "
        f"with_news={summary['events_with_any_news']} "
        f"finbert={summary['events_with_finbert_score']} "
        f"low_cov_syms={len(summary['low_coverage_symbols'])}"
    )


def main() -> int:
    df = make_synthetic_frame()
    print(f"[smoke] synthetic frame: {len(df)} events")
    check_text_encoder()
    check_model_store(df)
    check_coverage(df)
    print("[smoke] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
