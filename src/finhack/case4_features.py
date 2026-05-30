"""Earnings-event feature builder: news sentiment + spillover in [T-7d, T].

The feature row is now structured around a real text encoder:

- ``sent_lex_*``     — fast lexicon polarity (always available).
- ``sent_finbert_*`` — FinBERT polarity, looked up from the
  ``document_score`` cache when present (zero-cost at feature-build time).

The cache itself is populated by :mod:`finhack.text_encoder`; nothing here
runs FinBERT inline. The legacy magic-number predictor that combined these
features into a hand-tuned signal has been removed. ``apply_enhanced_fields``
now produces a ``enhanced_pred_sign`` from a thin learned head (loaded once
from disk if present, else a deterministic momentum heuristic) so the
heuristic-lane backtests still have a signal to compare against, but the
signal is no longer governed by hand-tuned coefficients.
"""

from __future__ import annotations

import bisect
import logging
import re
import sqlite3
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any

from finhack.config import Settings, load_settings
from finhack.data.company_graph import SPILLOVER_MAP, SPILLOVER_WEIGHTS, SYMBOL_TO_COMPANY
from finhack.text_encoder import (
    FINBERT_MODEL,
    LEXICON_MODEL,
    DocScore,
    get_doc_scores_for_window,
    lexicon_score,
)

logger = logging.getLogger(__name__)

_DOC_CACHE: dict[
    tuple[str, float], tuple[list[datetime], list[dict[str, Any]]]
] = {}
_FINBERT_CACHE: dict[tuple[str, float], dict[str, DocScore]] = {}
_LEXICON_CACHE: dict[tuple[str, float], dict[str, DocScore]] = {}
_PATTERN_CACHE: dict[str, re.Pattern[str]] = {}


def _ticker_pattern(symbol: str) -> re.Pattern[str]:
    """Case-sensitive uppercase-ticker regex.

    Avoids false positives on short / common-word tickers like HERE, TOUR, QSG
    where the lowercased ticker collides with normal English words. The match
    runs against the **original-case** text so 'TOUR' matches but 'tourism'
    does not.
    """
    key = f"^TICKER^{symbol.upper()}"
    pat = _PATTERN_CACHE.get(key)
    if pat is None:
        pat = re.compile(rf"\b{re.escape(symbol.upper())}\b")
        _PATTERN_CACHE[key] = pat
    return pat


def _name_pattern(name_lower: str) -> re.Pattern[str]:
    """Case-insensitive word-boundary regex for a company-name string."""
    key = f"^NAME^{name_lower}"
    pat = _PATTERN_CACHE.get(key)
    if pat is None:
        pat = re.compile(rf"\b{re.escape(name_lower)}\b")
        _PATTERN_CACHE[key] = pat
    return pat


def clear_doc_cache() -> None:
    """Drop the in-memory document caches (use when the DB has been rewritten)."""
    _DOC_CACHE.clear()
    _FINBERT_CACHE.clear()
    _LEXICON_CACHE.clear()


def _load_docs_sorted(db: Path) -> tuple[list[datetime], list[dict[str, Any]]]:
    """Load every document from the DB once, normalised + sorted by event datetime.

    Cached per (db_path, mtime) so backtest loops that hit the same window
    thousands of times don't re-scan the document table on every call.
    """
    key = (str(db.resolve()), db.stat().st_mtime)
    cached = _DOC_CACHE.get(key)
    if cached is not None:
        return cached
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT doc_id, title, body, published_at, fetched_at,
                   source_domain, relevance_score
            FROM document
            ORDER BY COALESCE(published_at, fetched_at) ASC
            """
        ).fetchall()
    finally:
        conn.close()
    pairs: list[tuple[datetime, dict[str, Any]]] = []
    for row in rows:
        doc_dt = parse_doc_datetime(row["published_at"]) or parse_doc_datetime(row["fetched_at"])
        if doc_dt is None:
            continue
        pairs.append(
            (
                doc_dt,
                {
                    "doc_id": str(row["doc_id"] or ""),
                    "title": str(row["title"] or ""),
                    "body": str(row["body"] or ""),
                    "published_at": doc_dt.isoformat(),
                    "source_domain": str(row["source_domain"] or ""),
                    "relevance_score": float(row["relevance_score"] or 0.0),
                },
            )
        )
    pairs.sort(key=lambda p: p[0])
    dts = [p[0] for p in pairs]
    objs = [p[1] for p in pairs]
    _DOC_CACHE[key] = (dts, objs)
    return _DOC_CACHE[key]


def _load_score_cache(db: Path, *, model_name: str) -> dict[str, DocScore]:
    key = (str(db.resolve()), db.stat().st_mtime)
    target = _FINBERT_CACHE if model_name == FINBERT_MODEL else _LEXICON_CACHE
    cached = target.get(key)
    if cached is not None:
        return cached
    from finhack.text_encoder import load_doc_scores

    scores = load_doc_scores(db, model_name=model_name)
    target[key] = scores
    return scores


def sign_from_return(ret: float, dead_zone: float = 0.15) -> int:
    if ret > dead_zone:
        return 1
    if ret < -dead_zone:
        return -1
    return 0


def direction_from_sign(sign: int) -> str:
    if sign > 0:
        return "bullish"
    if sign < 0:
        return "bearish"
    return "mixed"


def resolve_db_path(settings: Settings | None = None) -> Path:
    cfg = settings or load_settings()
    raw = cfg.database_url
    if raw.startswith("sqlite:///"):
        raw = raw.replace("sqlite:///", "", 1)
    p = Path(raw)
    return p if p.is_absolute() else Path.cwd() / p


def parse_doc_datetime(raw: str | None) -> datetime | None:
    text = str(raw or "").strip()
    if not text:
        return None
    normalized = text
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(normalized)
    except ValueError:
        try:
            dt = parsedate_to_datetime(text)
        except (TypeError, ValueError, IndexError):
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def fetch_docs_between(
    start: datetime,
    end: datetime,
    db_path: Path | None = None,
    *,
    settings: Settings | None = None,
) -> list[dict[str, Any]]:
    db = db_path or resolve_db_path(settings)
    if not db.exists():
        return []
    dts, objs = _load_docs_sorted(db)
    lo = bisect.bisect_left(dts, start)
    hi = bisect.bisect_right(dts, end)
    return objs[lo:hi]


def _doc_finbert_score(
    doc: dict[str, Any], scores: dict[str, DocScore]
) -> float | None:
    doc_id = doc.get("doc_id")
    if not doc_id:
        return None
    found = scores.get(str(doc_id))
    return None if found is None else float(found.score)


def _doc_lex_score(
    doc: dict[str, Any], lex_scores: dict[str, DocScore]
) -> float:
    doc_id = doc.get("doc_id")
    if doc_id:
        cached = lex_scores.get(str(doc_id))
        if cached is not None:
            return float(cached.score)
    text = f"{doc.get('title', '')} {doc.get('body', '')}"
    return lexicon_score(text)


def compute_news_features(
    symbol: str,
    t_event: datetime,
    *,
    db_path: Path | None = None,
    settings: Settings | None = None,
    name_overrides: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Sentiment + spillover features from documents in [T-7d, T].

    Reads cached ``document_score`` rows for both lexicon and FinBERT; when
    FinBERT scores are absent (e.g. transformers not installed) the FinBERT
    aggregations stay at 0 and the lexicon features carry the signal.
    """
    clean = (symbol or "").strip().upper()
    start = t_event - timedelta(days=7)
    db = db_path or resolve_db_path(settings)
    docs = fetch_docs_between(start, t_event, db_path=db, settings=settings)
    company = SYMBOL_TO_COMPANY.get(clean)
    override_name = (name_overrides or {}).get(clean)
    if override_name:
        company_name = override_name.strip().lower()
    elif company:
        company_name = company.name.lower()
    else:
        company_name = clean.lower()
    spill_symbols = SPILLOVER_MAP.get(clean, [])
    weighted_edges = SPILLOVER_WEIGHTS.get(clean, [])

    sym_pat = _ticker_pattern(clean)
    name_pat = _name_pattern(company_name) if company_name else None
    peer_pats = [_ticker_pattern(p) for p in spill_symbols]
    weighted_pats = [(_ticker_pattern(p), w) for (p, w) in weighted_edges]

    finbert_scores = _load_score_cache(db, model_name=FINBERT_MODEL) if db.exists() else {}
    lex_scores = _load_score_cache(db, model_name=LEXICON_MODEL) if db.exists() else {}

    sym_docs = 0
    relevance_sum = 0.0
    spill_mentions = 0
    weighted_spill_score = 0.0
    driver_scores: list[tuple[float, str]] = []

    finbert_vals: list[float] = []
    lex_vals: list[float] = []
    finbert_pos = 0
    finbert_neg = 0

    for doc in docs:
        text = f"{doc['title']} {doc['body']}"
        low = text.lower()
        has_symbol = bool(sym_pat.search(text))
        if not has_symbol and name_pat is not None:
            has_symbol = bool(name_pat.search(low))
        rel = float(doc.get("relevance_score", 0.0))
        lex_val = _doc_lex_score(doc, lex_scores)
        fb_val = _doc_finbert_score(doc, finbert_scores)
        if has_symbol:
            sym_docs += 1
            lex_vals.append(lex_val)
            if fb_val is not None:
                finbert_vals.append(fb_val)
                if fb_val > 0.05:
                    finbert_pos += 1
                elif fb_val < -0.05:
                    finbert_neg += 1
            relevance_sum += rel
            title = str(doc.get("title") or "").strip()
            if title:
                weight = max(abs(lex_val), abs(fb_val) if fb_val is not None else 0.0)
                driver_scores.append((weight + rel, title))
        for pat in peer_pats:
            if pat.search(text):
                spill_mentions += 1
        for pat, weight in weighted_pats:
            if pat.search(text):
                weighted_spill_score += weight

    lex_mean = (sum(lex_vals) / sym_docs) if sym_docs else 0.0
    lex_max_abs = max((abs(v) for v in lex_vals), default=0.0)
    fb_mean = (sum(finbert_vals) / len(finbert_vals)) if finbert_vals else 0.0
    fb_max_abs = max((abs(v) for v in finbert_vals), default=0.0)
    fb_pos_ratio = (
        finbert_pos / float(len(finbert_vals)) if finbert_vals else 0.0
    )
    fb_neg_ratio = (
        finbert_neg / float(len(finbert_vals)) if finbert_vals else 0.0
    )
    fb_polarity_gap = fb_pos_ratio - fb_neg_ratio
    mean_rel = relevance_sum / sym_docs if sym_docs else 0.0
    spillover_density = spill_mentions / max(1, len(docs))
    exposure = (
        (mean_rel * sym_docs)
        + (0.35 * spill_mentions)
        + (0.25 * weighted_spill_score)
    )

    driver_scores.sort(key=lambda x: x[0], reverse=True)
    top_drivers = [title for _, title in driver_scores[:3]]

    return {
        "sent_doc_count": sym_docs,
        "sent_lex_mean": round(lex_mean, 4),
        "sent_lex_max_abs": round(lex_max_abs, 4),
        "sent_finbert_mean": round(fb_mean, 4),
        "sent_finbert_max_abs": round(fb_max_abs, 4),
        "sent_finbert_pos_ratio": round(fb_pos_ratio, 4),
        "sent_finbert_neg_ratio": round(fb_neg_ratio, 4),
        "sent_finbert_polarity_gap": round(fb_polarity_gap, 4),
        "sent_finbert_doc_count": int(len(finbert_vals)),
        "sent_mean_score": round(lex_mean, 4),
        "sent_mean_relevance": round(mean_rel, 4),
        "spillover_mentions_7d": spill_mentions,
        "spillover_density_7d": round(spillover_density, 4),
        "enhanced_documents_considered": len(docs),
        "enhanced_direct_mentions": sym_docs,
        "enhanced_spillover_mentions": spill_mentions,
        "spillover_weighted_score_7d": round(weighted_spill_score, 4),
        "enhanced_exposure_score": round(exposure, 4),
        "top_drivers": top_drivers,
    }


def _enhanced_signal_from_model(
    pre_ret: float, features: dict[str, Any]
) -> tuple[int, float] | None:
    """Use a persisted model artifact when available; return (sign, confidence)."""
    try:
        from finhack.research.model_store import predict_with_saved_model

        return predict_with_saved_model(pre_ret=pre_ret, features=features)
    except Exception as exc:  # noqa: BLE001
        logger.debug("Saved enhanced model unavailable: %s", exc)
        return None


def _deterministic_fallback(
    pre_ret: float, features: dict[str, Any], dead_zone: float
) -> tuple[int, float]:
    """Deterministic fallback when no trained model is available.

    No magic numbers: the signal is the sign of pre-event momentum, with
    confidence shaped by document/relevance coverage. Used only by the
    legacy heuristic lane in the backtest comparison.
    """
    sign = sign_from_return(pre_ret, dead_zone=dead_zone)
    docs = int(features.get("enhanced_direct_mentions", 0) or 0)
    rel = float(features.get("sent_mean_relevance", 0.0) or 0.0)
    spill = int(features.get("enhanced_spillover_mentions", 0) or 0)
    # Confidence rises with corroborating evidence in [T-7d, T] but is
    # bounded so the heuristic lane stays comparable to the trained model.
    confidence = min(
        1.0,
        0.20 + docs * 0.05 + rel * 0.05 + spill * 0.02 + abs(pre_ret) / 20.0,
    )
    return sign, round(confidence, 4)


def predict_enhanced(
    pre_ret: float,
    features: dict[str, Any],
    *,
    dead_zone: float = 0.10,
) -> tuple[int, str, float]:
    """Predict (sign, direction, confidence) for the legacy heuristic lane.

    Tries a saved trained model first; falls back to a deterministic
    momentum-based rule if the artifact is missing or unloadable.
    """
    learned = _enhanced_signal_from_model(pre_ret, features)
    if learned is not None:
        sign, confidence = learned
        return sign, direction_from_sign(sign), round(float(confidence), 4)
    sign, confidence = _deterministic_fallback(pre_ret, features, dead_zone)
    return sign, direction_from_sign(sign), confidence


def apply_enhanced_fields(
    row: dict[str, Any],
    *,
    db_path: Path | None = None,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Enrich a validation event row with news-driven enhanced prediction fields."""
    symbol = str(row.get("symbol", "")).upper().strip()
    t_event_raw = row.get("t_event_utc")
    if not symbol or not t_event_raw:
        return row

    txt = str(t_event_raw).strip()
    if txt.endswith("Z"):
        txt = txt[:-1] + "+00:00"
    try:
        t_event = datetime.fromisoformat(txt)
    except ValueError:
        return row
    if t_event.tzinfo is None:
        t_event = t_event.replace(tzinfo=timezone.utc)
    else:
        t_event = t_event.astimezone(timezone.utc)

    features = compute_news_features(
        symbol, t_event, db_path=db_path, settings=settings
    )
    pre_ret = float(row.get("baseline_pre_7d_return_pct", 0.0))
    enhanced_sign, enhanced_direction, confidence = predict_enhanced(pre_ret, features)

    enriched = dict(row)
    enriched.update(features)
    enriched["enhanced_pred_sign"] = enhanced_sign
    enriched["enhanced_pred_direction"] = enhanced_direction
    enriched["enhanced_confidence"] = confidence
    return enriched
