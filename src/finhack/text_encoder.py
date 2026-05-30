"""Document-level sentiment encoder with FinBERT + lexicon fallback.

The expensive part of sentiment scoring (FinBERT inference) runs once per
document and is cached in a `document_score` table inside the same SQLite DB
used for the news corpus. Re-running validation, training, or permutation
tests after an initial scoring pass costs only a SQL read.

Public surface:

- ``score_document(title, body)`` → cheap, in-process polarity in [-1, 1].
- ``ensure_doc_scores(db_path, model_name)`` → ensure every document has a
  cached encoder score; idempotent. Falls back to lexicon if FinBERT can't
  be loaded.
- ``load_doc_scores(db_path, model_name)`` → read scores keyed by ``doc_id``.

The module deliberately avoids importing transformers at module load so a
fresh checkout still works without torch installed.
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger(__name__)

LEXICON_MODEL = "lexicon-v1"
FINBERT_MODEL = "prosusai-finbert-v1"

POSITIVE_TERMS = (
    "beat",
    "beats",
    "exceed",
    "exceeded",
    "growth",
    "strong",
    "surge",
    "surged",
    "upgrade",
    "upgraded",
    "bullish",
    "record",
    "expansion",
    "outperform",
    "raise",
    "raised",
    "raises",
    "boost",
    "boosted",
    "rally",
    "rallied",
    "guidance higher",
    "tops estimates",
    "above expectations",
)

NEGATIVE_TERMS = (
    "miss",
    "missed",
    "misses",
    "downgrade",
    "downgraded",
    "weak",
    "decline",
    "declined",
    "bearish",
    "lawsuit",
    "delay",
    "delayed",
    "cut",
    "cuts",
    "lowered",
    "lowers",
    "underperform",
    "warning",
    "guidance cut",
    "below expectations",
    "fall",
    "fell",
    "plunge",
    "plunged",
    "loss",
    "losses",
    "probe",
    "investigation",
)


def _ensure_score_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS document_score (
            doc_id TEXT NOT NULL,
            model_name TEXT NOT NULL,
            score REAL NOT NULL,
            label TEXT,
            scored_at TEXT NOT NULL,
            PRIMARY KEY (doc_id, model_name)
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_document_score_model "
        "ON document_score (model_name)"
    )


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    _ensure_score_table(conn)
    return conn


def lexicon_score(text: str) -> float:
    """Cheap bag-of-words polarity in [-1, 1]."""
    if not text:
        return 0.0
    low = text.lower()
    pos = sum(low.count(k) for k in POSITIVE_TERMS)
    neg = sum(low.count(k) for k in NEGATIVE_TERMS)
    if pos == 0 and neg == 0:
        return 0.0
    return (pos - neg) / float(pos + neg)


def score_document(title: str, body: str) -> float:
    """Synchronous, in-process lexicon score (kept for callers that need
    a non-cached value)."""
    return lexicon_score(f"{title or ''} {body or ''}")


@dataclass(slots=True)
class DocScore:
    score: float
    label: str | None
    model_name: str


def load_doc_scores(
    db_path: Path,
    *,
    model_name: str = LEXICON_MODEL,
) -> dict[str, DocScore]:
    """Return cached scores keyed by ``doc_id`` for the given model."""
    if not Path(db_path).exists():
        return {}
    conn = _connect(Path(db_path))
    try:
        rows = conn.execute(
            "SELECT doc_id, score, label FROM document_score WHERE model_name = ?",
            (model_name,),
        ).fetchall()
    finally:
        conn.close()
    return {
        str(r["doc_id"]): DocScore(
            score=float(r["score"]),
            label=str(r["label"]) if r["label"] is not None else None,
            model_name=model_name,
        )
        for r in rows
    }


def _load_unscored_docs(
    conn: sqlite3.Connection,
    *,
    model_name: str,
    limit: int | None = None,
) -> list[sqlite3.Row]:
    sql = (
        "SELECT d.doc_id, d.title, d.body FROM document d "
        "LEFT JOIN document_score s "
        "ON s.doc_id = d.doc_id AND s.model_name = ? "
        "WHERE s.doc_id IS NULL"
    )
    params: tuple[Any, ...] = (model_name,)
    if limit:
        sql += " LIMIT ?"
        params = params + (int(limit),)
    return list(conn.execute(sql, params).fetchall())


def _persist_scores(
    conn: sqlite3.Connection,
    *,
    model_name: str,
    rows: Iterable[tuple[str, float, str | None]],
) -> int:
    now = datetime.now(timezone.utc).isoformat()
    payload = [(doc_id, model_name, float(score), label, now) for doc_id, score, label in rows]
    if not payload:
        return 0
    conn.executemany(
        "INSERT OR REPLACE INTO document_score "
        "(doc_id, model_name, score, label, scored_at) VALUES (?, ?, ?, ?, ?)",
        payload,
    )
    return len(payload)


def ensure_lexicon_scores(db_path: Path, *, batch_size: int = 1000) -> int:
    """Score every uncached document with the lexicon scorer. Returns rows written."""
    if not Path(db_path).exists():
        return 0
    written = 0
    conn = _connect(Path(db_path))
    try:
        while True:
            rows = _load_unscored_docs(conn, model_name=LEXICON_MODEL, limit=batch_size)
            if not rows:
                break
            triples: list[tuple[str, float, str | None]] = []
            for r in rows:
                text = f"{r['title'] or ''} {r['body'] or ''}"
                score = lexicon_score(text)
                if score > 0.05:
                    label = "positive"
                elif score < -0.05:
                    label = "negative"
                else:
                    label = "neutral"
                triples.append((str(r["doc_id"]), score, label))
            written += _persist_scores(conn, model_name=LEXICON_MODEL, rows=triples)
            conn.commit()
    finally:
        conn.close()
    return written


def _try_load_finbert():  # noqa: ANN202 - lazy import return
    """Try to load FinBERT via transformers. Return (tokenizer, model, device) or None."""
    try:
        import torch  # type: ignore
        from transformers import (  # type: ignore
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )
    except Exception as exc:  # noqa: BLE001
        logger.info("FinBERT unavailable (transformers/torch not installed): %s", exc)
        return None
    try:
        name = "ProsusAI/finbert"
        tokenizer = AutoTokenizer.from_pretrained(name)
        model = AutoModelForSequenceClassification.from_pretrained(name)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()
        return tokenizer, model, device, torch
    except Exception as exc:  # noqa: BLE001
        logger.warning("FinBERT load failed: %s", exc)
        return None


def _finbert_score_batch(
    bundle,
    texts: list[str],
    *,
    max_length: int = 256,
) -> list[tuple[float, str]]:
    """Run FinBERT on a list of texts. Returns (score, label) per text.

    score in [-1, 1] = P(positive) - P(negative); label is the argmax class
    among {positive, negative, neutral}.
    """
    tokenizer, model, device, torch = bundle
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        logits = model(**enc).logits
    probs = logits.softmax(dim=-1).cpu().numpy()
    id2label = model.config.id2label
    positive_idx = next(
        (i for i, lab in id2label.items() if str(lab).lower().startswith("pos")), 0
    )
    negative_idx = next(
        (i for i, lab in id2label.items() if str(lab).lower().startswith("neg")), 1
    )
    out: list[tuple[float, str]] = []
    for row in probs:
        score = float(row[positive_idx] - row[negative_idx])
        label_idx = int(row.argmax())
        label = str(id2label.get(label_idx, label_idx)).lower()
        out.append((score, label))
    return out


def ensure_finbert_scores(
    db_path: Path,
    *,
    batch_size: int = 32,
    max_documents: int | None = None,
    progress: bool = False,
) -> tuple[int, str]:
    """Ensure every document has a FinBERT score cached.

    Returns ``(rows_written, model_used)`` where ``model_used`` is either
    ``FINBERT_MODEL`` on success or ``LEXICON_MODEL`` if FinBERT could not
    be loaded and we wrote lexicon scores instead.
    """
    if not Path(db_path).exists():
        return 0, LEXICON_MODEL
    bundle = _try_load_finbert()
    if bundle is None:
        if progress:
            print(
                "[encoder] FinBERT not available; falling back to lexicon scores",
                flush=True,
            )
        return ensure_lexicon_scores(db_path), LEXICON_MODEL

    written = 0
    conn = _connect(Path(db_path))
    try:
        remaining = max_documents
        while True:
            limit = batch_size
            if remaining is not None:
                if remaining <= 0:
                    break
                limit = min(batch_size, remaining)
            rows = _load_unscored_docs(conn, model_name=FINBERT_MODEL, limit=limit)
            if not rows:
                break
            texts = [
                f"{r['title'] or ''} {r['body'] or ''}".strip() or " "
                for r in rows
            ]
            scored = _finbert_score_batch(bundle, texts)
            triples = [
                (str(rows[i]["doc_id"]), score, label)
                for i, (score, label) in enumerate(scored)
            ]
            written += _persist_scores(conn, model_name=FINBERT_MODEL, rows=triples)
            conn.commit()
            if progress:
                print(f"[encoder] FinBERT scored {written} documents...", flush=True)
            if remaining is not None:
                remaining -= len(rows)
    finally:
        conn.close()
    return written, FINBERT_MODEL


def get_doc_scores_for_window(
    db_path: Path,
    doc_ids: Iterable[str],
    *,
    model_name: str,
) -> dict[str, DocScore]:
    """Look up cached scores for a specific set of documents."""
    ids = [str(d) for d in doc_ids]
    if not ids or not Path(db_path).exists():
        return {}
    out: dict[str, DocScore] = {}
    conn = _connect(Path(db_path))
    try:
        # SQLite has a limit on parameter count; chunk just in case.
        chunk = 500
        for i in range(0, len(ids), chunk):
            batch = ids[i : i + chunk]
            placeholders = ",".join("?" for _ in batch)
            rows = conn.execute(
                f"SELECT doc_id, score, label FROM document_score "
                f"WHERE model_name = ? AND doc_id IN ({placeholders})",
                (model_name, *batch),
            ).fetchall()
            for r in rows:
                out[str(r["doc_id"])] = DocScore(
                    score=float(r["score"]),
                    label=str(r["label"]) if r["label"] is not None else None,
                    model_name=model_name,
                )
    finally:
        conn.close()
    return out
