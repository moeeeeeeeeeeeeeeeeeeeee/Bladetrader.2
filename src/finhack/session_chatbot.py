"""Database-backed session chatbot memory for UI integration."""

from __future__ import annotations

from datetime import datetime, timezone
import sqlite3
from threading import Lock
from uuid import uuid4

from finhack.config import load_settings

_SESSIONS: dict[str, list[dict[str, str]]] = {}
_LOCK = Lock()


def _new_session_id() -> str:
    return f"sess_{uuid4().hex[:16]}"


def _db_path() -> str:
    raw = load_settings().database_url
    if raw.startswith("sqlite:///"):
        raw = raw.replace("sqlite:///", "", 1)
    return raw


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(_db_path())
    conn.row_factory = sqlite3.Row
    return conn


def _ensure_chat_table() -> None:
    with _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_turn (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_chat_turn_session
            ON chat_turn (session_id, id)
            """
        )


def _build_reply(message: str, context: dict[str, object] | None) -> str:
    lower = message.lower()
    if "news" in lower:
        return (
            "I can summarize recent AI-market news and map likely 5-7 day impacts."
        )
    if "sector" in lower or "spillover" in lower or "risk" in lower:
        return (
            "Sector intelligence uses metrics A/B/C/D across the 14 tracked stocks to estimate "
            "sector movement and spillover links."
        )
    if "earnings" in lower:
        return (
            "Case 4 mode is focused on a leakage-safe 7-day pre-earnings window and "
            "predicting five-day post-earnings direction."
        )
    if context:
        keys = ", ".join(sorted(str(k) for k in context.keys())[:5])
        return f"I received your context ({keys}). Ask for sector, news, risk, or earnings guidance."
    return "I am ready. Ask about sector outlook, spillover links, news, or earnings-direction validation."


def answer_user_question(
    message: str,
    session_id: str | None = None,
    context: dict[str, object] | None = None,
) -> tuple[str, str]:
    """Return chatbot reply and session id."""
    cleaned = (message or "").strip()
    if not cleaned:
        raise ValueError("Message cannot be empty.")

    sid = (session_id or "").strip() or _new_session_id()
    reply = _build_reply(cleaned, context)

    _ensure_chat_table()
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    with _connect() as conn:
        conn.execute(
            "INSERT INTO chat_turn (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (sid, "user", cleaned, now),
        )
        conn.execute(
            "INSERT INTO chat_turn (session_id, role, content, created_at) VALUES (?, ?, ?, ?)",
            (sid, "assistant", reply, now),
        )

    with _LOCK:
        history = _SESSIONS.setdefault(sid, [])
        history.append({"role": "user", "content": cleaned})
        history.append({"role": "assistant", "content": reply})
    return reply, sid


def get_conversation_history(session_id: str) -> list[dict[str, str]]:
    """Return prior turns for a session id."""
    sid = (session_id or "").strip()
    if not sid:
        return []
    _ensure_chat_table()
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT role, content
            FROM chat_turn
            WHERE session_id = ?
            ORDER BY id ASC
            LIMIT 200
            """,
            (sid,),
        ).fetchall()
    if rows:
        return [{"role": str(r["role"]), "content": str(r["content"])} for r in rows]
    with _LOCK:
        turns = _SESSIONS.get(sid, [])
        return [dict(t) for t in turns]
