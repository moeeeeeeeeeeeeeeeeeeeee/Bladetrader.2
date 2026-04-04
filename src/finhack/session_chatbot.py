"""Simple session chatbot memory for UI integration.

This keeps the API deployable while backend LLM behavior evolves.
"""

from __future__ import annotations

from threading import Lock
from uuid import uuid4

_SESSIONS: dict[str, list[dict[str, str]]] = {}
_LOCK = Lock()


def _new_session_id() -> str:
    return f"sess_{uuid4().hex[:16]}"


def _build_reply(message: str, context: dict[str, object] | None) -> str:
    lower = message.lower()
    if "news" in lower:
        return (
            "I can help summarize recent news and explain likely market direction over "
            "the next five trading days once documents are available."
        )
    if "exposure" in lower or "spillover" in lower:
        return (
            "Exposure analysis uses direct mentions plus cross-company spillover signals. "
            "Share a ticker and I can guide the next API call."
        )
    if "earnings" in lower:
        return (
            "Case 4 mode is focused on a leakage-safe 7-day pre-earnings window and "
            "predicting five-day post-earnings direction."
        )
    if context:
        keys = ", ".join(sorted(str(k) for k in context.keys())[:5])
        return f"I received your context ({keys}). Ask for news, exposure, or earnings guidance."
    return "I am ready. Ask about portfolio news, ticker exposure, or earnings-direction validation."


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
    with _LOCK:
        turns = _SESSIONS.get(sid, [])
        return [dict(t) for t in turns]
