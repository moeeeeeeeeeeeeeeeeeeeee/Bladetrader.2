"""Persistent futures trade journal (SQLite).

Manual log for trades you take on TopStep / Tradovate / etc. while this
program supplies the signal, levels, and backtest context. Not broker sync.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from finhack.case4_features import resolve_db_path
from finhack.config import Settings, load_settings


@dataclass(slots=True)
class FuturesTrade:
    trade_id: str
    created_at_utc: str
    session_date: str | None
    instrument: str
    direction: str
    contracts: int
    entry_price: float | None
    exit_price: float | None
    stop_price: float | None
    target_price: float | None
    pnl_usd: float | None
    pnl_pct: float | None
    status: str
    signal_confidence: float | None
    signal_contributors: list[str]
    notes: str
    tags: list[str]


def _db_path(settings: Settings | None = None) -> Path:
    return resolve_db_path(settings or load_settings())


def ensure_journal_schema(db: Path | None = None) -> None:
    path = db or _db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS futures_trade (
                trade_id TEXT PRIMARY KEY,
                created_at_utc TEXT NOT NULL,
                session_date TEXT,
                instrument TEXT NOT NULL,
                direction TEXT NOT NULL,
                contracts INTEGER NOT NULL DEFAULT 1,
                entry_price REAL,
                exit_price REAL,
                stop_price REAL,
                target_price REAL,
                pnl_usd REAL,
                pnl_pct REAL,
                status TEXT NOT NULL DEFAULT 'planned',
                signal_confidence REAL,
                signal_contributors TEXT NOT NULL DEFAULT '[]',
                notes TEXT NOT NULL DEFAULT '',
                tags TEXT NOT NULL DEFAULT '[]'
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_futures_trade_session
            ON futures_trade (session_date)
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_futures_trade_status
            ON futures_trade (status)
            """
        )
        conn.commit()
    finally:
        conn.close()


def _row_to_trade(row: sqlite3.Row) -> FuturesTrade:
    try:
        contribs = json.loads(row["signal_contributors"] or "[]")
    except json.JSONDecodeError:
        contribs = []
    try:
        tags = json.loads(row["tags"] or "[]")
    except json.JSONDecodeError:
        tags = []
    return FuturesTrade(
        trade_id=str(row["trade_id"]),
        created_at_utc=str(row["created_at_utc"]),
        session_date=row["session_date"],
        instrument=str(row["instrument"]),
        direction=str(row["direction"]),
        contracts=int(row["contracts"] or 1),
        entry_price=row["entry_price"],
        exit_price=row["exit_price"],
        stop_price=row["stop_price"],
        target_price=row["target_price"],
        pnl_usd=row["pnl_usd"],
        pnl_pct=row["pnl_pct"],
        status=str(row["status"]),
        signal_confidence=row["signal_confidence"],
        signal_contributors=[str(x) for x in contribs] if isinstance(contribs, list) else [],
        notes=str(row["notes"] or ""),
        tags=[str(x) for x in tags] if isinstance(tags, list) else [],
    )


def list_trades(
    *,
    limit: int = 100,
    status: str | None = None,
    settings: Settings | None = None,
) -> list[FuturesTrade]:
    ensure_journal_schema()
    db = _db_path(settings)
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    try:
        if status:
            rows = conn.execute(
                """
                SELECT * FROM futures_trade
                WHERE status = ?
                ORDER BY created_at_utc DESC
                LIMIT ?
                """,
                (status, max(1, min(limit, 500))),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT * FROM futures_trade
                ORDER BY created_at_utc DESC
                LIMIT ?
                """,
                (max(1, min(limit, 500)),),
            ).fetchall()
    finally:
        conn.close()
    return [_row_to_trade(r) for r in rows]


def create_trade(
    *,
    instrument: str,
    direction: str,
    session_date: str | None = None,
    contracts: int = 1,
    entry_price: float | None = None,
    exit_price: float | None = None,
    stop_price: float | None = None,
    target_price: float | None = None,
    pnl_usd: float | None = None,
    pnl_pct: float | None = None,
    status: str = "planned",
    signal_confidence: float | None = None,
    signal_contributors: list[str] | None = None,
    notes: str = "",
    tags: list[str] | None = None,
    settings: Settings | None = None,
) -> FuturesTrade:
    ensure_journal_schema()
    db = _db_path(settings)
    trade_id = uuid.uuid4().hex[:16]
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    clean_dir = direction.strip().lower()
    if clean_dir not in {"long", "short"}:
        raise ValueError("direction must be 'long' or 'short'")
    clean_status = status.strip().lower()
    if clean_status not in {"planned", "open", "closed", "cancelled"}:
        raise ValueError("invalid status")

    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    try:
        conn.execute(
            """
            INSERT INTO futures_trade (
                trade_id, created_at_utc, session_date, instrument, direction,
                contracts, entry_price, exit_price, stop_price, target_price,
                pnl_usd, pnl_pct, status, signal_confidence,
                signal_contributors, notes, tags
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                trade_id,
                now,
                session_date,
                instrument.upper(),
                clean_dir,
                max(1, int(contracts)),
                entry_price,
                exit_price,
                stop_price,
                target_price,
                pnl_usd,
                pnl_pct,
                clean_status,
                signal_confidence,
                json.dumps(signal_contributors or []),
                notes,
                json.dumps(tags or []),
            ),
        )
        conn.commit()
        row = conn.execute(
            "SELECT * FROM futures_trade WHERE trade_id = ?", (trade_id,)
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        raise RuntimeError("trade insert failed")
    return _row_to_trade(row)


def update_trade(
    trade_id: str,
    *,
    updates: dict[str, Any],
    settings: Settings | None = None,
) -> FuturesTrade | None:
    ensure_journal_schema()
    allowed = {
        "session_date",
        "instrument",
        "direction",
        "contracts",
        "entry_price",
        "exit_price",
        "stop_price",
        "target_price",
        "pnl_usd",
        "pnl_pct",
        "status",
        "signal_confidence",
        "signal_contributors",
        "notes",
        "tags",
    }
    payload: dict[str, Any] = {}
    for key, val in updates.items():
        if key not in allowed:
            continue
        if key in {"signal_contributors", "tags"} and isinstance(val, list):
            payload[key] = json.dumps(val)
        else:
            payload[key] = val
    if not payload:
        trades = list_trades(limit=500, settings=settings)
        for t in trades:
            if t.trade_id == trade_id:
                return t
        return None

    sets = ", ".join(f"{k} = ?" for k in payload)
    db = _db_path(settings)
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    try:
        conn.execute(
            f"UPDATE futures_trade SET {sets} WHERE trade_id = ?",
            (*payload.values(), trade_id),
        )
        conn.commit()
        row = conn.execute(
            "SELECT * FROM futures_trade WHERE trade_id = ?", (trade_id,)
        ).fetchone()
    finally:
        conn.close()
    return _row_to_trade(row) if row else None


def delete_trade(trade_id: str, *, settings: Settings | None = None) -> bool:
    ensure_journal_schema()
    db = _db_path(settings)
    conn = sqlite3.connect(str(db))
    try:
        cur = conn.execute(
            "DELETE FROM futures_trade WHERE trade_id = ?", (trade_id,)
        )
        conn.commit()
        return cur.rowcount > 0
    finally:
        conn.close()


def journal_stats(*, settings: Settings | None = None) -> dict[str, Any]:
    trades = list_trades(limit=500, settings=settings)
    closed = [t for t in trades if t.status == "closed"]
    wins = [t for t in closed if (t.pnl_usd or 0) > 0]
    losses = [t for t in closed if (t.pnl_usd or 0) < 0]
    total_pnl = sum(t.pnl_usd or 0.0 for t in closed)
    open_count = sum(1 for t in trades if t.status == "open")
    planned_count = sum(1 for t in trades if t.status == "planned")
    return {
        "trades_total": len(trades),
        "trades_closed": len(closed),
        "trades_open": open_count,
        "trades_planned": planned_count,
        "win_rate": round(len(wins) / len(closed), 4) if closed else None,
        "total_pnl_usd": round(total_pnl, 2),
        "avg_win_usd": round(sum(t.pnl_usd or 0 for t in wins) / len(wins), 2)
        if wins
        else None,
        "avg_loss_usd": round(sum(t.pnl_usd or 0 for t in losses) / len(losses), 2)
        if losses
        else None,
        "profit_factor": (
            round(
                sum(t.pnl_usd or 0 for t in wins)
                / abs(sum(t.pnl_usd or 0 for t in losses)),
                2,
            )
            if losses and sum(t.pnl_usd or 0 for t in losses) != 0
            else None
        ),
    }


def trade_to_dict(trade: FuturesTrade) -> dict[str, Any]:
    return asdict(trade)
