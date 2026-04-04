"""
FastAPI app: session chatbot for the UI.

Run from repo root:
  uvicorn finhack.api:app --app-dir src --reload --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

import csv
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import secrets
import sqlite3
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

load_dotenv()
from pydantic import BaseModel, Field

from finhack.agents.news_intake_agent import NewsIntakeAgent
from finhack.agents.sector_intelligence_agent import CompanyImpact, SectorIntelligenceAgent
from finhack.config import load_settings
from finhack.data.company_graph import CASE4_SYMBOLS, SECTOR_BUCKETS, SYMBOL_TO_COMPANY
from finhack.market_data import get_case4_market_points, get_close_series, get_market_symbols
from finhack.session_chatbot import (
    answer_user_question,
    get_conversation_history,
)

app = FastAPI(
    title="BladeTrader API",
    description="Case 4 sentiment + spillover API for hackathon workflow.",
    version="0.1.0",
)

def _allowed_origins_from_env() -> list[str]:
    raw = os.getenv("CORS_ALLOWED_ORIGINS", "").strip()
    if raw:
        return [o.strip() for o in raw.split(",") if o.strip()]
    # Safe local defaults. Add platform URL via CORS_ALLOWED_ORIGINS in deployment.
    origins = ["http://127.0.0.1:8000", "http://localhost:8000"]
    return origins


ALLOWED_ORIGINS = _allowed_origins_from_env()

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

WEB_DIR = Path(__file__).resolve().parent / "web"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
app.mount("/app", StaticFiles(directory=WEB_DIR, html=True), name="web")


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=8000)
    session_id: str | None = None
    context: dict[str, Any] | None = None


class ChatResponse(BaseModel):
    reply: str
    session_id: str


class ChatHistoryResponse(BaseModel):
    session_id: str
    turns: list[dict[str, str]]


class NewsIngestRequest(BaseModel):
    max_queries: int = Field(default=8, ge=1, le=20)
    max_per_query: int = Field(default=10, ge=1, le=50)
    hours_back: int = Field(default=24 * 7, ge=1, le=24 * 180)
    trusted_sources_only: bool | None = None
    require_gnews: bool | None = None
    require_primary_api: bool | None = None
    enable_gdelt: bool | None = None
    enable_rss_fallback: bool | None = None


class NewsBackfillRequest(BaseModel):
    days_back: int = Field(default=365, ge=7, le=3650)
    chunk_days: int = Field(default=7, ge=1, le=30)
    max_queries: int = Field(default=14, ge=1, le=20)
    max_per_query: int = Field(default=50, ge=1, le=50)
    max_pages: int = Field(default=2, ge=1, le=10)
    trusted_sources_only: bool | None = None
    require_gnews: bool | None = None
    require_primary_api: bool | None = None
    enable_gdelt: bool | None = None


class NewsDocumentResponse(BaseModel):
    doc_id: str
    url: str
    published_at: str | None
    fetched_at: str
    source: str
    source_url: str | None
    source_domain: str
    title: str
    body: str
    keyword_hits: list[str]
    relevance_score: float
    query: str


class NewsIngestResponse(BaseModel):
    queries_used: list[str]
    fetched_articles: int
    inserted_documents: int
    skipped_documents: int
    transport: str
    primary_api_enforced: bool
    source_counts: dict[str, int]
    documents: list[NewsDocumentResponse]


class AuthRegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=8, max_length=120)
    display_name: str | None = Field(default=None, max_length=80)


class AuthLoginRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=8, max_length=120)


class AuthResponse(BaseModel):
    token: str
    user_id: int
    username: str
    display_name: str | None
    expires_at_utc: str


class AuthMeResponse(BaseModel):
    user_id: int
    username: str
    display_name: str | None
    created_at_utc: str


class DashboardSummaryResponse(BaseModel):
    run_at_utc: str | None
    offline_only: bool
    snapshot_path: str | None
    stock_universe_size: int
    earnings_events_evaluated: int
    baseline_accuracy: float | None
    enhanced_accuracy: float | None
    uplift_vs_baseline_pp: float | None
    enhanced_feature_coverage: float | None
    spillover_feature_coverage_within_enhanced: float | None


class DashboardEventPreview(BaseModel):
    symbol: str
    t_event_utc: str
    actual_5d_return_pct: float
    actual_sign: int
    baseline_pred_sign: int
    enhanced_pred_sign: int
    enhanced_pred_direction: str
    enhanced_confidence: float


class ReadinessCheck(BaseModel):
    name: str
    ok: bool
    detail: str


class DeployReadinessResponse(BaseModel):
    ready: bool
    mode: str
    checks: list[ReadinessCheck]
    allowed_origins: list[str]
    recommendations: list[str]


class MarketPointResponse(BaseModel):
    symbol: str
    price: float | None
    previous_close: float | None
    change: float | None
    change_percent: float | None
    as_of_utc: str
    source: str
    impacted_symbols: list[str]


class Case4MarketResponse(BaseModel):
    provider: str
    run_at_utc: str
    symbols: list[MarketPointResponse]


class MarketProviderResponse(BaseModel):
    provider: str
    has_eodhd_key: bool
    is_live_ready: bool
    notes: list[str]


class MarketSymbolResponse(BaseModel):
    symbol: str
    company_name: str
    source: str


class MarketSymbolsResponse(BaseModel):
    provider: str
    run_at_utc: str
    symbols: list[MarketSymbolResponse]


class MarketHistoryPointResponse(BaseModel):
    t_utc: str
    close: float


class MarketHistoryResponse(BaseModel):
    symbol: str
    provider: str
    run_at_utc: str
    points: list[MarketHistoryPointResponse]


class SectorAnalyzeRequest(BaseModel):
    sector: str
    owned_symbols: list[str] | None = None
    horizon_days: int = Field(default=5, ge=5, le=7)


class SectorCompanyResponse(BaseModel):
    symbol: str
    company_name: str
    role: str
    current_price: float | None
    predicted_direction: str
    predicted_move_pct: float
    correlation_to_sector: float
    leverage_or_hedge: str
    rationale: str
    connected_to: str | None


class SectorAnalyzeResponse(BaseModel):
    sector: str
    horizon_days: int
    predicted_sector_move_pct: float
    confidence: float
    metric_a_news_pressure: float
    metric_b_correlation_strength: float
    metric_c_content_impact: float
    metric_d_network_spillover: float
    news_articles_7d: int
    top_owned: list[SectorCompanyResponse]
    movers_not_owned: list[SectorCompanyResponse]
    generated_at_utc: str


class SectorCatalogResponse(BaseModel):
    sectors: list[str]
    tracked_symbols: list[str]


class SectorAnalyzeAllRequest(BaseModel):
    horizon_days: int = Field(default=5, ge=5, le=7)


class SectorAnalyzeAllStockResponse(BaseModel):
    symbol: str
    company_name: str
    sector: str
    predicted_direction: str
    predicted_move_pct: float
    correlation_to_sector: float
    leverage_or_hedge: str
    connected_to: str | None
    rationale: str


class SectorAnalyzeAllResponse(BaseModel):
    horizon_days: int
    confidence: float
    generated_at_utc: str
    sectors: list[dict[str, Any]]
    stocks: list[SectorAnalyzeAllStockResponse]


class UploadedHolding(BaseModel):
    ticker: str
    name: str
    value: float
    sector: str


class FileUploadResponse(BaseModel):
    filename: str
    content_type: str
    saved_path: str
    size_bytes: int
    preview_lines: list[str]
    detected_holdings: list[UploadedHolding]


CASE4_PATH = PROJECT_ROOT / "data" / "case4_earnings_validation.json"
WEB_INDEX_PATH = WEB_DIR / "index.html"
UPLOAD_DIR = PROJECT_ROOT / "data" / "uploads"


def _load_case4_summary_data() -> dict[str, Any]:
    if not CASE4_PATH.exists():
        raise HTTPException(status_code=404, detail="case4_earnings_validation.json not found")
    try:
        raw = CASE4_PATH.read_text(encoding="utf-8")
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=500, detail=f"Invalid case4 summary JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise HTTPException(status_code=500, detail="Invalid case4 summary payload")
    return payload


def _database_path() -> str:
    raw = load_settings().database_url
    if raw.startswith("sqlite:///"):
        raw = raw.replace("sqlite:///", "", 1)
    if "://" in raw:
        raw = "data/finhack.db"
    p = Path(raw)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    p.parent.mkdir(parents=True, exist_ok=True)
    return str(p)


def _connect_db() -> sqlite3.Connection:
    conn = sqlite3.connect(_database_path())
    conn.row_factory = sqlite3.Row
    return conn


def _ensure_auth_tables() -> None:
    with _connect_db() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS app_user (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                password_salt TEXT NOT NULL,
                display_name TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS app_session (
                token TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                FOREIGN KEY(user_id) REFERENCES app_user(id)
            )
            """
        )


def _hash_password(password: str, salt_hex: str) -> str:
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        bytes.fromhex(salt_hex),
        200_000,
    )
    return digest.hex()


def _auth_from_header(authorization: str | None) -> tuple[int, str, str | None]:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Bearer token")
    token = authorization.replace("Bearer ", "", 1).strip()
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    with _connect_db() as conn:
        row = conn.execute(
            """
            SELECT s.user_id, s.expires_at, u.username, u.display_name
            FROM app_session s
            JOIN app_user u ON u.id = s.user_id
            WHERE s.token = ?
            """,
            (token,),
        ).fetchone()
    if row is None:
        raise HTTPException(status_code=401, detail="Invalid session token")
    if str(row["expires_at"]) < now:
        raise HTTPException(status_code=401, detail="Session expired")
    return int(row["user_id"]), str(row["username"]), row["display_name"]


def _readiness_snapshot() -> DeployReadinessResponse:
    checks: list[ReadinessCheck] = []

    case4_exists = CASE4_PATH.exists()
    checks.append(
        ReadinessCheck(
            name="case4_summary_file",
            ok=case4_exists,
            detail=str(CASE4_PATH) if case4_exists else "Missing data/case4_earnings_validation.json",
        )
    )

    web_exists = WEB_INDEX_PATH.exists()
    checks.append(
        ReadinessCheck(
            name="web_index_file",
            ok=web_exists,
            detail=str(WEB_INDEX_PATH) if web_exists else "Missing src/finhack/web/index.html",
        )
    )

    cors_ok = bool(ALLOWED_ORIGINS) and "*" not in ALLOWED_ORIGINS
    checks.append(
        ReadinessCheck(
            name="cors_config",
            ok=cors_ok,
            detail="Restricted origins configured" if cors_ok else "CORS is too permissive",
        )
    )

    try:
        payload = _load_case4_summary_data() if case4_exists else {}
    except HTTPException:
        payload = {}
    offline_only = bool(payload.get("offline_only", False)) if isinstance(payload, dict) else False
    mode = "offline" if offline_only else "live"
    checks.append(
        ReadinessCheck(
            name="mode",
            ok=True,
            detail=f"Running in {mode} mode",
        )
    )

    recommendations: list[str] = []
    if mode == "offline":
        recommendations.append("Set GNEWS_API_KEY for live ingestion mode when ready.")
    if os.getenv("GNEWS_API_KEY", "").strip() == "":
        recommendations.append("GNEWS_API_KEY is not set; news ingestion may stay offline.")
    if os.getenv("CORS_ALLOWED_ORIGINS", "").strip() == "":
        recommendations.append(
            "Set CORS_ALLOWED_ORIGINS to your Railway frontend URL for stricter production policy."
        )

    ready = all(check.ok for check in checks if check.name != "mode")
    return DeployReadinessResponse(
        ready=ready,
        mode=mode,
        checks=checks,
        allowed_origins=ALLOWED_ORIGINS,
        recommendations=recommendations,
    )


def _normalize_header(name: str) -> str:
    return "".join(ch for ch in (name or "").strip().lower() if ch.isalnum())


def _extract_holdings_from_csv(raw_text: str) -> list[UploadedHolding]:
    reader = csv.DictReader(io.StringIO(raw_text))
    if not reader.fieldnames:
        return []
    header_map = {_normalize_header(h): h for h in reader.fieldnames if h}
    ticker_key = next(
        (
            header_map[k]
            for k in ("ticker", "symbol", "securitysymbol")
            if k in header_map
        ),
        None,
    )
    name_key = next(
        (
            header_map[k]
            for k in ("company", "name", "securityname", "description")
            if k in header_map
        ),
        None,
    )
    value_key = next(
        (
            header_map[k]
            for k in ("marketvalue", "currentvalue", "value", "positionvalue")
            if k in header_map
        ),
        None,
    )
    if ticker_key is None:
        return []

    out: list[UploadedHolding] = []
    for row in reader:
        ticker = str(row.get(ticker_key, "")).strip().upper()
        if not ticker:
            continue
        if len(ticker) > 12:
            continue
        name = str(row.get(name_key, "")).strip() if name_key else ticker
        value_raw = str(row.get(value_key, "")).replace(",", "").replace("$", "") if value_key else ""
        try:
            value = float(value_raw) if value_raw else 0.0
        except ValueError:
            value = 0.0
        out.append(
            UploadedHolding(
                ticker=ticker,
                name=name or ticker,
                value=round(value, 2),
                sector="Other AI",
            )
        )
        if len(out) >= 50:
            break
    return out


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/")
def web_root() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.get("/api/deploy/readiness", response_model=DeployReadinessResponse)
def get_deploy_readiness() -> DeployReadinessResponse:
    return _readiness_snapshot()


@app.post("/api/files/upload", response_model=FileUploadResponse)
async def upload_file(file: UploadFile = File(...)) -> FileUploadResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")
    safe_name = Path(file.filename).name
    suffix = Path(safe_name).suffix.lower()
    allowed = {".csv", ".txt", ".json", ".jsonl"}
    if suffix not in allowed:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    payload = await file.read()
    size_bytes = len(payload)
    if size_bytes == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    if size_bytes > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File exceeds 10MB limit")

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    stored_name = f"{stamp}_{safe_name}"
    saved_path = UPLOAD_DIR / stored_name
    saved_path.write_bytes(payload)

    text = payload.decode("utf-8", errors="replace")
    lines = [ln[:240] for ln in text.splitlines()[:8]]
    detected: list[UploadedHolding] = []
    if suffix == ".csv":
        detected = _extract_holdings_from_csv(text)

    return FileUploadResponse(
        filename=safe_name,
        content_type=file.content_type or "application/octet-stream",
        saved_path=saved_path.as_posix(),
        size_bytes=size_bytes,
        preview_lines=lines,
        detected_holdings=detected,
    )


@app.post("/api/auth/register", response_model=AuthResponse)
def register_user(body: AuthRegisterRequest) -> AuthResponse:
    _ensure_auth_tables()
    now = datetime.now(timezone.utc).replace(microsecond=0)
    salt = secrets.token_hex(16)
    pw_hash = _hash_password(body.password, salt)
    username = body.username.strip().lower()
    if not username:
        raise HTTPException(status_code=400, detail="Invalid username")
    try:
        with _connect_db() as conn:
            cur = conn.execute(
                """
                INSERT INTO app_user (username, password_hash, password_salt, display_name, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (username, pw_hash, salt, body.display_name, now.isoformat()),
            )
            user_id = int(cur.lastrowid)
    except sqlite3.IntegrityError as exc:
        raise HTTPException(status_code=409, detail="Username already exists") from exc

    token = secrets.token_urlsafe(32)
    expires = (now + timedelta(days=14)).isoformat()
    with _connect_db() as conn:
        conn.execute(
            """
            INSERT INTO app_session (token, user_id, created_at, expires_at)
            VALUES (?, ?, ?, ?)
            """,
            (token, user_id, now.isoformat(), expires),
        )
    return AuthResponse(
        token=token,
        user_id=user_id,
        username=username,
        display_name=body.display_name,
        expires_at_utc=expires,
    )


@app.post("/api/auth/login", response_model=AuthResponse)
def login_user(body: AuthLoginRequest) -> AuthResponse:
    _ensure_auth_tables()
    username = body.username.strip().lower()
    with _connect_db() as conn:
        user = conn.execute(
            """
            SELECT id, username, password_hash, password_salt, display_name
            FROM app_user
            WHERE username = ?
            """,
            (username,),
        ).fetchone()
    if user is None:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    expected = _hash_password(body.password, str(user["password_salt"]))
    if expected != str(user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    now = datetime.now(timezone.utc).replace(microsecond=0)
    token = secrets.token_urlsafe(32)
    expires = (now + timedelta(days=14)).isoformat()
    with _connect_db() as conn:
        conn.execute(
            """
            INSERT INTO app_session (token, user_id, created_at, expires_at)
            VALUES (?, ?, ?, ?)
            """,
            (token, int(user["id"]), now.isoformat(), expires),
        )
    return AuthResponse(
        token=token,
        user_id=int(user["id"]),
        username=str(user["username"]),
        display_name=user["display_name"],
        expires_at_utc=expires,
    )


@app.get("/api/auth/me", response_model=AuthMeResponse)
def get_me(authorization: str | None = Header(default=None)) -> AuthMeResponse:
    _ensure_auth_tables()
    user_id, _, _ = _auth_from_header(authorization)
    with _connect_db() as conn:
        row = conn.execute(
            """
            SELECT id, username, display_name, created_at
            FROM app_user
            WHERE id = ?
            """,
            (user_id,),
        ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="User not found")
    return AuthMeResponse(
        user_id=int(row["id"]),
        username=str(row["username"]),
        display_name=row["display_name"],
        created_at_utc=str(row["created_at"]),
    )


@app.get("/api/market/provider", response_model=MarketProviderResponse)
def get_market_provider() -> MarketProviderResponse:
    settings = load_settings()
    provider = settings.market_data_provider.value
    has_eodhd_key = bool((settings.eodhd_api_key or "").strip())
    notes: list[str] = []
    if provider == "eodhd" and not has_eodhd_key:
        notes.append("EODHD provider selected but EODHD_API_KEY is missing.")
    if provider == "yahoo":
        notes.append("Provider is set to Yahoo; set MARKET_DATA_PROVIDER=eodhd for live hackathon mode.")
    return MarketProviderResponse(
        provider=provider,
        has_eodhd_key=has_eodhd_key,
        is_live_ready=(provider == "eodhd" and has_eodhd_key),
        notes=notes,
    )


@app.get("/api/market/case4/stocks", response_model=Case4MarketResponse)
def get_case4_market_stocks() -> Case4MarketResponse:
    provider, points = get_case4_market_points()
    return Case4MarketResponse(
        provider=provider,
        run_at_utc=points[0].as_of_utc if points else "",
        symbols=[MarketPointResponse(**asdict(p)) for p in points],
    )


@app.get("/api/market/symbols", response_model=MarketSymbolsResponse)
def get_market_symbols_catalog(limit: int = 1500) -> MarketSymbolsResponse:
    provider, symbols = get_market_symbols(limit=limit)
    return MarketSymbolsResponse(
        provider=provider,
        run_at_utc=datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        symbols=[MarketSymbolResponse(**asdict(row)) for row in symbols],
    )


@app.get("/api/market/history/{symbol}", response_model=MarketHistoryResponse)
def get_market_history(symbol: str, days: int = 60) -> MarketHistoryResponse:
    safe_symbol = (symbol or "").strip().upper()
    if not safe_symbol:
        raise HTTPException(status_code=400, detail="symbol is required")
    safe_days = max(10, min(days, 365))
    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=safe_days + 20)
    series = get_close_series(
        safe_symbol,
        start=start_dt.date().isoformat(),
        end=(end_dt + timedelta(days=1)).date().isoformat(),
    )
    points: list[MarketHistoryPointResponse] = []
    if not series.empty:
        tail = series.tail(safe_days)
        for idx, value in tail.items():
            dt = idx.to_pydatetime() if hasattr(idx, "to_pydatetime") else idx
            if isinstance(dt, datetime):
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                dt_txt = dt.astimezone(timezone.utc).replace(microsecond=0).isoformat()
            else:
                dt_txt = str(dt)
            points.append(MarketHistoryPointResponse(t_utc=dt_txt, close=float(value)))
    return MarketHistoryResponse(
        symbol=safe_symbol,
        provider=load_settings().market_data_provider.value,
        run_at_utc=end_dt.replace(microsecond=0).isoformat(),
        points=points,
    )


@app.get("/api/agents/sector/catalog", response_model=SectorCatalogResponse)
def get_sector_catalog() -> SectorCatalogResponse:
    return SectorCatalogResponse(
        sectors=list(SECTOR_BUCKETS),
        tracked_symbols=list(CASE4_SYMBOLS),
    )


@app.post("/api/agents/sector/analyze", response_model=SectorAnalyzeResponse)
def analyze_sector(body: SectorAnalyzeRequest) -> SectorAnalyzeResponse:
    try:
        agent = SectorIntelligenceAgent()
        result = agent.predict_sector(
            sector=body.sector,
            owned_symbols=body.owned_symbols,
            horizon_days=body.horizon_days,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    def _as_company(c: CompanyImpact) -> SectorCompanyResponse:
        return SectorCompanyResponse(**asdict(c))

    return SectorAnalyzeResponse(
        sector=result.sector,
        horizon_days=result.horizon_days,
        predicted_sector_move_pct=result.predicted_sector_move_pct,
        confidence=result.confidence,
        metric_a_news_pressure=result.metric_a_news_pressure,
        metric_b_correlation_strength=result.metric_b_correlation_strength,
        metric_c_content_impact=result.metric_c_content_impact,
        metric_d_network_spillover=result.metric_d_network_spillover,
        news_articles_7d=result.news_articles_7d,
        top_owned=[_as_company(c) for c in result.top_owned],
        movers_not_owned=[_as_company(c) for c in result.movers_not_owned],
        generated_at_utc=result.generated_at_utc,
    )


@app.post("/api/agents/sector/analyze-all", response_model=SectorAnalyzeAllResponse)
def analyze_all_case4_stocks(body: SectorAnalyzeAllRequest) -> SectorAnalyzeAllResponse:
    try:
        agent = SectorIntelligenceAgent()
        result = agent.predict_case4_universe(horizon_days=body.horizon_days)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    stocks: list[SectorAnalyzeAllStockResponse] = []
    for row in result.stocks:
        comp = SYMBOL_TO_COMPANY.get(row.symbol)
        stocks.append(
            SectorAnalyzeAllStockResponse(
                symbol=row.symbol,
                company_name=row.company_name,
                sector=comp.sector_bucket if comp else "Unknown",
                predicted_direction=row.predicted_direction,
                predicted_move_pct=row.predicted_move_pct,
                correlation_to_sector=row.correlation_to_sector,
                leverage_or_hedge=row.leverage_or_hedge,
                connected_to=row.connected_to,
                rationale=row.rationale,
            )
        )

    return SectorAnalyzeAllResponse(
        horizon_days=result.horizon_days,
        confidence=result.confidence,
        generated_at_utc=result.generated_at_utc,
        sectors=result.sector_summary,
        stocks=stocks,
    )


@app.get("/api/dashboard/summary", response_model=DashboardSummaryResponse)
def get_dashboard_summary() -> DashboardSummaryResponse:
    payload = _load_case4_summary_data()
    baseline = payload.get("baseline") if isinstance(payload.get("baseline"), dict) else {}
    enhanced = payload.get("enhanced") if isinstance(payload.get("enhanced"), dict) else {}
    return DashboardSummaryResponse(
        run_at_utc=payload.get("run_at_utc"),
        offline_only=bool(payload.get("offline_only", False)),
        snapshot_path=payload.get("snapshot_path"),
        stock_universe_size=int(payload.get("stock_universe_size", 0)),
        earnings_events_evaluated=int(payload.get("earnings_events_evaluated", 0)),
        baseline_accuracy=baseline.get("accuracy"),
        enhanced_accuracy=enhanced.get("accuracy"),
        uplift_vs_baseline_pp=payload.get("uplift_vs_baseline_pp"),
        enhanced_feature_coverage=payload.get("enhanced_feature_coverage"),
        spillover_feature_coverage_within_enhanced=payload.get(
            "spillover_feature_coverage_within_enhanced"
        ),
    )


@app.get("/api/dashboard/events", response_model=list[DashboardEventPreview])
def get_dashboard_events(limit: int = 8) -> list[DashboardEventPreview]:
    safe_limit = max(1, min(limit, 30))
    payload = _load_case4_summary_data()
    events = payload.get("events", [])
    if not isinstance(events, list):
        return []
    previews: list[DashboardEventPreview] = []
    for event in events[:safe_limit]:
        if not isinstance(event, dict):
            continue
        previews.append(
            DashboardEventPreview(
                symbol=str(event.get("symbol", "")),
                t_event_utc=str(event.get("t_event_utc", "")),
                actual_5d_return_pct=float(event.get("actual_5d_return_pct", 0.0)),
                actual_sign=int(event.get("actual_sign", 0)),
                baseline_pred_sign=int(event.get("baseline_pred_sign", 0)),
                enhanced_pred_sign=int(event.get("enhanced_pred_sign", 0)),
                enhanced_pred_direction=str(event.get("enhanced_pred_direction", "mixed")),
                enhanced_confidence=float(event.get("enhanced_confidence", 0.0)),
            )
        )
    return previews


@app.post("/api/chat", response_model=ChatResponse)
def post_chat(body: ChatRequest) -> ChatResponse:
    try:
        reply, sid = answer_user_question(
            body.message.strip(),
            session_id=body.session_id,
            context=body.context,
        )
    except Exception as exc:  # noqa: BLE001 — surface as 502 for demo
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return ChatResponse(reply=reply, session_id=sid)


@app.get("/api/chat/history/{session_id}", response_model=ChatHistoryResponse)
def get_chat_history(session_id: str) -> ChatHistoryResponse:
    turns = get_conversation_history(session_id)
    return ChatHistoryResponse(session_id=session_id, turns=turns)


@app.post("/api/agents/news-intake/run", response_model=NewsIngestResponse)
def run_news_intake(body: NewsIngestRequest) -> NewsIngestResponse:
    try:
        agent = NewsIntakeAgent()
        result = agent.run_ingest(
            max_queries=body.max_queries,
            max_per_query=body.max_per_query,
            hours_back=body.hours_back,
            trusted_sources_only=body.trusted_sources_only,
            require_gnews=body.require_gnews,
            require_primary_api=body.require_primary_api,
            enable_gdelt=body.enable_gdelt,
            enable_rss_fallback=body.enable_rss_fallback,
        )
    except Exception as exc:  # noqa: BLE001 - bubble up for demo iteration
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    docs = [NewsDocumentResponse(**asdict(d)) for d in result.documents]
    return NewsIngestResponse(
        queries_used=result.queries_used,
        fetched_articles=result.fetched_articles,
        inserted_documents=result.inserted_documents,
        skipped_documents=result.skipped_documents,
        transport=result.transport,
        primary_api_enforced=result.primary_api_enforced,
        source_counts=result.source_counts,
        documents=docs,
    )


@app.get("/api/agents/news-intake/documents", response_model=list[NewsDocumentResponse])
def list_news_documents(limit: int = 50) -> list[NewsDocumentResponse]:
    safe_limit = max(1, min(limit, 250))
    try:
        agent = NewsIntakeAgent()
        docs = agent.list_documents(limit=safe_limit)
    except Exception as exc:  # noqa: BLE001 - bubble up for demo iteration
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return [NewsDocumentResponse(**asdict(d)) for d in docs]


@app.post("/api/agents/news-intake/backfill", response_model=NewsIngestResponse)
def backfill_news_intake(body: NewsBackfillRequest) -> NewsIngestResponse:
    try:
        agent = NewsIntakeAgent()
        result = agent.run_historical_backfill(
            days_back=body.days_back,
            chunk_days=body.chunk_days,
            max_queries=body.max_queries,
            max_per_query=body.max_per_query,
            max_pages=body.max_pages,
            trusted_sources_only=body.trusted_sources_only,
            require_gnews=body.require_gnews,
            require_primary_api=body.require_primary_api,
            enable_gdelt=body.enable_gdelt,
        )
    except Exception as exc:  # noqa: BLE001 - bubble up for demo iteration
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    docs = [NewsDocumentResponse(**asdict(d)) for d in result.documents]
    return NewsIngestResponse(
        queries_used=result.queries_used,
        fetched_articles=result.fetched_articles,
        inserted_documents=result.inserted_documents,
        skipped_documents=result.skipped_documents,
        transport=result.transport,
        primary_api_enforced=result.primary_api_enforced,
        source_counts=result.source_counts,
        documents=docs,
    )


