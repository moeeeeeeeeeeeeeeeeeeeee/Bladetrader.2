"""
FastAPI app: session chatbot for the UI.

Run from repo root:
  uvicorn finhack.api:app --app-dir src --reload --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

from dataclasses import asdict
import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

load_dotenv()
from pydantic import BaseModel, Field

from finhack.agents.exposure_agent import ExposureAgent, StockProfile
from finhack.agents.news_intake_agent import NewsIntakeAgent
from finhack.session_chatbot import (
    answer_user_question,
    get_conversation_history,
)

app = FastAPI(
    title="FinHack26 API",
    description="Session chatbot and future data/model endpoints.",
    version="0.1.0",
)

def _allowed_origins_from_env() -> list[str]:
    raw = os.getenv("CORS_ALLOWED_ORIGINS", "").strip()
    if raw:
        return [o.strip() for o in raw.split(",") if o.strip()]
    # Safe local defaults. Add Render URL when present.
    origins = ["http://127.0.0.1:8000", "http://localhost:8000"]
    render_url = os.getenv("RENDER_EXTERNAL_URL", "").strip()
    if render_url:
        origins.append(render_url)
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


class ExposureAnalyzeRequest(BaseModel):
    symbol: str = Field(..., min_length=1, max_length=20)
    company_name: str | None = None
    sector: str | None = None
    aliases: list[str] | None = None
    ai_themes: list[str] | None = None
    hours_back: int = Field(default=24 * 14, ge=1, le=24 * 365)
    max_documents: int = Field(default=250, ge=10, le=1000)
    top_k: int = Field(default=8, ge=1, le=15)


class ExposureDriverResponse(BaseModel):
    doc_id: str
    title: str
    url: str
    source: str
    published_at: str | None
    impact_direction: str
    impact_score: float
    direct_mention: bool
    matched_themes: list[str]
    why: str


class ExposureAnalyzeResponse(BaseModel):
    symbol: str
    looked_back_hours: int
    documents_considered: int
    direct_mentions: int
    spillover_mentions: int
    exposure_score: float
    direct_exposure_score: float
    spillover_exposure_score: float
    impact_direction: str
    confidence: float
    matched_themes: list[str]
    top_drivers: list[ExposureDriverResponse]


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


CASE4_PATH = Path("data") / "case4_earnings_validation.json"
WEB_INDEX_PATH = WEB_DIR / "index.html"


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
            "Set CORS_ALLOWED_ORIGINS to your Render frontend URL for stricter production policy."
        )

    ready = all(check.ok for check in checks if check.name != "mode")
    return DeployReadinessResponse(
        ready=ready,
        mode=mode,
        checks=checks,
        allowed_origins=ALLOWED_ORIGINS,
        recommendations=recommendations,
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/")
def web_root() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.get("/api/deploy/readiness", response_model=DeployReadinessResponse)
def get_deploy_readiness() -> DeployReadinessResponse:
    return _readiness_snapshot()


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


@app.post("/api/agents/exposure/analyze", response_model=ExposureAnalyzeResponse)
def analyze_exposure(body: ExposureAnalyzeRequest) -> ExposureAnalyzeResponse:
    try:
        agent = ExposureAgent()
        profile = StockProfile(
            symbol=body.symbol.strip().upper(),
            company_name=body.company_name,
            sector=body.sector,
            aliases=body.aliases,
            ai_themes=body.ai_themes,
        )
        result = agent.analyze_stock_exposure(
            profile,
            hours_back=body.hours_back,
            max_documents=body.max_documents,
            top_k=body.top_k,
        )
    except Exception as exc:  # noqa: BLE001 - bubble up for demo iteration
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    drivers = [ExposureDriverResponse(**asdict(d)) for d in result.top_drivers]
    return ExposureAnalyzeResponse(
        symbol=result.symbol,
        looked_back_hours=result.looked_back_hours,
        documents_considered=result.documents_considered,
        direct_mentions=result.direct_mentions,
        spillover_mentions=result.spillover_mentions,
        exposure_score=result.exposure_score,
        direct_exposure_score=result.direct_exposure_score,
        spillover_exposure_score=result.spillover_exposure_score,
        impact_direction=result.impact_direction,
        confidence=result.confidence,
        matched_themes=result.matched_themes,
        top_drivers=drivers,
    )
