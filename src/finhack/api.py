"""
FastAPI app: session chatbot for the UI.

Run from repo root:
  uvicorn finhack.api:app --app-dir src --reload --host 127.0.0.1 --port 8000
"""

from __future__ import annotations

from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

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

# Tighten origins before production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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
    enable_rss_fallback: bool | None = None


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


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


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
            enable_rss_fallback=body.enable_rss_fallback,
        )
    except Exception as exc:  # noqa: BLE001 - bubble up for demo iteration
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    docs = [NewsDocumentResponse(**d.__dict__) for d in result.documents]
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
    return [NewsDocumentResponse(**d.__dict__) for d in docs]


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
    drivers = [ExposureDriverResponse(**d.__dict__) for d in result.top_drivers]
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
