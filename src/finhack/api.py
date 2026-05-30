"""
FastAPI surface for the research dashboard.

Single-user research tool: post-earnings direction prediction from AI news
sentiment + cross-company spillover. Endpoints expose research outputs only
(model summary, backtest, upcoming-earnings predictions, market data,
sector/spillover analysis, news document inspector).

Run from repo root:
  py -m uvicorn finhack.api:app --app-dir src --reload --host 127.0.0.1 --port 8080
  (or: .\\scripts\\run_dev.ps1)
"""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_PROJECT_ROOT / ".env")
from pydantic import BaseModel, Field

from finhack.agents.news_intake_agent import NewsIntakeAgent
from finhack.agents.sector_intelligence_agent import CompanyImpact, SectorIntelligenceAgent
from finhack.config import load_settings
from finhack.data.company_graph import SECTOR_BUCKETS, SYMBOL_TO_COMPANY
from finhack.data.trading_universe import trading_symbols
from finhack.market_data import (
    get_case4_market_points,
    get_market_symbols,
    get_ohlc_intraday,
    get_ohlc_series,
    ohlc_frame_to_point_rows,
)
from finhack.paper_signals import build_earnings_paper_signals, paper_signals_to_dict

app = FastAPI(
    title="BladeTrader Research API",
    description="Research-only API for post-earnings sentiment and spillover prediction.",
    version="0.2.0",
)


def _allowed_origins_from_env() -> list[str]:
    raw = os.getenv("CORS_ALLOWED_ORIGINS", "").strip()
    if raw:
        return [o.strip() for o in raw.split(",") if o.strip()]
    return [
        "http://127.0.0.1:8080",
        "http://localhost:8080",
        "http://127.0.0.1:8000",
        "http://localhost:8000",
    ]


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


# ---------- News document inspector (read-only) ----------


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


# ---------- Dashboard summary models ----------


class DashboardSummaryResponse(BaseModel):
    run_at_utc: str | None
    mode: str
    stock_universe_size: int
    earnings_events_evaluated: int
    baseline_accuracy: float | None
    enhanced_accuracy: float | None
    uplift_vs_baseline_pp: float | None
    enhanced_feature_coverage: float | None
    spillover_feature_coverage_within_enhanced: float | None
    backtest_best_strategy: str | None = None
    backtest_cumulative_return_pct: float | None = None
    backtest_hit_rate: float | None = None
    backtest_trades: int | None = None
    path_win_rate: float | None = None
    path_stop_hit_rate: float | None = None
    path_target_hit_rate: float | None = None


class DashboardBacktestStrategyMetrics(BaseModel):
    trades: int
    hit_rate: float | None
    cumulative_return_pct: float | None
    sharpe_ratio: float | None
    max_drawdown_pct: float | None
    mean_trade_return_pct: float | None


class DashboardBacktestResponse(BaseModel):
    generated_at_utc: str | None
    events_labeled: int
    best_out_of_sample_strategy: str | None
    parameters: dict[str, Any]
    strategies: dict[str, DashboardBacktestStrategyMetrics]
    recent_trades: list[dict[str, Any]]


class DashboardEventPreview(BaseModel):
    symbol: str
    t_event_utc: str
    actual_5d_return_pct: float
    actual_sign: int
    baseline_pred_sign: int
    enhanced_pred_sign: int
    enhanced_pred_direction: str
    enhanced_confidence: float


# ---------- Market models ----------


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
    open: float
    high: float
    low: float
    close: float


class MarketHistoryResponse(BaseModel):
    symbol: str
    provider: str
    run_at_utc: str
    interval: str = "1d"
    window_note: str | None = None
    points: list[MarketHistoryPointResponse]


MARKET_HISTORY_INTERVALS = frozenset({"1d", "1wk", "1mo", "1m", "2m", "5m", "15m", "30m", "1h"})
MARKET_HISTORY_EOD_INTERVALS = frozenset({"1d", "1wk", "1mo"})
MARKET_HISTORY_DEFAULT_YEARS = {"1d": 5, "1wk": 10, "1mo": 20}
MARKET_HISTORY_MAX_YEARS = {"1d": 15, "1wk": 20, "1mo": 30}


# ---------- Sector / spillover models ----------


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


# ---------- Forward prediction models ----------


class PaperSignalsRequest(BaseModel):
    horizon_days: int = Field(default=14, ge=1, le=45)
    min_confidence: float = Field(default=0.15, ge=0.0, le=1.0)
    universe_limit: int | None = Field(default=None, ge=10, le=5000)
    actionable_only: bool = False


# ---------- Data paths ----------


CASE4_PATH = PROJECT_ROOT / "data" / "case4_earnings_validation.json"
CASE4_BACKTEST_PATH = PROJECT_ROOT / "data" / "case4_backtest_summary.json"


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


def _load_backtest_summary_data() -> dict[str, Any]:
    if not CASE4_BACKTEST_PATH.exists():
        raise HTTPException(status_code=404, detail="case4_backtest_summary.json not found")
    try:
        raw = CASE4_BACKTEST_PATH.read_text(encoding="utf-8")
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=500, detail=f"Invalid backtest JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise HTTPException(status_code=500, detail="Invalid backtest payload")
    return payload


def _backtest_strategy_metrics(payload: dict[str, Any], key: str) -> DashboardBacktestStrategyMetrics | None:
    strategies = payload.get("strategies")
    if not isinstance(strategies, dict):
        return None
    block = strategies.get(key)
    if not isinstance(block, dict):
        return None
    metrics = block.get("metrics")
    if not isinstance(metrics, dict):
        return None
    return DashboardBacktestStrategyMetrics(
        trades=int(metrics.get("trades", 0)),
        hit_rate=metrics.get("hit_rate"),
        cumulative_return_pct=metrics.get("cumulative_return_pct"),
        sharpe_ratio=metrics.get("sharpe_ratio"),
        max_drawdown_pct=metrics.get("max_drawdown_pct"),
        mean_trade_return_pct=metrics.get("mean_trade_return_pct"),
    )


# ---------- Routes: health + static ----------


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/")
def web_root() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


# ---------- Routes: market data ----------


@app.get("/api/market/provider", response_model=MarketProviderResponse)
def get_market_provider() -> MarketProviderResponse:
    settings = load_settings()
    provider = settings.market_data_provider.value
    has_eodhd_key = bool((settings.eodhd_api_key or "").strip())
    notes: list[str] = []
    if not has_eodhd_key:
        notes.append("EODHD_API_KEY is missing — market data and earnings calendar disabled.")
    else:
        notes.append(
            f"Live quotes cached {settings.market_data_live_cache_ttl_seconds}s; "
            f"history cached {settings.market_data_cache_ttl_seconds}s."
        )
    return MarketProviderResponse(
        provider=provider,
        has_eodhd_key=has_eodhd_key,
        is_live_ready=has_eodhd_key,
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
def get_market_history(
    symbol: str,
    days: int | None = None,
    years: float | None = None,
    interval: str = "1d",
) -> MarketHistoryResponse:
    safe_symbol = (symbol or "").strip().upper()
    if not safe_symbol:
        raise HTTPException(status_code=400, detail="symbol is required")
    raw_iv = (interval or "1d").strip().lower()
    if raw_iv not in MARKET_HISTORY_INTERVALS:
        raw_iv = "1d"

    end_dt = datetime.now(timezone.utc)
    points: list[MarketHistoryPointResponse] = []
    cfg = load_settings()
    provider_note = cfg.market_data_provider.value
    window_note: str | None = None

    if raw_iv in MARKET_HISTORY_EOD_INTERVALS:
        if years is not None:
            lookback_days = int(max(0.25, years) * 366)
        elif days is not None:
            lookback_days = max(30, days)
        else:
            lookback_days = MARKET_HISTORY_DEFAULT_YEARS.get(raw_iv, 5) * 366

        max_days = MARKET_HISTORY_MAX_YEARS.get(raw_iv, 15) * 366
        lookback_days = max(30, min(lookback_days, max_days))
        buffer_days = 45 if raw_iv == "1d" else 14
        start_dt = end_dt - timedelta(days=lookback_days + buffer_days)
        frame = get_ohlc_series(
            safe_symbol,
            start=start_dt.date().isoformat(),
            end=(end_dt + timedelta(days=1)).date().isoformat(),
            interval=raw_iv,
        )
        bar_count = len(frame) if frame is not None and not frame.empty else 0
        approx_years = max(1, round(lookback_days / 366))
        iv_label = {"1d": "daily", "1wk": "weekly", "1mo": "monthly"}.get(raw_iv, raw_iv)
        window_note = f"{bar_count} bars · {iv_label} · ~{approx_years}y window"
        for dt_txt, o, h, l, c in ohlc_frame_to_point_rows(frame):
            points.append(
                MarketHistoryPointResponse(
                    t_utc=dt_txt,
                    open=o,
                    high=h,
                    low=l,
                    close=c,
                )
            )
    else:
        intraday, window_note, provider_note = get_ohlc_intraday(
            safe_symbol, raw_iv, settings=cfg
        )
        for dt_txt, o, h, l, c in ohlc_frame_to_point_rows(intraday, intraday=True):
            points.append(
                MarketHistoryPointResponse(
                    t_utc=dt_txt,
                    open=o,
                    high=h,
                    low=l,
                    close=c,
                )
            )

    return MarketHistoryResponse(
        symbol=safe_symbol,
        provider=provider_note,
        run_at_utc=end_dt.replace(microsecond=0).isoformat(),
        interval=raw_iv,
        window_note=window_note,
        points=points,
    )


# ---------- Routes: sector / spillover analysis ----------


@app.get("/api/agents/sector/catalog", response_model=SectorCatalogResponse)
def get_sector_catalog() -> SectorCatalogResponse:
    settings = load_settings()
    return SectorCatalogResponse(
        sectors=list(SECTOR_BUCKETS),
        tracked_symbols=trading_symbols(settings=settings),
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


# ---------- Routes: dashboard (research outputs) ----------


@app.get("/api/dashboard/summary", response_model=DashboardSummaryResponse)
def get_dashboard_summary() -> DashboardSummaryResponse:
    payload = _load_case4_summary_data()
    baseline = payload.get("baseline") if isinstance(payload.get("baseline"), dict) else {}
    enhanced = payload.get("enhanced") if isinstance(payload.get("enhanced"), dict) else {}

    backtest_best: str | None = None
    backtest_cum: float | None = None
    backtest_hit: float | None = None
    backtest_trades: int | None = None
    if CASE4_BACKTEST_PATH.exists():
        try:
            bt = json.loads(CASE4_BACKTEST_PATH.read_text(encoding="utf-8"))
            if isinstance(bt, dict):
                backtest_best = bt.get("best_out_of_sample_strategy")
                strategies = bt.get("strategies")
                if isinstance(strategies, dict) and backtest_best and backtest_best in strategies:
                    block = strategies[backtest_best]
                    if isinstance(block, dict):
                        metrics = block.get("metrics")
                        if isinstance(metrics, dict):
                            backtest_cum = metrics.get("cumulative_return_pct")
                            backtest_hit = metrics.get("hit_rate")
                            backtest_trades = metrics.get("trades")
        except (json.JSONDecodeError, OSError):
            pass

    path_metrics = payload.get("trade_path_metrics")
    path_win = path_stop = path_target = None
    if isinstance(path_metrics, dict):
        path_win = path_metrics.get("path_win_rate")
        path_stop = path_metrics.get("stop_hit_rate")
        path_target = path_metrics.get("target_hit_rate")

    return DashboardSummaryResponse(
        run_at_utc=payload.get("run_at_utc"),
        mode=str(payload.get("mode", "live")),
        stock_universe_size=int(payload.get("stock_universe_size", 0)),
        earnings_events_evaluated=int(payload.get("earnings_events_evaluated", 0)),
        baseline_accuracy=baseline.get("accuracy"),
        enhanced_accuracy=enhanced.get("accuracy"),
        uplift_vs_baseline_pp=payload.get("uplift_vs_baseline_pp"),
        enhanced_feature_coverage=payload.get("enhanced_feature_coverage"),
        spillover_feature_coverage_within_enhanced=payload.get(
            "spillover_feature_coverage_within_enhanced"
        ),
        backtest_best_strategy=backtest_best,
        backtest_cumulative_return_pct=backtest_cum,
        backtest_hit_rate=backtest_hit,
        backtest_trades=backtest_trades,
        path_win_rate=path_win,
        path_stop_hit_rate=path_stop,
        path_target_hit_rate=path_target,
    )


@app.get("/api/dashboard/backtest", response_model=DashboardBacktestResponse)
def get_dashboard_backtest() -> DashboardBacktestResponse:
    payload = _load_backtest_summary_data()
    strategies_raw = payload.get("strategies")
    strategies: dict[str, DashboardBacktestStrategyMetrics] = {}
    recent: list[dict[str, Any]] = []
    if isinstance(strategies_raw, dict):
        for key in (
            "baseline",
            "enhanced_heuristic",
            "model_enhanced",
            "walk_forward_model",
            "enhanced_path_stop_target",
            "baseline_path_stop_target",
        ):
            parsed = _backtest_strategy_metrics(payload, key)
            if parsed is not None:
                strategies[key] = parsed
        best = payload.get("best_out_of_sample_strategy")
        if isinstance(best, str) and best in strategies_raw:
            block = strategies_raw[best]
            if isinstance(block, dict):
                trades = block.get("recent_trades")
                if isinstance(trades, list):
                    recent = [t for t in trades if isinstance(t, dict)]

    params = payload.get("parameters")
    return DashboardBacktestResponse(
        generated_at_utc=payload.get("generated_at_utc"),
        events_labeled=int(payload.get("events_labeled", 0)),
        best_out_of_sample_strategy=payload.get("best_out_of_sample_strategy"),
        parameters=params if isinstance(params, dict) else {},
        strategies=strategies,
        recent_trades=recent,
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


# ---------- Routes: forward predictions on upcoming earnings ----------


@app.get("/api/signals/earnings/paper")
@app.post("/api/signals/earnings/paper")
def get_paper_earnings_signals(
    horizon_days: int = 14,
    min_confidence: float = 0.15,
    universe_limit: int | None = None,
    actionable_only: bool = False,
    body: PaperSignalsRequest | None = None,
) -> dict[str, Any]:
    req = body or PaperSignalsRequest(
        horizon_days=horizon_days,
        min_confidence=min_confidence,
        universe_limit=universe_limit,
        actionable_only=actionable_only,
    )
    try:
        bundle = build_earnings_paper_signals(
            horizon_days=req.horizon_days,
            universe_limit=req.universe_limit,
            min_confidence=req.min_confidence,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    payload = paper_signals_to_dict(bundle)
    if req.actionable_only:
        payload["signals"] = [s for s in payload["signals"] if s.get("actionable")]
        payload["upcoming_earnings_count"] = len(payload["signals"])
    return payload


# ---------- Routes: news document inspector (read-only) ----------


@app.get("/api/agents/news-intake/documents", response_model=list[NewsDocumentResponse])
def list_news_documents(limit: int = 50) -> list[NewsDocumentResponse]:
    """Read-only inspector for news documents. To ingest news, run CLI scripts."""
    safe_limit = max(1, min(limit, 250))
    try:
        agent = NewsIntakeAgent()
        docs = agent.list_documents(limit=safe_limit)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    return [NewsDocumentResponse(**asdict(d)) for d in docs]
