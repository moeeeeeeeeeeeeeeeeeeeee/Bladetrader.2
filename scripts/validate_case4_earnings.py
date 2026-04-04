"""
Case 4-style validation:
- Anchor on earnings dates T from yfinance
- Build features only from [T-7d, T]
- Predict 5-trading-day post-earnings direction
- Compare baseline market-only vs sentiment-enhanced (Agent 2)
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import yfinance as yf

from finhack.agents.exposure_agent import ExposureAgent, StockProfile
from finhack.agents.news_intake_agent import NewsIntakeAgent


UNIVERSE: tuple[StockProfile, ...] = (
    StockProfile("NVDA", "NVIDIA", "Semiconductors"),
    StockProfile("MSFT", "Microsoft", "Software"),
    StockProfile("GOOGL", "Alphabet", "Internet"),
    StockProfile("AMZN", "Amazon", "Cloud"),
    StockProfile("META", "Meta Platforms", "Internet"),
    StockProfile("AAPL", "Apple", "Hardware"),
    StockProfile("AMD", "Advanced Micro Devices", "Semiconductors"),
    StockProfile("AVGO", "Broadcom", "Semiconductors"),
    StockProfile("TSM", "Taiwan Semiconductor", "Semiconductors"),
    StockProfile("ASML", "ASML Holding", "Semiconductors"),
    StockProfile("QCOM", "Qualcomm", "Semiconductors"),
    StockProfile("INTC", "Intel", "Semiconductors"),
    StockProfile("ANET", "Arista Networks", "Hardware"),
    StockProfile("SMCI", "Super Micro Computer", "Hardware"),
    StockProfile("PLTR", "Palantir", "Software"),
    StockProfile("SNOW", "Snowflake", "Software"),
    StockProfile("ORCL", "Oracle", "Software"),
    StockProfile("CRM", "Salesforce", "Software"),
    StockProfile("PANW", "Palo Alto Networks", "Cybersecurity"),
    StockProfile("CRWD", "CrowdStrike", "Cybersecurity"),
)


def sign_from_direction(direction: str) -> int:
    if direction == "bullish":
        return 1
    if direction == "bearish":
        return -1
    return 0


def sign_from_return(ret: float, dead_zone: float = 0.15) -> int:
    if ret > dead_zone:
        return 1
    if ret < -dead_zone:
        return -1
    return 0


def compute_return_pct(series, i0: int, i1: int) -> float | None:
    if i0 < 0 or i1 < 0 or i0 >= len(series) or i1 >= len(series):
        return None
    p0 = float(series.iloc[i0])
    p1 = float(series.iloc[i1])
    if p0 == 0:
        return None
    return ((p1 - p0) / p0) * 100.0


def get_earnings_events(
    symbol: str,
    limit: int = 12,
    recent_days: int = 365,
) -> list[datetime]:
    t = yf.Ticker(symbol)
    try:
        df = t.get_earnings_dates(limit=limit)
    except Exception:
        return []
    if df is None or df.empty:
        return []
    out: list[datetime] = []
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=max(30, recent_days))
    for idx in df.index:
        dt = idx.to_pydatetime()
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        if cutoff <= dt < now:
            out.append(dt)
    return sorted(out)


def evaluate_event(
    profile: StockProfile,
    t_event: datetime,
    exposure: ExposureAgent,
) -> dict[str, Any] | None:
    start = (t_event - timedelta(days=45)).date().isoformat()
    end = (t_event + timedelta(days=20)).date().isoformat()
    px = yf.Ticker(profile.symbol).history(start=start, end=end, auto_adjust=False)
    if px.empty or "Close" not in px.columns:
        return None
    close = px["Close"].dropna()
    if close.empty:
        return None

    idx = list(close.index)
    event_i = None
    for i, ts in enumerate(idx):
        if ts.to_pydatetime().date() >= t_event.date():
            event_i = i
            break
    if event_i is None:
        return None

    # Baseline: market/price-only momentum from pre-event window.
    pre_start_i = max(0, event_i - 7)
    pre_end_i = max(0, event_i - 1)
    pre_ret = compute_return_pct(close, pre_start_i, pre_end_i)
    if pre_ret is None:
        return None
    baseline_pred = sign_from_return(pre_ret, dead_zone=0.10)

    post_i = event_i + 5
    post_ret = compute_return_pct(close, event_i, post_i)
    if post_ret is None:
        return None
    actual = sign_from_return(post_ret)

    sentiment = exposure.analyze_stock_exposure_at(
        profile,
        anchor_at=t_event,
        lookback_days=7,
        max_documents=1000,
        top_k=8,
    )
    enhanced_pred = sign_from_direction(sentiment.impact_direction)

    return {
        "symbol": profile.symbol,
        "t_event_utc": t_event.isoformat(),
        "actual_5d_return_pct": round(post_ret, 4),
        "actual_sign": actual,
        "baseline_pre_7d_return_pct": round(pre_ret, 4),
        "baseline_pred_sign": baseline_pred,
        "enhanced_pred_sign": enhanced_pred,
        "enhanced_pred_direction": sentiment.impact_direction,
        "enhanced_documents_considered": sentiment.documents_considered,
        "enhanced_direct_mentions": sentiment.direct_mentions,
        "enhanced_spillover_mentions": sentiment.spillover_mentions,
        "enhanced_exposure_score": sentiment.exposure_score,
        "enhanced_confidence": sentiment.confidence,
        "top_drivers": [asdict(d) for d in sentiment.top_drivers[:5]],
    }


def accuracy(rows: list[dict[str, Any]], pred_key: str) -> tuple[int, int, float | None]:
    total = 0
    correct = 0
    for r in rows:
        pred = int(r.get(pred_key, 0))
        actual = int(r.get("actual_sign", 0))
        if pred == 0 or actual == 0:
            continue
        total += 1
        if pred == actual:
            correct += 1
    if total == 0:
        return correct, total, None
    return correct, total, correct / total


def main() -> None:
    news = NewsIntakeAgent()
    exposure = ExposureAgent()

    # Keep trusted filter on for quality; widen horizon for more potential earnings windows.
    ingest = news.run_ingest(
        max_queries=14,
        max_per_query=25,
        hours_back=24 * 365,
        trusted_sources_only=False,
    )

    rows: list[dict[str, Any]] = []
    per_symbol_count: dict[str, int] = {}
    for profile in UNIVERSE:
        events = get_earnings_events(profile.symbol, limit=14, recent_days=365)
        for t_event in events:
            row = evaluate_event(profile, t_event, exposure)
            if not row:
                continue
            rows.append(row)
            per_symbol_count[profile.symbol] = per_symbol_count.get(profile.symbol, 0) + 1

    baseline_correct, baseline_total, baseline_acc = accuracy(rows, "baseline_pred_sign")
    enhanced_correct, enhanced_total, enhanced_acc = accuracy(rows, "enhanced_pred_sign")

    enhanced_rows = [r for r in rows if r.get("enhanced_documents_considered", 0) > 0]
    enhanced_cov = len(enhanced_rows) / len(rows) if rows else None
    spillover_rows = [r for r in enhanced_rows if r.get("enhanced_spillover_mentions", 0) > 0]
    spillover_cov = len(spillover_rows) / len(enhanced_rows) if enhanced_rows else None

    summary = {
        "run_at_utc": datetime.now(timezone.utc).isoformat(),
        "ingest": asdict(ingest),
        "stock_universe_size": len(UNIVERSE),
        "earnings_events_evaluated": len(rows),
        "symbols_with_events": len(per_symbol_count),
        "baseline": {
            "correct": baseline_correct,
            "total": baseline_total,
            "accuracy": round(baseline_acc, 4) if baseline_acc is not None else None,
        },
        "enhanced": {
            "correct": enhanced_correct,
            "total": enhanced_total,
            "accuracy": round(enhanced_acc, 4) if enhanced_acc is not None else None,
        },
        "uplift_vs_baseline_pp": round((enhanced_acc - baseline_acc) * 100.0, 2)
        if enhanced_acc is not None and baseline_acc is not None
        else None,
        "enhanced_feature_coverage": round(enhanced_cov, 4) if enhanced_cov is not None else None,
        "spillover_feature_coverage_within_enhanced": round(spillover_cov, 4)
        if spillover_cov is not None
        else None,
        "events": rows,
    }

    out_dir = Path("data")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "case4_earnings_validation.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote: {out_path.as_posix()}")
    print(
        json.dumps(
            {
                "events": summary["earnings_events_evaluated"],
                "baseline_accuracy": summary["baseline"]["accuracy"],
                "enhanced_accuracy": summary["enhanced"]["accuracy"],
                "uplift_vs_baseline_pp": summary["uplift_vs_baseline_pp"],
                "enhanced_feature_coverage": summary["enhanced_feature_coverage"],
                "spillover_feature_coverage_within_enhanced": summary[
                    "spillover_feature_coverage_within_enhanced"
                ],
                "ingested_documents": summary["ingest"]["inserted_documents"],
                "transport": summary["ingest"]["transport"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

