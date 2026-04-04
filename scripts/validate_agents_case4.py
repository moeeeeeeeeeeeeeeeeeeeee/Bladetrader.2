"""
Validate Agent 1 + Agent 2 against yfinance 5-trading-day movement.

This script is a practical validation harness for iterative testing.
It does not replace a full earnings-anchored dataset yet.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from statistics import mean
from typing import Any

import yfinance as yf

from finhack.agents.exposure_agent import ExposureAgent, StockProfile
from finhack.agents.news_intake_agent import NewsIntakeAgent


STOCK_UNIVERSE: tuple[StockProfile, ...] = (
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


def _parse_iso(raw: str | None) -> datetime | None:
    if not raw:
        return None
    txt = raw.strip()
    if not txt:
        return None
    if txt.endswith("Z"):
        txt = txt[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(txt)
    except ValueError:
        try:
            dt = parsedate_to_datetime(txt)
        except Exception:  # noqa: BLE001
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def five_day_return_pct(symbol: str, event_dt: datetime) -> float | None:
    start = (event_dt - timedelta(days=2)).date().isoformat()
    end = (event_dt + timedelta(days=20)).date().isoformat()
    df = yf.Ticker(symbol).history(start=start, end=end, auto_adjust=False)
    if df.empty:
        return None
    closes = df["Close"].dropna()
    if closes.empty:
        return None
    idx = list(closes.index)
    # first trading day on/after event timestamp date
    base_i = None
    event_date = event_dt.date()
    for i, ts in enumerate(idx):
        if ts.date() >= event_date:
            base_i = i
            break
    if base_i is None:
        return None
    target_i = base_i + 5
    if target_i >= len(idx):
        return None
    p0 = float(closes.iloc[base_i])
    p5 = float(closes.iloc[target_i])
    if p0 == 0:
        return None
    return ((p5 - p0) / p0) * 100.0


def predicted_sign(direction: str) -> int:
    if direction == "bullish":
        return 1
    if direction == "bearish":
        return -1
    return 0


def realized_sign(ret_pct: float, dead_zone: float = 0.15) -> int:
    if ret_pct > dead_zone:
        return 1
    if ret_pct < -dead_zone:
        return -1
    return 0


def corr(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mx) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - my) ** 2 for y in ys))
    den = den_x * den_y
    if den == 0:
        return None
    return num / den


def main() -> None:
    news = NewsIntakeAgent()
    exposure = ExposureAgent()

    ingest = news.run_ingest(
        max_queries=12,
        max_per_query=20,
        hours_back=24 * 60,
        trusted_sources_only=False,
    )

    stock_results: list[dict[str, Any]] = []
    event_level: list[dict[str, Any]] = []
    exposure_scores: list[float] = []
    abs_returns: list[float] = []
    directional_total = 0
    directional_correct = 0
    stock_directional_total = 0
    stock_directional_correct = 0

    for profile in STOCK_UNIVERSE:
        analysis = exposure.analyze_stock_exposure(
            profile,
            hours_back=24 * 60,
            max_documents=500,
            top_k=10,
        )
        predicted = predicted_sign(analysis.impact_direction)
        stock_event_returns: list[float] = []
        stock_event_hits = 0
        stock_event_count = 0

        for d in analysis.top_drivers:
            dt = _parse_iso(d.published_at)
            if dt is None:
                continue
            ret = five_day_return_pct(profile.symbol, dt)
            if ret is None:
                continue
            actual = realized_sign(ret)
            pred = predicted_sign(d.impact_direction)
            hit = pred != 0 and actual != 0 and pred == actual
            if pred != 0 and actual != 0:
                directional_total += 1
                stock_event_count += 1
                if hit:
                    directional_correct += 1
                    stock_event_hits += 1
            stock_event_returns.append(ret)
            event_level.append(
                {
                    "symbol": profile.symbol,
                    "doc_id": d.doc_id,
                    "published_at": d.published_at,
                    "driver_direction": d.impact_direction,
                    "five_day_return_pct": round(ret, 4),
                    "hit": hit,
                    "title": d.title,
                    "url": d.url,
                }
            )

        mean_ret = mean(stock_event_returns) if stock_event_returns else None
        stock_actual = realized_sign(mean_ret) if mean_ret is not None else 0
        stock_hit = predicted != 0 and stock_actual != 0 and predicted == stock_actual
        if predicted != 0 and stock_actual != 0:
            stock_directional_total += 1
            if stock_hit:
                stock_directional_correct += 1
        if mean_ret is not None:
            exposure_scores.append(analysis.exposure_score)
            abs_returns.append(abs(mean_ret))

        stock_results.append(
            {
                "symbol": profile.symbol,
                "analysis": asdict(analysis),
                "events_tested": len(stock_event_returns),
                "directional_events": stock_event_count,
                "directional_hits": stock_event_hits,
                "directional_accuracy": round(
                    stock_event_hits / stock_event_count, 4
                )
                if stock_event_count > 0
                else None,
                "mean_five_day_return_pct": round(mean_ret, 4) if mean_ret is not None else None,
                "predicted_direction": analysis.impact_direction,
                "predicted_sign": predicted,
                "stock_actual_sign": stock_actual,
                "stock_directional_hit": stock_hit,
            }
        )

    overall_acc = (
        directional_correct / directional_total if directional_total > 0 else None
    )
    stock_overall_acc = (
        stock_directional_correct / stock_directional_total
        if stock_directional_total > 0
        else None
    )
    score_absret_corr = corr(exposure_scores, abs_returns)

    result = {
        "run_at_utc": datetime.now(timezone.utc).isoformat(),
        "ingest": asdict(ingest),
        "stock_count": len(STOCK_UNIVERSE),
        "event_count_tested": len(event_level),
        "directional_total": directional_total,
        "directional_correct": directional_correct,
        "directional_accuracy": round(overall_acc, 4) if overall_acc is not None else None,
        "stock_directional_total": stock_directional_total,
        "stock_directional_correct": stock_directional_correct,
        "stock_directional_accuracy": round(stock_overall_acc, 4)
        if stock_overall_acc is not None
        else None,
        "exposure_score_vs_abs_5d_return_corr": round(score_absret_corr, 4)
        if score_absret_corr is not None
        else None,
        "stocks": stock_results,
        "events": event_level,
        "notes": [
            "Validation uses article timestamps as event anchors.",
            "Case objective specifies earnings anchors; this is an iterative proxy test.",
            "For final scoring, use earnings calendar T and strict [T-7d, T] feature windows.",
        ],
    }

    out_dir = Path("data")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "agent_case4_validation.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Wrote: {out_path.as_posix()}")
    print(
        json.dumps(
            {
                "stock_count": result["stock_count"],
                "event_count_tested": result["event_count_tested"],
                "directional_accuracy": result["directional_accuracy"],
                "stock_directional_accuracy": result["stock_directional_accuracy"],
                "exposure_score_vs_abs_5d_return_corr": result[
                    "exposure_score_vs_abs_5d_return_corr"
                ],
                "transport": result["ingest"]["transport"],
                "inserted_documents": result["ingest"]["inserted_documents"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

