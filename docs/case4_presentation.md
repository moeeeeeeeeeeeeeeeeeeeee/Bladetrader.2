# BladeTrader Case-4 Prototype Summary

## 1) Data Pipeline
- News ingestion (`news_intake_agent`) collects AI-market documents into SQLite (`document` table) from GNews/GDELT/RSS.
- Earnings event validation uses event anchors `T` and evaluates post-earnings 5-trading-day direction.
- Prototype modeling script: `scripts/train_case4_prototype_model.py`.
- Outputs:
  - `data/case4_model_comparison.json`
  - `data/case4_visuals/sentiment_trend.png`
  - `data/case4_visuals/sentiment_vs_return.png`
  - `data/case4_visuals/spillover_relationships.png`
  - `data/case4_visuals/model_comparison.png`

## 2) Sentiment Methodology
- Company-level sentiment score is computed from 7-day pre-earnings news using term polarity:
  - Positive terms (e.g., beat, growth, upgrade, bullish)
  - Negative terms (e.g., miss, downgrade, weak, bearish)
- Features include:
  - `sent_doc_count`
  - `sent_mean_score`
  - `sent_mean_relevance`

## 3) Feature Design

### Baseline Model (price-only)
- `baseline_pre_7d_return_pct`
- `baseline_pre_7d_abs_return`

### Sentiment-Enhanced Model
- All baseline features, plus:
  - `sent_doc_count`
  - `sent_mean_score`
  - `sent_mean_relevance`
  - `spillover_mentions_7d`
  - `spillover_density_7d`

### Spillover Feature
- Cross-company spillover is derived from `SPILLOVER_MAP` in `src/finhack/data/company_graph.py`.
- For each event, mentions of linked symbols within `[T-7d, T]` form spillover features.

## 4) Validation Approach
- Time-aware split only (chronological split by event timestamp).
- Binary direction target: sign of post-earnings 5-day return.
- Metrics:
  - Accuracy
  - Precision
  - Recall
  - F1
- Report compares baseline vs sentiment-enhanced model directly.

## 5) Data Leakage Controls
- Strict feature window: **only** information from the 7-day window ending on earnings date `T`.
- Target constructed from returns **after** `T` (5 trading days).
- No random shuffling in validation; chronological split preserves temporal order.
- No post-event text/price fields are used as inputs.

## 6) Main Findings (Prototype Stage)
- Baseline provides a stable directional proxy from pre-earnings momentum.
- Sentiment + spillover features improve directional context and explainability.
- Spillover features help identify where adjacent company narratives influence expected move.
- Prediction quality is sensitive to document coverage and source mix.

## 7) Improvement Roadmap (Real-World Investment Use)
- Expand dataset:
  - More years, more symbols, richer earnings history.
  - Stronger event normalization across providers.
- Improve sentiment labeling:
  - Domain-adapted finance NLP labels instead of keyword polarity.
  - Model confidence calibration with uncertainty intervals.
- Enrich event relationships:
  - Weighted graph edges from empirical co-movement and narrative linkage.
  - Dynamic spillover edges by regime and volatility state.
- Model upgrades:
  - Sequence models for event timing and persistence.
  - Ensemble with probabilistic calibration and walk-forward retraining.
- Production controls:
  - Better schema validation, monitoring, and drift detection.
  - Versioned feature store and reproducible backtest runs.

