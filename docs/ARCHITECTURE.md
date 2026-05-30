# Architecture overview

Personal **Python research tool** built on **FastAPI** (HTTP surface), a single static **research dashboard** (`src/finhack/web/index.html`), and **SQLite** for persistence. The goal is to test whether AI news sentiment plus cross-company spillover features improve **post-earnings direction** prediction, under strict no-leakage windows around each earnings date `T`.

## Layout

| Path | Role |
|------|------|
| `src/finhack/api.py` | FastAPI surface: market data, sector/spillover, dashboard summaries, forward predictions, news inspector. |
| `src/finhack/config.py` | Settings loader (env / `.env`). |
| `src/finhack/market_data.py` | Provider abstraction (EODHD / Yahoo) for prices, earnings calendar, OHLC. |
| `src/finhack/agents/news_intake_agent.py` | News ingestion (GNews, GDELT, Yahoo, RSS, EODHD) into the SQLite `document` table. |
| `src/finhack/agents/sector_intelligence_agent.py` | Sector-level prediction + spillover scoring. |
| `src/finhack/data/company_graph.py` | Static spillover edges between tickers and sector buckets. |
| `src/finhack/data/trading_universe.py` | Tracked symbol universe. |
| `src/finhack/text_encoder.py` | Document sentiment encoder. Lexicon scorer (always available) + FinBERT (`ProsusAI/finbert`) when `transformers` + `torch` are installed. Scores cached in the `document_score` SQLite table. |
| `src/finhack/case4_features.py` | `[T-7d, T]` feature builder (lexicon + FinBERT + spillover) reading the cached scores. Produces the heuristic-lane `enhanced_pred_sign` via a saved trained model when present. |
| `src/finhack/paper_signals.py` | Forward-prediction engine for upcoming earnings; loads the trained artifact. |
| `src/finhack/research/case4_dataset.py` | Feature frame builder + train/test fitting; routes through `model_store`. Owns `news_coverage_summary`. |
| `src/finhack/research/market_features.py` | Multi-window pre-event returns, prior post-earnings drift, sector cohort return, neighbor earnings clustering. Computed once at validation time. |
| `src/finhack/research/model_store.py` | Train/persist/load the enhanced classifier. LightGBM preferred, scaled logistic fallback. |
| `src/finhack/research/case4_backtest.py` | Walk-forward, top-K, threshold-tuned, and locked-holdout backtests. |
| `src/finhack/research/case4_permutation.py` | Permutation tests with FinBERT coverage warnings in the data-quality block. |
| `src/finhack/web/index.html` | Single-page research dashboard (vanilla JS). |
| `scripts/` | CLI utilities: snapshot, validation, training, backtest, permutation. |

## Data flow

1. **Ingest** — CLI scripts call `NewsIntakeAgent` to pull news into SQLite (`document` table). HTTP endpoints expose only a read-only inspector.
2. **Encoder scoring** — `validate_case4_earnings` (and `train_case4_prototype_model`) call `text_encoder.ensure_lexicon_scores` and `ensure_finbert_scores`, populating `document_score` once per doc/model. Re-runs are zero-cost reads.
3. **Events + prices** — `market_data.py` pulls earnings dates and OHLC from the configured provider. The validate script fetches a 120-day pre-event window so multi-window features are exact.
4. **Per-event features** — `case4_features.py` aggregates cached document scores in `[T-7d, T]`. `research/market_features.py` adds price-derived and cross-symbol features. Targets use returns measured **after** `T`.
5. **Train + validation + backtest** — research scripts emit JSON artifacts under `data/`; training also writes `data/models/case4_enhanced.pkl`.
6. **Forward predictions** — `paper_signals.build_earnings_paper_signals` scans the next N days of upcoming earnings and produces predicted direction + confidence per symbol. It loads the persisted model artifact; if the artifact is missing it falls back to a deterministic momentum-based heuristic (no magic-number combiner).
7. **Dashboard** — reads the JSON artifacts and calls `paper_signals` on demand.

## Key design choices

- **Single-user, local-first.** No auth, no deploy framing; SQLite is the only data store.
- **CLI for writes, HTTP for reads.** Ingestion runs as scripts; HTTP only exposes research outputs.
- **Leakage discipline.** All features built from `[T-7d, T]` (or earlier); backtest is walk-forward, never random split. Encoder scoring is per-document and time-independent so it can be reused across folds without leakage.
- **Learned weights, not hand-tuned.** Every signal that drives a prediction goes through a trained model; the legacy magic-number combiner has been removed.
- **Explainability.** Each document keeps URL, domain, query, keyword hits, and now a cached encoder score so any prediction can be traced back to source.

## Related docs

- `CASE4_NORTH_STAR.md` — research methodology and validation rules.
- `../README.md` — install, run, pipeline commands, API list.
