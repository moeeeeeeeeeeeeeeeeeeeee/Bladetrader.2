# BladeTrader

Personal research tool that studies whether **AI-related news sentiment** and **cross-company spillover** signals improve **post-earnings direction** prediction.

- Target: sign of the **5-trading-day return** after each earnings date `T`.
- Inputs respect a **no-future-data** window: features come from `[T-7d, T]` (or earlier) only.
- Validation is **time-aware** (walk-forward), never random splits.
- Sentiment uses a per-document **FinBERT** encoder cached in SQLite, with a lexicon fallback when `transformers` + `torch` aren't installed.
- The runtime signal goes through a **trained classifier** (LightGBM, with logistic regression as a fallback). No hand-tuned magic-number combiner sits in the prediction path.

See `docs/CASE4_NORTH_STAR.md` for the methodology and `docs/ARCHITECTURE.md` for layout.

## Quick start

1. Install: `python -m pip install -r requirements.txt`
   - `transformers` + `torch` are listed but optional; without them the encoder falls back to the lexicon scorer.
   - `lightgbm` is listed but optional; without it the model falls back to scaled logistic regression.
2. Copy `.env.example` to `.env` and set at least:
   - `MARKET_DATA_PROVIDER=eodhd` (recommended) and `EODHD_API_KEY=<key>`
   - Optional: `GNEWS_API_KEY` to expand news coverage beyond EODHD/GDELT/RSS
3. Launch the dashboard:
   - `.\scripts\run_dev.ps1`
   - or `py -m uvicorn finhack.api:app --app-dir src --reload --host 127.0.0.1 --port 8080`
4. Open [http://127.0.0.1:8080/](http://127.0.0.1:8080/)

## Research pipeline (CLI)

Typical end-to-end run:

```
# 1. Snapshot the current news corpus into JSONL (also (re)hydrates SQLite if empty)
python scripts/build_case4_dataset_snapshot.py --profile quick

# 2. Validate baseline vs sentiment-enhanced on historical earnings
#    (also tops up document_score cache: lexicon always, FinBERT if installed,
#     and adds market-derived features like sector cohort + prior drift)
python scripts/validate_case4_earnings.py

# 3. Train the prototype model (LightGBM > logistic fallback) and persist
#    data/models/case4_enhanced.pkl + data/case4_model_comparison.json
python scripts/train_case4_prototype_model.py

# 4. Walk-forward backtest (writes data/case4_backtest_summary.json)
python scripts/backtest_case4_earnings.py

# 5. Permutation significance test (writes data/case4_permutation_results.json)
python scripts/run_case4_permutation_test.py
```

Once those files exist, the dashboard at `/` reads them automatically and the "predict" button on the upcoming-earnings card produces forward predictions via `paper_signals.build_earnings_paper_signals` using the persisted model.

## API (only research surfaces)

- `GET /health`
- `GET /api/market/provider` · `GET /api/market/case4/stocks` · `GET /api/market/symbols` · `GET /api/market/history/{symbol}`
- `GET /api/agents/sector/catalog` · `POST /api/agents/sector/analyze` · `POST /api/agents/sector/analyze-all`
- `GET /api/dashboard/summary` · `GET /api/dashboard/backtest` · `GET /api/dashboard/events`
- `GET|POST /api/signals/earnings/paper` — forward predictions on upcoming earnings
- `GET /api/agents/news-intake/documents` — read-only inspector of the SQLite news corpus

News ingestion runs from the CLI scripts above, not from HTTP.

## Layout

| Path | Role |
|------|------|
| `src/finhack/api.py` | FastAPI surface (dashboard, sector, forward prediction, news inspector). |
| `src/finhack/web/index.html` | Single-page research dashboard. |
| `src/finhack/agents/news_intake_agent.py` | News ingest (GNews, GDELT, Yahoo, RSS, EODHD) into SQLite. |
| `src/finhack/agents/sector_intelligence_agent.py` | Sector-level prediction + spillover scoring. |
| `src/finhack/text_encoder.py` | FinBERT + lexicon document scorer with SQLite cache. |
| `src/finhack/case4_features.py` | `[T-7d, T]` feature builder (sentiment + spillover). |
| `src/finhack/paper_signals.py` | Forward-prediction engine for upcoming earnings. |
| `src/finhack/research/` | Dataset, market features, model store, walk-forward backtest, trade-path, permutation test. |
| `src/finhack/data/company_graph.py` | Sector buckets and spillover edges. |
| `scripts/` | CLI utilities for the pipeline above. |

Generated artifacts (DB snapshot, validation JSON, backtest JSON, model `.pkl`, run logs) are gitignored. See `data/README.md`.
