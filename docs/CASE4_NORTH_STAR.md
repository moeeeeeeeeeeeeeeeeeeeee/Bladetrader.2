# Research methodology (north star)

This project tests one question and one only:

> **Do AI-related news sentiment and cross-company spillover signals improve prediction of post-earnings direction?**

## Target

For each earnings date `T`, predict the **sign of the 5-trading-day return** measured **after** `T`.

## Inputs (strict)

- All features come from the **7 calendar days ending at `T`**, i.e. `[T-7d, T]`. Multi-window momentum features (`pre_30d`, `pre_60d`, prior post-earnings drift) end at the bar **before** `T` and never look past it.
- Nothing from `(T, T+5d]` may touch the feature row — that window is the label only.
- Validation is **time-aware** (walk-forward). No random splits.

## Feature families

- **Direct sentiment** — per-document scores aggregated over `[T-7d, T]` for documents that mention the issuer or its company name. Two encoders run: a lexicon scorer (always on) and FinBERT (`ProsusAI/finbert`, when `transformers` + `torch` are installed). Both write into a `document_score` SQLite cache so the encoder runs once per document, not once per backtest fold.
- **Spillover** — mentions of the issuer's neighbors in the company graph (`data/company_graph.py`), including weighted edges, in the same window.
- **Price priors** — `baseline_pre_7d_return_pct`, multi-window pre-event returns (`pre_30d`, `pre_60d`), realized 30-day vol, and prior-event post-earnings drift (5d return after the previous earnings date for the same symbol). All ending at `T`.
- **Cross-symbol context** — `sector_cohort_5d_return_pct` (mean 5d pre-event return of sector peers ending at `T`) and `neighbor_earnings_count_30d` (how many graph neighbors reported earnings in `[T-30d, T]`).

## Model

- A single classifier (`LightGBM` when available, scaled `LogisticRegression` as fallback) is fit per train fold. Standardization is part of the model artifact, so prediction is deterministic.
- The artifact lives at `data/models/case4_enhanced.pkl` and is consumed by `paper_signals` and the heuristic-lane backtests so the entire pipeline reads its signal from a learned model — no hand-tuned magic-number combiner anywhere in the runtime path.

## Evaluation surfaces

- `scripts/validate_case4_earnings.py` → `data/case4_earnings_validation.json` — per-event rows with all features, plus per-symbol news coverage diagnostics.
- `scripts/train_case4_prototype_model.py` → `data/case4_model_comparison.json` and `data/models/case4_enhanced.pkl` — baseline (price-only) vs sentiment-enhanced metrics + persisted model.
- `scripts/backtest_case4_earnings.py` → `data/case4_backtest_summary.json` — walk-forward strategy comparison (hit rate, cumulative return, Sharpe, drawdown, locked holdout).
- `scripts/run_case4_permutation_test.py` → `data/case4_permutation_results.json` — null distribution under label shuffles, including FinBERT coverage warnings.

## Principles

- End-to-end pipeline before model tuning.
- Every prediction must be traceable: source URLs, keyword hits, spillover paths, encoder model.
- Keep direct vs spillover channels separable so attribution stays honest.
- Encoder scoring runs once per document and is cached; re-running validation, training, or permutation tests re-uses cached scores.
- If the enhanced model does not beat the baseline by a statistically meaningful margin under the permutation test, that is the finding — don't paper over it.
