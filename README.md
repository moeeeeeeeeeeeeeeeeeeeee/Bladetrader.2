# BladeTrader

Case-4 hackathon app for AI-sentiment, cross-company spillover, and 5-7 day post-event direction workflows across 14 tracked stocks.

## Run locally

1. Install dependencies: `python -m pip install -r requirements.txt`
2. Copy `.env.example` to `.env`
3. Set `.env` values:
   - `MARKET_DATA_PROVIDER=eodhd`
   - `EODHD_API_KEY=<your key>`
   - `GNEWS_API_KEY=<optional but recommended>`
4. Start API/UI: `uvicorn finhack.api:app --app-dir src --reload --host 127.0.0.1 --port 8000`
5. Open `http://127.0.0.1:8000/`

## Core agents

- **Agent 1 (`news_intake_agent`)**: ingests AI-market news from GNews/GDELT/Yahoo/RSS into SQLite.
- **Agent 3/4 (`sector_intelligence_agent`)**: learns 5-year pattern links between AI news and sector/company movement using metrics A/B/C/D and produces 5-7 day forecasts.

## Deploy

### Railway

[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/new)

1. Open [Railway](https://railway.app/new) and create a new project from this GitHub repository.
2. Railway auto-detects `railway.json` and uses the start command.
3. Set environment variables:
   - `CORS_ALLOWED_ORIGINS=https://<your-railway-domain>`
   - `GNEWS_API_KEY=<optional-for-live-news>`
   - `MARKET_DATA_PROVIDER=eodhd`
   - `EODHD_API_KEY=<required-for-eodhd-market-and-news>`
4. Open the generated domain from Railway project settings.

## API overview

- `GET /health`
- `GET /api/deploy/readiness`
- `GET /api/market/provider`
- `GET /api/market/case4/stocks`
- `GET /api/dashboard/summary`
- `GET /api/dashboard/events`
- `POST /api/agents/news-intake/run`
- `GET /api/agents/news-intake/documents`
- `POST /api/agents/news-intake/backfill`
- `GET /api/agents/sector/catalog`
- `POST /api/agents/sector/analyze`
- `POST /api/agents/sector/analyze-all`
- `POST /api/chat`
- `GET /api/chat/history/{session_id}`
- `POST /api/auth/register`
- `POST /api/auth/login`
- `GET /api/auth/me`

## Notes

- Frontend is served from `src/finhack/web/index.html`.
- Chat history and auth sessions are stored in SQLite (`DATABASE_URL`).
- Never commit `.env` or API keys.