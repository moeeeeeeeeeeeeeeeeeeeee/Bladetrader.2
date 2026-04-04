# BladeTrader

FastAPI backend plus a deployable frontend entrypoint for Case 4-style sentiment and exposure workflows.

## Run locally

1. Install dependencies:
   - `python -m pip install -r requirements.txt`
2. Copy `.env.example` to `.env` and set keys as needed.
3. Start API + web app:
   - `uvicorn finhack.api:app --app-dir src --reload --host 127.0.0.1 --port 8000`
4. Open:
   - `http://127.0.0.1:8000/`

## Deploy (public app link)

### Render (recommended)

1. Push your latest `main` branch to GitHub.
2. In Render, choose **New +** -> **Blueprint**.
3. Select this repository. Render will use `render.yaml`.
4. Deploy and open the generated URL.

Quick deploy button:
- [![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/moeeeeeeeeeeeeeeeeeeeee/Bladetrader.2)

Start command used:
- `uvicorn finhack.api:app --app-dir src --host 0.0.0.0 --port $PORT`

Recommended environment variables:
- `CORS_ALLOWED_ORIGINS=https://<your-render-domain>`
- `GNEWS_API_KEY=<optional-for-live-news>`
- `MARKET_DATA_PROVIDER=eodhd` and `EODHD_API_KEY=<required-for-eodhd-market-and-news>`

### Railway

1. Create a new Railway project from this GitHub repository.
2. Railway auto-detects `railway.json` and uses the start command.
3. Open the generated domain from Railway project settings.

## Available endpoints

- `GET /health`
- `GET /api/deploy/readiness`
- `POST /api/chat`
- `GET /api/chat/history/{session_id}`
- `POST /api/agents/news-intake/run`
- `GET /api/agents/news-intake/documents`
- `POST /api/agents/news-intake/backfill`
- `POST /api/agents/exposure/analyze`
- `POST /api/research/annual/build`
- `GET /api/research/annual/events`
- `GET /api/research/annual/spillover`

## Annual AI-report dataset (14 primary companies + spillovers)

Build historical annual-anchor sentiment/spillover features into SQLite:

- `py scripts/build_annual_ai_dataset.py --years-back 5 --pre-days 30 --post-days 20 --max-news-per-event 25`

What this writes:

- `annual_company_universe` table (14 primary AI names + spillover names)
- `annual_event` table (one annual anchor event per symbol/year, with pre/post return and target)
- `annual_event_news` table (AI-event news items around each annual anchor)
- `annual_spillover_effect` table (weighted spillover paths from source AI leaders to related targets)

## Notes

- The chatbot backend currently uses in-memory session state in `src/finhack/session_chatbot.py`.
- Frontend entrypoint is served from `src/finhack/web/index.html`.