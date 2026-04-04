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

## Available endpoints

- `GET /health`
- `POST /api/chat`
- `GET /api/chat/history/{session_id}`
- `POST /api/agents/news-intake/run`
- `GET /api/agents/news-intake/documents`
- `POST /api/agents/news-intake/backfill`
- `POST /api/agents/exposure/analyze`

## Notes

- The chatbot backend currently uses in-memory session state in `src/finhack/session_chatbot.py`.
- Frontend entrypoint is served from `src/finhack/web/index.html`.