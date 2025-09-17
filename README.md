<div align="center">
<img width="1200" height="475" alt="GHBanner" src="https://github.com/user-attachments/assets/0aa67016-6eaf-458a-adb2-6e31a0763ed6" />
</div>

# AI Virtual Try‑On (FastAPI + React/Vite)

Monorepo with a Python FastAPI backend (`backend_py`) and a React + Vite + TypeScript frontend (`frontend`). The legacy Node backend has been removed to avoid confusion.

View in AI Studio: https://ai.studio/apps/drive/1ORGriwJMQVw1Sd-cSjddK7sGBrrm_B6D

## What’s Included
- Backend: FastAPI with routes for generate, recommendations, style tips, history evaluation, proxy/image tools
- Frontend: React 19 + Vite 6 + Tailwind
- Product data: PostgreSQL-backed recommender cache (loaded via `db_recommender`)
- Docker: Dev and Prod compose files

## Prerequisites
- Python 3.11+
- Node.js 18+ (frontend tooling only)

## Run Locally

Backend (FastAPI)
- Terminal A
  - `cd backend_py`
  - `python -m venv .venv`
  - Activate venv (Windows) `.venv\Scripts\activate` / (Linux/macOS) `source .venv/bin/activate`
  - `pip install -r requirements.txt`
  - `uvicorn app.main:app --reload --host 0.0.0.0 --port 3001`

Frontend (Vite)
- Terminal B
  - `cd frontend`
  - `npm install` (or `npm ci`)
  - `npm run dev` (opens on 5173)

Tip: `scripts/quickstart.ps1` installs frontend deps, prepares the backend venv, and starts both services.

## Endpoints
- `GET /health`
- `POST /api/generate`
- `POST /api/recommend`
- `POST /api/recommend/from-fitting`
- `POST /api/tips`
- `POST /api/try-on/video`
- `POST /api/try-on/video/status`
- `POST /api/evaluate`
- `POST /api/recommend/by-positions` (DB-backed similar items by numeric positions)

## Configuration
Copy `backend_py/.env.example` to `backend_py/.env` and set keys as needed.
- CORS: `FRONTEND_URL`
- Azure OpenAI (optional): `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_KEY`, `AZURE_OPENAI_DEPLOYMENT_ID`
- Gemini (optional): `GEMINI_API_KEY`, `GEMINI_FIXED_PROMPT`, `GEMINI_TEMPERATURE`
- Vertex AI video generation (optional): `VERTEX_PROJECT_ID`, `VERTEX_LOCATION`, `VERTEX_MODEL_ID`, `VERTEX_API_ENDPOINT`, `GOOGLE_APPLICATION_CREDENTIALS`

Frontend dev uses `VITE_API_URL` when provided; otherwise defaults to `http://localhost:3001` via `vite.config.ts`.
Optional frontend flags: `VITE_FEATURE_VIDEO`, `VITE_VIDEO_PROMPT`, `VITE_VIDEO_ASPECT`, `VITE_VIDEO_DURATION`, `VITE_VIDEO_RESOLUTION`.

## Docker
- Dev: `docker compose -f docker-compose.dev.yml up`
- Prod: `docker compose up -d`

## Troubleshooting
- Vite not found: `cd frontend && npm install`
- npm ERESOLVE: ensure `@testing-library/react@^16` then reinstall; as fallback use `npm install --legacy-peer-deps`
- Import error `ModuleNotFoundError: app`: run from `backend_py` or use `uvicorn backend_py.app.main:app` at repo root
- Windows EOL warning on `git add`: harmless; use `.gitattributes` to normalize
- Root Node workspace was removed — run all Node commands inside `frontend/`. Use `backend_py/.venv` only
