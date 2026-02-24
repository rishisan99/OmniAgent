# OmniAgent

OmniAgent is a multimodal agent system with a FastAPI backend and Next.js frontend.
It supports text responses plus tool blocks for image, audio, document, web, RAG, knowledge-base RAG, and vision flows.

## Project Structure

- `backend/`: FastAPI API, graph orchestration, tool agents, SSE streaming, session + asset handling.
- `frontend/omniagent-ui/`: Next.js app that renders streamed assistant text and generated blocks.
- `docker-compose.yml`: Full-stack local/prod container orchestration (frontend + backend).

## Core Architecture

- API entrypoint: `backend/src/api/app.py`
- Streaming chat endpoint: `POST /api/chat/stream`
- Session endpoints: `/api/session/meta`, `/api/session/state`, `/api/session/clear`
- Model metadata endpoint: `GET /api/models`
- Graph execution:
  - v1 path: `backend/src/graph/runner.py`
  - v2 path: `backend/src/graph/v2_flow.py` (enabled via `GRAPH_V2_ENABLED=true`)
- Stream format: Server-Sent Events (`token`, `block_start`, `block_token`, `block_end`, `error`, etc.)

## Features

- Streaming assistant text tokens into chat UI.
- Tool blocks with per-task rendering:
  - `image_gen`: image preview + download/view actions
  - `tts`: in-app audio player
  - `doc`: document download/view actions
  - `web`, `rag`, `kb_rag`, `vision`: knowledge/retrieval context support
- File upload + session artifact memory.
- Knowledge-base indexing/warmup on backend startup.

## Environment Variables

Create `.env.prod` from `.env.prod.example`.

Required for common flows:

- `OPENAI_API_KEY`
- `TAVILY_API_KEY` (for web/news tasks)

Optional providers:

- `ANTHROPIC_API_KEY`
- `GOOGLE_API_KEY`

Frontend/backend wiring:

- `NEXT_PUBLIC_API_BASE=` (keep empty for same-origin `/api` proxy)
- `BACKEND_INTERNAL_URL=http://backend:8000`

Runtime/model controls:

- `GRAPH_V2_ENABLED`
- `TEXT_PROVIDER`, `TEXT_MODEL`
- `SUPPORT_PROVIDER`, `SUPPORT_MODEL`
- `PLANNER_PROVIDER`, `PLANNER_MODEL`
- `IMAGE_TASK_TIMEOUT_SEC`, `IMAGE_API_TIMEOUT_SEC`

## Local Development (without Docker)

Backend (from repo root):

```bash
uv run uvicorn backend.src.api.app:app --host 0.0.0.0 --port 8000 --reload
```

Frontend:

```bash
cd frontend/omniagent-ui
npm ci
npm run dev
```

App URLs:

- Frontend: `http://localhost:3000`
- Backend API: `http://localhost:8000`

## Docker Run

```bash
cp .env.prod.example .env.prod
# edit .env.prod with real keys
docker compose --env-file .env.prod up -d --build
```

Open `http://localhost:3000`.

## Short Run Instructions

1. `cp .env.prod.example .env.prod`
2. Add API keys in `.env.prod`
3. `docker compose --env-file .env.prod up -d --build`
4. Open `http://localhost:3000`
