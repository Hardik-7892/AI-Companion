# server.py
#
# New from-scratch HTML/CSS/JS frontend for AI Companion.
# Run with:  python server.py   (opens http://127.0.0.1:8000)
#
# This file is ADDITIVE only: it wraps the shared logic in app_utils.py
# via a small FastAPI JSON API and serves the static files in web/.
# gradio_app.py and streamlit_demo/streamlit_app.py are NOT touched and
# keep working exactly as before:
#   - Gradio  -> local default (`python gradio_app.py`, root requirements.txt)
#   - Streamlit demo -> Streamlit Community Cloud
#     (entrypoint streamlit_demo/streamlit_app.py + its own lean
#     streamlit_demo/requirements.txt, which takes precedence on Cloud)

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

load_dotenv()

from app_utils import (  # noqa: E402
    BACKEND_CHOICES,
    PERSONALITY_CHOICES,
    available_model_files,
    create_chat_id,
    get_engine,
    load_all_history,
    load_chat_ids,
    load_persona,
    load_recent_history,
    reset_chat,
    run_chat,
    save_details,
)
from model import ClaudeLLM  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent
WEB_DIR = REPO_ROOT / "web"

app = FastAPI(title="AI Companion — Custom UI")


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #

class CreateChatIn(BaseModel):
    name: str = Field(default="")
    current: str | None = None


class PersonaIn(BaseModel):
    chat_id: str
    user_name: str = ""
    companion_name: str = ""
    user_gender: str = ""
    companion_gender: str = ""
    traits: list[str] = Field(default_factory=list)
    custom_personality: str = ""


class ChatIn(BaseModel):
    message: str
    chat_id: str
    model_name: str | None = None
    gpu_layers: int = 0
    backend: str = BACKEND_CHOICES[0]
    claude_model: str = ClaudeLLM.DEFAULT_MODEL


# --------------------------------------------------------------------------- #
# API
# --------------------------------------------------------------------------- #

@app.get("/api/health")
def health() -> dict:
    return {"ok": True}


@app.get("/api/config")
def config() -> dict:
    return {
        "backends": BACKEND_CHOICES,
        "personalities": PERSONALITY_CHOICES,
        "default_claude_model": ClaudeLLM.DEFAULT_MODEL,
        "has_openrouter_key": bool(os.environ.get("OPENROUTER_API_KEY")),
    }


@app.get("/api/chats")
def list_chats() -> dict:
    return {"chats": load_chat_ids()}


@app.post("/api/chats")
def create_chat(payload: CreateChatIn) -> dict:
    chat_ids, selected, message = create_chat_id(payload.name, payload.current)
    return {"chats": chat_ids, "selected": selected, "message": message}


@app.get("/api/models")
def list_models() -> dict:
    return {"models": available_model_files()}


@app.get("/api/history")
def history(chat_id: str, mode: str = "recent") -> dict:
    try:
        if mode == "all":
            hist, status = load_all_history(chat_id)
        else:
            hist, status = load_recent_history(chat_id)
    except Exception as exc:  # keep UI resilient on fresh/empty chats
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"history": hist, "status": status}


@app.delete("/api/history")
def clear_history(chat_id: str) -> dict:
    hist, status = reset_chat(chat_id)
    return {"history": hist, "status": status}


@app.get("/api/persona")
def get_persona(chat_id: str) -> dict:
    persona = load_persona(chat_id)
    return {"persona": persona.data, "choices": PERSONALITY_CHOICES}


@app.post("/api/persona")
def post_persona(payload: PersonaIn) -> dict:
    message = save_details(
        payload.user_name,
        payload.companion_name,
        payload.user_gender,
        payload.companion_gender,
        payload.traits,
        payload.custom_personality,
        payload.chat_id,
    )
    # Drop any cached LLM state by rebuilding lazily on next /api/chat call.
    # (No global engine cache here, so nothing else to clear.)
    return {"message": message}


@app.post("/api/chat")
def chat(payload: ChatIn) -> dict:
    history, status, handled = run_chat(
        payload.message,
        None,  # history is rebuilt server-side; client sends message only
        payload.chat_id,
        payload.model_name or "",
        payload.gpu_layers or 0,
        payload.backend,
        payload.claude_model or ClaudeLLM.DEFAULT_MODEL,
        engine=None,
    )
    # run_chat with history=None returns only the new pair; fetch recent for UI.
    # To keep the contract simple, return the single exchange + status.
    if handled:
        recent, status = load_recent_history(payload.chat_id)
        # Return just the last pair to append client-side.
        return {"exchange": history[-2:], "status": status, "handled": True}
    return {"exchange": [], "status": status, "handled": False}


# --------------------------------------------------------------------------- #
# Static frontend
# --------------------------------------------------------------------------- #

if WEB_DIR.is_dir():
    app.mount("/assets", StaticFiles(directory=WEB_DIR), name="assets")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    index_file = WEB_DIR / "index.html"
    if not index_file.is_file():
        raise HTTPException(status_code=500, detail="web/index.html missing")
    return FileResponse(index_file)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=False)
