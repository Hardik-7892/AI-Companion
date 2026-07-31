# app_utils.py
#
# Shared, UI-agnostic logic used by both the Gradio app (gradio_app.py) and
# the Streamlit demo (streamlit_demo/streamlit_app.py).
# Keep everything here free of gradio / streamlit imports so the logic can be
# reused (and tested) by either frontend.

import json
import os
from pathlib import Path

from model import ClaudeLLM, LLM, ChatEngine, Memory, Persona

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

CHATS_FILE  = "chats.json"
CHATS_BASE  = "chats"
MODELS_DIR  = Path("./models")

# Number of user-assistant pairs shown in the chat panel on load.
# The full log is always in memory.json; this only controls the UI default.
N_RECENT_UI: int = 10

PERSONALITY_CHOICES = [
    "Playful", "Affectionate", "Shy", "Confident", "Teasing",
    "Supportive", "Jealous", "Clingy", "Mature", "Tsundere",
]

BACKEND_CHOICES = ["Local (GGUF)", "Claude (OpenRouter)"]


def available_model_files() -> list[str]:
    """GGUF models present in models/ — empty means the Local backend is unusable."""
    return sorted(f.name for f in MODELS_DIR.glob("*.gguf"))


# --------------------------------------------------------------------------- #
# Engine factory
# --------------------------------------------------------------------------- #

def get_engine(
    chat_id: str,
    model_path: str | Path,
    n_gpu_layers: int = 0,
    backend: str = "Local (GGUF)",
    claude_model: str = ClaudeLLM.DEFAULT_MODEL,
) -> ChatEngine:
    if backend == "Claude (OpenRouter)":
        llm = ClaudeLLM(model=claude_model or ClaudeLLM.DEFAULT_MODEL)
    else:
        llm = LLM.get_instance(str(model_path), n_gpu_layers=n_gpu_layers)
    persona = Persona(path=f"{CHATS_BASE}/{chat_id}/persona.json")
    memory  = Memory(path=f"{CHATS_BASE}/{chat_id}/memory.json")
    return ChatEngine(llm=llm, memory=memory, persona=persona)


def load_persona(chat_id: str) -> Persona:
    """Load the persisted persona for a chat (used by the UIs to prefill forms)."""
    return Persona(path=f"{CHATS_BASE}/{chat_id}/persona.json")


# --------------------------------------------------------------------------- #
# Chat-list helpers
# --------------------------------------------------------------------------- #

def load_chat_ids() -> list[str]:
    try:
        with open(CHATS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return ["default"]


def save_chat_ids(chat_ids: list[str]) -> None:
    with open(CHATS_FILE, "w", encoding="utf-8") as f:
        json.dump(chat_ids, f, ensure_ascii=False, indent=4)


def create_chat_id(new_name: str, current_value: str | None) -> tuple[list[str], str, str]:
    """
    Add a new chat id (if unique) and return (updated ids, selected id, status).
    Selecting an existing name just switches to it; blank names are rejected.
    """
    new_name = (new_name or "").strip()
    chat_ids = load_chat_ids()

    if new_name and new_name not in chat_ids:
        chat_ids.append(new_name)
        save_chat_ids(chat_ids)
        return chat_ids, new_name, f"Chat '{new_name}' created."

    if new_name in chat_ids:
        return chat_ids, new_name, f"Switched to '{new_name}'."

    selected = current_value or (chat_ids[0] if chat_ids else "default")
    return chat_ids, selected, "Enter a unique chat name."


# --------------------------------------------------------------------------- #
# History helpers
# --------------------------------------------------------------------------- #

def _history_status(shown: int, total: int) -> str:
    if shown >= total:
        return f"All {total} exchanges loaded."
    return f"Showing latest {shown} of {total} exchanges — click 'Load All History' to see more."


def load_recent_history(chat_id: str) -> tuple[list[dict], str]:
    """Load the last N_RECENT_UI pairs for a chat."""
    memory = Memory(path=f"{CHATS_BASE}/{chat_id}/memory.json")
    total  = memory.pair_count()
    recent = memory.get_recent(N_RECENT_UI)
    shown  = len(recent) // 2
    return recent, _history_status(shown, total)


def load_all_history(chat_id: str) -> tuple[list[dict], str]:
    """Load every message for a chat."""
    memory = Memory(path=f"{CHATS_BASE}/{chat_id}/memory.json")
    total  = memory.pair_count()
    return memory.get_all(), _history_status(total, total)


def reset_chat(chat_id: str) -> tuple[list, str]:
    """Wipe a chat's archive + FAISS index."""
    memory = Memory(path=f"{CHATS_BASE}/{chat_id}/memory.json")
    memory.clear()
    return [], "History cleared."


# --------------------------------------------------------------------------- #
# Persona / details
# --------------------------------------------------------------------------- #

def save_details(
    user_name: str,
    companion_name: str,
    user_gender: str,
    companion_gender: str,
    traits: list[str],
    custom_personality: str,
    chat_id: str,
) -> str:
    persona = Persona(path=f"{CHATS_BASE}/{chat_id}/persona.json")
    persona.update(
        user_name          = user_name or persona.data["user_name"],
        companion_name    = companion_name or persona.data["companion_name"],
        personality_traits = traits or [],
        custom_personality = custom_personality or "",
        user_gender        = user_gender or persona.data.get("user_gender", ""),
        companion_gender   = companion_gender or persona.data.get("companion_gender", ""),
    )
    return f"Details saved for chat '{chat_id}'!"


# --------------------------------------------------------------------------- #
# Chat
# --------------------------------------------------------------------------- #

def run_chat(
    user_input: str,
    history: list[dict] | None,
    chat_id: str,
    model_name: str,
    gpu_layers: float,
    backend: str,
    claude_model: str,
    engine: ChatEngine | None = None,
) -> tuple[list[dict], str]:
    """
    Core chat turn, shared by the Gradio and Streamlit frontends.

    Returns (history, status, handled). History is unchanged (with an
    explanatory status and handled=False) when there is nothing to send, the
    OpenRouter key is missing, or a Local backend is chosen without a model
    file — so the frontend can keep the user's input for a retry.

    Pass a pre-built `engine` (e.g. a cached one from Streamlit) to skip the
    expensive per-call construction (embedder load, client setup). When it is
    None the engine is built with get_engine().
    """
    if not user_input.strip():
        return history or [], "", False

    if backend == "Claude (OpenRouter)":
        if not os.environ.get("OPENROUTER_API_KEY"):
            return history or [], (
                "Set your key first: add OPENROUTER_API_KEY to .env "
                "(see README) or to the Streamlit secret, then restart."
            ), False

    history = history or []

    if backend != "Claude (OpenRouter)":
        model_path = MODELS_DIR / (model_name or "")
        if not model_name or not model_path.is_file():
            return history, (
                "No GGUF model selected. Add a .gguf file to the models/ "
                "folder (and restart) to use the Local (GGUF) backend."
            ), False

    model_path = MODELS_DIR / (model_name or "")
    n_gpu      = max(int(gpu_layers or 0), 0)
    if engine is None:
        engine = get_engine(
            chat_id, model_path, n_gpu_layers=n_gpu,
            backend=backend, claude_model=claude_model,
        )
    reply = engine.chat(user_input)

    history.append({"role": "user",      "content": user_input})
    history.append({"role": "assistant", "content": reply})

    memory = Memory(path=f"{CHATS_BASE}/{chat_id}/memory.json")
    total  = memory.pair_count()
    shown  = len(history) // 2
    return history, _history_status(shown, total), True
