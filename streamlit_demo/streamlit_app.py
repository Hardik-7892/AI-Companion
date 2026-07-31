# streamlit_demo/streamlit_app.py
#
# Streamlit UI for AI Companion. Run locally with:
#     streamlit run streamlit_demo/streamlit_app.py
#
# Shared logic lives in app_utils.py at the repo root (also used by the
# Gradio app, gradio_app.py).
#
# Deploy on Streamlit Community Cloud:
#   * point the "Main file" at streamlit_demo/streamlit_app.py
#   * add OPENROUTER_API_KEY under the app's Secrets
# The sibling requirements.txt is used by the Cloud build (lean deps).

import os
import sys
from pathlib import Path

# Make the repo root importable so `app_utils` and `model` resolve.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import streamlit as st

from app_utils import (
    BACKEND_CHOICES,
    MODELS_DIR,
    PERSONALITY_CHOICES,
    available_model_files,
    create_chat_id,
    get_engine,
    load_all_history,
    load_chat_ids,
    load_recent_history,
    reset_chat,
    run_chat,
    save_details,
)
from model import ClaudeLLM

# --------------------------------------------------------------------------- #
# Secrets -> environment
# --------------------------------------------------------------------------- #

# ClaudeLLM reads OPENROUTER_API_KEY from os.environ. Streamlit exposes
# st.secrets; mirror it into the env so the shared logic works unchanged.
if "OPENROUTER_API_KEY" not in os.environ:
    try:
        os.environ["OPENROUTER_API_KEY"] = st.secrets.get("OPENROUTER_API_KEY", "")
    except Exception:
        pass  # no secrets configured (e.g. purely local run without the key)


@st.cache_resource(show_spinner=False)
def get_cached_engine(chat_id: str, model_name: str, gpu_layers: int,
                      backend: str, claude_model: str):
    """Cache the ChatEngine per (chat, backend, model) so the MiniLM embedder
    and API client are not rebuilt on every Streamlit rerun."""
    model_path = MODELS_DIR / (model_name or "") if backend != "Claude (OpenRouter)" else MODELS_DIR / ""
    return get_engine(
        chat_id, model_path,
        n_gpu_layers=int(gpu_layers or 0),
        backend=backend, claude_model=claude_model,
    )


st.set_page_config(page_title="AI Companion", page_icon="💗", layout="wide")

# --------------------------------------------------------------------------- #
# State helpers
# --------------------------------------------------------------------------- #

st.session_state.setdefault("_hist", {})    # chat_id -> list[dict]
st.session_state.setdefault("_status", {})  # chat_id -> status string


def chat_state(chat_id: str) -> tuple[list[dict], str]:
    """Return (history, status) for a chat, loading recent history on first visit."""
    histories = st.session_state["_hist"]
    statuses  = st.session_state["_status"]
    if chat_id not in histories:
        history, status = load_recent_history(chat_id)
        histories[chat_id] = history
        statuses[chat_id]  = status
    return histories[chat_id], statuses[chat_id]


# --------------------------------------------------------------------------- #
# Layout
# --------------------------------------------------------------------------- #

st.warning(
    "⚠️ **Public demo — not private.** This app has no accounts or "
    "encryption: all visitors share the same chat history and the owner can "
    "read it. Don't share personal or sensitive information. Messages are "
    "also sent to OpenRouter."
)

chat_ids = load_chat_ids()

def on_create_chat() -> None:
    ids, selected, message = create_chat_id(
        st.session_state.get("new_chat_name", ""),
        st.session_state.get("chat_id"),
    )
    st.session_state["chat_id"] = selected
    st.session_state["_flash"] = message
    st.session_state["new_chat_name"] = ""

# ---- Sidebar ------------------------------------------------------------- #
with st.sidebar:
    st.title("💗 AI Companion")

    current = st.session_state.get("chat_id")
    index = chat_ids.index(current) if current in chat_ids else 0
    chat_id = st.selectbox("Select chat", chat_ids, index=index, key="chat_id")

    st.text_input("New chat name", placeholder="e.g. Chat 2", key="new_chat_name")
    st.button("Create Chat", type="primary", use_container_width=True, on_click=on_create_chat)

    if "_flash" in st.session_state:
        st.success(st.session_state.pop("_flash"))

    st.divider()
    st.subheader("Backend")

    model_files = available_model_files()
    if model_files:
        backend = st.radio("Backend", BACKEND_CHOICES, key="backend")
    else:
        backend = "Claude (OpenRouter)"
        st.info(
            "No GGUF model found in `models/` — the Local (GGUF) backend is "
            "disabled. Drop a `.gguf` file there and restart to enable it."
        )

    if backend == "Local (GGUF)":
        model_name = st.selectbox("Choose Model", model_files, key="model_name")
        gpu_layers = st.number_input(
            "GPU Layers", min_value=0, step=1, value=0, key="gpu_layers",
            help="0 = CPU (works anywhere). Raise to offload to your GPU.",
        )
        claude_model = ClaudeLLM.DEFAULT_MODEL  # unused for the local backend
    else:
        claude_model = st.text_input(
            "OpenRouter Model", value=ClaudeLLM.DEFAULT_MODEL, key="claude_model"
        )
        model_name, gpu_layers = None, 0

    st.divider()
    if st.button("📜 Load All History", use_container_width=True):
        history, status = load_all_history(chat_id)
        st.session_state["_hist"][chat_id]   = history
        st.session_state["_status"][chat_id] = status
        st.rerun()

    if st.button("Delete Chat Data", use_container_width=True):
        reset_chat(chat_id)
        get_cached_engine.clear()  # drop any stale in-memory engine/memory
        st.session_state["_hist"][chat_id]   = []
        st.session_state["_status"][chat_id] = "History cleared."
        st.rerun()

# ---- Main ---------------------------------------------------------------- #
st.title("Chat with Your AI Companion")

history, status = chat_state(chat_id)
st.caption(status)

with st.expander("Enter Details (Optional)"):
    c1, c2 = st.columns(2)
    user_name_input   = c1.text_input("Your Name (Optional)", key="d_user_name")
    companion_name_input = c2.text_input("Companion's Name (Optional)", key="d_companion_name")
    c3, c4 = st.columns(2)
    user_gender_input = c3.text_input("Your Gender (Optional)", key="d_user_gender")
    companion_gender_input = c4.text_input("Companion's Gender (Optional)", key="d_companion_gender")
    traits_input = st.multiselect("Select one or more traits", PERSONALITY_CHOICES, key="d_traits")
    custom_personality_input = st.text_area(
        "Custom personality (Optional)",
        placeholder="Describe how you want her to behave, tone, style, etc.",
        key="d_custom",
    )
    if st.button("Save Details", type="secondary"):
        st.success(save_details(
            user_name_input, companion_name_input,
            user_gender_input, companion_gender_input,
            traits_input, custom_personality_input, chat_id,
        ))
        get_cached_engine.clear()  # drop stale engine so the persona reloads from disk

for message in history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Say something..."):
    engine = get_cached_engine(chat_id, model_name, int(gpu_layers or 0),
                               backend, claude_model)
    new_history, new_status, handled = run_chat(
        prompt, history, chat_id, model_name, gpu_layers,
        backend, claude_model, engine=engine,
    )
    st.session_state["_hist"][chat_id]   = new_history
    st.session_state["_status"][chat_id] = new_status
    if not handled:
        st.error(new_status)
    st.rerun()
