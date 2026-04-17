# app.py

import json
from pathlib import Path

import gradio as gr

from model import LLM, ChatEngine, Memory, Persona

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


# --------------------------------------------------------------------------- #
# Engine factory
# --------------------------------------------------------------------------- #

def get_engine(chat_id: str, model_path: str | Path) -> ChatEngine:
    llm     = LLM.get_instance(str(model_path))
    persona = Persona(path=f"{CHATS_BASE}/{chat_id}/persona.json")
    memory  = Memory(path=f"{CHATS_BASE}/{chat_id}/memory.json")
    return ChatEngine(llm=llm, memory=memory, persona=persona)


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


def refresh_chat_selector() -> gr.update:
    """
    Called by app.load() every time the browser loads the page.
    Re-reads chats.json so newly-created chats survive a reload.
    """
    chat_ids = load_chat_ids()
    return gr.update(choices=chat_ids, value=chat_ids[0])


def create_chat(new_name: str, current_value: str):
    """
    current_value is the Dropdown's *selected value* (a string), not its choices.
    We always read the authoritative list from disk.
    """
    new_name  = (new_name or "").strip()
    chat_ids  = load_chat_ids()

    if new_name and new_name not in chat_ids:
        chat_ids.append(new_name)
        save_chat_ids(chat_ids)
        return gr.update(choices=chat_ids, value=new_name), f"Chat '{new_name}' created."

    if new_name in chat_ids:
        # User typed an existing name — just switch to it silently
        return gr.update(choices=chat_ids, value=new_name), f"Switched to '{new_name}'."

    return gr.update(choices=chat_ids, value=current_value), "Enter a unique chat name."


# --------------------------------------------------------------------------- #
# History helpers
# --------------------------------------------------------------------------- #

def _history_status(shown: int, total: int) -> str:
    if shown >= total:
        return f"All {total} exchanges loaded."
    return f"Showing latest {shown} of {total} exchanges — click 'Load All History' to see more."


def load_recent_history(chat_id: str) -> tuple[list[dict], str]:
    """Load the last N_RECENT_UI pairs into the chatbot panel."""
    memory     = Memory(path=f"{CHATS_BASE}/{chat_id}/memory.json")
    total      = memory.pair_count()
    recent     = memory.get_recent(N_RECENT_UI)
    shown      = len(recent) // 2
    return recent, _history_status(shown, total)


def load_all_history(chat_id: str) -> tuple[list[dict], str]:
    """Load every message for this chat into the chatbot panel."""
    memory = Memory(path=f"{CHATS_BASE}/{chat_id}/memory.json")
    total  = memory.pair_count()
    return memory.get_all(), _history_status(total, total)


# --------------------------------------------------------------------------- #
# Persona / details tab
# --------------------------------------------------------------------------- #

def save_details(
    user_name: str,
    girlfriend_name: str,
    traits: list[str],
    custom_personality: str,
    chat_id: str,
) -> str:
    persona = Persona(path=f"{CHATS_BASE}/{chat_id}/persona.json")
    persona.update(
        user_name          = user_name or persona.data["user_name"],
        girlfriend_name    = girlfriend_name or persona.data["girlfriend_name"],
        personality_traits = traits or [],
        custom_personality = custom_personality or "",
    )
    return f"Details saved for chat '{chat_id}'!"


# --------------------------------------------------------------------------- #
# Chat callback
# --------------------------------------------------------------------------- #

def chat_page(
    user_input: str,
    history: list[dict],
    chat_id: str,
    model_name: str,
) -> tuple[str, list[dict], str]:
    if not user_input.strip():
        return user_input, history or [], ""

    history    = history or []
    model_path = MODELS_DIR / model_name
    engine     = get_engine(chat_id, model_path)
    reply      = engine.chat(user_input)

    history.append({"role": "user",      "content": user_input})
    history.append({"role": "assistant", "content": reply})

    # Update the status line with new totals
    memory = Memory(path=f"{CHATS_BASE}/{chat_id}/memory.json")
    total  = memory.pair_count()
    shown  = len(history) // 2
    status = _history_status(shown, total)

    return "", history, status


def reset_chat(chat_id: str) -> tuple[list, str]:
    memory = Memory(path=f"{CHATS_BASE}/{chat_id}/memory.json")
    memory.clear()
    return [], "History cleared."


# --------------------------------------------------------------------------- #
# Theme
# --------------------------------------------------------------------------- #

theme = gr.themes.Soft(
    primary_hue="pink",
    secondary_hue="violet",
    neutral_hue="gray",
).set(
    body_background_fill    = "#fdf2f8",
    body_text_color         = "#1f2933",
    block_background_fill   = "#ffffff",
    block_border_width      = "1px",
    block_border_color      = "#f9a8d4",
    input_background_fill   = "#ffffff",
    input_border_color      = "#f472b6",
    button_primary_background_fill       = "linear-gradient(90deg, *primary_300, *secondary_300)",
    button_primary_background_fill_hover = "linear-gradient(90deg, *primary_200, *secondary_200)",
    button_primary_text_color            = "#ffffff",
)

custom_css = """
body[data-theme="pink"] #chat-selector-row { background:#fce7f3; border-color:#f9a8d4; }
body[data-theme="pink"] #tabs-container    { background:#fdf2f8; border-color:#f9a8d4; }
body[data-theme="pink"] #chatbot-block .gradio-chatbot { background:#fff7fb; border-color:#fecdd3; }

body[data-theme="blue"] #chat-selector-row { background:#dbeafe; border-color:#60a5fa; }
body[data-theme="blue"] #tabs-container    { background:#eff6ff; border-color:#60a5fa; }
body[data-theme="blue"] #chatbot-block .gradio-chatbot { background:#e0f2fe; border-color:#93c5fd; }

body[data-theme="dark"] #chat-selector-row { background:#1f2933; border-color:#4b5563; }
body[data-theme="dark"] #tabs-container    { background:#0f172a; border-color:#4b5563; }
body[data-theme="dark"] #chatbot-block .gradio-chatbot { background:#020617; border-color:#475569; }

#chat-selector-row { padding:.75rem 1rem; border-radius:.75rem; border-width:1px; }
#tabs-container    { border-radius:.75rem; padding:.75rem; border-width:1px; }
#chatbot-block .gradio-chatbot { border-radius:.75rem; border-width:1px; }
button { border-radius:9999px !important; }
"""

apply_theme_js = """
(theme_name) => {
    const body = document.querySelector('body');
    if (!body) return;
    const map = { Pink: 'pink', Blue: 'blue', Dark: 'dark' };
    body.setAttribute('data-theme', map[theme_name] ?? 'pink');
    return theme_name;
}
"""


# --------------------------------------------------------------------------- #
# Gradio app
# --------------------------------------------------------------------------- #

def launch_gradio_app() -> None:
    with gr.Blocks(theme=theme, css=custom_css) as app:

        chat_ids = load_chat_ids()

        # ---- Top bar -------------------------------------------------------
        with gr.Row(elem_id="top-bar"):
            with gr.Row(elem_id="chat-selector-row"):
                chat_selector = gr.Dropdown(
                    choices=chat_ids, value=chat_ids[0], label="Select chat"
                )
                new_chat_name = gr.Textbox(
                    label="New chat name", placeholder="e.g. Chat 2"
                )
                create_chat_btn = gr.Button("Create Chat", variant="primary")
                create_status   = gr.Textbox(label="Chat status", interactive=False)

            theme_choice = gr.Dropdown(
                choices=["Pink", "Blue", "Dark"], value="Pink",
                label="Theme", scale=0, elem_id="theme-dropdown",
            )

        # Re-read chat list from disk every time the browser loads the page.
        # This is the fix for chats disappearing after a reload.
        app.load(fn=refresh_chat_selector, outputs=[chat_selector])

        theme_choice.change(None, inputs=theme_choice, outputs=None, js=apply_theme_js)

        create_chat_btn.click(
            fn=create_chat,
            inputs=[new_chat_name, chat_selector],
            outputs=[chat_selector, create_status],
        )

        # ---- Main tabs -----------------------------------------------------
        with gr.Column(elem_id="tabs-container"):
            with gr.Tabs():

                # ---- Tab 1: Details ----------------------------------------
                with gr.TabItem("Enter Details"):
                    gr.Markdown("### Please enter your details (Optional):")
                    with gr.Row():
                        user_name_input = gr.Textbox(
                            label="Your Name (Optional)", placeholder="Enter your name"
                        )
                        girlfriend_name_input = gr.Textbox(
                            label="Girlfriend's Name (Optional)", placeholder="Enter her name"
                        )
                    gr.Markdown("### Choose personality traits (Optional):")
                    traits_input = gr.CheckboxGroup(
                        choices=PERSONALITY_CHOICES, label="Select one or more traits"
                    )
                    custom_personality_input = gr.Textbox(
                        label="Custom personality (Optional)",
                        placeholder="Describe how you want her to behave, tone, style, etc.",
                        lines=3,
                    )
                    initialize_button = gr.Button("Initialize Chat", variant="secondary")
                    message_box = gr.Textbox(label="Status", interactive=False)

                    initialize_button.click(
                        fn=save_details,
                        inputs=[
                            user_name_input, girlfriend_name_input,
                            traits_input, custom_personality_input, chat_selector,
                        ],
                        outputs=message_box,
                    )

                # ---- Tab 2: Chat -------------------------------------------
                with gr.TabItem("Chat"):
                    gr.Markdown("### Chat with Your AI Girlfriend")

                    # Status bar: shows how many exchanges are loaded vs total
                    history_status = gr.Textbox(
                        value="", interactive=False, show_label=False,
                        container=False, elem_id="history-status"
                    )
                    load_all_btn = gr.Button(
                        "📜 Load All History", variant="secondary", size="sm"
                    )

                    with gr.Column(elem_id="chatbot-block"):
                        chatbot = gr.Chatbot(
                            label="Conversation", height=520
                        )

                    model_files = [f.name for f in MODELS_DIR.glob("*.gguf")]
                    model_selector = gr.Dropdown(
                        choices=model_files, label="Choose Model",
                        value=model_files[0] if model_files else None,
                    )
                    chat_input = gr.Textbox(
                        label="Your Message", placeholder="Say something...", lines=2
                    )
                    with gr.Row():
                        send_btn  = gr.Button("Send",              variant="primary")
                        reset_btn = gr.Button("Reset Chat",        variant="secondary")

                    # When the selected chat changes → load latest N into chatbot
                    chat_selector.change(
                        fn=load_recent_history,
                        inputs=[chat_selector],
                        outputs=[chatbot, history_status],
                    )

                    # Load all history button
                    load_all_btn.click(
                        fn=load_all_history,
                        inputs=[chat_selector],
                        outputs=[chatbot, history_status],
                    )

                    # Also load recent history when the page first loads
                    app.load(
                        fn=load_recent_history,
                        inputs=[chat_selector],
                        outputs=[chatbot, history_status],
                    )

                    shared_inputs  = [chat_input, chatbot, chat_selector, model_selector]
                    shared_outputs = [chat_input, chatbot, history_status]

                    chat_input.submit(fn=chat_page, inputs=shared_inputs, outputs=shared_outputs)
                    send_btn.click(   fn=chat_page, inputs=shared_inputs, outputs=shared_outputs)

                    reset_btn.click(
                        fn=reset_chat,
                        inputs=[chat_selector],
                        outputs=[chatbot, history_status],
                    )

        app.launch(inbrowser=True)


if __name__ == "__main__":
    launch_gradio_app()