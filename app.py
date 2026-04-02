# app.py

import json
import gradio as gr
from model import chat, load_persona, save_persona
from audio import transcribe_audio, tts_from_text
from pathlib import Path

# ---------- chat list helpers ----------

CHATS_FILE = "chats.json"

def load_chat_ids():
    try:
        with open(CHATS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        # default first chat
        return ["default"]

def save_chat_ids(chat_ids):
    with open(CHATS_FILE, "w", encoding="utf-8") as f:
        json.dump(chat_ids, f, ensure_ascii=False, indent=4)


def create_chat(new_name, current_chat_ids):
    new_name = (new_name or "").strip()
    # current_chat_ids will be the current value of the dropdown
    if isinstance(current_chat_ids, list):
        chat_ids = current_chat_ids
    else:
        chat_ids = load_chat_ids()

    if new_name and new_name not in chat_ids:
        chat_ids.append(new_name)
        save_chat_ids(chat_ids)
        msg = f"Chat '{new_name}' created."
        value = new_name
    else:
        msg = "Enter a unique chat name."
        value = chat_ids[0] if chat_ids else "default"

    return gr.update(choices=chat_ids, value=value), msg


# ---------- persona + traits ----------

PERSONALITY_CHOICES = [
    "Playful",
    "Affectionate",
    "Shy",
    "Confident",
    "Teasing",
    "Supportive",
    "Jealous",
    "Clingy",
    "Mature",
    "Tsundere",
]


def ask_names_and_traits(user_name, girlfriend_name, traits, custom_personality, chat_id):
    persona = load_persona(chat_id)

    if user_name:
        persona["user_name"] = user_name
    if girlfriend_name:
        persona["girlfriend_name"] = girlfriend_name

    persona["personality_traits"] = traits or []
    persona["custom_personality"] = custom_personality or ""

    save_persona(persona, chat_id)

    return f"Details saved for chat '{chat_id}'!"


# ---------- chat callback (messages format) ----------

def chat_page(user_input, history, chat_id, model_name):
    """
    user_input: current message from textbox
    history: list of dicts, each with {'role': 'user'|'assistant', 'content': str}
    """
    if history is None:
        history = []

    persona = load_persona(chat_id)
    user_name = persona.get("user_name", "")
    girlfriend_name = persona.get("girlfriend_name", "")
    
    # Construct model path
    MODELS_DIR = Path("./models")
    model_path = MODELS_DIR / model_name

    assistant_reply = chat(
        user_input,
        chat_id=chat_id,
        user_name=user_name,
        girlfriend_name=girlfriend_name,
        model_path = model_path
    )

    history.append({"role": "user", "content": user_input})
    history.append({"role": "assistant", "content": assistant_reply})

    return "", history


# ---------- reset conversation for a chat (optional) ----------

def reset_chat(chat_id):
    # Only resets UI history; your model.py can expose a clear function if you also
    # want to wipe memory.json/persona.json for that chat.
    return [], f"Cleared visible history for chat '{chat_id}'."


# ---------- theme + CSS ----------

# Simple built-in theme tweak (you can swap to Glass/Soft/etc.)
theme = gr.themes.Soft(
    primary_hue="pink",
    secondary_hue="violet",
    neutral_hue="gray",
).set(
    body_background_fill="#fdf2f8",
    body_text_color="#1f2933",
    block_background_fill="#ffffff",
    block_border_width="1px",
    block_border_color="#f9a8d4",
    input_background_fill="#ffffff",
    input_border_color="#f472b6",
    button_primary_background_fill="linear-gradient(90deg, *primary_300, *secondary_300)",
    button_primary_background_fill_hover="linear-gradient(90deg, *primary_200, *secondary_200)",
    button_primary_text_color="#ffffff",
)


custom_css = """
body[data-theme="pink"] #chat-selector-row {
  background: #fce7f3;
  border-color: #f9a8d4;
}
body[data-theme="pink"] #tabs-container {
  background: #fdf2f8;
  border-color: #f9a8d4;
}
body[data-theme="pink"] #chatbot-block .gradio-chatbot {
  background: #fff7fb;
  border-color: #fecdd3;
}

/* Blue theme */
body[data-theme="blue"] #chat-selector-row {
  background: #dbeafe;
  border-color: #60a5fa;
}
body[data-theme="blue"] #tabs-container {
  background: #eff6ff;
  border-color: #60a5fa;
}
body[data-theme="blue"] #chatbot-block .gradio-chatbot {
  background: #e0f2fe;
  border-color: #93c5fd;
}

/* Dark theme */
body[data-theme="dark"] #chat-selector-row {
  background: #1f2933;
  border-color: #4b5563;
}
body[data-theme="dark"] #tabs-container {
  background: #0f172a;
  border-color: #4b5563;
}
body[data-theme="dark"] #chatbot-block .gradio-chatbot {
  background: #020617;
  border-color: #475569;
}

/* Shared tweaks */
#chat-selector-row {
  padding: 0.75rem 1rem;
  border-radius: 0.75rem;
  border-width: 1px;
}
#tabs-container {
  border-radius: 0.75rem;
  padding: 0.75rem;
  border-width: 1px;
}
#chatbot-block .gradio-chatbot {
  border-radius: 0.75rem;
  border-width: 1px;
}
button {
  border-radius: 9999px !important;
}
"""

apply_theme_js = """
(theme_name) => {
  const body = document.querySelector('body');
  if (!body) return;
  if (theme_name === "Pink") {
    body.setAttribute('data-theme', 'pink');
  } else if (theme_name === "Blue") {
    body.setAttribute('data-theme', 'blue');
  } else if (theme_name === "Dark") {
    body.setAttribute('data-theme', 'dark');
  } else {
    body.setAttribute('data-theme', 'pink');
  }
  return theme_name;
}
"""


def apply_theme(theme_name):
    if theme_name == "Pink":
        return "theme-pink"
    if theme_name == "Blue":
        return "theme-blue"
    if theme_name == "Dark":
        return "theme-dark"
    return "theme-pink"

# ---------- Gradio app ----------
def launch_gradio_app():
    with gr.Blocks(theme=theme, css=custom_css) as app:
        chat_ids = load_chat_ids()

        # Top bar: left = chat selection, right = theme selector
        with gr.Row(elem_id="top-bar"):
            with gr.Row(elem_id="chat-selector-row"):
                chat_selector = gr.Dropdown(
                    choices=chat_ids,
                    value=chat_ids[0],
                    label="Select chat",
                )
                new_chat_name = gr.Textbox(
                    label="New chat name",
                    placeholder="e.g. Chat 2",
                )
                create_chat_btn = gr.Button("Create Chat", variant="primary")
                create_status = gr.Textbox(
                    label="Chat status",
                    interactive=False,
                )

            # Right side: theme selector
            theme_choice = gr.Dropdown(
                choices=["Pink", "Blue", "Dark"],
                value="Pink",
                label="Theme",
                scale=0,
                elem_id="theme-dropdown",
            )

        # # hidden anchor element to carry the theme class
        # theme_anchor = gr.HTML(value="", elem_id="theme-anchor")

        # # When theme changes, write class name into theme_anchor
        # theme_choice.change(
        #     fn=apply_theme,
        #     inputs=theme_choice,
        #     outputs=theme_anchor,
        # )
        theme_choice.change(
            None,
            inputs=theme_choice,
            outputs=None,
            js=apply_theme_js,
        )

        # Chat creation wiring
        create_chat_btn.click(
            fn=create_chat,
            inputs=[new_chat_name, chat_selector],
            outputs=[chat_selector, create_status],
        )

        # Main content (tabs)
        with gr.Column(elem_id="tabs-container"):
            with gr.Tabs():
                # ---------- First Tab: Enter Details ----------
                with gr.TabItem("Enter Details"):
                    gr.Markdown("### Please enter your details (Optional):")

                    with gr.Row():
                        user_name_input = gr.Textbox(
                            label="Your Name (Optional)",
                            placeholder="Enter your name",
                        )
                        girlfriend_name_input = gr.Textbox(
                            label="Girlfriend's Name (Optional)",
                            placeholder="Enter her name",
                        )

                    gr.Markdown("### Choose personality traits (Optional):")

                    traits_input = gr.CheckboxGroup(
                        choices=PERSONALITY_CHOICES,
                        label="Select one or more traits",
                    )

                    custom_personality_input = gr.Textbox(
                        label="Custom personality (Optional)",
                        placeholder="Describe how you want her to behave, tone, style, etc.",
                        lines=3,
                    )

                    initialize_button = gr.Button("Initialize Chat", variant="secondary")
                    message_box = gr.Textbox(
                        label="Status",
                        interactive=False,
                    )

                    initialize_button.click(
                        fn=ask_names_and_traits,
                        inputs=[
                            user_name_input,
                            girlfriend_name_input,
                            traits_input,
                            custom_personality_input,
                            chat_selector,
                        ],
                        outputs=message_box,
                    )

                # ---------- Second Tab: Chat Interface ----------
                
                with gr.TabItem("Chat"):
                    gr.Markdown("### Chat with Your AI Girlfriend")

                    with gr.Column(elem_id="chatbot-block"):
                        chatbot = gr.Chatbot(
                            label="Conversation",
                        )
                    MODELS_DIR = Path("./models")
                    model_files = [f.name for f in MODELS_DIR.glob("*.gguf")]

                    model_selector = gr.Dropdown(
                        choices=model_files,
                        label="Choose Model",
                        value=model_files[0] if model_files else None
                    )
                    chat_input = gr.Textbox(
                        label="Your Message",
                        placeholder="Say something...",
                        lines=2,
                    )

                    send_btn = gr.Button("Send", variant="primary")
                    reset_btn = gr.Button("Reset Visible Chat", variant="secondary")

                    chat_input.submit(
                        fn=chat_page,
                        inputs=[chat_input, chatbot, chat_selector, model_selector],
                        outputs=[chat_input, chatbot],
                    )

                    send_btn.click(
                        fn=chat_page,
                        inputs=[chat_input, chatbot, chat_selector, model_selector],
                        outputs=[chat_input, chatbot],
                    )

                    reset_btn.click(
                        fn=reset_chat,
                        inputs=[chat_selector],
                        outputs=[chatbot, message_box],
                    )

        app.launch(inbrowser=True)


if __name__ == "__main__":
    launch_gradio_app()
