# gradio_app.py
#
# Gradio UI for AI Companion. Launch with: python gradio_app.py
# Shared logic lives in app_utils.py (also used by streamlit_demo/streamlit_app.py).

import gradio as gr

from app_utils import (
    BACKEND_CHOICES,
    PERSONALITY_CHOICES,
    available_model_files,
    create_chat_id,
    load_all_history,
    load_chat_ids,
    load_recent_history,
    reset_chat,
    run_chat,
    save_details,
)
from model import ClaudeLLM


# --------------------------------------------------------------------------- #
# Chat-list helpers
# --------------------------------------------------------------------------- #

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
    chat_ids, selected, message = create_chat_id(new_name, current_value)
    return gr.update(choices=chat_ids, value=selected), message


# --------------------------------------------------------------------------- #
# Chat callback
# --------------------------------------------------------------------------- #

def chat_page(
    user_input: str,
    history: list[dict],
    chat_id: str,
    model_name: str,
    gpu_layers: float,
    backend: str,
    claude_model: str,
) -> tuple[str, list[dict], str]:
    history, status, handled = run_chat(
        user_input, history, chat_id, model_name, gpu_layers, backend, claude_model
    )
    # Keep the user's text on error (key missing, no model, empty input).
    return ("" if handled else user_input), history, status


def update_settings(backend: str) -> tuple:
    """
    Enable/disable sidebar controls depending on the selected backend.
    Local (GGUF) needs the model + GPU layers; Claude (OpenRouter) needs
    the model slug.
    """
    is_claude = backend == "Claude (OpenRouter)"
    return (
        gr.update(interactive=is_claude),       # claude_model_input
        gr.update(interactive=not is_claude),   # model_selector
        gr.update(interactive=not is_claude),   # gpu_layers_input
    )


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
#chat-sidebar { background:#ffffff; border:1px solid #f9a8d4; border-radius:.75rem; padding:.75rem; }
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
                        companion_name_input = gr.Textbox(
                            label="Companion's Name (Optional)", placeholder="Enter their name"
                        )
                    with gr.Row():
                        user_gender_input = gr.Textbox(
                            label="Your Gender (Optional)", placeholder="Enter your gender"
                        )
                        companion_gender_input = gr.Textbox(
                            label="Companion's Gender (Optional)", placeholder="Enter their gender (Female by default)"
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
                            user_name_input, companion_name_input,
                            user_gender_input, companion_gender_input,
                            traits_input, custom_personality_input, chat_selector,
                        ],
                        outputs=message_box,
                    )

                # ---- Tab 2: Chat -------------------------------------------
                with gr.TabItem("Chat"):
                    with gr.Row():
                        # Left sidebar: config + actions
                        with gr.Column(scale=0, min_width=300, elem_id="chat-sidebar"):
                            model_files = available_model_files()
                            backend_selector = gr.Dropdown(
                                choices=BACKEND_CHOICES, label="Backend",
                                value=BACKEND_CHOICES[0],
                                info="Local (GGUF) runs on your computer for "
                                     "free. Claude (OpenRouter) uses "
                                     "Anthropic's Claude SDK routed through "
                                     "OpenRouter — set OPENROUTER_API_KEY in "
                                     ".env first.",
                            )
                            model_selector = gr.Dropdown(
                                choices=model_files, label="Choose Model",
                                value=model_files[0] if model_files else None,
                            )
                            claude_model_input = gr.Textbox(
                                label="OpenRouter Model",
                                value=ClaudeLLM.DEFAULT_MODEL,
                                interactive=False,
                                info="Only used with the Claude backend. "
                                     "Default is a free Gemma model. Try "
                                     "'openrouter/free' to auto-pick, or any "
                                     "slug, e.g. "
                                     "'anthropic/claude-sonnet-5'.",
                            )
                            gpu_layers_input = gr.Number(
                                label="GPU Layers",
                                value=0,
                                precision=0,
                                info="Speed setting. 0 = run on your processor "
                                     "(CPU), which works on any computer. "
                                     "Increase the number to use your graphics "
                                     "card (GPU) for faster replies. If replies "
                                     "crash or freeze, set it back to 0. A good "
                                     "starting point for most laptops is 15.",
                            )
                            load_all_btn = gr.Button(
                                "📜 Load All History", variant="secondary", size="sm"
                            )
                            reset_btn = gr.Button(
                                "Delete Chat Data", variant="secondary"
                            )

                        # Right: the conversation
                        with gr.Column(elem_id="chat-main"):
                            gr.Markdown("### Chat with Your AI Companion")

                            # Status bar: shows how many exchanges are loaded vs total
                            history_status = gr.Textbox(
                                value="", interactive=False, show_label=False,
                                container=False, elem_id="history-status"
                            )

                            with gr.Column(elem_id="chatbot-block"):
                                chatbot = gr.Chatbot(
                                    label="Conversation", height=520
                                )

                            chat_input = gr.Textbox(
                                label="Your Message", placeholder="Say something...", lines=2
                            )
                            with gr.Row():
                                send_btn = gr.Button("Send", variant="primary")

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

                    # Enable/disable controls based on the selected backend
                    backend_selector.change(
                        fn=update_settings,
                        inputs=[backend_selector],
                        outputs=[claude_model_input, model_selector, gpu_layers_input],
                    )

                    shared_inputs  = [chat_input, chatbot, chat_selector, model_selector, gpu_layers_input, backend_selector, claude_model_input]
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
