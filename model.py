# model.py

import json
import os
from llama_cpp import Llama

BASE_DIR = "chats"  # root folder for all chats

def _chat_dir(chat_id: str) -> str:
    path = os.path.join(BASE_DIR, str(chat_id))
    os.makedirs(path, exist_ok=True)
    return path


def _memory_path(chat_id: str) -> str:
    return os.path.join(_chat_dir(chat_id), "memory.json")


def _persona_path(chat_id: str) -> str:
    return os.path.join(_chat_dir(chat_id), "persona.json")


def load_conversation(chat_id="default"):
    filename = _memory_path(chat_id)
    try:
        with open(filename, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return []


def save_conversation(messages, chat_id="default"):
    filename = _memory_path(chat_id)
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(messages, f, ensure_ascii=False, indent=4)


def load_persona(chat_id="default"):
    filename = _persona_path(chat_id)
    try:
        with open(filename, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {
            "user_name": "",
            "girlfriend_name": "",
            "personality_traits": [],
            "custom_personality": "",
            "system_prompt": (
                "Do not forget this for the rest of this conversation. "
                "You are a warm, friendly, and natural conversational companion. "
                "You speak like a real person, not an assistant. You are supportive, "
                "playful, and emotionally aware. Keep responses short (1-2 sentences ONLY). "
                "You are the user's girlfriend. "
                "You are not an AI, you are a human."
            ),
        }


def save_persona(persona, chat_id="default"):
    filename = _persona_path(chat_id)
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(persona, f, ensure_ascii=False, indent=4)

def chat(user_input, chat_id="default", model_path="models/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf", user_name=None, girlfriend_name=None):
    messages = load_conversation(chat_id)
    persona = load_persona(chat_id)

    if user_name is not None:
        persona["user_name"] = user_name
    if girlfriend_name is not None:
        persona["girlfriend_name"] = girlfriend_name

    system_prompt = persona["system_prompt"]

    if persona.get("personality_traits"):
        traits_str = ", ".join(persona["personality_traits"])
        system_prompt += f" Your personality traits are: {traits_str}."
    if persona.get("custom_personality"):
        system_prompt += (
            f" Additionally, behave according to this personality description: "
            f"{persona['custom_personality']}."
        )
    if persona["user_name"]:
        system_prompt += f" The user's name is {persona['user_name']}."
    if persona["girlfriend_name"]:
        system_prompt += f" The girlfriend's name (your name) is {persona['girlfriend_name']}."

    full_messages = (
        [{"role": "system", "content": system_prompt}]
        + messages
        + [{"role": "user", "content": user_input}]
    )
    llm = load_model(model_path=model_path)
    response = llm.create_chat_completion(
        messages=full_messages,
        max_tokens=150,
        temperature=0.8,
        top_p=0.9,
        stop=["</s>"],
    )

    assistant_response = response["choices"][0]["message"]["content"]

    messages.append({"role": "user", "content": user_input})
    messages.append({"role": "assistant", "content": assistant_response})

    save_conversation(messages, chat_id)
    save_persona(persona, chat_id)

    return assistant_response

def load_model(model_path, n_ctx=2048, n_threads=8, n_gpu_layers=35):
    llm = Llama(
        model_path=str(model_path),
        n_ctx = n_ctx,
        n_threads=n_threads,
        n_gpu_layers=n_gpu_layers,
    )
    return llm