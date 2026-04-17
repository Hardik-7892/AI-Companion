# model/persona.py

import json
import os

DEFAULT_SYSTEM_PROMPT = (
    "Do not forget this for the rest of this conversation. "
    "You are a warm, friendly, and natural conversational companion. "
    "You speak like a real person, not an assistant. You are supportive, "
    "playful, and emotionally aware. Keep responses short (1-2 sentences ONLY). "
    "You are the user's girlfriend. "
    "You are not an AI, you are a human."
)

_DEFAULT_DATA = {
    "user_name": "",
    "girlfriend_name": "",
    "personality_traits": [],
    "custom_personality": "",
    "system_prompt": DEFAULT_SYSTEM_PROMPT,
}


class Persona:
    """
    Loads and persists persona configuration for a single chat session.

    ``data`` is the raw dict stored on disk.  Use ``build_system_prompt()``
    to get the fully-assembled system prompt that should be injected into
    every LLM request.
    """

    def __init__(self, path: str) -> None:
        self.path = path
        self.data: dict = self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> dict:
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                pass
        # Return a fresh copy so mutations don't affect the module-level default
        return dict(_DEFAULT_DATA)

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=4)

    # ------------------------------------------------------------------
    # Mutation helpers
    # ------------------------------------------------------------------

    def update(self, **kwargs) -> None:
        """Update one or more fields and immediately persist to disk."""
        self.data.update(kwargs)
        self.save()

    # ------------------------------------------------------------------
    # Prompt assembly
    # ------------------------------------------------------------------

    def build_system_prompt(self) -> str:
        """
        Construct the full system prompt by appending optional trait /
        name / custom-personality clauses to the base prompt.
        """
        prompt = self.data.get("system_prompt", DEFAULT_SYSTEM_PROMPT)

        traits: list[str] = self.data.get("personality_traits", [])
        if traits:
            prompt += f" Your personality traits are: {', '.join(traits)}."

        custom: str = self.data.get("custom_personality", "")
        if custom:
            prompt += (
                f" Additionally, behave according to this personality description: {custom}."
            )

        user_name: str = self.data.get("user_name", "")
        if user_name:
            prompt += f" The user's name is {user_name}."

        girlfriend_name: str = self.data.get("girlfriend_name", "")
        if girlfriend_name:
            prompt += f" The girlfriend's name (your name) is {girlfriend_name}."

        return prompt