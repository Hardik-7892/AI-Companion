# model/persona.py

import json
import os

DEFAULT_SYSTEM_PROMPT = (
    "You are a warm, friendly conversational companion. "
    "Keep replies to 1-2 short sentences.\n"
    "Always end your reply with ' || ' followed by one short fact about the "
    "user, or the word 'None' if nothing important was said.\n"
    "Examples:\n"
    "- 'I love that color! || User likes brown.'\n"
    "- 'How are you today? || None'\n"
    "Only save important facts about the user or your relationship, "
    "never a summary of the chat."
)

_DEFAULT_DATA = {
    "user_name": "",
    "companion_name": "",
    "user_gender": "",
    "companion_gender": "",
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
            prompt += (
                f" The user's name is {user_name}. Always address the user as {user_name}."
            )

        companion_name: str = self.data.get("companion_name", "")
        if companion_name:
            prompt += (
                f" The companion's name (your name) is {companion_name}. "
                f"Always refer to yourself as {companion_name}."
            )

        user_gender: str = self.data.get("user_gender", "")
        if user_gender:
            prompt += f" The user's gender is {user_gender}."

        companion_gender: str = self.data.get("companion_gender", "")
        if companion_gender:
            prompt += f" The companion's gender (your gender) is {companion_gender}."

        return prompt