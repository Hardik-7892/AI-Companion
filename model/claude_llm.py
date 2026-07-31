# model/claude_llm.py
#
# Claude-compatible LLM backend, routed through OpenRouter.
#
# The Anthropic Python SDK talks to any Anthropic-compatible endpoint via its
# `base_url`. We use OpenRouter (https://openrouter.ai/api) because it needs no
# Anthropic dev account and works pay-as-you-go — including free models.
#
# IMPORTANT URL detail: the SDK appends "/v1/messages" to `base_url` itself,
# so for OpenRouter you must pass https://openrouter.ai/api (WITHOUT "/v1").
# The "/v1" suffix is only used by OpenAI-style SDKs against the OpenAI-compatible
# endpoint (https://openrouter.ai/api/v1); using it here produces a 404/405.

import os

from dotenv import load_dotenv

load_dotenv()  # reads OPENROUTER_API_KEY / ANTHROPIC_API_KEY from .env if present


class ClaudeLLM:
    """
    Chat LLM backed by the Anthropic SDK, pointed at OpenRouter by default.

    Implements the same `chat()` interface as model.llm.LLM, so it can be
    swapped into ChatEngine / Memory / Persona without any other changes —
    including the '|| [fact]' RAG parsing.
    """

    # Default model. Tested and reliable free instruct model on OpenRouter
    # (follows the persona prompt + '|| [fact]' RAG format well, $0 cost).
    # Set any slug via the app's model field, e.g. "openrouter/free" to let
    # OpenRouter auto-pick, or "anthropic/claude-sonnet-5" for paid Claude.
    DEFAULT_MODEL = "google/gemma-4-26b-a4b-it:free"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        base_url: str = "https://openrouter.ai/api",
    ) -> None:
        self.model = model

        # --- Active config: OpenRouter (no Anthropic account / credits needed) ---
        if api_key is None:
            api_key = os.environ.get("OPENROUTER_API_KEY", "")

        # --- Original Anthropic-first config (kept for reference) ---
        # Using Anthropic's own API instead of OpenRouter:
        #   base_url = None                  # defaults to https://api.anthropic.com
        #   api_key  = os.environ.get("ANTHROPIC_API_KEY", "")
        # requires an Anthropic dev account with billing, and model IDs without
        # a provider prefix (e.g. "claude-sonnet-5").
        # base_url = "https://openrouter.ai/api"

        self._client = None  # created lazily on first chat() call
        self._base_url = base_url
        self._api_key = api_key

    def chat(
        self,
        messages: list[dict],
        max_tokens: int = 150,
        temperature: float = 0.8,
        top_p: float = 0.9,
    ) -> str:
        """
        Run a chat completion and return the assistant text.

        `messages` uses the app's format: an optional leading
        {"role": "system", ...} plus user/assistant turns. The Anthropic API
        has no "system" message role, so the system prompt is extracted and
        passed via the top-level `system=` argument.
        """
        if self._client is None:
            from anthropic import Anthropic  # lazy import: app works without it

            self._client = Anthropic(
                api_key=self._api_key,
                base_url=self._base_url,
            )

        # Split the app's system message out of the turn list.
        system_parts = [
            m["content"] for m in messages if m["role"] == "system"
        ]
        turns = [m for m in messages if m["role"] != "system"]

        kwargs: dict = {}
        if system_parts:
            kwargs["system"] = "\n\n".join(system_parts)

        response = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            messages=turns,  # type: ignore[arg-type]
            **kwargs,
        )

        # The response is a list of content blocks; keep the text ones.
        return "".join(b.text for b in response.content if b.type == "text")
