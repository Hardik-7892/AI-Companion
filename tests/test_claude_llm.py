# tests/test_claude_llm.py
"""
Tests for the Claude (OpenRouter) backend using a fake Anthropic client —
no API key or network needed.

Run:  python tests/test_claude_llm.py
"""
import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model import ChatEngine, ClaudeLLM, Memory, Persona


class FakeTextBlock:
    def __init__(self, text: str):
        self.type = "text"
        self.text = text


class FakeContentBlock:
    def __init__(self):
        self.type = "tool_use"  # should be skipped when joining text


class FakeMessage:
    def __init__(self, text: str):
        self.content = [FakeContentBlock(), FakeTextBlock(text)]


class FakeAnthropic:
    """Stand-in for anthropic.Anthropic. Records the last request."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.last = None

    class messages:
        @staticmethod
        def create(**kwargs):
            msg = FakeMessage(kwargs.pop("expected_reply", "hi || None"))
            FakeAnthropic.last = kwargs
            return msg


def test_claude_llm_system_split_and_blocks():
    import anthropic

    fake = FakeAnthropic()
    original = anthropic.Anthropic
    anthropic.Anthropic = lambda **kw: fake
    try:
        llm = ClaudeLLM(model="openrouter/free", api_key="sk-or-v1-test")
        assert llm._base_url == "https://openrouter.ai/api"

        reply = llm.chat(
            [
                {"role": "system", "content": "You are Luna."},
                {"role": "user", "content": "Hi"},
            ]
        )
        assert reply == "hi || None"

        sent = FakeAnthropic.last
        assert sent["model"] == "openrouter/free"
        assert sent["system"] == "You are Luna."
        assert sent["messages"] == [{"role": "user", "content": "Hi"}]
        assert sent["max_tokens"] == 150
        print("PASS: claude_llm system split + content-block join")
    finally:
        anthropic.Anthropic = original


def test_claude_llm_chat_engine_fact():
    import anthropic

    fake = FakeAnthropic()
    original = anthropic.Anthropic
    anthropic.Anthropic = lambda **kw: fake
    try:
        llm = ClaudeLLM(model="openrouter/free", api_key="sk-or-v1-test")

        def patched_create(**kwargs):
            kwargs["expected_reply"] = "I love butter too! || User likes butter popcorn."
            return FakeMessage(kwargs["expected_reply"])

        FakeAnthropic.messages.create = staticmethod(patched_create)

        base = tempfile.mkdtemp(prefix="aicompanion_claude_")
        try:
            memory = Memory(path=os.path.join(base, "memory.json"))
            memory.clear()
            persona = Persona(path=os.path.join(base, "persona.json"))
            engine = ChatEngine(llm=llm, memory=memory, persona=persona)
            reply = engine.chat("What popcorn flavour do I like?")
            assert reply == "I love butter too!", reply
            assert memory.pair_count() == 1
            assert memory.index.ntotal == 1
            assert memory.facts == ["User likes butter popcorn."]

            sent = FakeAnthropic.last
            roles = [m["role"] for m in sent["messages"]]
            assert roles == ["user"], roles
            print("PASS: claude_llm -> ChatEngine '||' parsing + double-write")
        finally:
            shutil.rmtree(base, ignore_errors=True)
    finally:
        anthropic.Anthropic = original


def main() -> None:
    test_claude_llm_system_split_and_blocks()
    test_claude_llm_chat_engine_fact()
    print("\nCLAUDE-LLM TESTS DONE")


if __name__ == "__main__":
    main()
