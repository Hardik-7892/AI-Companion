# tests/test_mock_llm.py
"""
End-to-end component tests using a mocked LLM (no GGUF model required).

Run:  python tests/test_mock_llm.py
"""
import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model import ChatEngine, Memory, Persona


class MockLLM:
    """Stands in for model.llm.LLM. chat() returns whatever reply is set."""

    def __init__(self, reply: str = ""):
        self.reply = reply
        self.calls = []

    def chat(self, messages: list[dict]) -> str:
        self.calls.append(messages)
        return self.reply


def test_persona(base: str) -> None:
    p = Persona(path=os.path.join(base, "persona.json"))
    p.update(
        user_name="Sam",
        companion_name="Luna",
        user_gender="male",
        companion_gender="female",
        personality_traits=["Playful"],
        custom_personality="",
    )

    reloaded = Persona(path=os.path.join(base, "persona.json"))
    prompt = reloaded.build_system_prompt()
    assert "The user's name is Sam." in prompt
    assert "The companion's name (your name) is Luna." in prompt
    assert "Your personality traits are: Playful." in prompt
    assert "The user's gender is male." in prompt
    assert "The companion's gender (your gender) is female." in prompt
    assert "girlfriend" not in prompt.lower()
    print("PASS: persona save/load/build")


def test_memory_rag(base: str) -> None:
    path = os.path.join(base, "chat_m", "memory.json")
    m = Memory(path=path)
    m.clear()

    m.add_to_archive("user", "My favorite popcorn flavour is butter.")
    m.add_to_archive("assistant", "Nice choice!")
    m.add_to_index("User likes butter popcorn")
    m.add_to_archive("user", "I have a dog named Rex.")
    m.add_to_archive("assistant", "Rex is cute!")
    m.add_to_index("User has a dog named Rex")

    assert m.pair_count() == 2
    assert m.index.ntotal == 2
    assert m.facts == ["User likes butter popcorn", "User has a dog named Rex"]

    reloaded = Memory(path=path)
    assert reloaded.pair_count() == 2
    assert reloaded.index.ntotal == 2
    assert reloaded.facts == m.facts

    # Regression: search must return the *fact* text, not the assistant reply.
    results = reloaded.search("What popcorn flavour do I like?", k=2)
    assert results, "expected at least one retrieved fact"
    assert "butter" in results[0].lower(), f"got {results!r}"
    assert results[0] == "User likes butter popcorn"

    print(f"PASS: memory archive + FAISS + search (got {results!r})")


def test_memory_old_chat_empty_facts(base: str) -> None:
    """A chat with vectors but no facts store must degrade gracefully."""
    path = os.path.join(base, "chat_old", "memory.json")
    m = Memory(path=path)
    m.clear()
    m.add_to_archive("user", "hi")
    m.add_to_archive("assistant", "hello")
    m.index.add(m.embedder.encode(["orphan vector"]).astype("float32"))
    m.facts = []
    m.save()

    reloaded = Memory(path=path)
    assert reloaded.search("anything", k=3) == []
    print("PASS: old chat with empty facts degrades gracefully")


def test_chat_engine_fact_and_no_fact(base: str) -> None:
    path = os.path.join(base, "chat_eng", "memory.json")
    memory = Memory(path=path)
    memory.clear()
    persona = Persona(path=os.path.join(base, "chat_eng", "persona.json"))

    # Mock that emits a fact after the '||' delimiter.
    llm = MockLLM(reply="I love butter too! || User likes butter popcorn.")
    engine = ChatEngine(llm=llm, memory=memory, persona=persona)
    reply = engine.chat("What popcorn flavour do I like?")
    assert reply == "I love butter too!", reply
    assert memory.pair_count() == 1
    assert memory.index.ntotal == 1
    assert memory.facts == ["User likes butter popcorn."]
    assert "User likes butter popcorn" in memory.search("What popcorn do I like?", k=1)[0]

    # Mock that emits no fact -> archive grows but index does not.
    llm2 = MockLLM(reply="Just chatting casually.")
    engine2 = ChatEngine(llm=llm2, memory=memory, persona=persona)
    reply2 = engine2.chat("Hi")
    assert reply2 == "Just chatting casually."
    assert memory.pair_count() == 2
    assert memory.index.ntotal == 1, "no fact should mean no index write"

    # The prompt sent to the LLM must include retrieved memories + recent pairs.
    last_call = llm2.calls[-1]
    roles = [msg["role"] for msg in last_call]
    assert roles == ["system", "user", "assistant", "user"], roles
    assert "Relevant memories" in last_call[0]["content"]

    print("PASS: chat engine '||' parsing + double-write")


def main() -> None:
    base = tempfile.mkdtemp(prefix="aicompanion_test_")
    print("Temp dir:", base)
    try:
        test_persona(base)
        test_memory_rag(base)
        test_memory_old_chat_empty_facts(base)
        test_chat_engine_fact_and_no_fact(base)
    finally:
        shutil.rmtree(base, ignore_errors=True)
    print("\nALL TESTS PASSED")


if __name__ == "__main__":
    main()
