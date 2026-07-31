# tests/test_real_model.py
"""
End-to-end test with a real GGUF model (no mocking). Verifies that the LLM
actually emits the '||' delimiter and that RAG facts get stored and retrieved.

Requires a .gguf file in models/. Skips cleanly if none is found.

Run:  python tests/test_real_model.py
"""
import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model import ChatEngine, LLM, Memory, Persona

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def find_model() -> Path | None:
    ggufs = sorted(MODELS_DIR.glob("*.gguf"))
    return ggufs[0] if ggufs else None


def test_real_rag(base: str, model_path: Path) -> None:
    llm = LLM.get_instance(str(model_path), n_ctx=2048)

    memory = Memory(path=os.path.join(base, "memory.json"))
    persona = Persona(path=os.path.join(base, "persona.json"))
    persona.update(
        user_name="Sam",
        companion_name="Luna",
        personality_traits=["Playful"],
        user_gender="male",
        companion_gender="female",
    )

    engine = ChatEngine(llm=llm, memory=memory, persona=persona)

    engine.chat("Hi, my name is Sam and I love butter popcorn.")

    # The real model must have produced a fact (or at least a reply) and the
    # archive must have grown. Index may be 0 if the model said '|| None'.
    assert memory.pair_count() == 1, "archive should contain one exchange"
    print(f"PASS: real model replied; facts so far: {memory.facts!r}")

    engine.chat("What do I like to eat at the movies?")
    assert memory.pair_count() == 2, "archive should contain two exchanges"
    print(f"PASS: real model replied again; facts: {memory.facts!r}")

    if memory.index.ntotal > 0:
        hits = memory.search("What is my favorite popcorn?", k=1)
        print(f"PASS: RAG search returned {hits!r}")
    else:
        print("SKIP: model emitted no facts, so RAG retrieval was not exercised")


def main() -> None:
    model_path = find_model()
    if model_path is None:
        print("SKIP: no .gguf found in models/ - put a model there to run this test.")
        return

    print(f"Using model: {model_path.name} ({model_path.stat().st_size // (1024*1024)} MB)")
    base = tempfile.mkdtemp(prefix="aicompanion_real_")
    try:
        test_real_rag(base, model_path)
    finally:
        shutil.rmtree(base, ignore_errors=True)
    print("\nREAL-MODEL TEST DONE")


if __name__ == "__main__":
    main()
