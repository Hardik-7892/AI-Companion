# model/chat_engine.py

from .llm import LLM
from .memory import Memory
from .persona import Persona

# How many user-assistant exchanges to include in every LLM prompt.
# Increase if you want the model to remember further back at the cost of
# a larger context window.
N_RECENT_PAIRS: int = 10


class ChatEngine:
    """
    Orchestrates a single chat session.

    * Builds the message list  →  [system] + recent history + [user turn]
    * Delegates inference to LLM
    * Persists every new turn via Memory

    In Step 2, build_messages() will also inject semantically-retrieved
    long-term memories returned by Memory.search().
    """

    def __init__(self, llm: LLM, memory: Memory, persona: Persona) -> None:
        self.llm = llm
        self.memory = memory
        self.persona = persona

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def build_messages(self, user_input: str) -> list[dict]:
        system_prompt = self.persona.build_system_prompt()

        # Step 2 hook: inject semantically-similar older memories
        retrieved = self.memory.search(user_input)
        if retrieved:
            formatted = "\n".join(f"- {m}" for m in retrieved)
            system_prompt += f"\n\nRelevant memories:\n{formatted}"

        return (
            [{"role": "system", "content": system_prompt}]
            + self.memory.get_recent(N_RECENT_PAIRS)   # <-- only last N pairs
            + [{"role": "user", "content": user_input}]
        )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def chat(self, user_input: str) -> str:
        # 1. Prepare messages with strict instructions
        # Your persona/system prompt MUST say: "Always end your response with '||[FACT]' if a new fact is mentioned. If not, end with '||None'."
        messages = self.build_messages(user_input)
        raw_output = self.llm.chat(messages)

        # 2. The Parsing Logic (Safety first!)
        reply = raw_output
        fact = "None"

        if "||" in raw_output:
            parts = raw_output.split("||", 1)
            reply = parts[0].strip()
            fact = parts[1].strip().replace("Fact: ", "")
        
        # 3. The Double-Write (Archive + Index)
        self.memory.add_to_archive("user", user_input)
        self.memory.add_to_archive("assistant", reply)

        if fact.lower() != "none":
            self.memory.add_to_index(fact)

        return reply