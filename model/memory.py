# model/memory.py

import json
import os


class Memory:
    """
    Conversation memory backed by a JSON file.

    The file always contains the **full** log.  ``get_recent(n)`` returns only
    the last *n* user-assistant pairs so the LLM prompt stays compact.
    ``get_all()`` returns the complete history for UI display.

    Step 2 will add ``search()`` (FAISS) so that older turns are retrievable
    by semantic similarity rather than recency alone.
    """

    def __init__(self, path: str) -> None:
        self.path = path
        self.messages: list[dict] = self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> list[dict]:
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                pass
        return []

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.messages, f, ensure_ascii=False, indent=4)

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def add(self, role: str, content: str) -> None:
        """Append one message and persist immediately."""
        self.messages.append({"role": role, "content": content})
        self.save()

    def clear(self) -> None:
        self.messages = []
        self.save()

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get_all(self) -> list[dict]:
        """Full conversation log (for UI 'Load All')."""
        return list(self.messages)

    def get_recent(self, n_pairs: int = 10) -> list[dict]:
        """
        Last *n_pairs* user-assistant exchanges (2 x n_pairs messages).
        Used by ChatEngine to keep the LLM context window compact.
        """
        return self.messages[-(n_pairs * 2):]

    def pair_count(self) -> int:
        """Total number of stored user-assistant pairs."""
        return len(self.messages) // 2

    # ------------------------------------------------------------------
    # Placeholder for Step 2
    # ------------------------------------------------------------------

    def search(self, query: str, k: int = 4) -> list[str]:
        """Semantic search -- no-op until FAISS is wired in (Step 2)."""
        return []