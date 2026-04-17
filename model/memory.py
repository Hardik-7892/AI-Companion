# model/memory.py

import json
import os
from pathlib import Path
import faiss
from sentence_transformers import SentenceTransformer

class Memory:
    """
    Conversation memory with a 'Librarian' (FAISS) component.
    
    The JSON file acts as the Archive (full text).
    The FAISS index acts as the Index (vectorized searchable metadata).
    """

    def __init__(self, path: str, embedding_model_name = "all-MiniLM-L6-v2") -> None:
        self.path = path
        # The archive path (JSON)
        self.archive_path = path 
        # The index path (FAISS)
        self.index_path = path.replace(".json", ".index")
        # A mapping file to link FAISS vector IDs back to JSON message indices
        self.mapping_path = path.replace(".json", "_map.json")

        self.messages: list[dict] = self._load_archive()

        # Initialize Librarian (Embedding Model)
        embedding_cache_dir = Path("./models/embeddings")
        embedding_cache_dir.mkdir(parents=True, exist_ok=True)
        self.embedder = SentenceTransformer(
            embedding_model_name, 
            cache_folder=str(embedding_cache_dir)
        )
        
        # Initialize Index and Mapping
        self.index = None
        self.mapping = self._load_mapping()
        self._load_index()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def _load_archive(self) -> list[dict]:
        if os.path.exists(self.archive_path):
            try:
                with open(self.archive_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                pass
        return []
    
    def _load_mapping(self) -> list[int]:
        """Loads the list of JSON message indices that are currently in the FAISS index."""
        if os.path.exists(self.mapping_path):
            with open(self.mapping_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return []
    
    def _load_index(self) -> None:
        """Loads the FAISS index from disk."""
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
        else:
            # Create a simple FlatL2 index (good for small/medium chat histories)
            dimension = 384 # Dimension for all-MiniLM-L6-v2
            self.index = faiss.IndexFlatL2(dimension)

    def save(self) -> None:
        """Saves both the Archive and the Index mapping."""
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        # 1. Save JSON Archive
        with open(self.archive_path, "w", encoding="utf-8") as f:
            json.dump(self.messages, f, ensure_ascii=False, indent=4)
        # 2. Save Mapping (which message index in JSON corresponds to which vector in FAISS)
        with open(self.mapping_path, "w", encoding="utf-8") as f:
            json.dump(self.mapping, f)
        # 3. Save the FAISS Index itself
        if self.index is not None:
            faiss.write_index(self.index, self.index_path)

    def clear(self) -> None:
        """Wipes both the Archive and the Index."""
        self.messages = []
        self.mapping = []
        self.index = faiss.IndexFlatL2(384)
        self.save()

    # ------------------------------------------------------------------
    # Read (Retrieval)
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

    def search(self, query: str, k: int = 3) -> list[str]:
        """
        The Librarian's core function.
        Finds the most semantically similar messages from the past.
        """
        if not self.index or self.index.ntotal == 0:
            return []

        # 1. Embed the user query
        # query_vector = self.embedder.encode([query]).astype('float3rypt')
        query_vector = self.embedder.encode([query]).astype('float32')
        
        # 2. Search FAISS for top K neighbors
        distances, indices = self.index.search(query_vector, k)
        
        # 3. Use the mapping to find which JSON messages those vectors refer to
        results = []
        for idx in indices[0]:
            if idx != -1 and idx < len(self.mapping):
                json_msg_idx = self.mapping[idx]
                message_text = self.messages[json_msg_idx]["content"]
                results.append(message_text)
        
        return results
    
    def add_to_archive(self, role: str, content: str) -> None:
        """Always save every message to the JSON file."""
        self.messages.append({"role": role, "content": content})
        self.save()

    def add_to_index(self, fact: str) -> None:
        """Only save 'knowledge nuggets' to FAISS."""
        print("----------------------------------------------------------------------")
        print(fact)
        if not fact or fact.lower() == "none":
            return
            
        embedding = self.embedder.encode([fact]).astype('float32')
        self.index.add(embedding)
        # We link this vector to the index of the latest message in the archive
        self.mapping.append(len(self.messages) - 1) 
        self.save()