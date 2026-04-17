# test.py
import os
from pathlib import Path
from model.memory import Memory

def run_verification(chat_id: str):
    print(f"--- Verifying Chat: {chat_id} ---")
    
    # 1. Define the path to your archive (the JSON)
    target_path = Path(f"chats/{chat_id}/memory.json")

    if not target_path.exists():
        print(f"❌ Error: File {target_path} does not exist. Did you run a chat first?")
        return

    try:
        # 2. Load the Memory class (this loads JSON, FAISS, and Mapping)
        memory = Memory(str(target_path))

        # --- CHECK 1: The Archive (JSON) ---
        print(f"✅ Archive loaded.")
        print(f"📊 Total messages in JSON: {len(memory.messages)}")
        print(f"🔄 Total pairs in JSON: {memory.pair_count()}")

        # --- CHECK 2: The Librarian (FAISS) ---
        if memory.index is not None:
            print(f"✅ FAISS Index loaded.")
            print(f"🔍 Vectors stored in FAISS: {memory.index.ntotal}")
        else:
            print("❌ Error: FAISS index failed to load.")

        # --- CHECK 3: The Semantic Search (The Real Test) ---
        # query = "What is my favorite color?"
        query = "What popcorn flavour do I like?"
        print(f"\n🤖 Testing Librarian with query: '{query}'")
        
        results = memory.search(query, k=3)

        if results:
            print("✨ SUCCESS! The Librarian found these facts:")
            for i, res in enumerate(results):
                print(f"  {i+1}. {res}")
        else:
            print("⚠️ Search returned nothing. Either the fact wasn't saved, or the query was too different.")

    except Exception as e:
        print(f"❌ An error occurred during testing: {e}")

if __name__ == "__main__":
    # Change 'default' to whatever chat ID you were testing with
    TEST_CHAT_ID = "test" 
    run_verification(TEST_CHAT_ID)
