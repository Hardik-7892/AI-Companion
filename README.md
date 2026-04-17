
# AI Companion – Modular GGUF-Powered RAG Chat App

A locally-running conversational AI companion built with **Gradio** and **GGUF models** using `llama-cpp-python`, featuring a **RAG (Retrieval-Augmented Generation)** pipeline for long-term semantic memory.

This version introduces a **modular architecture** with separate components for:

* LLM handling
* Memory management (Vector DB & Archive)
* Persona customization
* Chat orchestration

---

## ✨ Features

* 🧠 **Local LLM (GGUF)** via `llama-cpp-python`
* 🔍 **RAG-Enabled Memory**: Uses **FAISS** vector database and **Sentence Transformers** to retrieve semantically relevant past facts from long-term storage.
* 💬 **Persistent chat memory**: Dual-layer storage (JSON Archive for full history + FAISS Index for fast semantic retrieval).
* ❤️ **Customizable AI persona**
* 🌈 **Gender Inclusary** (Fully customizable gender for both User and Companion)
* 🔁 **Multiple chat sessions**
* 🎨 **UI themes** (Pink, Blue, Dark)
* ⚡ **Efficient model caching** (load once, reuse)

---

## 📁 Project Structure
```
.
├── app.py                      # Main Gradio app
│
├── models/
│   └── model.gguf             # Your GGUF model(s)
│
├── model/
│   ├── __init__.py
│   ├── chat_engine.py         # Orchestrates chat flow & RAG augmentation
│   ├── llm.py                 # LLaMA wrapper (with caching)
│   ├── memory.py              # Persistent conversation memory (Archive + FAISS Index)
│   └── persona.py             # Persona + system prompt builder
│
├── chats/                     # Auto-created per chat session
│   └── <chat_id>/
│       ├── memory.json        # The 'Archive' (Full text history)
│       ├── memory.index       # The 'Vector Index' (FAISS)
│       ├── memory_map.json    # Vector-to-Text mapping
│       └── persona.json
│
├── chats.json                 # Stores list of chat IDs
├── requirements.txt
└── README.md
```
---

## ⚙️ Setup

### 1. Clone the repo
```bash
git clone https://github.com/Hardik-7892/AI-Companion.git
cd AI-Companion
```

---

### 2. Create virtual environment
```bash
python -m venv venv
```
Activate it:
* **Windows**: `venv\Scripts\activate`
* **macOS/Linux**: `source venv/bin/activate`

---

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

### 4. Add GGUF model(s)

Place your `.gguf` file(s) inside:

```
models/
```

Example:

```
models/
├── llama-3-8b-instruct.Q4_K_M.gguf
```

> ⚠️ Models are NOT included due to size constraints.

---

### 5. Run the app

```bash
python app.py
```

The app will open in your browser:

```
http://127.0.0.1:7860
```

---

## 🚀 How It Works (RAG Pipeline)

```
User Input
   ↓
[Retriever] → Search FAISS Index for semantically similar "knowledge nuggets"
   ↓
[Augmenter] → Inject retrieved facts into the System Prompt
   ↓
[ChatEngine] → [System Prompt + Retrieved Context + Recent History + User Message]
   ↓
[LLM] (llama.cpp) → Generates Response
   ↓
[Parser] → Extracts new "Facts" from response using '||' delimiter
   ↓
[Archiver] → Saves full text to JSON and updates FAISS Vector Index
```

---

## 🛠️ Core Components

#### 🔹 `LLM` (model/llm.py)
* Wraps `llama_cpp.Llama`
* Uses **class-level caching** → model loads only once

#### 🔹 `Memory` (model/memory.py) - **The RAG Engine**
* **Archive**: JSON file containing the complete conversation log.
* **Librarian (Retriever)**: Uses `SentenceTransformer` to vectorize queries and facts.
* **Index (Vector DB)**: `FAISS` index for high-speed semantic similarity search.

#### 🔹 `Persona` (model/persona.py)
* Builds dynamic **system prompt**
* Supports: Names, Genders, Personality Traits, and Custom Descriptions.

#### 🔹 `ChatEngine` (model/chat_engine.py) - **The Orchestrator**
* Performs the **RAG augmentation** step by calling `Memory.search()` and appending results to the context window.
* Handles the logic of parsing "Facts" from LLM output to update the Vector Index.

---

## 💡 Usage
1. Select or create a chat
2. (Optional) Configure:
   * Your name and gender
   * Companion's name and gender
   * Personality traits
3. Choose a model
4. Start chatting

---

## 🧠 Memory Behavior
* **Short-term Context**: The last **N pairs** are always loaded into the LLM context window for immediate flow.
* **Long-term Retrieval (RAG)**: When you mention something from much earlier in the chat, the system retrieves the relevant "knowledge nugget" from the FAISS index and injects it into the current prompt.

---

## 🔮 Roadmap
* 🔍 Semantic memory expansion (larger vector chunks)
* 🎤 Speech-to-text (Whisper)
* 🔊 Text-to-speech
* 🧠 Long-term personality evolution
* 🌐 Remote model support