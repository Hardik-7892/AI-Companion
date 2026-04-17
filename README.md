# AI Companion – Modular GGUF-Powered Chat App

A locally-running conversational AI companion built with **Gradio** and **GGUF models** using `llama-cpp-python`.

This version introduces a **modular architecture** with separate components for:

* LLM handling
* Memory management
* Persona customization
* Chat orchestration

---

## ✨ Features

* 🧠 **Local LLM (GGUF)** via `llama-cpp-python`
* 💬 **Persistent chat memory** (JSON-based)
* ❤️ **Customizable AI persona**
* 🌈 **Gender Inclusivity** (Fully customizable gender for both User and Companion)
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
│   ├── chat_engine.py         # Orchestrates chat flow
│   ├── llm.py                 # LLaMA wrapper (with caching)
│   ├── memory.py              # Persistent conversation memory
│   └── persona.py             # Persona + system prompt builder
│
├── chats/                     # Auto-created per chat session
│   └── <chat_id>/
│       ├── memory.json
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

* **Windows**

```bash
venv\Scripts\activate
```

* **macOS/Linux**

```bash
source venv/bin/activate
```

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

## 🚀 How It Works

### 1. Chat Flow

```
User Input
   ↓
ChatEngine
   ↓
[System Prompt (Persona)]
+ [Recent Memory]
+ [User Message]
   ↓
LLM (llama.cpp)
   ↓
Response
   ↓
Saved to Memory
```

---

### 2. Core Components

#### 🔹 `LLM` (model/llm.py)

* Wraps `llama_cpp.Llama`
* Uses **class-level caching** → model loads only once

#### 🔹 `Memory` (model/memory.py)

* Stores chat history in JSON
* Provides:
  * `get_recent(n)` → for context window
  * `get_all()` → full history
* Future: semantic search (FAISS)

#### 🔹 `Persona` (model/persona.py)

* Builds dynamic **system prompt**
* Supports:
  * Names
    * User and Companion genders
  * Traits
  * Custom personality text

#### 🔹 `ChatEngine` (model/chat_engine.py)

* Central orchestrator
* Combines:
  * Persona + Memory + User input
* Handles:
  * Prompt construction
  * LLM call
  * Persistence

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

* Full conversation → saved in `memory.json`
* Only last **N pairs (default: 10)** sent to LLM
* UI can:
  * Load recent history
  * Load full history

---

## ⚠️ Notes & Tips

### Model Issues

* Ensure model path is a **string**, not `Path`
* Already handled in code via:

```python
LLM.get_instance(str(model_path))
```

---

### Performance Tuning

You can tweak in `LLM`:

```python
n_ctx=2048
n_threads=8
n_gpu_layers=35
```

* Increase `n_gpu_layers` → better GPU usage
* Adjust `n_threads` → CPU optimization

---

### No Model Showing?

* Make sure `.gguf` file exists in:

```
models/
```

---

### Port Already in Use

Change launch:

```bash
app.launch(server_port=7861)
```

---

## 🔮 Roadmap

* 🔍 Semantic memory (FAISS)
* 🎤 Speech-to-text (Whisper)
* 🔊 Text-to-speech
* 🧠 Long-term personality evolution
* 🌐 Remote model support

---

## ⚡ Summary

This project is a **clean, modular local LLM chat system** with:

* Separation of concerns
* Persistent memory
* Dynamic persona control
* Efficient model reuse

Perfect base for:

* AI companions
* Roleplay systems
* Local LLM experimentation