
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
* 🖥️ **Two UIs, one brain**: Gradio (local) + an optional Streamlit demo — both share the same logic via `app_utils.py`.

---

## 📁 Project Structure

```bash
.
├── gradio_app.py               # Main Gradio app (python gradio_app.py)
├── app_utils.py                # Shared logic used by both UIs
│
├── streamlit_demo/             # Optional Streamlit UI (see below)
│   ├── streamlit_app.py        #   streamlit run streamlit_demo/streamlit_app.py
│   └── requirements.txt        #   Lean deps for the Cloud deploy
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
│       ├── memory_facts.json  # Fact text, 1:1 with the FAISS vectors
│       ├── memory_map.json    # Legacy vector-to-message mapping (unused for retrieval)
│       └── persona.json
│
├── chats.json                 # Stores list of chat IDs
├── tests/
│   ├── test_mock_llm.py       # Component tests (mocked LLM, no model needed)
│   └── test_real_model.py     # End-to-end RAG test (needs a .gguf in models/)
├── requirements.txt
└── README.md
```

---
> 💡 The embedding model (`all-MiniLM-L6-v2`) is auto-downloaded into `models/embeddings/` on first run.
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

```bash
models/
```

Example:

```bash
models/
├── llama-3-8b-instruct.Q4_K_M.gguf
```

> ⚠️ Models are NOT included due to size constraints.
> 💡 Model size matters for RAG — see [Model Recommendations](#-model-recommendations--limitations).

---

### 5. Run the app

**Gradio UI** (default):

```bash
python gradio_app.py
```

The app will open in your browser:

```bash
http://127.0.0.1:7860
```

**Optional Streamlit UI** — same logic, different frontend:

```bash
pip install -r streamlit_demo/requirements.txt
streamlit run streamlit_demo/streamlit_app.py
```

It uses the same `.env`/`OPENROUTER_API_KEY` for the Claude backend.

---

## 🧪 Testing

Run the component tests (they use a mocked LLM, so no GGUF model is required):

```bash
python tests/test_mock_llm.py
```

Covers: Persona persistence/prompt building, the Memory archive + FAISS + semantic search round-trip, and ChatEngine `||` fact parsing with the double-write (archive + index).

With a GGUF model in `models/`, also run the end-to-end RAG test (skips cleanly if no model is present):

```bash
python tests/test_real_model.py
```

---

## 🤖 Model Recommendations & Limitations

The app runs any GGUF **instruct** model, but model size matters a lot — especially for RAG:

| Model size            | RAG (`||` fact extraction) | Notes |
|-----------------------|----------------------------|-------|
| ≤ 0.5B                | ❌ Usually fails            | Chats fine, but ignores the `||` fact format → long-term memory is silently disabled. |
| 1.5B (Q4_K_M, ~1 GB)  | ✅ Works                    | **Minimum tested**: `Qwen2.5-1.5B-Instruct` Q4_K_M. |
| 3B – 8B               | ✅ Best                     | Recommended for natural replies + reliable fact extraction. |

Key facts:

* **Parameter count**: RAG depends on the model reliably following the strict `[chat] || [fact]` output format. Models under ~1B often ignore it — they chat, but never store memories.
* **Quantization**: prefer `Q4_K_M` / `Q5` GGUFs for a good quality-vs-size balance.
* **Context window**: defaults to `n_ctx=2048` (`model/llm.py`).
* **GPU speed setting (GPU Layers field)**: By default the app runs on your processor (CPU) with `0` — this works on any computer. If you have an NVIDIA graphics card (GPU) and want faster replies, raise the number in the **GPU Layers** field in the Chat tab (e.g. `15`). The number controls how much work is handed to your GPU; higher is faster but uses more graphics card memory. If replies crash or freeze, or you see an "out of memory" error, set it back to `0`. It caps automatically at your model's limit, so picking a big number just means "use all of the GPU you can". The setting takes effect on the next model load; changing it mid-session loads a second instance.
* **RAM**: rough guide ≈ 1 GB per 1B parameters at Q4 quantization.

Example — download the tested minimum model:

```bash
python -c "from huggingface_hub import hf_hub_download; hf_hub_download('Qwen/Qwen2.5-1.5B-Instruct-GGUF', 'qwen2.5-1.5b-instruct-q4_k_m.gguf', local_dir='models')"
```

---

## ☁️ Claude (OpenRouter) backend

The app can also use **Anthropic's Claude SDK** routed through **OpenRouter** —
no local GGUF model, no Anthropic account or credit card required.

1. Create an account + API key at [openrouter.ai/keys](https://openrouter.ai/keys) (starts with `sk-or-...`; shown only once).
2. Copy `.env.example` to `.env` and paste your key (`.env` is git-ignored):

   ```bash
   # .env
   OPENROUTER_API_KEY=sk-or-...
   ```

3. In the app, set **Backend** to **Claude (OpenRouter)**. The **OpenRouter Model** field lets you pick any slug:
   * `google/gemma-4-26b-a4b-it:free` — default; a tested, reliable free model ($0)
   * `openrouter/free` — auto-selects a free model at random (quality varies)
   * `anthropic/claude-sonnet-5` — real Claude via OpenRouter (paid, cheap)
   * any other slug from [openrouter.ai/models](https://openrouter.ai/models)

Notes:

* **Why OpenRouter?** The Anthropic SDK accepts a `base_url`; OpenRouter exposes an Anthropic-compatible endpoint at `https://openrouter.ai/api`. The SDK appends `/v1/messages` itself, so the URL must **not** include `/v1` (that suffix is for OpenAI-style SDKs and would 404). The original Anthropic-first config is kept commented in `model/claude_llm.py` if you ever want to point at `api.anthropic.com` directly.
* Free models have low rate limits (~50 requests/day without credits; higher if you add credits) — fine for exploring, not for production.
* The Claude backend reuses the same `ChatEngine`/`Memory`/`Persona` pipeline, including `|| [fact]` RAG extraction.

---

### GPU setup (optional)

The app runs on CPU out of the box. To use your NVIDIA GPU:

1. Create a dedicated environment:

   ```bash
   conda create -n aigf_gpu python=3.11 -y
   conda activate aigf_gpu
   ```

2. Follow the install steps in [`requirements-gpu.txt`](requirements-gpu.txt) (CUDA-enabled torch + `llama-cpp-python` from conda-forge, then the app dependencies — **not** `requirements.txt`, which would downgrade the CUDA torch).
3. Launch the app and raise the **GPU Layers** field in the Chat tab (see above).

Models in `models/` are shared with the CPU setup — no re-download.

---

## ☁️ Deploy the Streamlit demo (free)

The `streamlit_demo/` folder is a self-contained Streamlit port that can be
hosted for free on [Streamlit Community Cloud](https://streamlit.io/cloud):

1. Push this repo to GitHub.
2. At Streamlit Cloud, **Create app** → connect the GitHub repo → set the
   **Main file** to `streamlit_demo/streamlit_app.py`.
3. Add your API key under the app's **Secrets** (Settings → Secrets):

   ```toml
   OPENROUTER_API_KEY = "sk-or-..."
   ```

4. Deploy. You get a public URL like `your-name/ai-companion.streamlit.app`.

Notes:

* The Cloud build uses `streamlit_demo/requirements.txt` (it sits next to the
  entrypoint, which takes precedence over the root `requirements.txt`) — a
  lean set that skips the heavier local-only deps.
* Chat data lives in the app's filesystem and is **shared by all visitors**
  (no per-user accounts). Don't use it for private data; messages are also
  sent to OpenRouter.
* The Local (GGUF) backend is hidden when no `.gguf` is in `models/` — the
  demo then defaults to the Claude (OpenRouter) backend.

---

## System Architecture

The core of this application is a RAG (Retrieval-Augmented Generation) pipeline that enables long-term semantic memory. The diagram below illustrates the flow from user input to context-augmented inference:
![System Architecture]<img width="551" height="453" alt="image" src="https://github.com/user-attachments/assets/b5aa3016-393a-431d-8fb8-22f9d98e0295" />

### 🚀 How It Works (RAG Pipeline)

```bash
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

### 🔹 `LLM` (model/llm.py)

* Wraps `llama_cpp.Llama`
* Uses **class-level caching** → model loads only once

### 🔹 `Memory` (model/memory.py) - **The RAG Engine**

* **Archive**: JSON file containing the complete conversation log.
* **Librarian (Retriever)**: Uses `SentenceTransformer` to vectorize queries and facts.
* **Index (Vector DB)**: `FAISS` index for high-speed semantic similarity search.

### 🔹 `Persona` (model/persona.py)

* Builds dynamic **system prompt**
* Supports: Names, Genders, Personality Traits, and Custom Descriptions.

### 🔹 `ChatEngine` (model/chat_engine.py) - **The Orchestrator**

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
