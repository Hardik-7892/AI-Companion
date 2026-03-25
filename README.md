# AI Girlfriend (AIGF) - Gradio App

A conversational AI girlfriend app built with **Gradio** and **LLaMA models** (GGUF format).  
Users can select different models, customize personalities, and chat in a friendly, natural style.


## Features

- Select between multiple GGUF models.
- Customize your AI girlfriend’s personality traits.
- Enter names for yourself and your AI companion.
- Chat interface with persistent conversation memory.
- Audio support for transcription and TTS (later).
- Theme options: Pink, Blue, Dark.


## Folder Structure

```

AIGF/
│
├─ app.py                 # Main Gradio app
├─ model.py               # Model loading & chat logic
├─ audio.py               # Audio features (transcription/TTS)
├─ chats/                 # Auto-generated chat histories
├─ models/                # Your GGUF models (not included in repo)
├─ requirements.txt       # Python dependencies
├─ run.bat                # Windows launcher
└─ .gitignore             # Files/folders to ignore in Git

````


## Setup Instructions

1. **Clone the repository**

```bash
git clone https://github.com/YOUR_USERNAME/AIGF.git
cd AIGF
````

2. **Create a virtual environment**

```bash
python -m venv venv
```

3. **Activate the virtual environment**

* Windows:

```bash
venv\Scripts\activate
```

* macOS / Linux:

```bash
source venv/bin/activate
```

4. **Install dependencies**

```bash
pip install -r requirements.txt
```

5. **Add your GGUF models**

Place your `.gguf` model files in the `models/` folder.
Example:

```
models/
├─ Meta-Llama-3-8B-Instruct-Q4_K_M.gguf
├─ Llama-3.2-3B-Instruct-Q6_K.gguf
├─ Any other model.gguf file supported by llama-cpp-python
```

> ⚠️ Models are **not included** in this repository due to size restrictions.

6. **Run the app**

```bash
python app.py
```

Or on Windows:

```bash
run.bat
```

Your default browser should open automatically to the chat interface (usually `http://127.0.0.1:7860`).


## Usage

1. Select an existing chat or create a new one.
2. Enter optional details:

   * Your name
   * AI girlfriend’s name
   * Personality traits or custom description
3. Select a model from the dropdown.
4. Start chatting!
5. Use the **Reset** button to clear visible chat history (keeps saved memory).


## Troubleshooting & Tips

* **WindowsPath errors with LLaMA models**:
  Make sure `model_path` passed to `Llama()` is a **string**, not a `Path` object.
  Example fix in `model.py`:

  ```python
  llm = Llama(model_path=str(model_path), n_ctx=2048, n_threads=8)
  ```

* **Missing GGUF models**:
  The app won’t work without proper GGUF files in the `models/` folder.

* **Port already in use**:
  If `7860` is busy, change the port in `app.launch(server_port=7861)`.

* **Virtual environment issues**:
  Always activate the venv before running:
  `venv\Scripts\activate` (Windows) or `source venv/bin/activate` (macOS/Linux).

* **Audio features**:
  Ensure all audio dependencies (`soundfile`, `whisper`, `TTS`) are installed if you plan to use transcription or TTS.

* **Performance tips**:
  Large models (8B+) may need GPU or fewer threads for faster responses. Adjust `n_threads` and `n_gpu_layers` in `load_model()`.

