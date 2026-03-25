from transformers import pipeline
import numpy as np
from gtts import gTTS
import tempfile
from pathlib import Path

asr = pipeline("automatic-speech-recognition", model="openai/whisper-tiny.en")  # or any small model

def transcribe_audio(audio):
    if audio is None:
        return ""
    sr, y = audio
    if y.ndim > 1:
        y = y.mean(axis=1)
    y = y.astype(np.float32)
    if np.max(np.abs(y)) > 0:
        y = y / np.max(np.abs(y))
    text = asr({"sampling_rate": sr, "raw": y})["text"]
    return text

def tts_from_text(text: str, lang: str = "en"):
    text = (text or "").strip()
    if not text:
        return None  # Gradio Audio will show nothing

    # Create a temporary mp3 file
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tmp_path = Path(tmp.name)
    tmp.close()

    tts = gTTS(text=text, lang=lang, slow=False)
    tts.save(str(tmp_path))  # gTTS writes MP3

    # Gradio Audio accepts a file path as str/Path[web:113]
    return str(tmp_path)
