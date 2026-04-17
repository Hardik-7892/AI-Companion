import os
import tempfile

# ---------- Offline Speech-to-Text (Whisper) ----------
try:
    from faster_whisper import WhisperModel
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False

_whisper_models = {}
def _load_whisper_model(model_size="small"):
    if model_size not in _whisper_models:
        if not WHISPER_AVAILABLE:
            raise RuntimeError(
                "faster-whisper not installed. Install via `pip install faster-whisper`"
            )
        _whisper_models[model_size] = WhisperModel(
            model_size, device="cpu", compute_type="int8"
        )
    return _whisper_models[model_size]

def transcribe_audio(audio_path: str, model_size="small") -> str:
    model = _load_whisper_model(model_size)
    segments, _info = model.transcribe(audio_path, beam_size=5)
    return " ".join(segment.text for segment in segments)