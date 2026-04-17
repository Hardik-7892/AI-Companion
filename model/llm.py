# model/llm.py

from llama_cpp import Llama


class LLM:
    """
    Thin wrapper around llama_cpp.Llama.

    Uses a class-level cache so the same model file is only loaded once per
    process, regardless of how many times `get_instance` is called.
    """

    _cache: dict[str, "LLM"] = {}

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        model_path: str,
        n_ctx: int = 2048,
        n_threads: int = 8,
        n_gpu_layers: int = 35,
    ) -> None:
        self.model_path = model_path
        self._model = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_gpu_layers=n_gpu_layers,
        )

    # ------------------------------------------------------------------
    # Cache helper
    # ------------------------------------------------------------------

    @classmethod
    def get_instance(
        cls,
        model_path: str,
        n_ctx: int = 2048,
        n_threads: int = 8,
        n_gpu_layers: int = 35,
    ) -> "LLM":
        """Return a cached LLM for *model_path*, creating one if needed."""
        key = str(model_path)
        if key not in cls._cache:
            cls._cache[key] = cls(
                model_path=key,
                n_ctx=n_ctx,
                n_threads=n_threads,
                n_gpu_layers=n_gpu_layers,
            )
        return cls._cache[key]

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def chat(
        self,
        messages: list[dict],
        max_tokens: int = 150,
        temperature: float = 0.8,
        top_p: float = 0.9,
    ) -> str:
        """Run a chat completion and return the assistant text."""
        response = self._model.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=["</s>"],
        )
        return response["choices"][0]["message"]["content"]