from __future__ import annotations

import argparse
import base64
import mimetypes
import sys
from pathlib import Path

from llama_cpp import Llama
from llama_cpp.llama_chat_format import Llava15ChatHandler


def file_to_data_uri(path: Path) -> str:
    mime, _ = mimetypes.guess_type(str(path))
    mime = mime or "application/octet-stream"
    data = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{data}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Simple multimodal GGUF image test for llama-cpp-python"
    )
    parser.add_argument("image", help="Path to the input image")
    parser.add_argument(
        "--model",
        default="../models/Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-Q5_K_M.gguf",
        help="Path to the main GGUF model file",
    )
    parser.add_argument(
        "--mmproj",
        default="../models/mmproj-Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-f16.gguf",
        help="Path to the multimodal projector GGUF file",
    )
    parser.add_argument(
        "--prompt",
        default="Describe this image in detail. Mention the main objects, setting, and any visible text.",
        help="Question to ask about the image",
    )
    parser.add_argument("--n-ctx", type=int, default=4096, help="Context size")
    parser.add_argument("--n-threads", type=int, default=8, help="CPU threads")
    parser.add_argument("--n-gpu-layers", type=int, default=35, help="GPU layers to offload")
    args = parser.parse_args()

    image_path = Path(args.image)
    model_path = Path(args.model)
    mmproj_path = Path(args.mmproj)

    for p, label in [(image_path, "image"), (model_path, "model"), (mmproj_path, "mmproj")]:
        if not p.exists():
            print(f"Error: {label} file not found: {p}", file=sys.stderr)
            return 1

    chat_handler = Llava15ChatHandler(clip_model_path=str(mmproj_path))
    llm = Llama(
        model_path=str(model_path),
        chat_handler=chat_handler,
        n_ctx=args.n_ctx,
        n_threads=args.n_threads,
        n_gpu_layers=args.n_gpu_layers,
        verbose=True,
    )

    image_data_uri = file_to_data_uri(image_path)

    response = llm.create_chat_completion(
        messages=[
            {
                "role": "system",
                "content": "You are a helpful vision assistant. Answer only from what is visible in the image.",
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": args.prompt},
                    {"type": "image_url", "image_url": {"url": image_data_uri}},
                ],
            },
        ],
        max_tokens=300,
        temperature=0.8,
    )

    content = response["choices"][0]["message"]["content"]
    print("\n=== MODEL RESPONSE ===\n")
    print(content)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())