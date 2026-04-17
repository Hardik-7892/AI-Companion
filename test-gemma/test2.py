# image_support_gguf.py
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Llava15ChatHandler  # or swap this out

# ==== CONFIG – ADAPT THESE PATHS ====
MODEL_PATH = "../models/Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-Q5_K_M.gguf"
MMPROJ_PATH = "../models/mmproj-Gemma-4-E4B-Uncensored-HauhauCS-Aggressive-f16.gguf"

# simple hardcoded inputs
IMAGE_PATH = "image.jpg"  # put any image here
QUESTION = "Describe the image in detail."

import base64, mimetypes
mime_type, _ = mimetypes.guess_type(IMAGE_PATH)  # e.g. "image/jpeg", "image/png"
with open(IMAGE_PATH, "rb") as f:
    img_b64 = base64.b64encode(f.read()).decode()
image_url = f"data:{mime_type};base64,{img_b64}"

handler = Llava15ChatHandler(clip_model_path=MMPROJ_PATH)

llm = Llama(
    model_path=MODEL_PATH,
    chat_handler=handler,
    n_ctx=4096,
    n_gpu_layers=1,
    logits_all=True,  # required for multimodal
)

llm.create_chat_completion(messages=[{
    "role": "user",
    "content": [
        {"type": "image_url", "image_url": {"url": image_url}},
        {"type": "text", "text": "Describe this image."}
    ]
}])
response = llm.create_chat_completion(...)
print(response["choices"][0]["message"]["content"])