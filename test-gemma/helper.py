from transformers import AutoModelForImageTextToText, AutoProcessor
from PIL import Image
# import torch

GEMMA_PATH = "google/gemma-3n-E2B-it"  # or E4B-it

processor = AutoProcessor.from_pretrained(GEMMA_PATH)
model = AutoModelForImageTextToText.from_pretrained(
    GEMMA_PATH,
    # torch_dtype="auto",
    device_map="auto",
)

def ask_image_question(image_path: str, question: str, max_new_tokens: int = 64) -> str:
    image = Image.open(image_path).convert("RGB")

    messages = [
        {"role": "user", "content": [
            {"type": "text", "text": question},
            {"type": "image", "image": image},
        ]}
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
    ).to(model.device)

    output_ids = model.generate(
        inputs,
        max_new_tokens=max_new_tokens,
    )

    # skip the prompt part when decoding
    input_len = inputs.shape[-1]
    text = processor.batch_decode(
        output_ids[:, input_len:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )
    return text