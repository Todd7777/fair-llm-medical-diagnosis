import torch
from transformers import (
    AutoProcessor,
    AutoModelForVision2Seq,
    LlavaNextForConditionalGeneration,
    AutoProcessor as LlavaNextProcessor,
)
from PIL import Image
import os

MODEL_INFOS = {
    "qwen": {
        "model_id": "Qwen/Qwen2.5-VL-7B-Instruct",
        "model_class": AutoModelForVision2Seq,
        "processor_class": AutoProcessor,
    },
    "llava": {
        "model_id": "llava-hf/llava-v1.6-mistral-7b-hf",
        "model_class": LlavaNextForConditionalGeneration,
        "processor_class": LlavaNextProcessor,
    },
}


class Model:
    def __init__(self, model_name):
        model_name = model_name.lower().strip()
        if model_name not in MODEL_INFOS:
            raise ValueError(...)
        info = MODEL_INFOS[model_name]
        self.model_name = model_name
        self.model_id = info["model_id"]
        if not torch.cuda.is_available():
            raise Exception("CUDA is not available.")
        self.device = "cuda"
        print(f"Loading model {self.model_id} on {self.device}...")
        self.model = info["model_class"].from_pretrained(
            self.model_id, torch_dtype=torch.float16, device_map="auto"
        )
        self.processor = info["processor_class"].from_pretrained(self.model_id)
        print(f"Model and processor for '{self.model_name}' loaded.")

    def infer(self, prompt: str, image_paths: list[str], max_new_tokens: int = 256):
        images = []
        for path in image_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Image not found: {path}")
            img = Image.open(path).convert("RGB")
            images.append(img)

        if self.model_name == "qwen":
            inputs = self.processor(
                prompt=prompt, images=images, return_tensors="pt"
            ).to(self.model.device)
        elif self.model_name == "llava":
            inputs = self.processor(
                prompt=prompt, images=images, return_tensors="pt"
            ).to(self.model.device)
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")

        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        decoded = self.processor.tokenizer.batch_decode(
            outputs, skip_special_tokens=True
        )

        return decoded[0]
