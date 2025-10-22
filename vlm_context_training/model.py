import torch
from transformers import (
    AutoTokenizer,
    AutoProcessor,
    AutoModelForImageTextToText,
    LlavaNextForConditionalGeneration,
    AutoProcessor as LlavaNextProcessor,
)
from PIL import Image
import os

MODEL_INFOS = {
    "qwen": {
        "model_id": "Qwen/Qwen2.5-VL-7B-Instruct",
        "model_class": AutoModelForImageTextToText,
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
            raise ValueError(f"Unsupported model name: {model_name}")

        info = MODEL_INFOS[model_name]
        self.model_name = model_name
        self.model_id = info["model_id"]

        if not torch.cuda.is_available():
            raise Exception("CUDA is not available.")
        self.device = "cuda"

        print(f"Loading tokenizer for {self.model_id}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)

        if "<|image|>" not in self.tokenizer.get_vocab():
            print("Adding <|image|> token to tokenizer...")
            self.tokenizer.add_special_tokens({"additional_special_tokens": ["<|image|>"]})
            self.tokenizer.pad_token = self.tokenizer.eos_token  # Optional but helps in some cases

        print(f"Loading model {self.model_id} on {self.device}...")
        self.model = info["model_class"].from_pretrained(
            self.model_id,
            device_map="auto",
            torch_dtype=torch.float16
        )

        self.model.resize_token_embeddings(len(self.tokenizer))

        self.processor = info["processor_class"].from_pretrained(self.model_id)
        self.processor.tokenizer = self.tokenizer

        self.model.eval()
        print(f"Model and processor for '{self.model_name}' loaded.\n")

    def infer(self, prompt: str, image_paths: list[str], max_new_tokens: int = 256):
        images = []
        for path in image_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Image not found: {path}")
            img = Image.open(path).convert("RGB")
            images.append(img)

        if self.model_name == "qwen":
            num_images = len(images)
            image_tokens = ' '.join(["<|image|>"] * num_images)
            formatted_prompt = f"{image_tokens}\n{prompt}"

            print("\n[Qwen Prompt Debug]")
            print(f"Formatted prompt:\n{formatted_prompt}")
            print(f"Number of <|image|> tokens: {formatted_prompt.count('<|image|>')}")
            print(f"Number of images: {len(images)}\n")

            if formatted_prompt.count("<|image|>") != len(images):
                raise ValueError("Mismatch between number of images and <|image|> tokens in prompt!")

            processor_inputs = self.processor(
                text=formatted_prompt,
                images=images,
                return_tensors="pt"
            )

        elif self.model_name == "llava":
            formatted_prompt = prompt
            processor_inputs = self.processor(
                prompt=formatted_prompt,
                images=images,
                return_tensors="pt"
            )

        else:
            raise ValueError(f"Unsupported model: {self.model_name}")

        print("Does tokenizer have <|image|> token? ", "<|image|>" in self.processor.tokenizer.get_vocab())
        print(f"Input pixel_values shape: {processor_inputs['pixel_values'].shape if 'pixel_values' in processor_inputs else 'N/A'}")
        print(f"Input input_ids shape: {processor_inputs['input_ids'].shape if 'input_ids' in processor_inputs else 'N/A'}")

        inputs = processor_inputs.to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)

        decoded = self.processor.tokenizer.batch_decode(
            outputs, skip_special_tokens=True
        )
        return decoded[0]

    def batch_infer(self, prompt: str, image_paths: list[str], batch_size: int = 32, max_new_tokens: int = 256):
        all_outputs = []
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            print(f"\nProcessing batch {i // batch_size + 1} with {len(batch_paths)} images...")
            
            # Run infer method on this batch
            output = self.infer(prompt=prompt, image_paths=batch_paths, max_new_tokens=max_new_tokens)
            all_outputs.append(output)
        return all_outputs

