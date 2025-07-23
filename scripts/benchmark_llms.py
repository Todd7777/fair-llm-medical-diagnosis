#!/usr/bin/env python
"""
Medical Imaging Benchmarking Script for Vision-Language Models

This script benchmarks various VLMs and LLMs on medical imaging diagnosis tasks
using datasets like CheXpert, with a focus on fairness metrics.
"""

import os
import argparse
import yaml
import json
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
from PIL import Image
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# Add project root to path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

'''from fairtune.metrics.fairness import (
    compute_fairness_metrics,
    compute_equity_scaled_auc,
    compute_subgroup_metrics,
)'''

# Dictionary of strong open-source models and their configurations
# Only fully open-weight models are included to ensure local, reproducible research.
SUPPORTED_MODELS = {
    # ------------------------------------------------------------------
    # Vision-only models (zero-shot or fine-tuned CNN/Transformer backbones)
    # ------------------------------------------------------------------
    "chexpert_densenet121": {
        "type": "vision",
        "model_name": "stanfordmlgroup/chexpert-densenet121-disease-only",
        "processor_name": None,   # Standard torchvision transforms
        "is_local": True,
        "description": "DenseNet-121 fine-tuned on CheXpert (disease labels)",
    },
    "chexzero_vitl16": {
        "type": "vision",
        "model_name": "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        "processor_name": "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        "is_local": True,
        "description": "BiomedCLIP ViT-B/16 (public)",
    },

    # ------------------------------------------------------------------
    # Vision-Language models
    # ------------------------------------------------------------------
    "biomedclip_vit_base": {
        "type": "vision_language",
        "model_name": "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        "processor_name": "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
        "is_local": True,
        "description": "BiomedCLIP (CLIP variant) – strong zero-shot vision-language model for biomedical imaging",
    },
    "llava_v1_5_13b": {
        "type": "vision",
        "model_name": "llava-hf/llava-1.5-13b-hf",
        "processor_name": "llava-hf/llava-1.5-13b-hf",
        "is_local": True,
        "description": "LLaVA-1.5 13B – strong general VLM, useful baseline for medical VQA",
    },
    "instructblip_vicuna_7b": {
        "type": "vision_language",
        "model_name": "Salesforce/instructblip-vicuna-7b",
        "processor_name": "Salesforce/instructblip-vicuna-7b",
        "is_local": True,
        "description": "InstructBLIP (Vicuna-7B) – instruction-tuned BLIP-2 model",
    },
    "medflamingo_9b": {
        "type": "vision_language",
        "model_name": "microsoft/med-flamingo-9b",
        "processor_name": "microsoft/med-flamingo-9b",
        "is_local": True,
        "description": "Med-Flamingo 9B – multimodal continuation model adapted for medical images",
    },

    # ------------------------------------------------------------------
    # Text-only biomedical LLMs (for report generation / comparison)
    # ------------------------------------------------------------------
    "biogpt_large": {
        "type": "text",
        "model_name": "microsoft/BioGPT-Large",
        "processor_name": "microsoft/BioGPT-Large",
        "is_local": True,
        "description": "BioGPT-Large – 1.5B parameter biomedical language model",
    },
    "pubmedgpt_2_7b": {
        "type": "text",
        "model_name": "stanford-crfm/pubmedgpt-1.2B",
        "processor_name": "stanford-crfm/pubmedgpt-1.2B",
        "is_local": True,
        "description": "PubMedGPT – language model trained solely on PubMed abstracts",
    },

    # ------------------------------------------------------------------
    # Cloud/API multimodal LLMs (requires external API keys)
    # ------------------------------------------------------------------
    "chatgpt_gpt4v": {
        "type": "api",
        "model_name": "gpt-4o-mini",  # GPT-4o Vision endpoint
        "provider": "openai",
        "api_base": "https://api.openai.com/v1",
        "api_key_env": "OPENAI_API_KEY",
        "description": "GPT-4o with vision (multimodal). Requires OpenAI API key.",
    },
    "claude_opus_vision": {
        "type": "api",
        "model_name": "claude-3-opus-20240229",
        "provider": "anthropic",
        "api_base": "https://api.anthropic.com",
        "api_key_env": "ANTHROPIC_API_KEY",
        "description": "Claude 3 Opus with vision. Requires Anthropic API key.",
    },
}

class MedicalImagingBenchmarker:
    """Benchmarking class for medical imaging models on diagnosis tasks."""
    
    def __init__(self, config_path: str):
        """Initialize the benchmarker with configuration."""
        self.config = self._load_config(config_path)
        self.device = torch.device(self.config["hardware"]["device"])
        self.results_dir = Path(self.config["output"]["results_dir"])
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if self.config["hardware"]["mixed_precision"] else None
        
        # Load datasets
        self.datasets = self._load_datasets()
        self.dataloaders = self._create_dataloaders()
        
    def _load_config(self, config_path: str) -> Dict:
        """Load and validate configuration."""
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        
        # Set default values if not specified
        config["hardware"].setdefault("device", "cuda" if torch.cuda.is_available() else "cpu")
        config["hardware"].setdefault("mixed_precision", True)
        config["hardware"].setdefault("gradient_checkpointing", True)
        
        return config
    
    def _load_datasets(self) -> Dict[str, Dict]:
        """Load all configured medical imaging datasets."""
        datasets = {}
        
        for dataset_name, dataset_cfg in self.config["datasets"].items():
            print(f"Loading {dataset_name} dataset...")
            
            if dataset_cfg["type"] == "image":
                if dataset_name == "chexpert":
                    dataset = self._load_chexpert(dataset_cfg)
                elif dataset_name == "mimic_cxr":
                    dataset = self._load_mimic_cxr(dataset_cfg)
                elif dataset_name == "padchest":
                    dataset = self._load_padchest(dataset_cfg)
                elif dataset_name == "retinal_disease_kaggle":
                    dataset = self._load_retinal_disease_kaggle(dataset_cfg)
                elif dataset_name == "odir2019":
                    dataset = self._load_odir2019(dataset_cfg)
                else:
                    raise ValueError(f"Unsupported dataset: {dataset_name}")
                
                datasets[dataset_name] = dataset
                print(f"Loaded {len(dataset['train'])} training, {len(dataset['val'])} validation, "
                      f"{len(dataset['test'])} test samples")
        
        return datasets
    
    def _load_chexpert(self, dataset_cfg: Dict) -> Dict:
            from torchvision import transforms
            from torch.utils.data import Dataset, DataLoader
            from pathlib import Path

            transform = transforms.Compose([
                transforms.Resize((self.config["evaluation"]["image_size"], 
                                self.config["evaluation"]["image_size"])),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.config["evaluation"]["normalize_mean"],
                                    std=self.config["evaluation"]["normalize_std"])
            ])

            image_root = dataset_cfg["image_root"]
            csv = dataset_cfg["csv"]

            dataset_val = CheXpertDataset(
                root_dir=dataset_cfg["path"],
                csv=Path(csv).name,  # just the filename, assuming it's inside `path`
                image_root=image_root,
                transform=transform
            )

            return {
                "train": [],
                "val": dataset_val,
                "test": dataset_val  # reuse val as test for benchmarking
            }

    
    def _load_mimic_cxr(self, dataset_cfg: Dict) -> Dict:
        """Load MIMIC-CXR dataset."""
        # Similar structure to _load_chexpert
        # Implementation depends on how you want to process MIMIC-CXR
        return {"train": [], "val": [], "test": []}
    
    def _load_padchest(self, dataset_cfg: Dict) -> Dict:
        """Load PadChest dataset."""
        # TODO: Implement actual loading
        return {"train": [], "val": [], "test": []}
    
    def _load_retinal_disease_kaggle(self, dataset_cfg: Dict) -> Dict:
        """Load Retinal Disease Classification Kaggle dataset."""
        # TODO: Implement actual loading logic (parse folder structure: Train/Test/<class>/*.jpg)
        return {"train": [], "val": [], "test": []}
    
    def _load_odir2019(self, dataset_cfg: Dict) -> Dict:
        """Load ODIR-2019 retinal dataset."""
        # TODO: Implement actual loading logic (uses CSV with multi-label annotations)
        return {"train": [], "val": [], "test": []}
    
    def _create_dataloaders(self) -> Dict[str, DataLoader]:
        """Create dataloaders for all datasets."""
        dataloaders = {}
        
        for dataset_name, dataset in self.datasets.items():
            dataloaders[dataset_name] = {}
            for split in ['train', 'val', 'test']:
                if dataset[split]:  # only create DataLoader if split is non-empty
                    dataloaders[dataset_name][split] = DataLoader(
                        dataset[split],
                        batch_size=self.config["evaluation"]["batch_size"],
                        num_workers=self.config["evaluation"]["num_workers"],
                        shuffle=(split == 'train'),
                        pin_memory=torch.cuda.is_available()
                    )
        
        return dataloaders
    
    def _load_model(self, model_name: str):
        """Load the specified model based on its type."""
        model_config = SUPPORTED_MODELS.get(model_name.lower())
        if not model_config:
            print(f"Model {model_name} not in SUPPORTED_MODELS.")
            return None
        
        # Get API key from environment if needed
        if "api_key_env" in model_config:
            model_config["api_key"] = os.getenv(model_config["api_key_env"])
        
        try:
            if model_config["type"] in ["vision", "vision_language"]:
                return self._load_vision_model(model_name, model_config)
            elif model_config["type"] == "text":
                return self._load_text_model(model_name, model_config)
            elif model_config["type"] == "api":
                return self._load_api_model(model_name, model_config)
            else:
                raise ValueError(f"Unsupported model type: {model_config['type']}")
        except Exception as e:
            print(f"Error loading model {model_name}: {str(e)}")
            raise
    
    def _load_vision_model(self, model_name: str, model_config: Dict):
        """Load a vision or vision-language model."""
        from transformers import AutoModelForImageClassification, AutoFeatureExtractor
        lname = model_name.lower()
        
        if "chexzero" in lname:
            # BiomedCLIP model
            from transformers import AutoProcessor, AutoModel
            hf_token = os.getenv("HF_TOKEN", None)
            processor = AutoProcessor.from_pretrained(
                model_config["processor_name"],
                trust_remote_code=True,
                token=hf_token
            )
            model = AutoModel.from_pretrained(
                model_config["model_name"],
                trust_remote_code=True,
                token=hf_token
            ).to(self.device)
            
            if self.config["hardware"]["gradient_checkpointing"]:
                model.gradient_checkpointing_enable()
                
            return {"model": model, "processor": processor, "type": "vision"}
            
        elif "chexpert" in lname and "densenet" in lname:
            # Standard CheXpert model
            feature_extractor = AutoFeatureExtractor.from_pretrained(
                model_config["processor_name"]
            )
            model = AutoModelForImageClassification.from_pretrained(
                model_config["model_name"],
                num_labels=len(self.config["datasets"]["chexpert"]["classes"])
            ).to(self.device)
            
            return {"model": model, "processor": feature_extractor, "type": "vision"}
            
        elif "llava" in lname:
            # LLaVA model for medical imaging
            from transformers import AutoProcessor, LlavaForConditionalGeneration
            
            processor = AutoProcessor.from_pretrained(
                model_config["processor_name"],
                trust_remote_code=True
            )
            model = LlavaForConditionalGeneration.from_pretrained(
                model_config["model_name"],
                torch_dtype=torch.float16 if self.config["hardware"]["mixed_precision"] else torch.float32,
                device_map="auto"
            )
            
            return {"model": model, "processor": processor, "type": "vision_language"}
        
        else:
            raise ValueError(f"Unsupported vision model name: {model_name}")
    
    def _load_api_model(self, model_name: str, model_config: Dict):
        """Initialize an API-based multimodal LLM client."""
        provider = model_config.get("provider")
        api_key = model_config.get("api_key")
        if not api_key:
            raise RuntimeError(
                f"API key for {model_name} not found. Set {model_config['api_key_env']} in environment variables."
            )
        if provider == "openai":
            try:
                from openai import OpenAI
                client = OpenAI(api_key=api_key, base_url=model_config.get("api_base"))
                return {"client": client, "model_name": model_config["model_name"], "type": "api", "provider": "openai"}
            except Exception as e:
                raise RuntimeError(f"Failed to initialize OpenAI client: {str(e)}")
        elif provider == "anthropic":
            try:
                from anthropic import Anthropic
                client = Anthropic(api_key=api_key, base_url=model_config.get("api_base"))
                return {"client": client, "model_name": model_config["model_name"], "type": "api", "provider": "anthropic"}
            except Exception as e:
                raise RuntimeError(f"Failed to initialize Anthropic client: {str(e)}")
        else:
            raise ValueError(f"Unsupported API provider: {provider}")
    
    def _load_text_model(self, model_name: str, model_config: Dict):
        """Load a text-based model."""
        if model_name == "biomedlm":
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            tokenizer = AutoTokenizer.from_pretrained(
                model_config["model_name"],
                trust_remote_code=True
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_config["model_name"],
                torch_dtype=torch.float16 if self.config["hardware"]["mixed_precision"] else torch.float32,
                device_map="auto"
            )
            
            return {"model": model, "tokenizer": tokenizer, "type": "text"}
            
        elif model_name == "claude":
            from anthropic import Anthropic
            return {
                "client": Anthropic(api_key=model_config["api_key"]),
                "type": "api",
                "model_name": model_config["model_name"]
            }
            
        elif model_name == "chatgpt":
            from openai import OpenAI
            return {
                "client": OpenAI(api_key=model_config["api_key"]),
                "type": "api",
                "model_name": model_config["model_name"]
            }
    
    def _init_qwen(self, config: Dict):
        """Initialize Qwen model."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            tokenizer = AutoTokenizer.from_pretrained(
                config["model_name"], 
                trust_remote_code=True
            )
            model = AutoModelForCausalLM.from_pretrained(
                config["model_name"],
                device_map="auto",
                trust_remote_code=True
            )
            return {"model": model, "tokenizer": tokenizer}
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Qwen model: {str(e)}")
    
    def _init_claude(self, config: Dict):
        """Initialize Claude model."""
        try:
            from anthropic import Anthropic
            return Anthropic(api_key=config["api_key"])
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Claude model: {str(e)}")
    
    def _init_deepseek(self, config: Dict):
        """Initialize Deepseek model."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            tokenizer = AutoTokenizer.from_pretrained(
                config["model_name"], 
                trust_remote_code=True
            )
            model = AutoModelForCausalLM.from_pretrained(
                config["model_name"],
                device_map="auto",
                trust_remote_code=True
            )
            return {"model": model, "tokenizer": tokenizer}
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Deepseek model: {str(e)}")
    
    def _init_chatgpt(self, config: Dict):
        """Initialize ChatGPT model."""
        try:
            from openai import OpenAI
            return OpenAI(api_key=config["api_key"])
        except Exception as e:
            raise RuntimeError(f"Failed to initialize ChatGPT model: {str(e)}")
    
    def _generate_prompt(self, sample: Dict) -> str:
        """Generate a prompt for the LLM based on the sample."""
        # TODO: Implement prompt engineering for medical diagnosis
        # This should include patient information, medical history, and the question
        return ""
    
    def _process_llm_response(self, response: str) -> Dict:
        """Process the LLM response to extract predictions and confidence scores."""
        # TODO: Implement response parsing logic
        # This should extract the predicted class and confidence scores
        return {"prediction": None, "confidence": None}
    
    def evaluate_model(self, model_name: str, dataset_name: str = "chexpert", split: str = "test") -> Dict:
        """Evaluate a model on the specified dataset split."""
        print(f"\nEvaluating {model_name} on {dataset_name} {split} split...")
        
        # Initialize model
        model_info = self._load_model(model_name)
        if model_info is None:
            print(f"Failed to load model {model_name}")
            return None
        
        # skip empty datasets
        if split not in self.dataloaders[dataset_name]:
            print(f"Skipping {model_name} on {dataset_name} {split} (no data)")
            return None
        
        # Get dataloader
        dataloader = self.dataloaders[dataset_name][split]
        
        results = {
            "model": model_name,
            "dataset": dataset_name,
            "split": split,
            "timestamp": datetime.now().isoformat(),
            "samples": []
        }
        
        # Set model to evaluation mode
        if hasattr(model_info.get("model"), 'eval'):
            model_info["model"].eval()
        
        # Evaluation loop
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Evaluating {model_name}")):
                try:
                    # Move batch to device
                    images = batch["image"].to(self.device)
                    labels = batch["label"].to(self.device)
                    demographics = batch["demographics"]
                    
                    # Forward pass
                    if model_info["type"] == "vision":
                        outputs = self._forward_vision_model(model_info, images)
                    elif model_info["type"] == "vision_language":
                        outputs = self._forward_vision_language_model(model_info, images, batch)
                    elif model_info["type"] == "text":
                        outputs = self._forward_text_model(model_info, batch)
                    elif model_info["type"] == "api":
                        outputs = self._forward_api_model(model_info, batch)
                    
                    # Process outputs
                    predictions = self._process_outputs(outputs, model_info)
                    
                    # Store results
                    batch_results = []
                    for i in range(len(images)):
                        batch_results.append({
                            "image_path": batch["image_path"][i],
                            "demographics": {k: v[i] for k, v in demographics.items()},
                            "ground_truth": labels[i].cpu().numpy().tolist(),
                            "prediction": predictions["logits"][i].cpu().numpy().tolist(),
                            "confidence": predictions["probs"][i].cpu().numpy().tolist(),
                            "attention_maps": predictions.get("attention_maps", [None] * len(images))[i]
                        })
                    
                    results["samples"].extend(batch_results)
                    
                except Exception as e:
                    print(f"Error processing batch {batch_idx}: {str(e)}")
                    continue
        
        # Compute metrics
        metrics = self._compute_metrics(results)
        results["metrics"] = metrics
        
        # Save results
        self._save_results(results, f"{model_name}_{dataset_name}_{split}")
        
        return metrics
    
    def _forward_vision_model(self, model_info: Dict, images: torch.Tensor) -> Dict:
        """Forward pass for vision models."""
        with torch.cuda.amp.autocast(enabled=self.scaler is not None):
            outputs = model_info["model"](images)
        return {"logits": outputs.logits if hasattr(outputs, 'logits') else outputs}
    
    def _forward_vision_language_model(self, model_info: Dict, images: torch.Tensor, batch: Dict) -> Dict:
        """Forward pass for vision-language models."""
        processor = model_info["processor"]
        model = model_info["model"]
        
        # Prepare prompts
        prompts = [
            "Question: What abnormalities are present in this chest X-ray? Answer:"
            for _ in range(len(images))
        ]

        pil_images = [Image.open(p).convert("RGB") for p in batch["image_path"]]
        
        # Process inputs
        inputs = processor(
            text=pil_images,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)
        
        # Generate outputs
        with torch.cuda.amp.autocast(enabled=self.scaler is not None):
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
                return_dict_in_generate=True,
                output_attentions=True,
                output_scores=True
            )
        
        # Process outputs
        generated_texts = processor.batch_decode(
            outputs.sequences,
            skip_special_tokens=True
        )
        
        return {
            "generated_texts": generated_texts,
            "logits": outputs.scores[-1] if outputs.scores else None,
            "attention": outputs.attentions[-1] if outputs.attentions else None
        }
    
    def _forward_text_model(self, model_info: Dict, batch: Dict) -> Dict:
        """Forward pass for text-based models."""
        # Implement text model inference
        pass
    
    def _forward_api_model(self, model_info: Dict, batch: Dict) -> Dict:
        """Forward pass for API-based models (OpenAI, Anthropic)."""
        client = model_info["client"]
        provider = model_info["provider"]
        model_name = model_info["model_name"]
        images = batch["image"]  # Tensor BxCxhxw
        # For demo purposes we take first image from batch
        # Real implementation should loop over batch and send each image – beware of rate limits and costs.
        import base64, io
        from PIL import Image
        generated_texts = []
        from torchvision import transforms  # Lazy import to avoid dependency for text-only runs
        for img_tensor in images:
            pil_img = transforms.ToPILImage()(img_tensor.cpu())
            buffered = io.BytesIO()
            pil_img.save(buffered, format="PNG")
            img_b64 = base64.b64encode(buffered.getvalue()).decode()
            prompt = "Identify any abnormalities in this retinal/chest image. Respond with JSON {\"labels\": [...]}"
            if provider == "openai":
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "user", "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}}
                        ]}
                    ],
                    max_tokens=256
                )
                generated_texts.append(response.choices[0].message.content)
            elif provider == "anthropic":
                response = client.messages.create(
                    model=model_name,
                    max_tokens=256,
                    messages=[
                        {"role": "user", "content": prompt},
                        {"role": "user", "content": {"type": "image", "source": {"media_type": "image/png", "data": img_b64}}}
                    ]
                )
                generated_texts.append(response.content[0].text)
        return {"generated_texts": generated_texts}

    
    def _process_outputs(self, outputs: Dict, model_info: Dict) -> Dict:
        """Process model outputs into predictions and confidences."""
        if model_info["type"] in ["vision", "text"]:
            logits = outputs["logits"]
            probs = torch.sigmoid(logits) if logits is not None else None
            return {
                "logits": logits,
                "probs": probs,
                "attention_maps": outputs.get("attention")
            }
        elif model_info["type"] in ["vision_language", "api"]:
            # Process generated text into structured outputs
            return {
                "generated_texts": outputs.get("generated_texts"),
                "logits": outputs.get("logits"),
                "attention_maps": outputs.get("attention")
            }
        return {}
    
    def _compute_metrics(self, results: Dict) -> Dict:
        """Compute evaluation metrics from results."""
        from sklearn.metrics import (
            accuracy_score, roc_auc_score, f1_score,
            precision_score, recall_score
        )
        
        # Extract predictions and labels
        y_true = np.array([sample["ground_truth"] for sample in results["samples"]])
        y_pred = np.array([sample["prediction"] for sample in results["samples"]])
        y_probs = np.array([sample["confidence"] for sample in results["samples"]])
        
        # Compute classification metrics
        metrics = {}
        
        # Binary metrics for each class
        for i, class_name in enumerate(self.config["datasets"][results["dataset"]]["classes"]):
            metrics[f"{class_name}_auc"] = roc_auc_score(y_true[:, i], y_probs[:, i])
            metrics[f"{class_name}_f1"] = f1_score(y_true[:, i], y_pred[:, i])
            metrics[f"{class_name}_precision"] = precision_score(y_true[:, i], y_pred[:, i])
            metrics[f"{class_name}_recall"] = recall_score(y_true[:, i], y_pred[:, i])
        
        # Average metrics
        metrics["macro_auc"] = np.mean([metrics[f"{c}_auc"] for c in self.config["datasets"][results["dataset"]]["classes"]])
        metrics["macro_f1"] = np.mean([metrics[f"{c}_f1"] for c in self.config["datasets"][results["dataset"]]["classes"]])
        
        # Compute fairness metrics
        if "demographics" in results["samples"][0]:
            fairness_metrics = self._compute_fairness_metrics(results)
            metrics.update(fairness_metrics)
        
        return metrics
    
    def _compute_fairness_metrics(self, results: Dict) -> Dict:
        """Compute fairness metrics across demographic groups."""
        from fairlearn.metrics import (
            demographic_parity_difference,
            equalized_odds_difference
        )
        
        metrics = {}
        demographics = results["samples"][0]["demographics"].keys()
        
        for attr in demographics:
            groups = [sample["demographics"][attr] for sample in results["samples"]]
            
            # Convert to binary for fairness metrics
            y_true = np.array([sample["ground_truth"] for sample in results["samples"]])
            y_pred = np.array([sample["prediction"] for sample in results["samples"]])
            
            # Compute metrics for each class
            for i, class_name in enumerate(self.config["datasets"][results["dataset"]]["classes"]):
                # Demographic parity difference
                dpd = demographic_parity_difference(
                    y_true[:, i],
                    y_pred[:, i],
                    sensitive_features=groups
                )
                metrics[f"{attr}_{class_name}_demographic_parity_diff"] = dpd
                
                # Equalized odds difference
                eod = equalized_odds_difference(
                    y_true[:, i],
                    y_pred[:, i],
                    sensitive_features=groups
                )
                metrics[f"{attr}_{class_name}_equalized_odds_diff"] = eod
        
        return metrics
    
    def _save_results(self, results: Dict, model_type: str):
        """Save evaluation results to disk."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.results_dir / f"{model_type}_{timestamp}.json"
        
        with open(filename, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to {filename}")
    
    def run_benchmark(self, models: List[str] = None):
        """Run the benchmark for the specified models."""
        if models is None:
            models = list(SUPPORTED_MODELS.keys())
        
        all_metrics = {}
        
        for model_type in models:
            if model_type.lower() not in SUPPORTED_MODELS:
                print(f"Skipping unsupported model: {model_type}")
                continue
                
            metrics = self._evaluate_model(model_type.lower())
            if metrics is not None:
                all_metrics[model_type] = metrics
        
        # Save summary of all metrics
        if all_metrics:
            summary_file = self.results_dir / "benchmark_summary.json"
            with open(summary_file, "w") as f:
                json.dump({
                    "timestamp": datetime.now().isoformat(),
                    "metrics": all_metrics
                }, f, indent=2)
            print(f"\nBenchmark summary saved to {summary_file}")

class CheXpertDataset(Dataset):
    def __init__(self, root_dir, csv="val_metadata.csv", image_root=None, transform=None):
        self.root_dir = Path(root_dir)
        self.csv_path = self.root_dir / csv
        self.image_root = Path(image_root)
        self.transform = transform
        self.samples = self._load_samples()

    def _load_samples(self):
        df = pd.read_csv(self.csv_path)
        samples = []
        for _, row in df.iterrows():
            img_path = (self.image_root / str(row["Path"]).lstrip("/")).as_posix()
            # if not os.path.exists(img_path):
            #     # skip bad rows
            #     continue
            labels = [row[c] for c in ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"]]
            labels = [0 if pd.isna(x) or x < 0 else int(x) for x in labels]
            sample = {
                "image_path": str(img_path),
                "label": torch.tensor(labels).float(),
                "demographics": {
                    "gender": row.get("Sex", "Unknown"),
                    "age": str(row.get("Age", "Unknown")),
                    "race": row.get("Race", "Unknown"),
                }
            }
            samples.append(sample)
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample["image_path"]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return {
            "image": image,
            "label": sample["label"],
            "image_path": sample["image_path"],
            "demographics": sample["demographics"]
        }

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Benchmark LLMs on medical diagnosis tasks")
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/llm_benchmark.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=None,
        help="List of models to benchmark (default: all supported models)"
    )
    parser.add_argument(
    "--vision-models",
    type=str,
    nargs="+",
    default=None,
    help="List of vision models to evaluate"
    )
    parser.add_argument(
    "--text-models",
    type=str,
    nargs="+",
    default=None,
    help="List of text models to evaluate"
    )
    return parser.parse_args()


def save_results(results: Dict, output_path: str):
    """Save benchmark results to disk."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save full results
    with open(output_path.with_suffix('.json'), 'w') as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    
    # Save metrics as CSV
    metrics_df = pd.json_normalize(results["metrics"], sep='_')
    metrics_df.to_csv(output_path.with_suffix('.csv'), index=False)
    
    # Generate and save plots
    generate_plots(results, output_path)

def generate_plots(results: Dict, output_path: Path):
    """Generate and save evaluation plots."""
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams.update({'font.size': 12})
    
    # Create output directory
    plots_dir = output_path.parent / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    # Plot ROC curves for each class
    plot_roc_curves(results, plots_dir)
    
    # Plot fairness metrics
    plot_fairness_metrics(results, plots_dir)
    
    # Plot attention maps if available
    if any("attention_maps" in sample for sample in results["samples"][:5]):
        plot_attention_maps(results, plots_dir)

def plot_roc_curves(results: Dict, output_dir: Path):
    """Plot ROC curves for each class."""
    from sklearn.metrics import roc_curve, auc
    import matplotlib.pyplot as plt
    
    y_true = np.array([sample["ground_truth"] for sample in results["samples"]])
    y_probs = np.array([sample["confidence"] for sample in results["samples"]])
    
    plt.figure(figsize=(10, 8))
    
    for i, class_name in enumerate(results["dataset"]["classes"]):
        fpr, tpr, _ = roc_curve(y_true[:, i], y_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f'{class_name} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(output_dir / 'roc_curves.png', bbox_inches='tight', dpi=300)
    plt.close()

def plot_fairness_metrics(results: Dict, output_dir: Path):
    """Plot fairness metrics across demographic groups."""
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    if "demographics" not in results["samples"][0]:
        return
    
    # Extract fairness metrics
    metrics = []
    for sample in results["samples"]:
        for attr, group in sample["demographics"].items():
            metrics.append({
                "attribute": attr,
                "group": group,
                **{k: v for k, v in sample.items() if k not in ["demographics", "attention_maps"]}
            })
    
    if not metrics:
        return
    
    df = pd.DataFrame(metrics)
    
    # Plot performance by group
    for metric in ["auc", "f1", "precision", "recall"]:
        plt.figure(figsize=(12, 6))
        sns.barplot(
            data=df,
            x="attribute",
            y=metric,
            hue="group"
        )
        plt.title(f"{metric.upper()} by Demographic Group")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_dir / f'{metric}_by_group.png', dpi=300)
        plt.close()

def plot_attention_maps(results: Dict, output_dir: Path):
    """Plot attention maps for sample images."""
    import matplotlib.pyplot as plt
    from torchvision.transforms.functional import to_pil_image
    
    samples_with_attention = [s for s in results["samples"] if s.get("attention_maps") is not None][:5]
    
    for i, sample in enumerate(samples_with_attention):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # Original image
        img = to_pil_image(sample["image"].cpu())
        ax1.imshow(img, cmap='gray')
        ax1.set_title("Original Image")
        ax1.axis('off')
        
        # Attention map
        attention_map = sample["attention_maps"][0].mean(dim=0).cpu().numpy()  # Average over heads
        ax2.imshow(img, cmap='gray', alpha=0.5)
        im = ax2.imshow(attention_map, cmap='viridis', alpha=0.5)
        ax2.set_title("Attention Map")
        ax2.axis('off')
        
        plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.savefig(output_dir / f'attention_map_{i}.png', dpi=300, bbox_inches='tight')
        plt.close()

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types."""
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                          np.int16, np.int32, np.int64, np.uint8,
                          np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def main():
    """Main function to run the benchmark."""
    args = parse_args()
    
    # Load configuration
    try:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading config file: {str(e)}")
        return
    
    # Initialize benchmarker
    benchmark = MedicalImagingBenchmarker(args.config)
    
    # Get models to evaluate
    models_to_evaluate = []
    if args.vision_models:
        models_to_evaluate.extend([m for m in args.vision_models if m in SUPPORTED_MODELS])
    if args.text_models:
        models_to_evaluate.extend([m for m in args.text_models if m in SUPPORTED_MODELS])
    
    if not models_to_evaluate:
        print("No valid models specified. Use --vision-models and/or --text-models to specify models to evaluate.")
        return
    
    # Run evaluation for each model and dataset
    all_metrics = {}
    
    for model_name in models_to_evaluate:
        model_metrics = {}
        
        for dataset_name in config["datasets"]:
            try:
                print(f"\n{'='*80}")
                print(f"Evaluating {model_name} on {dataset_name}")
                print(f"{'='*80}")
                
                # Evaluate on test split
                metrics = benchmark.evaluate_model(
                    model_name=model_name,
                    dataset_name=dataset_name,
                    split="test"
                )
                
                if metrics:
                    model_metrics[dataset_name] = metrics
                    
                    # Print summary
                    print("\nMetrics:")
                    for k, v in metrics.items():
                        if isinstance(v, (int, float)):
                            print(f"{k}: {v:.4f}")
                    
                    # Save results
                    output_dir = Path(config["output"]["results_dir"]) / model_name / dataset_name
                    output_dir.mkdir(parents=True, exist_ok=True)
                    
                    results = {
                        "model": model_name,
                        "dataset": dataset_name,
                        "metrics": metrics,
                        "config": config
                    }
                    
                    save_results(
                        results,
                        output_dir / f"{model_name}_{dataset_name}_results"
                    )
                
            except Exception as e:
                print(f"Error evaluating {model_name} on {dataset_name}: {str(e)}")
                import traceback
                traceback.print_exc()
        
        if model_metrics:
            all_metrics[model_name] = model_metrics
    
    # Save combined results
    if all_metrics:
        output_path = Path(config["output"]["results_dir"]) / "benchmark_results.json"
        with open(output_path, 'w') as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "config": config,
                "metrics": all_metrics
            }, f, indent=2, cls=NumpyEncoder)
        
        print(f"\nBenchmark completed. Results saved to {output_path}")


if __name__ == "__main__":
    main()
