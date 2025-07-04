"""
Model Adapters

This module provides adapters for different model types (LLMs, vision models, etc.)
to standardize their interface for medical image diagnosis tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Any
from transformers import PreTrainedModel, AutoModel, AutoModelForImageClassification
from transformers import CLIPModel, CLIPProcessor
import numpy as np


class ModelAdapter(nn.Module):
    """Base class for model adapters."""
    
    def __init__(self, model_name: str):
        """
        Initialize model adapter.
        
        Args:
            model_name: Name or path of the model
        """
        super().__init__()
        self.model_name = model_name
    
    def forward(
        self,
        images: torch.Tensor,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            images: Input images
            **kwargs: Additional arguments
        
        Returns:
            Dictionary of model outputs
        """
        raise NotImplementedError("Subclasses must implement forward")
    
    def get_embeddings(
        self,
        images: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        Get embeddings for input images.
        
        Args:
            images: Input images
            **kwargs: Additional arguments
        
        Returns:
            Image embeddings
        """
        raise NotImplementedError("Subclasses must implement get_embeddings")


class SpecializedMedicalModelAdapter(ModelAdapter):
    """Adapter for specialized medical AI models."""
    
    def __init__(
        self,
        model_name: str,
        num_classes: int = 2,
        pretrained: bool = True,
    ):
        """
        Initialize specialized medical model adapter.
        
        Args:
            model_name: Name or path of the model
            num_classes: Number of output classes
            pretrained: Whether to use pretrained weights
        """
        super().__init__(model_name)
        
        # Load model
        self.model = AutoModelForImageClassification.from_pretrained(
            model_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )
    
    def forward(
        self,
        images: torch.Tensor,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            images: Input images
            **kwargs: Additional arguments
        
        Returns:
            Dictionary of model outputs
        """
        outputs = self.model(pixel_values=images)
        
        return {
            "logits": outputs.logits,
            "embeddings": self.get_embeddings(images),
        }
    
    def get_embeddings(
        self,
        images: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        Get embeddings for input images.
        
        Args:
            images: Input images
            **kwargs: Additional arguments
        
        Returns:
            Image embeddings
        """
        # Extract features before the classification head
        with torch.no_grad():
            features = self.model.get_image_features(pixel_values=images)
        
        return features


class LLMVisionAdapter(ModelAdapter):
    """Adapter for LLMs with vision capabilities."""
    
    def __init__(
        self,
        model_name: str,
        num_classes: int = 2,
        use_clip: bool = True,
    ):
        """
        Initialize LLM vision adapter.
        
        Args:
            model_name: Name or path of the model
            num_classes: Number of output classes
            use_clip: Whether to use CLIP for image embeddings
        """
        super().__init__(model_name)
        
        # Load CLIP model for image embeddings
        if use_clip:
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            
            # Freeze CLIP parameters
            for param in self.clip_model.parameters():
                param.requires_grad = False
        else:
            self.clip_model = None
            self.clip_processor = None
        
        # Load LLM
        self.llm = AutoModel.from_pretrained(model_name)
        
        # Add classification head
        self.classifier = nn.Linear(self.llm.config.hidden_size, num_classes)
    
    def forward(
        self,
        images: torch.Tensor,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            images: Input images
            **kwargs: Additional arguments
        
        Returns:
            Dictionary of model outputs
        """
        # Get image embeddings
        image_embeddings = self.get_embeddings(images)
        
        # Process with LLM
        llm_outputs = self.llm(
            inputs_embeds=image_embeddings.unsqueeze(1),
            output_hidden_states=True,
        )
        
        # Get last hidden state
        last_hidden_state = llm_outputs.last_hidden_state[:, 0, :]
        
        # Apply classification head
        logits = self.classifier(last_hidden_state)
        
        return {
            "logits": logits,
            "embeddings": last_hidden_state,
        }
    
    def get_embeddings(
        self,
        images: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        Get embeddings for input images.
        
        Args:
            images: Input images
            **kwargs: Additional arguments
        
        Returns:
            Image embeddings
        """
        if self.clip_model is not None:
            # Process images with CLIP
            with torch.no_grad():
                # Convert images to format expected by CLIP
                if images.shape[1] == 3:  # Check if images are in [B, C, H, W] format
                    # CLIP expects pixel values in range [0, 255]
                    if images.max() <= 1.0:
                        images = images * 255.0
                    
                    # Process with CLIP
                    clip_outputs = self.clip_model.get_image_features(pixel_values=images)
                    
                    # Project to LLM embedding space
                    embeddings = self.llm.get_input_embeddings()(clip_outputs)
                else:
                    raise ValueError("Images must be in [B, C, H, W] format with C=3")
        else:
            # Use a simple projection if CLIP is not available
            embeddings = self.llm.get_input_embeddings()(images.flatten(1))
        
        return embeddings


class FairTuneModelAdapter(ModelAdapter):
    """Adapter for models fine-tuned with Fair LoRA."""
    
    def __init__(
        self,
        base_model: ModelAdapter,
        fair_lora_config: Dict[str, Any],
    ):
        """
        Initialize Fair LoRA model adapter.
        
        Args:
            base_model: Base model adapter
            fair_lora_config: Fair LoRA configuration
        """
        super().__init__(base_model.model_name + "-fair-lora")
        
        # Store base model
        self.base_model = base_model
        
        # Import Fair LoRA
        from .fair_lora import FairLoRAModel, prepare_fair_lora_config
        
        # Create Fair LoRA config
        self.fair_lora_config = prepare_fair_lora_config(**fair_lora_config)
        
        # Create Fair LoRA model
        if hasattr(base_model, "model"):
            self.model = FairLoRAModel(base_model.model, self.fair_lora_config)
        elif hasattr(base_model, "llm"):
            self.model = FairLoRAModel(base_model.llm, self.fair_lora_config)
        else:
            raise ValueError("Base model must have 'model' or 'llm' attribute")
    
    def forward(
        self,
        images: torch.Tensor,
        demographics: List[str] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with demographic-specific adaptation.
        
        Args:
            images: Input images
            demographics: List of demographic group identifiers for each sample
            **kwargs: Additional arguments
        
        Returns:
            Dictionary of model outputs
        """
        # Get base model outputs
        if hasattr(self.base_model, "model"):
            # For specialized medical models
            outputs = self.base_model.model(
                pixel_values=images,
                **kwargs
            )
            
            return {
                "logits": outputs.logits,
                "embeddings": self.get_embeddings(images),
            }
        elif hasattr(self.base_model, "llm"):
            # For LLM vision models
            image_embeddings = self.base_model.get_embeddings(images)
            
            # Process with Fair LoRA model
            llm_outputs = self.model(
                inputs_embeds=image_embeddings.unsqueeze(1),
                demographic_groups=demographics,
                output_hidden_states=True,
                **kwargs
            )
            
            # Get last hidden state
            last_hidden_state = llm_outputs.last_hidden_state[:, 0, :]
            
            # Apply classification head
            logits = self.base_model.classifier(last_hidden_state)
            
            return {
                "logits": logits,
                "embeddings": last_hidden_state,
            }
    
    def get_embeddings(
        self,
        images: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        Get embeddings for input images.
        
        Args:
            images: Input images
            **kwargs: Additional arguments
        
        Returns:
            Image embeddings
        """
        return self.base_model.get_embeddings(images, **kwargs)


def create_model_adapter(
    model_type: str,
    model_name: str,
    num_classes: int = 2,
    **kwargs
) -> ModelAdapter:
    """
    Create model adapter.
    
    Args:
        model_type: Type of model ("specialized" or "llm")
        model_name: Name or path of the model
        num_classes: Number of output classes
        **kwargs: Additional arguments for the model adapter
    
    Returns:
        Model adapter
    """
    if model_type == "specialized":
        return SpecializedMedicalModelAdapter(
            model_name=model_name,
            num_classes=num_classes,
            **kwargs
        )
    elif model_type == "llm":
        return LLMVisionAdapter(
            model_name=model_name,
            num_classes=num_classes,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
