#!/usr/bin/env python
"""
Multi-Modal Fair LoRA Module

This module implements an innovative multi-modal contrastive learning approach
for Fair LoRA, which enables better integration of medical images with clinical text
data while maintaining fairness across demographic groups.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from dataclasses import dataclass

from fairtune.models.fair_lora import FairLoRAConfig, FairLoRAModel


@dataclass
class MultiModalFairLoRAConfig(FairLoRAConfig):
    """Configuration for Multi-Modal Fair LoRA."""
    
    # Multi-modal specific parameters
    temperature: float = 0.07
    projection_dim: int = 256
    modality_fusion: str = "attention"  # "attention", "concat", or "gated"
    contrastive_weight: float = 0.5
    use_clinical_text: bool = True
    text_encoder_name: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
    shared_projection: bool = False


class MultiModalProjection(nn.Module):
    """
    Multi-Modal Projection Module.
    
    Projects features from different modalities into a shared embedding space
    for contrastive learning.
    """
    
    def __init__(
        self,
        image_dim: int,
        text_dim: int,
        projection_dim: int,
        shared: bool = False,
    ):
        """
        Initialize the multi-modal projection module.
        
        Args:
            image_dim: Dimension of image features
            text_dim: Dimension of text features
            projection_dim: Dimension of the projection space
            shared: Whether to use shared projection for both modalities
        """
        super().__init__()
        
        self.shared = shared
        
        if shared:
            # Shared projection
            self.projection = nn.Sequential(
                nn.Linear(max(image_dim, text_dim), projection_dim * 2),
                nn.LayerNorm(projection_dim * 2),
                nn.GELU(),
                nn.Linear(projection_dim * 2, projection_dim),
                nn.LayerNorm(projection_dim),
            )
            
            # Adapters for different input dimensions
            self.image_adapter = nn.Linear(image_dim, max(image_dim, text_dim)) if image_dim != max(image_dim, text_dim) else nn.Identity()
            self.text_adapter = nn.Linear(text_dim, max(image_dim, text_dim)) if text_dim != max(image_dim, text_dim) else nn.Identity()
        else:
            # Separate projections
            self.image_projection = nn.Sequential(
                nn.Linear(image_dim, projection_dim * 2),
                nn.LayerNorm(projection_dim * 2),
                nn.GELU(),
                nn.Linear(projection_dim * 2, projection_dim),
                nn.LayerNorm(projection_dim),
            )
            
            self.text_projection = nn.Sequential(
                nn.Linear(text_dim, projection_dim * 2),
                nn.LayerNorm(projection_dim * 2),
                nn.GELU(),
                nn.Linear(projection_dim * 2, projection_dim),
                nn.LayerNorm(projection_dim),
            )
    
    def forward_image(self, image_features: torch.Tensor) -> torch.Tensor:
        """Project image features."""
        if self.shared:
            return self.projection(self.image_adapter(image_features))
        else:
            return self.image_projection(image_features)
    
    def forward_text(self, text_features: torch.Tensor) -> torch.Tensor:
        """Project text features."""
        if self.shared:
            return self.projection(self.text_adapter(text_features))
        else:
            return self.text_projection(text_features)


class ModalityFusion(nn.Module):
    """
    Modality Fusion Module.
    
    Fuses features from different modalities for downstream tasks.
    """
    
    def __init__(
        self,
        image_dim: int,
        text_dim: int,
        output_dim: int,
        fusion_type: str = "attention",
    ):
        """
        Initialize the modality fusion module.
        
        Args:
            image_dim: Dimension of image features
            text_dim: Dimension of text features
            output_dim: Dimension of the output features
            fusion_type: Type of fusion ("attention", "concat", or "gated")
        """
        super().__init__()
        
        self.fusion_type = fusion_type
        
        if fusion_type == "attention":
            # Cross-attention fusion
            self.query_proj = nn.Linear(image_dim, output_dim)
            self.key_proj = nn.Linear(text_dim, output_dim)
            self.value_proj = nn.Linear(text_dim, output_dim)
            
            self.attention = nn.MultiheadAttention(
                embed_dim=output_dim,
                num_heads=8,
                batch_first=True,
            )
            
            self.output_proj = nn.Linear(output_dim + image_dim, output_dim)
        
        elif fusion_type == "concat":
            # Concatenation fusion
            self.image_proj = nn.Linear(image_dim, output_dim // 2)
            self.text_proj = nn.Linear(text_dim, output_dim // 2)
            self.output_proj = nn.Linear(output_dim, output_dim)
        
        elif fusion_type == "gated":
            # Gated fusion
            self.image_proj = nn.Linear(image_dim, output_dim)
            self.text_proj = nn.Linear(text_dim, output_dim)
            self.gate = nn.Linear(image_dim + text_dim, output_dim)
        
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")
    
    def forward(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass through the fusion module.
        
        Args:
            image_features: Image features [batch_size, image_dim]
            text_features: Text features [batch_size, text_dim]
            
        Returns:
            Fused features [batch_size, output_dim]
        """
        if self.fusion_type == "attention":
            # Cross-attention fusion
            query = self.query_proj(image_features).unsqueeze(1)  # [batch_size, 1, output_dim]
            key = self.key_proj(text_features).unsqueeze(1)  # [batch_size, 1, output_dim]
            value = self.value_proj(text_features).unsqueeze(1)  # [batch_size, 1, output_dim]
            
            attn_output, _ = self.attention(query, key, value)
            attn_output = attn_output.squeeze(1)  # [batch_size, output_dim]
            
            # Concatenate with original image features
            concat_features = torch.cat([attn_output, image_features], dim=1)
            fused_features = self.output_proj(concat_features)
        
        elif self.fusion_type == "concat":
            # Concatenation fusion
            image_proj = self.image_proj(image_features)
            text_proj = self.text_proj(text_features)
            
            concat_features = torch.cat([image_proj, text_proj], dim=1)
            fused_features = self.output_proj(concat_features)
        
        elif self.fusion_type == "gated":
            # Gated fusion
            image_proj = self.image_proj(image_features)
            text_proj = self.text_proj(text_features)
            
            concat_features = torch.cat([image_features, text_features], dim=1)
            gate = torch.sigmoid(self.gate(concat_features))
            
            fused_features = gate * image_proj + (1 - gate) * text_proj
        
        return fused_features


class ContrastiveLoss(nn.Module):
    """
    Contrastive Loss for multi-modal learning.
    
    Implements InfoNCE loss to align features from different modalities.
    """
    
    def __init__(self, temperature: float = 0.07):
        """
        Initialize the contrastive loss.
        
        Args:
            temperature: Temperature parameter for the softmax
        """
        super().__init__()
        
        self.temperature = temperature
    
    def forward(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            image_features: Image features [batch_size, feature_dim]
            text_features: Text features [batch_size, feature_dim]
            
        Returns:
            Contrastive loss
        """
        # Normalize features
        image_features = F.normalize(image_features, dim=1)
        text_features = F.normalize(text_features, dim=1)
        
        # Compute similarity matrix
        logits = torch.matmul(image_features, text_features.t()) / self.temperature
        
        # Labels are the diagonal elements (positive pairs)
        labels = torch.arange(logits.size(0), device=logits.device)
        
        # Compute loss (symmetric)
        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.t(), labels)
        
        return (loss_i2t + loss_t2i) / 2.0


class FairContrastiveLoss(nn.Module):
    """
    Fair Contrastive Loss for multi-modal learning.
    
    Extends contrastive learning with fairness constraints to ensure
    that the learned representations are fair across demographic groups.
    """
    
    def __init__(
        self,
        temperature: float = 0.07,
        fairness_weight: float = 0.5,
    ):
        """
        Initialize the fair contrastive loss.
        
        Args:
            temperature: Temperature parameter for the softmax
            fairness_weight: Weight for the fairness term
        """
        super().__init__()
        
        self.temperature = temperature
        self.fairness_weight = fairness_weight
        self.contrastive_loss = ContrastiveLoss(temperature)
    
    def forward(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
        demographics: List[str],
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            image_features: Image features [batch_size, feature_dim]
            text_features: Text features [batch_size, feature_dim]
            demographics: List of demographic groups for each sample
            
        Returns:
            Fair contrastive loss
        """
        # Standard contrastive loss
        contrastive_loss = self.contrastive_loss(image_features, text_features)
        
        # Group samples by demographics
        unique_demographics = list(set(demographics))
        demographic_indices = {demo: [] for demo in unique_demographics}
        
        for i, demo in enumerate(demographics):
            demographic_indices[demo].append(i)
        
        # Compute within-group contrastive loss
        within_group_loss = 0.0
        num_groups = 0
        
        for demo, indices in demographic_indices.items():
            if len(indices) < 2:
                continue
                
            # Extract features for this demographic group
            group_image_features = image_features[indices]
            group_text_features = text_features[indices]
            
            # Compute within-group contrastive loss
            group_loss = self.contrastive_loss(group_image_features, group_text_features)
            within_group_loss += group_loss
            num_groups += 1
        
        # Average within-group loss
        if num_groups > 0:
            within_group_loss /= num_groups
        
        # Compute between-group alignment
        between_group_loss = 0.0
        num_pairs = 0
        
        for i, demo_i in enumerate(unique_demographics):
            indices_i = demographic_indices[demo_i]
            if not indices_i:
                continue
                
            for j, demo_j in enumerate(unique_demographics[i+1:], i+1):
                indices_j = demographic_indices[demo_j]
                if not indices_j:
                    continue
                
                # Extract features for these demographic groups
                group_i_image_features = image_features[indices_i]
                group_i_text_features = text_features[indices_i]
                
                group_j_image_features = image_features[indices_j]
                group_j_text_features = text_features[indices_j]
                
                # Compute representation distance between groups
                i_centroids = torch.cat([
                    group_i_image_features.mean(dim=0, keepdim=True),
                    group_i_text_features.mean(dim=0, keepdim=True),
                ])
                
                j_centroids = torch.cat([
                    group_j_image_features.mean(dim=0, keepdim=True),
                    group_j_text_features.mean(dim=0, keepdim=True),
                ])
                
                # Normalize centroids
                i_centroids = F.normalize(i_centroids, dim=1)
                j_centroids = F.normalize(j_centroids, dim=1)
                
                # Compute distance
                distance = F.mse_loss(i_centroids, j_centroids)
                between_group_loss += distance
                num_pairs += 1
        
        # Average between-group loss
        if num_pairs > 0:
            between_group_loss /= num_pairs
        
        # Total loss: standard contrastive + fairness term
        # The fairness term encourages similar within-group alignment across groups
        # and minimizes representation differences between groups
        total_loss = contrastive_loss + self.fairness_weight * (within_group_loss + between_group_loss)
        
        return total_loss


class MultiModalFairLoRAModel(FairLoRAModel):
    """
    Multi-Modal Fair LoRA Model.
    
    Extends Fair LoRA with multi-modal contrastive learning capabilities,
    enabling better integration of medical images with clinical text data
    while maintaining fairness across demographic groups.
    """
    
    def __init__(
        self,
        base_model: nn.Module,
        config: MultiModalFairLoRAConfig,
        text_encoder: Optional[nn.Module] = None,
    ):
        """
        Initialize the multi-modal Fair LoRA model.
        
        Args:
            base_model: Base model to adapt
            config: Configuration for Multi-Modal Fair LoRA
            text_encoder: Text encoder model (optional)
        """
        super().__init__(base_model, config)
        
        self.config = config
        
        # Initialize text encoder if needed
        if config.use_clinical_text and text_encoder is None:
            from transformers import AutoModel
            text_encoder = AutoModel.from_pretrained(config.text_encoder_name)
        
        self.text_encoder = text_encoder
        
        # Get feature dimensions
        self.image_dim = self._get_image_feature_dim()
        self.text_dim = self._get_text_feature_dim()
        
        # Initialize multi-modal components
        self.projection = MultiModalProjection(
            image_dim=self.image_dim,
            text_dim=self.text_dim,
            projection_dim=config.projection_dim,
            shared=config.shared_projection,
        )
        
        self.fusion = ModalityFusion(
            image_dim=self.image_dim,
            text_dim=self.text_dim,
            output_dim=self.image_dim,
            fusion_type=config.modality_fusion,
        )
        
        # Initialize contrastive loss
        self.contrastive_loss = FairContrastiveLoss(
            temperature=config.temperature,
            fairness_weight=0.5,
        )
    
    def _get_image_feature_dim(self) -> int:
        """Get the dimension of image features."""
        # This is a placeholder; in practice, you would determine this
        # based on the base model architecture
        return 768
    
    def _get_text_feature_dim(self) -> int:
        """Get the dimension of text features."""
        if self.text_encoder is None:
            return 768  # Default dimension
        
        # This is a placeholder; in practice, you would determine this
        # based on the text encoder architecture
        return 768
    
    def encode_text(
        self,
        text_inputs: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Encode text inputs.
        
        Args:
            text_inputs: Text inputs for the text encoder
            
        Returns:
            Text features
        """
        if self.text_encoder is None:
            raise ValueError("Text encoder is not initialized")
        
        # Forward pass through the text encoder
        text_outputs = self.text_encoder(**text_inputs)
        
        # Extract features (usually the [CLS] token representation)
        text_features = text_outputs.last_hidden_state[:, 0]
        
        return text_features
    
    def forward(
        self,
        x: torch.Tensor,
        demographics: Optional[List[str]] = None,
        text_inputs: Optional[Dict[str, torch.Tensor]] = None,
        return_contrastive_loss: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through the model.
        
        Args:
            x: Input image tensor
            demographics: List of demographic groups for each sample
            text_inputs: Text inputs for the text encoder
            return_contrastive_loss: Whether to return contrastive loss
            
        Returns:
            Model outputs, optionally with contrastive loss
        """
        # Process image with Fair LoRA
        image_features = super().forward(x, demographics)
        
        # If no text inputs, return image features
        if text_inputs is None or not self.config.use_clinical_text:
            return image_features
        
        # Process text
        text_features = self.encode_text(text_inputs)
        
        # Project features to shared space
        projected_image = self.projection.forward_image(image_features)
        projected_text = self.projection.forward_text(text_features)
        
        # Compute contrastive loss if requested
        contrastive_loss = None
        if return_contrastive_loss:
            contrastive_loss = self.contrastive_loss(
                projected_image,
                projected_text,
                demographics,
            )
        
        # Fuse modalities
        fused_features = self.fusion(image_features, text_features)
        
        # Return outputs
        if return_contrastive_loss:
            return fused_features, contrastive_loss
        else:
            return fused_features


class MultiModalFairLoRATrainer:
    """
    Trainer for Multi-Modal Fair LoRA models.
    
    Implements the training procedure for multi-modal contrastive learning
    with fairness constraints.
    """
    
    def __init__(
        self,
        model: MultiModalFairLoRAModel,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        contrastive_weight: float = 0.5,
        demographic_key: str = "demographic",
        label_key: str = "labels",
        text_key: str = "text",
    ):
        """
        Initialize the multi-modal Fair LoRA trainer.
        
        Args:
            model: Multi-Modal Fair LoRA model
            optimizer: Optimizer for the model
            criterion: Loss function for the main task
            device: Device to use
            contrastive_weight: Weight for the contrastive loss
            demographic_key: Key for demographic information in batch
            label_key: Key for labels in batch
            text_key: Key for text inputs in batch
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.contrastive_weight = contrastive_weight
        self.demographic_key = demographic_key
        self.label_key = label_key
        self.text_key = text_key
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Perform a single training step.
        
        Args:
            batch: Batch of data
            
        Returns:
            Dictionary of metrics
        """
        # Extract data from batch
        images = batch["image"].to(self.device)
        labels = batch[self.label_key].to(self.device)
        demographics = batch[self.demographic_key]
        
        # Extract text inputs if available
        text_inputs = batch.get(self.text_key)
        if text_inputs is not None:
            text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
        
        # Forward pass
        outputs, contrastive_loss = self.model(
            x=images,
            demographics=demographics,
            text_inputs=text_inputs,
            return_contrastive_loss=True,
        )
        
        # Compute main loss
        main_loss = self.criterion(outputs, labels, demographics)
        
        # Combine losses
        total_loss = main_loss
        if contrastive_loss is not None:
            total_loss += self.contrastive_weight * contrastive_loss
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        # Return metrics
        metrics = {
            "main_loss": main_loss.item(),
            "total_loss": total_loss.item(),
        }
        
        if contrastive_loss is not None:
            metrics["contrastive_loss"] = contrastive_loss.item()
        
        return metrics
    
    def eval_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Perform a single evaluation step.
        
        Args:
            batch: Batch of data
            
        Returns:
            Dictionary of outputs
        """
        # Extract data from batch
        images = batch["image"].to(self.device)
        labels = batch[self.label_key].to(self.device)
        demographics = batch[self.demographic_key]
        
        # Extract text inputs if available
        text_inputs = batch.get(self.text_key)
        if text_inputs is not None:
            text_inputs = {k: v.to(self.device) for k, v in text_inputs.items()}
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(
                x=images,
                demographics=demographics,
                text_inputs=text_inputs,
                return_contrastive_loss=False,
            )
        
        return {
            "outputs": outputs,
            "labels": labels,
            "demographics": demographics,
        }


def create_multimodal_fair_lora_trainer(
    model: MultiModalFairLoRAModel,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    contrastive_weight: float = 0.5,
    demographic_key: str = "demographic",
    label_key: str = "labels",
    text_key: str = "text",
) -> MultiModalFairLoRATrainer:
    """
    Create a multi-modal Fair LoRA trainer.
    
    Args:
        model: Multi-Modal Fair LoRA model
        optimizer: Optimizer for the model
        criterion: Loss function for the main task
        device: Device to use
        contrastive_weight: Weight for the contrastive loss
        demographic_key: Key for demographic information in batch
        label_key: Key for labels in batch
        text_key: Key for text inputs in batch
        
    Returns:
        Multi-Modal Fair LoRA trainer
    """
    trainer = MultiModalFairLoRATrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        contrastive_weight=contrastive_weight,
        demographic_key=demographic_key,
        label_key=label_key,
        text_key=text_key,
    )
    
    return trainer
