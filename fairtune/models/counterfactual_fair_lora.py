#!/usr/bin/env python
"""
Counterfactual Fair LoRA Module

This module implements an innovative counterfactual fairness approach for Fair LoRA,
which ensures that model predictions remain invariant under counterfactual interventions
on protected attributes. This provides stronger fairness guarantees than standard
approaches by addressing causal relationships in the data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Callable
import numpy as np
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader

from fairtune.models.fair_lora import FairLoRAConfig, FairLoRAModel


@dataclass
class CounterfactualFairLoRAConfig(FairLoRAConfig):
    """Configuration for Counterfactual Fair LoRA."""
    
    # Counterfactual specific parameters
    counterfactual_weight: float = 1.0
    causal_strength: float = 0.5
    num_counterfactuals: int = 3
    counterfactual_augmentation: bool = True
    consistency_regularization: bool = True
    invariance_type: str = "prediction"  # "prediction", "representation", or "both"


class CounterfactualGenerator:
    """
    Counterfactual Generator for medical images.
    
    This class generates counterfactual examples by modifying demographic-specific
    features in the input images, allowing the model to learn invariant representations.
    """
    
    def __init__(
        self,
        demographic_groups: List[str],
        causal_strength: float = 0.5,
        feature_maps: Optional[Dict[str, torch.Tensor]] = None,
    ):
        """
        Initialize the counterfactual generator.
        
        Args:
            demographic_groups: List of demographic groups
            causal_strength: Strength of the causal intervention (0-1)
            feature_maps: Pre-computed feature maps for each demographic group
        """
        self.demographic_groups = demographic_groups
        self.causal_strength = causal_strength
        self.feature_maps = feature_maps or {}
        
        # Initialize demographic feature extractors
        self.feature_extractors = {}
    
    def fit(self, dataset: Dataset, feature_extractor: Callable):
        """
        Fit the counterfactual generator on a dataset.
        
        Args:
            dataset: Dataset containing images with demographic information
            feature_extractor: Function to extract features from images
        """
        # Group images by demographic
        demographic_images = {group: [] for group in self.demographic_groups}
        
        dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
        
        # Collect images by demographic
        for batch in dataloader:
            images = batch["image"]
            demographics = batch["demographic"]
            
            for i, demo in enumerate(demographics):
                if demo in demographic_images:
                    demographic_images[demo].append(images[i])
        
        # Extract features for each demographic group
        for group, images in demographic_images.items():
            if not images:
                continue
                
            # Stack images
            stacked_images = torch.stack(images)
            
            # Extract features
            with torch.no_grad():
                features = feature_extractor(stacked_images)
            
            # Compute mean feature map
            mean_features = features.mean(dim=0)
            
            # Store feature map
            self.feature_maps[group] = mean_features
    
    def generate_counterfactual(
        self,
        image: torch.Tensor,
        source_demographic: str,
        target_demographic: str,
    ) -> torch.Tensor:
        """
        Generate a counterfactual image by transforming from source to target demographic.
        
        Args:
            image: Input image
            source_demographic: Source demographic group
            target_demographic: Target demographic group
            
        Returns:
            Counterfactual image
        """
        if source_demographic == target_demographic:
            return image.clone()
        
        if source_demographic not in self.feature_maps or target_demographic not in self.feature_maps:
            return image.clone()
        
        # Get feature maps
        source_features = self.feature_maps[source_demographic]
        target_features = self.feature_maps[target_demographic]
        
        # Compute feature difference
        feature_diff = target_features - source_features
        
        # Apply causal intervention
        counterfactual = image + self.causal_strength * feature_diff
        
        # Ensure valid image range
        counterfactual = torch.clamp(counterfactual, 0, 1)
        
        return counterfactual
    
    def generate_all_counterfactuals(
        self,
        image: torch.Tensor,
        source_demographic: str,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate counterfactuals for all target demographics.
        
        Args:
            image: Input image
            source_demographic: Source demographic group
            
        Returns:
            Dictionary of counterfactual images for each target demographic
        """
        counterfactuals = {}
        
        for target_demographic in self.demographic_groups:
            if target_demographic != source_demographic:
                counterfactual = self.generate_counterfactual(
                    image=image,
                    source_demographic=source_demographic,
                    target_demographic=target_demographic,
                )
                
                counterfactuals[target_demographic] = counterfactual
        
        return counterfactuals


class CounterfactualConsistencyLoss(nn.Module):
    """
    Counterfactual Consistency Loss.
    
    This loss enforces that model predictions remain consistent across
    counterfactual examples, ensuring fairness through invariance.
    """
    
    def __init__(
        self,
        base_criterion: nn.Module,
        counterfactual_weight: float = 1.0,
        invariance_type: str = "prediction",
    ):
        """
        Initialize the counterfactual consistency loss.
        
        Args:
            base_criterion: Base criterion for the main task
            counterfactual_weight: Weight for the counterfactual consistency term
            invariance_type: Type of invariance to enforce ("prediction", "representation", or "both")
        """
        super().__init__()
        
        self.base_criterion = base_criterion
        self.counterfactual_weight = counterfactual_weight
        self.invariance_type = invariance_type
    
    def forward(
        self,
        outputs: torch.Tensor,
        labels: torch.Tensor,
        counterfactual_outputs: Dict[str, torch.Tensor],
        representations: Optional[torch.Tensor] = None,
        counterfactual_representations: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            outputs: Model outputs for original images
            labels: Ground truth labels
            counterfactual_outputs: Model outputs for counterfactual images
            representations: Model representations for original images
            counterfactual_representations: Model representations for counterfactual images
            
        Returns:
            Total loss
        """
        # Base loss
        base_loss = self.base_criterion(outputs, labels)
        
        # Counterfactual consistency loss
        consistency_loss = 0.0
        
        if self.invariance_type in ["prediction", "both"]:
            # Prediction consistency
            for cf_output in counterfactual_outputs.values():
                consistency_loss += F.mse_loss(outputs, cf_output)
        
        if self.invariance_type in ["representation", "both"] and representations is not None:
            # Representation consistency
            for cf_repr in counterfactual_representations.values():
                consistency_loss += F.mse_loss(representations, cf_repr)
        
        # Normalize by number of counterfactuals
        if counterfactual_outputs:
            consistency_loss /= len(counterfactual_outputs)
        
        # Total loss
        total_loss = base_loss + self.counterfactual_weight * consistency_loss
        
        return total_loss


class CounterfactualFairLoRAModel(FairLoRAModel):
    """
    Counterfactual Fair LoRA Model.
    
    This model extends Fair LoRA with counterfactual fairness guarantees,
    ensuring that predictions are invariant to demographic attributes.
    """
    
    def __init__(
        self,
        base_model: nn.Module,
        config: CounterfactualFairLoRAConfig,
        counterfactual_generator: Optional[CounterfactualGenerator] = None,
    ):
        """
        Initialize the counterfactual Fair LoRA model.
        
        Args:
            base_model: Base model to adapt
            config: Configuration for Counterfactual Fair LoRA
            counterfactual_generator: Generator for counterfactual examples
        """
        super().__init__(base_model, config)
        
        self.config = config
        self.counterfactual_generator = counterfactual_generator
        self.return_representations = False
    
    def set_return_representations(self, return_representations: bool):
        """Set whether to return representations."""
        self.return_representations = return_representations
    
    def forward(
        self,
        x: torch.Tensor,
        demographics: Optional[List[str]] = None,
        counterfactuals: Optional[Dict[str, torch.Tensor]] = None,
        return_counterfactuals: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor
            demographics: List of demographic groups for each sample
            counterfactuals: Pre-generated counterfactual images
            return_counterfactuals: Whether to return counterfactual outputs
            
        Returns:
            Model outputs, optionally with counterfactual outputs
        """
        # Standard forward pass
        outputs = super().forward(x, demographics)
        
        # Extract representations if needed
        representations = None
        if self.return_representations:
            # This is a simplified example; in practice, you would extract
            # representations from a specific layer of the model
            representations = outputs  # Placeholder
        
        # If not returning counterfactuals, return standard outputs
        if not return_counterfactuals:
            if self.return_representations:
                return outputs, representations
            return outputs
        
        # Generate counterfactuals if not provided
        if counterfactuals is None and self.counterfactual_generator is not None and demographics is not None:
            counterfactuals = {}
            
            for i, demo in enumerate(demographics):
                # Generate counterfactuals for this sample
                sample_counterfactuals = self.counterfactual_generator.generate_all_counterfactuals(
                    image=x[i].unsqueeze(0),
                    source_demographic=demo,
                )
                
                # Add to counterfactuals dictionary
                for target_demo, cf_image in sample_counterfactuals.items():
                    if target_demo not in counterfactuals:
                        counterfactuals[target_demo] = []
                    
                    counterfactuals[target_demo].append((i, cf_image))
        
        # Process counterfactuals
        counterfactual_outputs = {}
        counterfactual_representations = {}
        
        if counterfactuals:
            for target_demo, cf_samples in counterfactuals.items():
                # Skip if no counterfactuals for this demographic
                if not cf_samples:
                    continue
                
                # Extract indices and images
                indices = [idx for idx, _ in cf_samples]
                cf_images = torch.cat([img for _, img in cf_samples])
                
                # Create counterfactual demographics
                cf_demographics = [target_demo] * len(indices)
                
                # Forward pass for counterfactuals
                cf_output = super().forward(cf_images, cf_demographics)
                
                # Store outputs
                counterfactual_outputs[target_demo] = cf_output
                
                # Extract representations if needed
                if self.return_representations:
                    # This is a simplified example; in practice, you would extract
                    # representations from a specific layer of the model
                    counterfactual_representations[target_demo] = cf_output  # Placeholder
        
        # Return outputs and counterfactuals
        result = {
            "outputs": outputs,
            "counterfactual_outputs": counterfactual_outputs,
        }
        
        if self.return_representations:
            result["representations"] = representations
            result["counterfactual_representations"] = counterfactual_representations
        
        return result


class CounterfactualFairLoRATrainer:
    """
    Trainer for Counterfactual Fair LoRA models.
    
    This trainer implements the counterfactual fairness training procedure,
    including counterfactual generation and consistency regularization.
    """
    
    def __init__(
        self,
        model: CounterfactualFairLoRAModel,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        counterfactual_criterion: CounterfactualConsistencyLoss,
        device: torch.device,
        demographic_key: str = "demographic",
        label_key: str = "labels",
        counterfactual_augmentation: bool = True,
        num_counterfactuals: int = 3,
    ):
        """
        Initialize the counterfactual Fair LoRA trainer.
        
        Args:
            model: Counterfactual Fair LoRA model
            optimizer: Optimizer for the model
            criterion: Base criterion for the main task
            counterfactual_criterion: Criterion for counterfactual consistency
            device: Device to use
            demographic_key: Key for demographic information in batch
            label_key: Key for labels in batch
            counterfactual_augmentation: Whether to use counterfactual augmentation
            num_counterfactuals: Number of counterfactuals to generate per sample
        """
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.counterfactual_criterion = counterfactual_criterion
        self.device = device
        self.demographic_key = demographic_key
        self.label_key = label_key
        self.counterfactual_augmentation = counterfactual_augmentation
        self.num_counterfactuals = num_counterfactuals
    
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
        
        # Enable returning representations for counterfactual consistency
        self.model.set_return_representations(True)
        
        # Forward pass with counterfactuals
        outputs = self.model(
            x=images,
            demographics=demographics,
            return_counterfactuals=True,
        )
        
        # Extract outputs and counterfactuals
        model_outputs = outputs["outputs"]
        counterfactual_outputs = outputs["counterfactual_outputs"]
        representations = outputs.get("representations")
        counterfactual_representations = outputs.get("counterfactual_representations")
        
        # Compute loss
        loss = self.counterfactual_criterion(
            outputs=model_outputs,
            labels=labels,
            counterfactual_outputs=counterfactual_outputs,
            representations=representations,
            counterfactual_representations=counterfactual_representations,
        )
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Disable returning representations for inference
        self.model.set_return_representations(False)
        
        # Return metrics
        metrics = {
            "loss": loss.item(),
        }
        
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
        
        # Disable returning representations for inference
        self.model.set_return_representations(False)
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(images, demographics)
        
        return {
            "outputs": outputs,
            "labels": labels,
            "demographics": demographics,
        }


def create_counterfactual_fair_lora_trainer(
    model: CounterfactualFairLoRAModel,
    optimizer: torch.optim.Optimizer,
    base_criterion: nn.Module,
    device: torch.device,
    counterfactual_weight: float = 1.0,
    invariance_type: str = "both",
    demographic_key: str = "demographic",
    label_key: str = "labels",
    counterfactual_augmentation: bool = True,
    num_counterfactuals: int = 3,
) -> CounterfactualFairLoRATrainer:
    """
    Create a counterfactual Fair LoRA trainer.
    
    Args:
        model: Counterfactual Fair LoRA model
        optimizer: Optimizer for the model
        base_criterion: Base criterion for the main task
        device: Device to use
        counterfactual_weight: Weight for the counterfactual consistency term
        invariance_type: Type of invariance to enforce
        demographic_key: Key for demographic information in batch
        label_key: Key for labels in batch
        counterfactual_augmentation: Whether to use counterfactual augmentation
        num_counterfactuals: Number of counterfactuals to generate per sample
        
    Returns:
        Counterfactual Fair LoRA trainer
    """
    # Create counterfactual consistency loss
    counterfactual_criterion = CounterfactualConsistencyLoss(
        base_criterion=base_criterion,
        counterfactual_weight=counterfactual_weight,
        invariance_type=invariance_type,
    )
    
    # Create trainer
    trainer = CounterfactualFairLoRATrainer(
        model=model,
        optimizer=optimizer,
        criterion=base_criterion,
        counterfactual_criterion=counterfactual_criterion,
        device=device,
        demographic_key=demographic_key,
        label_key=label_key,
        counterfactual_augmentation=counterfactual_augmentation,
        num_counterfactuals=num_counterfactuals,
    )
    
    return trainer
