"""
Fair LoRA (FairTune) Implementation

This module implements the Fair LoRA approach for fine-tuning LLMs with demographic-specific
adaptations to improve both performance and fairness across different demographic groups.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from transformers import PreTrainedModel
import numpy as np


@dataclass
class FairLoRAConfig:
    """Configuration for Fair LoRA fine-tuning."""
    
    r: int = 8  # Rank of the low-rank matrices
    alpha: float = 16.0  # Scaling factor
    dropout: float = 0.05  # Dropout probability
    target_modules: List[str] = None  # List of module names to apply Fair LoRA
    bias: str = "none"  # Bias type: "none", "all", or "lora_only"
    demographic_groups: List[str] = None  # List of demographic group identifiers
    equity_weight: float = 0.5  # Weight for the equity component in the loss
    modules_to_save: List[str] = None  # List of modules to save fully


class FairLoRALayer(nn.Module):
    """
    Implementation of Fair LoRA layer with demographic-specific scaling matrices.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
        demographic_groups: List[str] = None,
    ):
        """
        Initialize Fair LoRA layer.
        
        Args:
            in_features: Input dimension
            out_features: Output dimension
            r: Rank of the low-rank matrices
            alpha: Scaling factor
            dropout: Dropout probability
            demographic_groups: List of demographic group identifiers
        """
        super().__init__()
        
        self.r = r
        self.alpha = alpha
        self.dropout = dropout
        self.demographic_groups = demographic_groups or ["default"]
        
        # Shared low-rank matrices across all demographic groups
        self.lora_A = nn.Parameter(torch.zeros((r, in_features)))
        self.lora_B = nn.Parameter(torch.zeros((out_features, r)))
        
        # Demographic-specific scaling matrices
        self.scaling_matrices = nn.ParameterDict({
            group: nn.Parameter(torch.eye(r))
            for group in self.demographic_groups
        })
        
        self.dropout_layer = nn.Dropout(dropout)
        
        # Initialize weights
        self.reset_parameters()
        
        # Scaling factor for training
        self.scaling = alpha / r
    
    def reset_parameters(self):
        """Initialize the parameters using Kaiming uniform initialization."""
        nn.init.kaiming_uniform_(self.lora_A, a=np.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
        # Initialize scaling matrices as identity
        for group in self.demographic_groups:
            nn.init.eye_(self.scaling_matrices[group])
    
    def forward(self, x: torch.Tensor, demographic: str = "default") -> torch.Tensor:
        """
        Forward pass with demographic-specific adaptation.
        
        Args:
            x: Input tensor
            demographic: Demographic group identifier
        
        Returns:
            Adapted output tensor
        """
        if demographic not in self.scaling_matrices:
            demographic = "default"
            
        # Apply shared low-rank transformation
        result = self.dropout_layer(x) @ self.lora_A.t()
        
        # Apply demographic-specific scaling
        result = result @ self.scaling_matrices[demographic]
        
        # Apply second shared low-rank transformation
        result = result @ self.lora_B.t()
        
        # Apply scaling factor
        return result * self.scaling


class FairLoRAModel(nn.Module):
    """
    Wrapper for a pre-trained model with Fair LoRA adaptation.
    """
    
    def __init__(
        self,
        base_model: PreTrainedModel,
        config: FairLoRAConfig,
    ):
        """
        Initialize Fair LoRA model.
        
        Args:
            base_model: Pre-trained model to adapt
            config: Fair LoRA configuration
        """
        super().__init__()
        
        self.base_model = base_model
        self.config = config
        
        # Keep track of adapted layers
        self.fair_lora_layers = {}
        
        # Apply Fair LoRA to target modules
        self._add_fair_lora_layers()
    
    def _add_fair_lora_layers(self):
        """Add Fair LoRA layers to target modules."""
        for name, module in self.base_model.named_modules():
            if any(target in name for target in self.config.target_modules):
                if isinstance(module, nn.Linear):
                    self._add_fair_lora_to_linear(module, name)
    
    def _add_fair_lora_to_linear(self, layer: nn.Linear, name: str):
        """Add Fair LoRA adaptation to a linear layer."""
        in_features, out_features = layer.in_features, layer.out_features
        
        # Create Fair LoRA layer
        fair_lora = FairLoRALayer(
            in_features=in_features,
            out_features=out_features,
            r=self.config.r,
            alpha=self.config.alpha,
            dropout=self.config.dropout,
            demographic_groups=self.config.demographic_groups,
        )
        
        # Store reference to original forward method
        layer._original_forward = layer.forward
        
        # Store reference to Fair LoRA layer
        self.fair_lora_layers[name] = fair_lora
        
        # Create new forward method with Fair LoRA adaptation
        def forward_with_fair_lora(x, demographic=None):
            original_output = layer._original_forward(x)
            fair_lora_output = self.fair_lora_layers[name](x, demographic)
            return original_output + fair_lora_output
        
        # Replace forward method
        layer.forward = forward_with_fair_lora
    
    def forward(
        self,
        input_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        demographic_groups: List[str] = None,
        **kwargs
    ):
        """
        Forward pass with demographic-specific adaptation.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            demographic_groups: List of demographic group identifiers for each sample
            **kwargs: Additional arguments for the base model
        
        Returns:
            Model outputs
        """
        # Set demographic for each layer
        if demographic_groups:
            for name, module in self.base_model.named_modules():
                if name in self.fair_lora_layers:
                    # Store demographic for this forward pass
                    module.demographic = demographic_groups[0]  # Using first demographic for simplicity
        
        # Forward pass through base model
        return self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )
    
    def get_demographic_parameters(self, demographic: str = None):
        """
        Get parameters specific to a demographic group.
        
        Args:
            demographic: Demographic group identifier
        
        Returns:
            List of parameters
        """
        params = []
        for layer in self.fair_lora_layers.values():
            if demographic is None:
                # Get shared parameters
                params.extend([layer.lora_A, layer.lora_B])
            else:
                # Get demographic-specific parameters
                if demographic in layer.scaling_matrices:
                    params.append(layer.scaling_matrices[demographic])
        
        return params


def prepare_fair_lora_config(
    r: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.05,
    target_modules: List[str] = None,
    bias: str = "none",
    demographic_groups: List[str] = None,
    equity_weight: float = 0.5,
    modules_to_save: List[str] = None,
) -> FairLoRAConfig:
    """
    Prepare Fair LoRA configuration.
    
    Args:
        r: Rank of the low-rank matrices
        alpha: Scaling factor
        dropout: Dropout probability
        target_modules: List of module names to apply Fair LoRA
        bias: Bias type: "none", "all", or "lora_only"
        demographic_groups: List of demographic group identifiers
        equity_weight: Weight for the equity component in the loss
        modules_to_save: List of modules to save fully
    
    Returns:
        Fair LoRA configuration
    """
    config = FairLoRAConfig(
        r=r,
        alpha=alpha,
        dropout=dropout,
        target_modules=target_modules or ["query", "key", "value", "dense"],
        bias=bias,
        demographic_groups=demographic_groups or ["default"],
        equity_weight=equity_weight,
        modules_to_save=modules_to_save,
    )
    
    return config
