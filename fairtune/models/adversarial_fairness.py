#!/usr/bin/env python
"""
Adversarial Fairness Module for Fair LoRA

This module implements an innovative adversarial fairness approach that combines
with Fair LoRA to create a more robust fairness-aware fine-tuning framework.
The adversarial component ensures that demographic information cannot be extracted
from the model's internal representations, enforcing stronger fairness guarantees.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import numpy as np


class GradientReversalLayer(torch.autograd.Function):
    """
    Gradient Reversal Layer for adversarial training.
    
    In the forward pass, this layer acts as an identity function.
    In the backward pass, it multiplies gradients by -lambda, effectively
    reversing the gradient direction for adversarial training.
    """
    
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()
    
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


class DemographicAdversary(nn.Module):
    """
    Demographic Adversary Network.
    
    This network tries to predict demographic information from the model's
    internal representations. Through adversarial training, the main model
    learns to remove demographic information from its representations.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_demographics: int,
        dropout: float = 0.2,
    ):
        """
        Initialize the demographic adversary network.
        
        Args:
            input_dim: Dimension of the input representation
            hidden_dim: Dimension of the hidden layer
            num_demographics: Number of demographic groups to predict
            dropout: Dropout probability
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_demographics = num_demographics
        
        # Network architecture
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_demographics),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the adversary network.
        
        Args:
            x: Input representation [batch_size, input_dim]
            
        Returns:
            Demographic logits [batch_size, num_demographics]
        """
        return self.network(x)


class AdversarialFairnessWrapper(nn.Module):
    """
    Adversarial Fairness Wrapper for models.
    
    This wrapper adds an adversarial fairness component to any model,
    ensuring that the model's internal representations do not contain
    demographic information.
    """
    
    def __init__(
        self,
        model: nn.Module,
        representation_dim: int,
        demographic_groups: List[str],
        hidden_dim: int = 128,
        lambda_: float = 1.0,
        extraction_layer: Optional[str] = None,
    ):
        """
        Initialize the adversarial fairness wrapper.
        
        Args:
            model: Base model to wrap
            representation_dim: Dimension of the representation to extract
            demographic_groups: List of demographic groups
            hidden_dim: Hidden dimension for the adversary network
            lambda_: Weight for the gradient reversal
            extraction_layer: Name of the layer to extract representations from
        """
        super().__init__()
        
        self.model = model
        self.representation_dim = representation_dim
        self.demographic_groups = demographic_groups
        self.num_demographics = len(demographic_groups)
        self.lambda_ = lambda_
        self.extraction_layer = extraction_layer
        
        # Create demographic adversary
        self.demographic_adversary = DemographicAdversary(
            input_dim=representation_dim,
            hidden_dim=hidden_dim,
            num_demographics=self.num_demographics,
            dropout=0.2,
        )
        
        # Create demographic group to index mapping
        self.demographic_to_idx = {group: i for i, group in enumerate(demographic_groups)}
        
        # Hook for extracting intermediate representations
        self.extracted_representation = None
        if extraction_layer is not None:
            self._register_extraction_hook()
    
    def _register_extraction_hook(self):
        """Register a hook to extract intermediate representations."""
        def get_extraction_hook(name):
            def hook(module, input, output):
                self.extracted_representation = output
            return hook
        
        # Find the target layer
        for name, module in self.model.named_modules():
            if name == self.extraction_layer:
                module.register_forward_hook(get_extraction_hook(name))
                break
    
    def _apply_gradient_reversal(self, x: torch.Tensor) -> torch.Tensor:
        """Apply gradient reversal to the input."""
        return GradientReversalLayer.apply(x, self.lambda_)
    
    def _get_representation(self, x: torch.Tensor) -> torch.Tensor:
        """Get the representation for the adversary."""
        if self.extraction_layer is not None:
            # If extraction layer is specified, use the extracted representation
            if self.extracted_representation is None:
                raise ValueError("No representation extracted. Make sure the extraction layer exists.")
            
            representation = self.extracted_representation
            
            # Reset extracted representation for the next forward pass
            self.extracted_representation = None
        else:
            # Otherwise, use the model's forward pass
            representation = self.model(x, return_representation=True)
        
        return representation
    
    def forward(
        self,
        x: torch.Tensor,
        demographics: Optional[List[str]] = None,
        training: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through the model and adversary.
        
        Args:
            x: Input tensor
            demographics: List of demographic groups for each sample
            training: Whether in training mode
            
        Returns:
            Tuple of (model_output, adversary_loss)
        """
        # Forward pass through the base model
        model_output = self.model(x)
        
        # If not in training mode or no demographics provided, return only model output
        if not training or demographics is None:
            return model_output, None
        
        # Get representation for the adversary
        representation = self._get_representation(x)
        
        # Apply gradient reversal
        reversed_representation = self._apply_gradient_reversal(representation)
        
        # Forward pass through the adversary
        demographic_logits = self.demographic_adversary(reversed_representation)
        
        # Convert demographics to indices
        demographic_indices = torch.tensor(
            [self.demographic_to_idx.get(d, 0) for d in demographics],
            device=x.device,
        )
        
        # Compute adversary loss
        adversary_loss = F.cross_entropy(demographic_logits, demographic_indices)
        
        return model_output, adversary_loss


class AdversarialFairLoRATrainer:
    """
    Trainer for Adversarial Fair LoRA models.
    
    This trainer combines Fair LoRA with adversarial fairness training,
    creating a more robust fairness-aware fine-tuning approach.
    """
    
    def __init__(
        self,
        model: nn.Module,
        adversary_wrapper: AdversarialFairnessWrapper,
        optimizer: torch.optim.Optimizer,
        adversary_optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        demographic_key: str = "demographic",
        label_key: str = "labels",
        lambda_scheduler: Optional[callable] = None,
    ):
        """
        Initialize the adversarial Fair LoRA trainer.
        
        Args:
            model: Fair LoRA model
            adversary_wrapper: Adversarial fairness wrapper
            optimizer: Optimizer for the main model
            adversary_optimizer: Optimizer for the adversary
            criterion: Loss function
            device: Device to use
            demographic_key: Key for demographic information in batch
            label_key: Key for labels in batch
            lambda_scheduler: Scheduler for the lambda parameter
        """
        self.model = model
        self.adversary_wrapper = adversary_wrapper
        self.optimizer = optimizer
        self.adversary_optimizer = adversary_optimizer
        self.criterion = criterion
        self.device = device
        self.demographic_key = demographic_key
        self.label_key = label_key
        self.lambda_scheduler = lambda_scheduler
    
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
        
        # Step 1: Train the adversary
        self.adversary_optimizer.zero_grad()
        
        # Forward pass through the model and adversary
        _, adversary_loss = self.adversary_wrapper(
            x=images,
            demographics=demographics,
            training=True,
        )
        
        # Backward pass for the adversary
        if adversary_loss is not None:
            adversary_loss.backward()
            self.adversary_optimizer.step()
        
        # Step 2: Train the main model
        self.optimizer.zero_grad()
        
        # Forward pass through the model
        outputs = self.model(images, demographics)
        
        # Compute main loss
        main_loss = self.criterion(outputs, labels, demographics)
        
        # Forward pass through the adversary wrapper
        _, adversary_loss = self.adversary_wrapper(
            x=images,
            demographics=demographics,
            training=True,
        )
        
        # Combine losses
        total_loss = main_loss
        if adversary_loss is not None:
            total_loss += adversary_loss
        
        # Backward pass for the main model
        total_loss.backward()
        self.optimizer.step()
        
        # Update lambda if scheduler is provided
        if self.lambda_scheduler is not None:
            self.adversary_wrapper.lambda_ = self.lambda_scheduler(self.adversary_wrapper.lambda_)
        
        # Return metrics
        metrics = {
            "main_loss": main_loss.item(),
            "total_loss": total_loss.item(),
        }
        
        if adversary_loss is not None:
            metrics["adversary_loss"] = adversary_loss.item()
        
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
        
        # Forward pass through the model
        with torch.no_grad():
            outputs = self.model(images, demographics)
        
        return {
            "outputs": outputs,
            "labels": labels,
            "demographics": demographics,
        }


class LambdaScheduler:
    """
    Scheduler for the lambda parameter in adversarial training.
    
    This scheduler gradually increases the lambda parameter during training,
    allowing the model to first learn the main task before focusing on fairness.
    """
    
    def __init__(
        self,
        initial_lambda: float = 0.0,
        final_lambda: float = 1.0,
        warmup_epochs: int = 1,
        total_epochs: int = 10,
        steps_per_epoch: int = 100,
    ):
        """
        Initialize the lambda scheduler.
        
        Args:
            initial_lambda: Initial lambda value
            final_lambda: Final lambda value
            warmup_epochs: Number of epochs for warmup
            total_epochs: Total number of epochs
            steps_per_epoch: Number of steps per epoch
        """
        self.initial_lambda = initial_lambda
        self.final_lambda = final_lambda
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.steps_per_epoch = steps_per_epoch
        
        self.current_step = 0
        self.warmup_steps = warmup_epochs * steps_per_epoch
        self.total_steps = total_epochs * steps_per_epoch
    
    def step(self) -> float:
        """
        Update the lambda value.
        
        Returns:
            Current lambda value
        """
        self.current_step += 1
        
        if self.current_step < self.warmup_steps:
            # Linear warmup
            lambda_value = self.initial_lambda + (self.final_lambda - self.initial_lambda) * (
                self.current_step / self.warmup_steps
            )
        else:
            # Constant lambda after warmup
            lambda_value = self.final_lambda
        
        return lambda_value
    
    def __call__(self, current_lambda: float) -> float:
        """
        Update and return the lambda value.
        
        Args:
            current_lambda: Current lambda value
            
        Returns:
            Updated lambda value
        """
        return self.step()


def create_adversarial_fair_lora_trainer(
    model: nn.Module,
    representation_dim: int,
    demographic_groups: List[str],
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    extraction_layer: Optional[str] = None,
    hidden_dim: int = 128,
    initial_lambda: float = 0.0,
    final_lambda: float = 1.0,
    warmup_epochs: int = 1,
    total_epochs: int = 10,
    steps_per_epoch: int = 100,
    demographic_key: str = "demographic",
    label_key: str = "labels",
) -> AdversarialFairLoRATrainer:
    """
    Create an adversarial Fair LoRA trainer.
    
    Args:
        model: Fair LoRA model
        representation_dim: Dimension of the representation to extract
        demographic_groups: List of demographic groups
        optimizer: Optimizer for the main model
        criterion: Loss function
        device: Device to use
        extraction_layer: Name of the layer to extract representations from
        hidden_dim: Hidden dimension for the adversary network
        initial_lambda: Initial lambda value
        final_lambda: Final lambda value
        warmup_epochs: Number of epochs for warmup
        total_epochs: Total number of epochs
        steps_per_epoch: Number of steps per epoch
        demographic_key: Key for demographic information in batch
        label_key: Key for labels in batch
        
    Returns:
        Adversarial Fair LoRA trainer
    """
    # Create adversarial wrapper
    adversary_wrapper = AdversarialFairnessWrapper(
        model=model,
        representation_dim=representation_dim,
        demographic_groups=demographic_groups,
        hidden_dim=hidden_dim,
        lambda_=initial_lambda,
        extraction_layer=extraction_layer,
    ).to(device)
    
    # Create adversary optimizer
    adversary_optimizer = torch.optim.Adam(
        adversary_wrapper.demographic_adversary.parameters(),
        lr=1e-4,
    )
    
    # Create lambda scheduler
    lambda_scheduler = LambdaScheduler(
        initial_lambda=initial_lambda,
        final_lambda=final_lambda,
        warmup_epochs=warmup_epochs,
        total_epochs=total_epochs,
        steps_per_epoch=steps_per_epoch,
    )
    
    # Create trainer
    trainer = AdversarialFairLoRATrainer(
        model=model,
        adversary_wrapper=adversary_wrapper,
        optimizer=optimizer,
        adversary_optimizer=adversary_optimizer,
        criterion=criterion,
        device=device,
        demographic_key=demographic_key,
        label_key=label_key,
        lambda_scheduler=lambda_scheduler,
    )
    
    return trainer
