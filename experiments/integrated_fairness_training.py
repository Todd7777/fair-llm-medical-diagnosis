#!/usr/bin/env python
"""
Integrated Fairness Training Script

This script implements a comprehensive training pipeline that integrates multiple
fairness approaches (Fair LoRA, Adversarial Fairness, Counterfactual Fairness,
and Multi-Modal Fairness) into a unified framework for medical image diagnosis.
"""

import os
import sys
import yaml
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import logging
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import project modules
from fairtune.models.fair_lora import FairLoRAModel, FairLoRAConfig
from fairtune.models.adversarial_fairness import AdversarialFairLoRAModel, AdversarialFairLoRAConfig
from fairtune.models.counterfactual_fair_lora import CounterfactualFairLoRAModel, CounterfactualFairLoRAConfig, CounterfactualGenerator
from fairtune.models.multimodal_fair_lora import MultiModalFairLoRAModel, MultiModalFairLoRAConfig
from fairtune.training.trainer import FairLoRATrainer
from fairtune.metrics.fairness import EquityScaledLoss, compute_fairness_metrics
from fairtune.data.datasets import get_dataset_loader

# Optional: Import wandb for experiment tracking
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def setup_logging(log_dir: str) -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        log_dir: Directory to save logs
        
    Returns:
        Logger instance
    """
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"training_{timestamp}.log")
    
    # Configure logger
    logger = logging.getLogger("integrated_fairness")
    logger.setLevel(logging.INFO)
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def load_config(config_path: str) -> Dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    return config


def get_model_and_trainer(
    config: Dict,
    device: torch.device,
    demographic_groups: List[str],
    dataset_loader: Optional[DataLoader] = None,
) -> Tuple[nn.Module, object]:
    """
    Create model and trainer based on configuration.
    
    Args:
        config: Configuration dictionary
        device: Device to use
        demographic_groups: List of demographic groups
        dataset_loader: Dataset loader for counterfactual generation
        
    Returns:
        Tuple of (model, trainer)
    """
    # Get base model
    from transformers import AutoModel
    base_model = AutoModel.from_pretrained(config["model"]["base_model"])
    
    # Determine fairness approach
    fairness_approach = config["fairness"]["approach"]
    
    if fairness_approach == "fair_lora":
        # Standard Fair LoRA
        lora_config = FairLoRAConfig(
            demographic_groups=demographic_groups,
            r=config["fairness"]["lora_r"],
            alpha=config["fairness"]["lora_alpha"],
            dropout=config["fairness"]["lora_dropout"],
            target_modules=config["fairness"]["target_modules"],
        )
        
        model = FairLoRAModel(base_model, lora_config)
        
    elif fairness_approach == "adversarial":
        # Adversarial Fair LoRA
        adv_config = AdversarialFairLoRAConfig(
            demographic_groups=demographic_groups,
            r=config["fairness"]["lora_r"],
            alpha=config["fairness"]["lora_alpha"],
            dropout=config["fairness"]["lora_dropout"],
            target_modules=config["fairness"]["target_modules"],
            adv_hidden_dims=config["fairness"]["adv_hidden_dims"],
            lambda_adv=config["fairness"]["lambda_adv"],
            grad_reversal_strength=config["fairness"]["grad_reversal_strength"],
        )
        
        model = AdversarialFairLoRAModel(base_model, adv_config)
        
    elif fairness_approach == "counterfactual":
        # Counterfactual Fair LoRA
        cf_config = CounterfactualFairLoRAConfig(
            demographic_groups=demographic_groups,
            r=config["fairness"]["lora_r"],
            alpha=config["fairness"]["lora_alpha"],
            dropout=config["fairness"]["lora_dropout"],
            target_modules=config["fairness"]["target_modules"],
            counterfactual_weight=config["fairness"]["counterfactual_weight"],
            causal_strength=config["fairness"]["causal_strength"],
            num_counterfactuals=config["fairness"]["num_counterfactuals"],
        )
        
        # Create counterfactual generator if dataset loader is provided
        counterfactual_generator = None
        if dataset_loader is not None:
            counterfactual_generator = CounterfactualGenerator(
                demographic_groups=demographic_groups,
                causal_strength=config["fairness"]["causal_strength"],
            )
            
            # Extract feature extractor from base model
            def feature_extractor(x):
                return base_model(x).last_hidden_state[:, 0]
            
            # Fit counterfactual generator
            counterfactual_generator.fit(dataset_loader.dataset, feature_extractor)
        
        model = CounterfactualFairLoRAModel(base_model, cf_config, counterfactual_generator)
        
    elif fairness_approach == "multimodal":
        # Multi-Modal Fair LoRA
        mm_config = MultiModalFairLoRAConfig(
            demographic_groups=demographic_groups,
            r=config["fairness"]["lora_r"],
            alpha=config["fairness"]["lora_alpha"],
            dropout=config["fairness"]["lora_dropout"],
            target_modules=config["fairness"]["target_modules"],
            temperature=config["fairness"]["temperature"],
            projection_dim=config["fairness"]["projection_dim"],
            modality_fusion=config["fairness"]["modality_fusion"],
            contrastive_weight=config["fairness"]["contrastive_weight"],
            use_clinical_text=config["fairness"]["use_clinical_text"],
            text_encoder_name=config["fairness"]["text_encoder_name"],
        )
        
        # Create text encoder if using clinical text
        text_encoder = None
        if config["fairness"]["use_clinical_text"]:
            from transformers import AutoModel
            text_encoder = AutoModel.from_pretrained(config["fairness"]["text_encoder_name"])
        
        model = MultiModalFairLoRAModel(base_model, mm_config, text_encoder)
        
    else:
        raise ValueError(f"Unknown fairness approach: {fairness_approach}")
    
    # Move model to device
    model = model.to(device)
    
    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    
    # Create loss function
    criterion = EquityScaledLoss(
        base_criterion=nn.CrossEntropyLoss(),
        demographic_groups=demographic_groups,
        scaling_factor=config["fairness"]["equity_scaling_factor"],
    )
    
    # Create trainer based on fairness approach
    if fairness_approach == "fair_lora":
        from fairtune.training.trainer import FairLoRATrainer
        trainer = FairLoRATrainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )
    
    elif fairness_approach == "adversarial":
        from fairtune.models.adversarial_fairness import AdversarialFairLoRATrainer
        trainer = AdversarialFairLoRATrainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            lambda_scheduler=None,  # Use default scheduler
        )
    
    elif fairness_approach == "counterfactual":
        from fairtune.models.counterfactual_fair_lora import CounterfactualFairLoRATrainer, CounterfactualConsistencyLoss
        
        # Create counterfactual consistency loss
        counterfactual_criterion = CounterfactualConsistencyLoss(
            base_criterion=criterion,
            counterfactual_weight=config["fairness"]["counterfactual_weight"],
            invariance_type=config["fairness"]["invariance_type"],
        )
        
        trainer = CounterfactualFairLoRATrainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            counterfactual_criterion=counterfactual_criterion,
            device=device,
            counterfactual_augmentation=config["fairness"]["counterfactual_augmentation"],
            num_counterfactuals=config["fairness"]["num_counterfactuals"],
        )
    
    elif fairness_approach == "multimodal":
        from fairtune.models.multimodal_fair_lora import MultiModalFairLoRATrainer
        trainer = MultiModalFairLoRATrainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            contrastive_weight=config["fairness"]["contrastive_weight"],
        )
    
    return model, trainer


def train(config: Dict, logger: logging.Logger) -> None:
    """
    Train the model based on configuration.
    
    Args:
        config: Configuration dictionary
        logger: Logger instance
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Set random seed for reproducibility
    seed = config["training"].get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Load dataset
    dataset_name = config["dataset"]["name"]
    dataset_path = config["dataset"]["path"]
    batch_size = config["training"]["batch_size"]
    
    logger.info(f"Loading dataset: {dataset_name} from {dataset_path}")
    
    train_loader, val_loader, test_loader, demographic_groups = get_dataset_loader(
        dataset_name=dataset_name,
        dataset_path=dataset_path,
        batch_size=batch_size,
    )
    
    logger.info(f"Demographic groups: {demographic_groups}")
    
    # Create model and trainer
    logger.info(f"Creating model with fairness approach: {config['fairness']['approach']}")
    model, trainer = get_model_and_trainer(
        config=config,
        device=device,
        demographic_groups=demographic_groups,
        dataset_loader=train_loader,
    )
    
    # Initialize wandb if available
    if WANDB_AVAILABLE and config["logging"].get("use_wandb", False):
        wandb.init(
            project=config["logging"]["wandb_project"],
            name=config["logging"]["wandb_run_name"],
            config=config,
        )
        
        # Watch model
        wandb.watch(model)
    
    # Training loop
    num_epochs = config["training"]["num_epochs"]
    save_dir = config["logging"]["save_dir"]
    os.makedirs(save_dir, exist_ok=True)
    
    logger.info(f"Starting training for {num_epochs} epochs")
    
    best_val_loss = float("inf")
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_losses = []
        
        for batch_idx, batch in enumerate(train_loader):
            metrics = trainer.train_step(batch)
            train_losses.append(metrics["loss"])
            
            if batch_idx % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {metrics['loss']:.4f}")
        
        train_loss = sum(train_losses) / len(train_losses)
        
        # Validation
        model.eval()
        val_losses = []
        val_outputs = []
        val_labels = []
        val_demographics = []
        
        with torch.no_grad():
            for batch in val_loader:
                outputs = trainer.eval_step(batch)
                
                val_outputs.append(outputs["outputs"].cpu())
                val_labels.append(outputs["labels"].cpu())
                val_demographics.extend(outputs["demographics"])
                
                # Compute validation loss
                loss = trainer.criterion(outputs["outputs"], outputs["labels"], outputs["demographics"])
                val_losses.append(loss.item())
        
        val_loss = sum(val_losses) / len(val_losses)
        
        # Compute fairness metrics
        val_outputs = torch.cat(val_outputs)
        val_labels = torch.cat(val_labels)
        
        fairness_metrics = compute_fairness_metrics(
            outputs=val_outputs,
            labels=val_labels,
            demographics=val_demographics,
            demographic_groups=demographic_groups,
        )
        
        # Log metrics
        logger.info(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        logger.info(f"Fairness Metrics: {fairness_metrics}")
        
        if WANDB_AVAILABLE and config["logging"].get("use_wandb", False):
            wandb_metrics = {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "epoch": epoch + 1,
            }
            
            # Add fairness metrics
            for metric_name, metric_value in fairness_metrics.items():
                wandb_metrics[f"fairness/{metric_name}"] = metric_value
            
            wandb.log(wandb_metrics)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            
            # Save model
            model_path = os.path.join(save_dir, f"best_model_{config['fairness']['approach']}.pt")
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": trainer.optimizer.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
                "fairness_metrics": fairness_metrics,
                "config": config,
            }, model_path)
            
            logger.info(f"Saved best model to {model_path}")
    
    # Final evaluation on test set
    logger.info("Evaluating on test set")
    
    model.eval()
    test_outputs = []
    test_labels = []
    test_demographics = []
    
    with torch.no_grad():
        for batch in test_loader:
            outputs = trainer.eval_step(batch)
            
            test_outputs.append(outputs["outputs"].cpu())
            test_labels.append(outputs["labels"].cpu())
            test_demographics.extend(outputs["demographics"])
    
    test_outputs = torch.cat(test_outputs)
    test_labels = torch.cat(test_labels)
    
    # Compute fairness metrics
    test_fairness_metrics = compute_fairness_metrics(
        outputs=test_outputs,
        labels=test_labels,
        demographics=test_demographics,
        demographic_groups=demographic_groups,
    )
    
    logger.info(f"Test Fairness Metrics: {test_fairness_metrics}")
    
    if WANDB_AVAILABLE and config["logging"].get("use_wandb", False):
        # Log test metrics
        test_wandb_metrics = {}
        
        for metric_name, metric_value in test_fairness_metrics.items():
            test_wandb_metrics[f"test/fairness/{metric_name}"] = metric_value
        
        wandb.log(test_wandb_metrics)
        
        # Finish wandb run
        wandb.finish()


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Integrated Fairness Training")
    parser.add_argument("--config", type=str, required=True, help="Path to configuration file")
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set up logging
    logger = setup_logging(config["logging"]["log_dir"])
    
    # Log configuration
    logger.info(f"Configuration: {config}")
    
    # Train model
    train(config, logger)


if __name__ == "__main__":
    main()
