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
from fairtune.models.fair_lora import LoRAModel, LoRAConfig
from fairtune.models.adversarial_fairness import (
    AdversarialFairLoRAModel,
    AdversarialLoRAConfig,
)
)
from fairtune.models.multimodal_lora import (
    MultiModalLoRAModel,
    MultiModalLoRAConfig,
)
from fairtune.training.trainer import LoRATrainer
from fairtune.metrics.fairness import EquityScaledLoss, compute_metrics
from fairtune.data.datasets import get_dataset_loader

from transformers import AutoModel

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
    logger = logging.getLogger("logger")
    logger.setLevel(logging.INFO)

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
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

    base_model = AutoModel.from_pretrained(config["model"]["base_model"])

    # for models with no image decoding attached
    if fairness_approach == "lora":
    # change nomes of first argument from fairness
        lora_config = LoRAConfig(
            demographic_groups=demographic_groups,
            r=config["fairness"]["lora_r"],
            alpha=config["fairness"]["lora_alpha"],
            dropout=config["fairness"]["lora_dropout"],
            target_modules=config["fairness"]["target_modules"],
        )

        model = LoRAModel(base_model, lora_config)


    elif fairness_approach == "multimodal":
        # Multi-Modal LoRA
        mm_config = MultiModalLoRAConfig(
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

            text_encoder = AutoModel.from_pretrained(
                config["fairness"]["text_encoder_name"]
            )

        model = MultiModalLoRAModel(base_model, mm_config, text_encoder)

    else:
        raise ValueError(f"Unknown approach: {approach}")

    # Move model to device
    model = model.to(device)

    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )

    # Create loss function
    criterion = Loss()

    # Create trainer based on approach
    if approach == "lora":
        from fairtune.training.trainer import LoRATrainer

        trainer = LoRATrainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )

    elif approach == "multimodal":
        from fairtune.models.multimodal_lora import MultiModalLoRATrainer

        trainer = MultiModalLoRATrainer(
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

    # Create model and trainer, need to edit the config
    logger.info(
        f"Creating model with approach: {config['fairness']['approach']}"
    )
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
                logger.info(
                    f"Epoch {epoch + 1}/{num_epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {metrics['loss']:.4f}"
                )

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
                loss = trainer.criterion(
                    outputs["outputs"], outputs["labels"], outputs["demographics"]
                )
                val_losses.append(loss.item())

        val_loss = sum(val_losses) / len(val_losses)

        # Compute metrics
        val_outputs = torch.cat(val_outputs)
        val_labels = torch.cat(val_labels)

        metrics = compute_metrics()

        # Log metrics
        logger.info(
            f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
        )
        logger.info(f"Metrics: {metrics}")

        if WANDB_AVAILABLE and config["logging"].get("use_wandb", False):
            wandb_metrics = {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "epoch": epoch + 1,
            }

            # Add fairness metrics
            for metric_name, metric_value in metrics.items():
                wandb_metrics[f"{metric_name}"] = metric_value

            wandb.log(wandb_metrics)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss

            # Save model
            model_path = os.path.join(
                save_dir, f"best_model_{config['fairness']['approach']}.pt"
            )
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": trainer.optimizer.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "metrics": metrics,
                    "config": config,
                },
                model_path,
            )

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
    test_metrics = compute_metrics(
        outputs=test_outputs,
        labels=test_labels,
        demographics=test_demographics,
        demographic_groups=demographic_groups,
    )

    logger.info(f"Test Metrics: {test_metrics}")

    if WANDB_AVAILABLE and config["logging"].get("use_wandb", False):
        # Log test metrics
        test_wandb_metrics = {}

        for metric_name, metric_value in test_metrics.items():
            test_wandb_metrics[f"test/{metric_name}"] = metric_value

        wandb.log(test_wandb_metrics)

        # Finish wandb run
        wandb.finish()


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Integrated Fairness Training")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to configuration file"
    )
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
