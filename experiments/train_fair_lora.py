#!/usr/bin/env python
"""
Fair LoRA Training Script

This script fine-tunes general LLMs using the Fair LoRA (FairTune) approach to improve
their performance and fairness on medical image diagnosis tasks.
"""

import os
import argparse
import yaml
import torch
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple, Union
from torch.utils.data import DataLoader, random_split

# Add project root to path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.loaders.dataset_loaders import ChestXRayDataset, PathologyImageDataset, RetinalImageDataset, create_data_loaders
from data.processors.image_processors import get_chest_xray_transform, get_pathology_transform, get_retinal_transform
from fairtune.models.model_adapters import create_model_adapter, FairTuneModelAdapter
from fairtune.models.fair_lora import prepare_fair_lora_config
from fairtune.training.trainer import create_fair_lora_trainer
from fairtune.metrics.fairness import compute_fairness_metrics


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train models with Fair LoRA")
    
    parser.add_argument("--config", type=str, required=True, help="Path to configuration file")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save results")
    parser.add_argument("--use_wandb", action="store_true", help="Whether to use Weights & Biases for logging")
    parser.add_argument("--wandb_project", type=str, default="fair-llm-medical", help="Weights & Biases project name")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="Weights & Biases run name")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--equity_weight", type=float, default=0.5, help="Weight for equity component in loss")
    
    return parser.parse_args()


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    return config


def set_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_dataset(config, split="train"):
    """Load dataset based on configuration."""
    dataset_type = config["dataset"]["type"]
    data_dir = config["dataset"]["data_dir"]
    metadata_path = config["dataset"]["metadata_path"]
    
    if dataset_type == "chest_xray":
        transform = get_chest_xray_transform(
            target_size=(config["dataset"].get("image_size", 224), config["dataset"].get("image_size", 224)),
            is_training=split == "train",
        )
        dataset = ChestXRayDataset(
            data_dir=data_dir,
            metadata_path=metadata_path,
            transform=transform,
            split=split,
            demographic_key=config["dataset"].get("demographic_key", "demographic"),
            label_cols=config["dataset"].get("label_cols", ["Pneumonia"]),
        )
    elif dataset_type == "pathology":
        transform = get_pathology_transform(
            target_size=(config["dataset"].get("image_size", 224), config["dataset"].get("image_size", 224)),
            is_training=split == "train",
        )
        dataset = PathologyImageDataset(
            data_dir=data_dir,
            metadata_path=metadata_path,
            transform=transform,
            split=split,
            demographic_key=config["dataset"].get("demographic_key", "demographic"),
            label_col=config["dataset"].get("label_col", "malignant"),
        )
    elif dataset_type == "retinal":
        transform = get_retinal_transform(
            target_size=(config["dataset"].get("image_size", 224), config["dataset"].get("image_size", 224)),
            is_training=split == "train",
        )
        dataset = RetinalImageDataset(
            data_dir=data_dir,
            metadata_path=metadata_path,
            transform=transform,
            split=split,
            demographic_key=config["dataset"].get("demographic_key", "demographic"),
            label_col=config["dataset"].get("label_col", "diabetic_retinopathy_grade"),
        )
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    return dataset


def create_fair_lora_model(base_model, config, device):
    """Create Fair LoRA model from base model."""
    # Get demographic groups from dataset
    demographic_groups = config["dataset"].get("demographic_groups", ["default"])
    
    # Create Fair LoRA config
    fair_lora_config = {
        "r": config["fair_lora"].get("r", 8),
        "alpha": config["fair_lora"].get("alpha", 16.0),
        "dropout": config["fair_lora"].get("dropout", 0.05),
        "target_modules": config["fair_lora"].get("target_modules", ["query", "key", "value", "dense"]),
        "bias": config["fair_lora"].get("bias", "none"),
        "demographic_groups": demographic_groups,
        "equity_weight": config["fair_lora"].get("equity_weight", 0.5),
    }
    
    # Create Fair LoRA model
    fair_lora_model = FairTuneModelAdapter(
        base_model=base_model,
        fair_lora_config=fair_lora_config,
    ).to(device)
    
    return fair_lora_model


def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Load configuration
    config = load_config(args.config)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load datasets
    train_dataset = load_dataset(config, split="train")
    val_dataset = load_dataset(config, split="val")
    
    # Create data loaders
    train_dataloader = create_data_loaders(
        dataset=train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
    )
    
    val_dataloader = create_data_loaders(
        dataset=val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )
    
    # Load base model
    base_model_name = config["model"]["name"]
    base_model_type = config["model"]["type"]
    
    base_model = create_model_adapter(
        model_type=base_model_type,
        model_name=base_model_name,
        num_classes=config["dataset"].get("num_classes", 2),
    ).to(args.device)
    
    # Create Fair LoRA model
    fair_lora_model = create_fair_lora_model(
        base_model=base_model,
        config=config,
        device=args.device,
    )
    
    # Create trainer
    trainer = create_fair_lora_trainer(
        model=fair_lora_model,
        train_dataloader=train_dataloader,
        eval_dataloader=val_dataloader,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        equity_weight=args.equity_weight,
        demographic_key=config["dataset"].get("demographic_key", "demographic"),
        label_key="labels",
        log_dir=os.path.join(args.output_dir, "logs"),
        checkpoint_dir=os.path.join(args.output_dir, "checkpoints"),
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
    )
    
    # Train model
    print("Training Fair LoRA model...")
    trainer.train()
    
    # Save final model
    final_model_path = os.path.join(args.output_dir, "final_model")
    os.makedirs(final_model_path, exist_ok=True)
    
    # Save model configuration
    with open(os.path.join(final_model_path, "config.yaml"), "w") as f:
        yaml.dump(config, f)
    
    print(f"Training complete. Model saved to {final_model_path}")


if __name__ == "__main__":
    main()
