#!/usr/bin/env python
"""
Model Evaluation Script

This script evaluates and compares the performance of general LLMs and specialized
medical AI models on medical image diagnosis tasks.
"""

import os
import argparse
import yaml
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union
from torch.utils.data import DataLoader
import wandb

# Add project root to path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.loaders.dataset_loaders import ChestXRayDataset, PathologyImageDataset, RetinalImageDataset, create_data_loaders
from data.processors.image_processors import get_chest_xray_transform, get_pathology_transform, get_retinal_transform
from fairtune.models.model_adapters import create_model_adapter
from fairtune.metrics.fairness import compute_fairness_metrics


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate models on medical image diagnosis tasks")
    
    parser.add_argument("--config", type=str, required=True, help="Path to configuration file")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save results")
    parser.add_argument("--use_wandb", action="store_true", help="Whether to use Weights & Biases for logging")
    parser.add_argument("--wandb_project", type=str, default="fair-llm-medical", help="Weights & Biases project name")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="Weights & Biases run name")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
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


def load_dataset(config, split="test"):
    """Load dataset based on configuration."""
    dataset_type = config["dataset"]["type"]
    data_dir = config["dataset"]["data_dir"]
    metadata_path = config["dataset"]["metadata_path"]
    
    if dataset_type == "chest_xray":
        transform = get_chest_xray_transform(
            target_size=(config["dataset"].get("image_size", 224), config["dataset"].get("image_size", 224)),
            is_training=False,
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
            is_training=False,
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
            is_training=False,
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


def load_models(config, device):
    """Load models based on configuration."""
    models = {}
    
    # Load specialized medical models
    for model_name, model_config in config["models"]["specialized"].items():
        models[model_name] = create_model_adapter(
            model_type="specialized",
            model_name=model_config["name"],
            num_classes=config["dataset"].get("num_classes", 2),
        ).to(device)
    
    # Load LLM models
    for model_name, model_config in config["models"]["llm"].items():
        models[model_name] = create_model_adapter(
            model_type="llm",
            model_name=model_config["name"],
            num_classes=config["dataset"].get("num_classes", 2),
            use_clip=model_config.get("use_clip", True),
        ).to(device)
    
    return models


def evaluate_model(model, dataloader, device, demographic_key="demographic"):
    """Evaluate model on a dataset."""
    model.eval()
    
    all_preds = []
    all_labels = []
    all_demographics = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            # Move batch to device
            batch_device = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            
            # Get model outputs
            outputs = model(batch_device["image"])
            
            # Get predictions
            preds = F.softmax(outputs["logits"], dim=-1)
            
            # Collect predictions, labels, and demographics
            all_preds.append(preds.cpu().numpy())
            all_labels.append(batch["labels"].cpu().numpy())
            all_demographics.extend(batch.get(demographic_key, ["unknown"] * len(batch["labels"])))
    
    # Concatenate predictions and labels
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    # Compute fairness metrics
    metrics = compute_fairness_metrics(
        predictions=all_preds,
        labels=all_labels,
        demographics=all_demographics,
    )
    
    return metrics


def plot_results(results, output_dir):
    """Plot evaluation results."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert results to DataFrame
    df = pd.DataFrame(results).T
    
    # Plot overall performance metrics
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x=df.index, y="overall_auc")
    plt.title("Overall AUC by Model")
    plt.xlabel("Model")
    plt.ylabel("AUC")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "overall_auc.png"))
    plt.close()
    
    # Plot equity-scaled AUC
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df, x=df.index, y="equity_scaled_auc")
    plt.title("Equity-Scaled AUC by Model")
    plt.xlabel("Model")
    plt.ylabel("Equity-Scaled AUC")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "equity_scaled_auc.png"))
    plt.close()
    
    # Plot fairness gaps
    fairness_gaps = ["accuracy_gap", "precision_gap", "recall_gap", "f1_gap", "auc_gap"]
    plt.figure(figsize=(12, 8))
    sns.heatmap(df[fairness_gaps], annot=True, cmap="YlGnBu")
    plt.title("Fairness Gaps by Model")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fairness_gaps.png"))
    plt.close()
    
    # Save results to CSV
    df.to_csv(os.path.join(output_dir, "results.csv"))


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
    
    # Initialize Weights & Biases
    if args.use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config={
                "dataset": config["dataset"],
                "models": config["models"],
                "batch_size": args.batch_size,
                "seed": args.seed,
            }
        )
    
    # Load dataset
    dataset = load_dataset(config)
    dataloader = create_data_loaders(
        dataset=dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
    )
    
    # Load models
    models = load_models(config, args.device)
    
    # Evaluate models
    results = {}
    for model_name, model in models.items():
        print(f"Evaluating {model_name}...")
        metrics = evaluate_model(
            model=model,
            dataloader=dataloader,
            device=args.device,
            demographic_key=config["dataset"].get("demographic_key", "demographic"),
        )
        results[model_name] = metrics
        
        # Log to Weights & Biases
        if args.use_wandb:
            wandb.log({f"{model_name}/{k}": v for k, v in metrics.items()})
    
    # Plot results
    plot_results(results, args.output_dir)
    
    # Print summary
    print("\nEvaluation Summary:")
    for model_name, metrics in results.items():
        print(f"\n{model_name}:")
        print(f"  Overall AUC: {metrics['overall_auc']:.4f}")
        print(f"  Equity-Scaled AUC: {metrics['equity_scaled_auc']:.4f}")
        print(f"  AUC Gap: {metrics['auc_gap']:.4f}")
    
    # Close Weights & Biases
    if args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
