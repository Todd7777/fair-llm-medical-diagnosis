#!/usr/bin/env python
"""
Benchmarking Script for Medical Image Diagnosis Models

This script provides a comprehensive benchmarking framework for evaluating
different models (specialized medical AI and general LLMs) on medical image
diagnosis tasks across multiple datasets, with a focus on fairness metrics.

Designed for interns to run standardized experiments for the research project.
"""

import os
import argparse
import yaml
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

# Add project root to path
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.loaders import dataset_loaders
from data.loaders.dataset_loaders import (
    ChestXRayDataset,
    PathologyImageDataset,
    RetinalImageDataset,
    create_data_loaders,
)
from data.processors.image_processors import (
    chest_xray_img_processor,
    pathology_img_processor,
    fundus_img_processor,
)
from fairtune.models.model_adapters import create_model_adapter
from fairtune.metrics.fairness import (
    compute_fairness_metrics,
    compute_equity_scaled_auc,
    compute_subgroup_metrics,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark models for medical image diagnosis"
    )

    parser.add_argument(
        "--config_dir",
        type=str,
        default="configs",
        help="Directory containing configuration files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/benchmarks",
        help="Directory to save results",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=["chexray", "pathology", "retinal"],
        help="Datasets to benchmark",
    )
    parser.add_argument(
        "--model_types",
        type=str,
        nargs="+",
        default=["specialized", "llm", "fairlora"],
        help="Model types to benchmark",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--num_workers", type=int, default=4, help="Number of workers for data loading"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Whether to use Weights & Biases for logging",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="fair-llm-medical-benchmarks",
        help="Weights & Biases project name",
    )

    return parser.parse_args()


def set_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


def load_dataset(config, split="test"):
    """Load dataset based on configuration."""
    dataset_type = config["dataset"]["type"]
    data_dir = config["dataset"]["data_dir"]
    metadata_path = config["dataset"]["metadata_path"]

    if dataset_type == "chest_xray":
        transform = chest_xray_img_processor(
            target_size=(
                config["dataset"].get("image_size", 224),
                config["dataset"].get("image_size", 224),
            ),
            is_training=False,  # Always use evaluation transforms for benchmarking
        )
        dataset = ChestXRayDataset(
            data_dir=data_dir,
            metadata_path=metadata_path,
            transform=transform,
            split=split,
            demographic_key=config["dataset"].get("demographic_key", "demographic"),
            label_col=config["dataset"].get("label_cols", ["Pneumonia"]),
        )
    elif dataset_type == "pathology":
        transform = pathology_img_processor(
            target_size=(
                config["dataset"].get("image_size", 224),
                config["dataset"].get("image_size", 224),
            ),
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
        # TODO: edit stuff on diabetic retinopathy, the actual dataset isn't just on that
        preprocessor = fundus_img_processor(
            target_size=(
                config["dataset"].get("image_size", 224),
                config["dataset"].get("image_size", 224),
            ),
            normalize=True,
            augment=False,
        )
        dataset = RetinalImageDataset(
            data_dir=data_dir,
            metadata_path=metadata_path,
            transform=preprocessor.img_process,  # Pass the preprocess method as the transform
            split=split,
            demographic_key=config["dataset"].get("demographic_key", "demographic"),
            label_col=config["dataset"].get("label_col", "diabetic_retinopathy_grade"),
        )

    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    return dataset


def load_model(model_type, model_name, config, device):
    """Load model based on type and name."""
    if model_type == "fairlora":
        # For Fair LoRA models, load the fine-tuned model from the specified path
        # The model_name in this case is the path to the fine-tuned model directory
        model_dir = model_name
        model_config_path = os.path.join(model_dir, "config.yaml")

        if not os.path.exists(model_config_path):
            raise ValueError(f"Model config not found: {model_config_path}")

        # Load model config
        with open(model_config_path, "r") as f:
            model_config = yaml.safe_load(f)

        # Create base model adapter
        base_model_type = model_config["model"]["type"]
        base_model_name = model_config["model"]["name"]

        # Create model adapter
        model = create_model_adapter(
            model_type=base_model_type,
            model_name=base_model_name,
            num_classes=config["dataset"].get("num_classes", 2),
            fairlora_path=model_dir,
        ).to(device)
    else:
        # For specialized and llm models, create the model adapter directly
        model = create_model_adapter(
            model_type=model_type,
            model_name=model_name,
            num_classes=config["dataset"].get("num_classes", 2),
        ).to(device)

    return model


def evaluate_model(model, dataloader, device, demographic_key="demographic"):
    """Evaluate model on dataloader."""
    model.eval()

    all_preds = []
    all_labels = []
    all_demographics = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            images = batch["image"].to(device)
            labels = batch["labels"].to(device)
            demographics = batch[demographic_key]

            # Forward pass
            outputs = model(images)

            # Get predictions
            if isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs

            # Convert logits to probabilities
            if logits.shape[1] > 1:  # Multi-class
                probs = torch.softmax(logits, dim=1)
            else:  # Binary
                probs = torch.sigmoid(logits)

            # Collect predictions, labels, and demographics
            all_preds.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
            all_demographics.extend(demographics)

    # Concatenate predictions and labels
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    return all_preds, all_labels, all_demographics


def plot_results(results, output_path):
    """Plot benchmark results."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Create DataFrame from results
    rows = []
    for dataset_name, dataset_results in results.items():
        for model_type, model_results in dataset_results.items():
            for model_name, metrics in model_results.items():
                row = {
                    "Dataset": dataset_name,
                    "Model Type": model_type,
                    "Model": model_name,
                    "AUC": metrics["auc"],
                    "Equity-Scaled AUC": metrics["equity_scaled_auc"],
                    "Fairness Gap": metrics["fairness_gap"],
                }
                rows.append(row)

    df = pd.DataFrame(rows)

    # Plot AUC and Equity-Scaled AUC
    plt.figure(figsize=(15, 10))

    # Create subplot for AUC
    plt.subplot(2, 1, 1)
    sns.barplot(x="Model", y="AUC", hue="Model Type", data=df)
    plt.title("AUC by Model and Dataset")
    plt.ylim(0.5, 1.0)
    plt.xticks(rotation=45)

    # Create subplot for Equity-Scaled AUC
    plt.subplot(2, 1, 2)
    sns.barplot(x="Model", y="Equity-Scaled AUC", hue="Model Type", data=df)
    plt.title("Equity-Scaled AUC by Model and Dataset")
    plt.ylim(0.5, 1.0)
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    # Plot Fairness Gap
    plt.figure(figsize=(15, 5))
    sns.barplot(x="Model", y="Fairness Gap", hue="Model Type", data=df)
    plt.title("Fairness Gap by Model and Dataset")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path.replace(".png", "_fairness_gap.png"))
    plt.close()


def main():
    """Main function."""
    # Parse arguments
    args = parse_args()

    # Set random seed
    set_seed(args.seed)

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)

    # Initialize Weights & Biases
    if args.use_wandb:
        import wandb

        wandb.init(project=args.wandb_project, name=f"benchmark_{timestamp}")

    # Initialize results dictionary
    results = {}

    # Iterate over datasets
    for dataset_name in args.datasets:
        print(f"\n=== Benchmarking on {dataset_name} dataset ===\n")

        # Load dataset configuration
        config_path = os.path.join(args.config_dir, f"{dataset_name}_evaluation.yaml")
        if not os.path.exists(config_path):
            print(f"Configuration file not found: {config_path}")
            continue

        config = load_config(config_path)

        # Load dataset
        dataset = load_dataset(config, split="test")

        # Create data loader
        dataloader = create_data_loaders(
            dataset=dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
        )

        # Initialize results for this dataset
        results[dataset_name] = {}

        # Iterate over model types
        for model_type in args.model_types:
            if model_type not in ["specialized", "llm", "fairlora"]:
                print(f"Unknown model type: {model_type}")
                continue

            # Initialize results for this model type
            results[dataset_name][model_type] = {}

            # Get models for this type
            if model_type == "fairlora":
                # For Fair LoRA models, look for fine-tuned models in the results directory
                fairlora_dir = os.path.join("results", dataset_name, "fairlora")
                if not os.path.exists(fairlora_dir):
                    print(f"Fair LoRA models directory not found: {fairlora_dir}")
                    continue

                # Get all model directories
                model_dirs = [
                    os.path.join(fairlora_dir, d)
                    for d in os.listdir(fairlora_dir)
                    if os.path.isdir(os.path.join(fairlora_dir, d))
                ]

                models = {os.path.basename(d): d for d in model_dirs}
            else:
                # For specialized and llm models, get models from config
                models = config["models"].get(model_type, {})

            # Iterate over models
            for model_name, model_info in models.items():
                print(f"\n--- Evaluating {model_type} model: {model_name} ---\n")

                try:
                    # Load model
                    if model_type == "fairlora":
                        model = load_model(model_type, model_info, config, args.device)
                    else:
                        model_name_or_path = model_info["name"]
                        model = load_model(
                            model_type, model_name_or_path, config, args.device
                        )

                    # Evaluate model
                    predictions, labels, demographics = evaluate_model(
                        model=model,
                        dataloader=dataloader,
                        device=args.device,
                        demographic_key=config["dataset"].get(
                            "demographic_key", "demographic"
                        ),
                    )

                    # Compute metrics
                    metrics = compute_fairness_metrics(
                        predictions=predictions,
                        labels=labels,
                        demographics=demographics,
                    )

                    # Compute equity-scaled AUC
                    equity_scaled_auc = compute_equity_scaled_auc(
                        predictions=predictions,
                        labels=labels,
                        demographics=demographics,
                    )

                    # Compute subgroup metrics
                    subgroup_metrics = compute_subgroup_metrics(
                        predictions=predictions,
                        labels=labels,
                        demographics=demographics,
                    )

                    # Calculate fairness gap
                    group_aucs = [
                        metrics["group_metrics"][group]["auc"]
                        for group in metrics["group_metrics"]
                    ]
                    fairness_gap = max(group_aucs) - min(group_aucs)

                    # Store results
                    results[dataset_name][model_type][model_name] = {
                        "auc": metrics["overall_metrics"]["auc"],
                        "equity_scaled_auc": equity_scaled_auc,
                        "fairness_gap": fairness_gap,
                        "group_metrics": metrics["group_metrics"],
                        "subgroup_metrics": subgroup_metrics,
                    }

                    # Log to Weights & Biases
                    if args.use_wandb:
                        wandb.log(
                            {
                                "dataset": dataset_name,
                                "model_type": model_type,
                                "model_name": model_name,
                                "auc": metrics["overall_metrics"]["auc"],
                                "equity_scaled_auc": equity_scaled_auc,
                                "fairness_gap": fairness_gap,
                            }
                        )

                    # Print results
                    print(f"\nResults for {model_type} model: {model_name}")
                    print(f"AUC: {metrics['overall_metrics']['auc']:.4f}")
                    print(f"Equity-Scaled AUC: {equity_scaled_auc:.4f}")
                    print(f"Fairness Gap: {fairness_gap:.4f}")

                    # Print group metrics
                    print("\nGroup Metrics:")
                    for group, group_metrics in metrics["group_metrics"].items():
                        print(
                            f"  {group}: AUC = {group_metrics['auc']:.4f}, "
                            f"Accuracy = {group_metrics['accuracy']:.4f}"
                        )

                except Exception as e:
                    print(f"Error evaluating {model_type} model {model_name}: {e}")

    # Save results
    results_path = os.path.join(output_dir, "benchmark_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    # Plot results
    plot_path = os.path.join(output_dir, "benchmark_results.png")
    plot_results(results, plot_path)

    print(f"\nBenchmarking complete. Results saved to {output_dir}")

    # Finish Weights & Biases run
    if args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
