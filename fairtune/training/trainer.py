"""
LoRA Trainer

This module uses specialized training utilities for LoRA.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple, Union, Callable
import numpy as np
from tqdm import tqdm
import wandb
from transformers import PreTrainedModel, get_scheduler
from ..models.lora import LoRAModel, LoRAConfig
from ..metrics import _


class LoRATrainer:
    """
    Trainer for LoRA fine-tuning with equity-focused objectives.
    """

    def __init__(
        self,
        model: LoRAModel,
        train_dataloader: DataLoader,
        eval_dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        num_epochs: int = 3,
        demographic_key: str = "demographic",
        label_key: str = "labels",
        log_dir: str = "logs",
        checkpoint_dir: str = "checkpoints",
        use_wandb: bool = False,
        wandb_project: str = "lora-medical",
        wandb_run_name: Optional[str] = None,
        evaluation_steps: int = 100,
        save_steps: int = 500,
        mixed_precision: bool = False,
    ):
        self.model = model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.device = device
        self.num_epochs = num_epochs
        self.demographic_key = demographic_key
        self.label_key = label_key
        self.log_dir = log_dir
        self.checkpoint_dir = checkpoint_dir
        self.use_wandb = use_wandb
        self.wandb_project = wandb_project
        self.wandb_run_name = wandb_run_name
        self.evaluation_steps = evaluation_steps
        self.save_steps = save_steps
        self.mixed_precision = mixed_precision

        # Create directories
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Initialize scaler for mixed precision training
        self.scaler = torch.cuda.amp.GradScaler() if mixed_precision else None

        # Move model to device
        self.model.to(device)

        # Initialize Weights & Biases
        if use_wandb:
            wandb.init(
                project=wandb_project,
                name=wandb_run_name,
                config={
                    "model_type": model.base_model.__class__.__name__,
                    "num_epochs": num_epochs,
                    "mixed_precision": mixed_precision,
                },
            )

    def train(self):
        """Train the model with LoRA."""
        global_step = 0
        best_eval_metric = float("-inf")

        for epoch in range(self.num_epochs):
            self.model.train()
            epoch_loss = 0.0

            progress_bar = tqdm(
                self.train_dataloader, desc=f"Epoch {epoch + 1}/{self.num_epochs}"
            )

            for step, batch in enumerate(progress_bar):
                # Move batch to device
                batch = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }

                # Extract demographic information
                demographics = batch.get(
                    self.demographic_key, ["default"] * batch["input_ids"].shape[0]
                )
                labels = batch[self.label_key]

                # Forward pass with mixed precision if enabled
                with (
                    torch.cuda.amp.autocast() if self.mixed_precision else nullcontext()
                ):
                    outputs = self.model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch.get("attention_mask"),
                        demographic_groups=demographics,
                    )

                    # adjust to compute loss without equity_loss
                    loss = self.equity_loss(outputs.logits, labels, demographics)

                # Backward pass with mixed precision if enabled
                if self.mixed_precision:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()

                # Update learning rate
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()

                # Zero gradients
                self.optimizer.zero_grad()

                # Update progress bar
                progress_bar.set_postfix({"loss": loss.item()})
                epoch_loss += loss.item()

                # Log to Weights & Biases
                if self.use_wandb:
                    wandb.log(
                        {
                            "train/loss": loss.item(),
                            "train/lr": self.lr_scheduler.get_last_lr()[0]
                            if self.lr_scheduler
                            else 0,
                        },
                        step=global_step,
                    )

                # Evaluate
                if global_step % self.evaluation_steps == 0:
                    eval_results = self.evaluate()

                    # Log evaluation results
                    if self.use_wandb:
                        wandb.log(
                            {f"eval/{k}": v for k, v in eval_results.items()},
                            step=global_step,
                        )

                    # Save best model
                    if eval_results["equity_scaled_auc"] > best_eval_metric:
                        best_eval_metric = eval_results["equity_scaled_auc"]
                        self.save_checkpoint(
                            os.path.join(self.checkpoint_dir, "best_model")
                        )

                # Save checkpoint
                if global_step % self.save_steps == 0:
                    self.save_checkpoint(
                        os.path.join(self.checkpoint_dir, f"checkpoint-{global_step}")
                    )

                global_step += 1

            # Log epoch metrics
            epoch_loss /= len(self.train_dataloader)
            print(f"Epoch {epoch + 1}/{self.num_epochs}, Loss: {epoch_loss:.4f}")

            if self.use_wandb:
                wandb.log(
                    {
                        "train/epoch_loss": epoch_loss,
                        "train/epoch": epoch + 1,
                    },
                    step=global_step,
                )

            # Evaluate at the end of each epoch
            eval_results = self.evaluate()

            # Log evaluation results
            if self.use_wandb:
                wandb.log(
                    {f"eval/{k}": v for k, v in eval_results.items()}, step=global_step
                )

            # Save checkpoint
            self.save_checkpoint(
                os.path.join(self.checkpoint_dir, f"checkpoint-epoch-{epoch + 1}")
            )

        # Save final model
        self.save_checkpoint(os.path.join(self.checkpoint_dir, "final_model"))

        # Close Weights & Biases
        if self.use_wandb:
            wandb.finish()

    def evaluate(self):
        """
        Evaluate the model on the evaluation dataset.

        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()

        all_preds = []
        all_labels = []
        all_demographics = []

        with torch.no_grad():
            for batch in tqdm(self.eval_dataloader, desc="Evaluating"):
                # Move batch to device
                batch = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }

                # Extract demographic information
                demographics = batch.get(
                    self.demographic_key, ["default"] * batch["input_ids"].shape[0]
                )
                labels = batch[self.label_key]

                # Forward pass
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch.get("attention_mask"),
                    demographic_groups=demographics,
                )

                # Get predictions
                preds = F.softmax(outputs.logits, dim=-1)

                # Collect predictions, labels, and demographics
                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                all_demographics.extend(demographics)

        # Concatenate predictions and labels
        all_preds = np.concatenate(all_preds, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        # TODO: Compute metrics
        metrics = compute_metrics()

        return metrics

    def save_checkpoint(self, path: str):
        """
        Save model checkpoint.

        Args:
            path: Path to save checkpoint
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # Save model state
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "lr_scheduler_state_dict": self.lr_scheduler.state_dict()
                if self.lr_scheduler
                else None,
            },
            f"{path}.pt",
        )

        # Save model configuration
        if hasattr(self.model, "config"):
            self.model.config.save_pretrained(path)

    def load_checkpoint(self, path: str):
        """
        Load model checkpoint.

        Args:
            path: Path to load checkpoint from
        """
        # Load checkpoint
        checkpoint = torch.load(f"{path}.pt", map_location=self.device)

        # Load model state
        self.model.load_state_dict(checkpoint["model_state_dict"])

        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Load learning rate scheduler state
        if (
            self.lr_scheduler is not None
            and checkpoint["lr_scheduler_state_dict"] is not None
        ):
            self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])


class nullcontext:
    """Context manager that does nothing."""

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


def create_lora_trainer(
    model: LoRAModel,
    train_dataloader: DataLoader,
    eval_dataloader: DataLoader,
    learning_rate: float = 5e-5,
    weight_decay: float = 0.01,
    num_epochs: int = 3,
    warmup_steps: int = 0,
    equity_weight: float = 0.5,
    **kwargs,
) -> LoRATrainer:
    """
    Create LoRA trainer with default settings.

    Args:
        model: LoRA model
        train_dataloader: Training data loader
        eval_dataloader: Evaluation data loader
        learning_rate: Learning rate
        weight_decay: Weight decay
        num_epochs: Number of training epochs
        warmup_steps: Number of warmup steps
        equity_weight: Weight for the equity component in the loss
        **kwargs: Additional arguments for LoRATrainer

    Returns:
        LoRA trainer
    """
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    # Create learning rate scheduler
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=len(train_dataloader) * num_epochs,
    )

    # Create trainer
    trainer = LoRATrainer(
        model=model,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        num_epochs=num_epochs,
        **kwargs,
    )

    return trainer
