"""
Fairness Metrics

This module implements fairness metrics for evaluating model performance across demographic groups,
including equity-scaled metrics as described in Luo et al. 2024.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Callable
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score


class EquityScaledLoss(nn.Module):
    """
    Equity-scaled loss function that balances performance across demographic groups.
    
    This loss function applies a demographic-specific scaling factor to the base loss,
    giving more weight to groups with historically worse performance.
    """
    
    def __init__(
        self,
        base_criterion: Callable,
        equity_weight: float = 0.5,
        smoothing_factor: float = 0.01,
    ):
        """
        Initialize equity-scaled loss.
        
        Args:
            base_criterion: Base loss criterion (e.g., CrossEntropyLoss)
            equity_weight: Weight for the equity component (0 = no equity scaling, 1 = full equity scaling)
            smoothing_factor: Smoothing factor for group weights to prevent instability
        """
        super().__init__()
        self.base_criterion = base_criterion
        self.equity_weight = equity_weight
        self.smoothing_factor = smoothing_factor
        
        # Keep track of group-specific performance
        self.group_error_rates = {}
        self.group_counts = {}
    
    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        demographics: List[str],
    ) -> torch.Tensor:
        """
        Compute equity-scaled loss.
        
        Args:
            logits: Model logits
            targets: Ground truth labels
            demographics: List of demographic group identifiers for each sample
        
        Returns:
            Equity-scaled loss
        """
        # Compute base loss for each sample
        base_loss = self.base_criterion(logits, targets)
        
        # If equity weight is 0, return the mean of the base loss
        if self.equity_weight == 0:
            return base_loss.mean()
        
        # Group samples by demographic
        unique_demographics = set(demographics)
        demographic_indices = {d: [] for d in unique_demographics}
        
        for i, d in enumerate(demographics):
            demographic_indices[d].append(i)
        
        # Compute group-specific error rates
        with torch.no_grad():
            for d in unique_demographics:
                if len(demographic_indices[d]) == 0:
                    continue
                
                # Get indices for this demographic group
                indices = demographic_indices[d]
                
                # Compute error rate for this group
                group_preds = logits[indices].argmax(dim=-1)
                group_targets = targets[indices]
                group_error_rate = (group_preds != group_targets).float().mean().item()
                
                # Update group error rates with exponential moving average
                if d not in self.group_error_rates:
                    self.group_error_rates[d] = group_error_rate
                    self.group_counts[d] = len(indices)
                else:
                    # Update with exponential moving average
                    alpha = 0.9  # Smoothing factor for EMA
                    self.group_error_rates[d] = alpha * self.group_error_rates[d] + (1 - alpha) * group_error_rate
                    self.group_counts[d] += len(indices)
        
        # Compute group weights based on error rates
        if len(self.group_error_rates) > 0:
            # Get maximum error rate across groups with sufficient samples
            valid_groups = [d for d in self.group_error_rates if self.group_counts[d] >= 10]
            if len(valid_groups) > 0:
                max_error_rate = max(self.group_error_rates[d] for d in valid_groups)
                
                # Compute weights: higher weight for groups with higher error rates
                group_weights = {
                    d: (self.group_error_rates[d] + self.smoothing_factor) / (max_error_rate + self.smoothing_factor)
                    for d in valid_groups
                }
                
                # Normalize weights to sum to number of groups
                weight_sum = sum(group_weights.values())
                if weight_sum > 0:
                    group_weights = {
                        d: w * len(group_weights) / weight_sum
                        for d, w in group_weights.items()
                    }
                
                # Apply weights to loss
                weighted_loss = torch.zeros_like(base_loss)
                for d, indices in demographic_indices.items():
                    if len(indices) == 0 or d not in group_weights:
                        continue
                    
                    # Apply group-specific weight
                    weighted_loss[indices] = base_loss[indices] * group_weights[d]
                
                # Combine base loss and weighted loss
                combined_loss = (1 - self.equity_weight) * base_loss + self.equity_weight * weighted_loss
                return combined_loss.mean()
        
        # Fallback to base loss if no group information is available
        return base_loss.mean()


def compute_equity_scaled_auc(
    predictions: np.ndarray,
    labels: np.ndarray,
    demographics: List[str],
    equity_weight: float = 0.5,
) -> float:
    """
    Compute equity-scaled AUC as described in Luo et al. 2024.
    
    Args:
        predictions: Model predictions (probabilities)
        labels: Ground truth labels
        demographics: List of demographic group identifiers for each sample
        equity_weight: Weight for the equity component (0 = standard AUC, 1 = fully equity-scaled)
    
    Returns:
        Equity-scaled AUC
    """
    # Compute overall AUC
    overall_auc = roc_auc_score(labels, predictions[:, 1] if predictions.shape[1] > 1 else predictions)
    
    # If equity weight is 0, return the overall AUC
    if equity_weight == 0:
        return overall_auc
    
    # Group samples by demographic
    unique_demographics = set(demographics)
    demographic_indices = {d: [] for d in unique_demographics}
    
    for i, d in enumerate(demographics):
        demographic_indices[d].append(i)
    
    # Compute group-specific AUCs
    group_aucs = {}
    for d in unique_demographics:
        if len(demographic_indices[d]) < 10:  # Skip groups with too few samples
            continue
        
        # Get indices for this demographic group
        indices = demographic_indices[d]
        
        # Compute AUC for this group
        try:
            group_auc = roc_auc_score(
                labels[indices],
                predictions[indices, 1] if predictions.shape[1] > 1 else predictions[indices]
            )
            group_aucs[d] = group_auc
        except ValueError:
            # Skip if all samples belong to the same class
            continue
    
    # If no group-specific AUCs could be computed, return the overall AUC
    if len(group_aucs) == 0:
        return overall_auc
    
    # Compute minimum AUC across groups
    min_group_auc = min(group_aucs.values())
    
    # Compute equity-scaled AUC
    equity_scaled_auc = (1 - equity_weight) * overall_auc + equity_weight * min_group_auc
    
    return equity_scaled_auc


def compute_fairness_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    demographics: List[str],
    equity_weight: float = 0.5,
) -> Dict[str, float]:
    """
    Compute fairness metrics for model evaluation.
    
    Args:
        predictions: Model predictions (probabilities)
        labels: Ground truth labels
        demographics: List of demographic group identifiers for each sample
        equity_weight: Weight for the equity component in equity-scaled metrics
    
    Returns:
        Dictionary of fairness metrics
    """
    # Convert probabilities to class predictions
    if predictions.shape[1] > 1:
        class_preds = predictions.argmax(axis=1)
        class_probs = predictions[:, 1]  # Probability of positive class for binary classification
    else:
        class_preds = (predictions > 0.5).astype(int)
        class_probs = predictions
    
    # Compute overall metrics
    overall_accuracy = accuracy_score(labels, class_preds)
    overall_precision = precision_score(labels, class_preds, average='weighted', zero_division=0)
    overall_recall = recall_score(labels, class_preds, average='weighted', zero_division=0)
    overall_f1 = f1_score(labels, class_preds, average='weighted', zero_division=0)
    
    try:
        overall_auc = roc_auc_score(labels, class_probs)
    except ValueError:
        overall_auc = 0.5  # Default value if AUC cannot be computed
    
    # Compute equity-scaled AUC
    equity_scaled_auc = compute_equity_scaled_auc(
        predictions=predictions,
        labels=labels,
        demographics=demographics,
        equity_weight=equity_weight,
    )
    
    # Group samples by demographic
    unique_demographics = set(demographics)
    demographic_indices = {d: [] for d in unique_demographics}
    
    for i, d in enumerate(demographics):
        demographic_indices[d].append(i)
    
    # Compute group-specific metrics
    group_metrics = {}
    for d in unique_demographics:
        if len(demographic_indices[d]) < 10:  # Skip groups with too few samples
            continue
        
        # Get indices for this demographic group
        indices = demographic_indices[d]
        
        # Compute metrics for this group
        group_accuracy = accuracy_score(labels[indices], class_preds[indices])
        group_precision = precision_score(labels[indices], class_preds[indices], average='weighted', zero_division=0)
        group_recall = recall_score(labels[indices], class_preds[indices], average='weighted', zero_division=0)
        group_f1 = f1_score(labels[indices], class_preds[indices], average='weighted', zero_division=0)
        
        try:
            group_auc = roc_auc_score(labels[indices], class_probs[indices])
        except ValueError:
            group_auc = 0.5  # Default value if AUC cannot be computed
        
        group_metrics[d] = {
            "accuracy": group_accuracy,
            "precision": group_precision,
            "recall": group_recall,
            "f1": group_f1,
            "auc": group_auc,
        }
    
    # Compute fairness gaps
    if len(group_metrics) >= 2:
        accuracy_gap = max(m["accuracy"] for m in group_metrics.values()) - min(m["accuracy"] for m in group_metrics.values())
        precision_gap = max(m["precision"] for m in group_metrics.values()) - min(m["precision"] for m in group_metrics.values())
        recall_gap = max(m["recall"] for m in group_metrics.values()) - min(m["recall"] for m in group_metrics.values())
        f1_gap = max(m["f1"] for m in group_metrics.values()) - min(m["f1"] for m in group_metrics.values())
        auc_gap = max(m["auc"] for m in group_metrics.values()) - min(m["auc"] for m in group_metrics.values())
    else:
        accuracy_gap = precision_gap = recall_gap = f1_gap = auc_gap = 0.0
    
    # Combine metrics
    metrics = {
        "overall_accuracy": overall_accuracy,
        "overall_precision": overall_precision,
        "overall_recall": overall_recall,
        "overall_f1": overall_f1,
        "overall_auc": overall_auc,
        "equity_scaled_auc": equity_scaled_auc,
        "accuracy_gap": accuracy_gap,
        "precision_gap": precision_gap,
        "recall_gap": recall_gap,
        "f1_gap": f1_gap,
        "auc_gap": auc_gap,
    }
    
    # Add group-specific metrics
    for d, group_metric in group_metrics.items():
        for metric_name, metric_value in group_metric.items():
            metrics[f"{d}_{metric_name}"] = metric_value
    
    return metrics
