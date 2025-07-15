"""
Metrics

This module uses metrics for evaluating model performance
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Callable
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


class EquityScaledLoss(nn.Module):
    def __init__(
        self,
        base_criterion: Callable,
        smoothing_factor: float = 0.01,
    ):
        super().__init__()
        self.base_criterion = base_criterion
        self.smoothing_factor = smoothing_factor

        # Keep track of group-specific performance
        self.group_error_rates = {}
        self.group_counts = {}

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            logits: Model logits
            targets: Ground truth labels

        Returns:
            loss
        """

        return loss.mean()


def compute_auc():
    pass


def compute_metrics():
    pass
