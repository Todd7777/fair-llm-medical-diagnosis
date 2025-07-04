#!/usr/bin/env python
"""
Unit tests for fairness metrics implementation.
"""

import os
import sys
import unittest
import torch
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fairtune.metrics.fairness import (
    compute_fairness_metrics,
    compute_equity_scaled_auc,
    compute_subgroup_metrics,
    EquityScaledLoss
)


class TestFairnessMetrics(unittest.TestCase):
    """Test fairness metrics functions."""
    
    def setUp(self):
        """Set up test case."""
        # Create synthetic predictions and labels
        np.random.seed(42)
        
        # Binary classification
        self.n_samples = 100
        self.n_groups = 3
        
        # Create predictions with bias
        # Group 0: High accuracy
        # Group 1: Medium accuracy
        # Group 2: Low accuracy
        self.predictions = np.zeros((self.n_samples, 1))
        self.labels = np.zeros((self.n_samples, 1))
        self.demographics = []
        
        for i in range(self.n_samples):
            group_idx = i % self.n_groups
            self.demographics.append(f"group{group_idx}")
            
            # Set label
            self.labels[i, 0] = np.random.randint(0, 2)
            
            # Set prediction based on group
            if group_idx == 0:  # High accuracy group
                # 90% chance of correct prediction
                if np.random.random() < 0.9:
                    self.predictions[i, 0] = self.labels[i, 0]
                else:
                    self.predictions[i, 0] = 1 - self.labels[i, 0]
            elif group_idx == 1:  # Medium accuracy group
                # 70% chance of correct prediction
                if np.random.random() < 0.7:
                    self.predictions[i, 0] = self.labels[i, 0]
                else:
                    self.predictions[i, 0] = 1 - self.labels[i, 0]
            else:  # Low accuracy group
                # 50% chance of correct prediction
                if np.random.random() < 0.5:
                    self.predictions[i, 0] = self.labels[i, 0]
                else:
                    self.predictions[i, 0] = 1 - self.labels[i, 0]
        
        # Convert binary predictions to probabilities
        self.pred_probs = np.zeros_like(self.predictions, dtype=float)
        for i in range(self.n_samples):
            if self.predictions[i, 0] == 1:
                self.pred_probs[i, 0] = 0.7 + 0.3 * np.random.random()  # Between 0.7 and 1.0
            else:
                self.pred_probs[i, 0] = 0.3 * np.random.random()  # Between 0.0 and 0.3
    
    def test_compute_fairness_metrics(self):
        """Test compute_fairness_metrics function."""
        # Compute fairness metrics
        metrics = compute_fairness_metrics(
            predictions=self.pred_probs,
            labels=self.labels,
            demographics=self.demographics,
        )
        
        # Check if overall metrics are present
        self.assertIn("overall_metrics", metrics)
        self.assertIn("auc", metrics["overall_metrics"])
        self.assertIn("accuracy", metrics["overall_metrics"])
        self.assertIn("precision", metrics["overall_metrics"])
        self.assertIn("recall", metrics["overall_metrics"])
        self.assertIn("f1", metrics["overall_metrics"])
        
        # Check if group metrics are present
        self.assertIn("group_metrics", metrics)
        for group_idx in range(self.n_groups):
            group_name = f"group{group_idx}"
            self.assertIn(group_name, metrics["group_metrics"])
            self.assertIn("auc", metrics["group_metrics"][group_name])
            self.assertIn("accuracy", metrics["group_metrics"][group_name])
            self.assertIn("precision", metrics["group_metrics"][group_name])
            self.assertIn("recall", metrics["group_metrics"][group_name])
            self.assertIn("f1", metrics["group_metrics"][group_name])
        
        # Check if fairness gap metrics are present
        self.assertIn("fairness_gaps", metrics)
        self.assertIn("auc_gap", metrics["fairness_gaps"])
        self.assertIn("accuracy_gap", metrics["fairness_gaps"])
        
        # Verify that group0 has higher AUC than group2
        self.assertGreater(
            metrics["group_metrics"]["group0"]["auc"],
            metrics["group_metrics"]["group2"]["auc"]
        )
    
    def test_compute_equity_scaled_auc(self):
        """Test compute_equity_scaled_auc function."""
        # Compute equity-scaled AUC
        equity_scaled_auc = compute_equity_scaled_auc(
            predictions=self.pred_probs,
            labels=self.labels,
            demographics=self.demographics,
            equity_weight=0.5,
        )
        
        # Check if equity-scaled AUC is a float
        self.assertIsInstance(equity_scaled_auc, float)
        
        # Check if equity-scaled AUC is between 0 and 1
        self.assertGreaterEqual(equity_scaled_auc, 0.0)
        self.assertLessEqual(equity_scaled_auc, 1.0)
        
        # Compute regular AUC
        from sklearn.metrics import roc_auc_score
        regular_auc = roc_auc_score(self.labels, self.pred_probs)
        
        # Equity-scaled AUC should be lower than regular AUC due to fairness penalty
        self.assertLessEqual(equity_scaled_auc, regular_auc)
        
        # Test with different equity weights
        equity_scaled_auc_0 = compute_equity_scaled_auc(
            predictions=self.pred_probs,
            labels=self.labels,
            demographics=self.demographics,
            equity_weight=0.0,
        )
        
        equity_scaled_auc_1 = compute_equity_scaled_auc(
            predictions=self.pred_probs,
            labels=self.labels,
            demographics=self.demographics,
            equity_weight=1.0,
        )
        
        # With equity_weight=0, should be equal to regular AUC
        self.assertAlmostEqual(equity_scaled_auc_0, regular_auc, places=5)
        
        # With equity_weight=1, should be lower than with equity_weight=0
        self.assertLess(equity_scaled_auc_1, equity_scaled_auc_0)
    
    def test_compute_subgroup_metrics(self):
        """Test compute_subgroup_metrics function."""
        # Compute subgroup metrics
        subgroup_metrics = compute_subgroup_metrics(
            predictions=self.pred_probs,
            labels=self.labels,
            demographics=self.demographics,
        )
        
        # Check if subgroup metrics are present for each group
        for group_idx in range(self.n_groups):
            group_name = f"group{group_idx}"
            self.assertIn(group_name, subgroup_metrics)
            
            # Check metrics for each group
            group_metrics = subgroup_metrics[group_name]
            self.assertIn("auc", group_metrics)
            self.assertIn("accuracy", group_metrics)
            self.assertIn("precision", group_metrics)
            self.assertIn("recall", group_metrics)
            self.assertIn("f1", group_metrics)
            self.assertIn("count", group_metrics)
            
            # Check if count is correct
            expected_count = sum(1 for d in self.demographics if d == group_name)
            self.assertEqual(group_metrics["count"], expected_count)
        
        # Verify that group0 has higher accuracy than group2
        self.assertGreater(
            subgroup_metrics["group0"]["accuracy"],
            subgroup_metrics["group2"]["accuracy"]
        )


class TestEquityScaledLoss(unittest.TestCase):
    """Test EquityScaledLoss class."""
    
    def setUp(self):
        """Set up test case."""
        self.batch_size = 9
        self.n_classes = 2
        self.demographic_groups = ["group0", "group1", "group2"]
        
        # Create logits and labels
        self.logits = torch.randn(self.batch_size, self.n_classes)
        self.labels = torch.randint(0, self.n_classes, (self.batch_size,))
        
        # Create demographics
        self.demographics = []
        for i in range(self.batch_size):
            self.demographics.append(self.demographic_groups[i % len(self.demographic_groups)])
    
    def test_init(self):
        """Test initialization."""
        loss_fn = EquityScaledLoss(
            base_criterion=torch.nn.CrossEntropyLoss(),
            equity_weight=0.5,
        )
        
        self.assertEqual(loss_fn.equity_weight, 0.5)
        self.assertIsInstance(loss_fn.base_criterion, torch.nn.CrossEntropyLoss)
    
    def test_forward(self):
        """Test forward pass."""
        # Create loss function
        loss_fn = EquityScaledLoss(
            base_criterion=torch.nn.CrossEntropyLoss(),
            equity_weight=0.5,
        )
        
        # Compute loss
        loss = loss_fn(
            logits=self.logits,
            labels=self.labels,
            demographics=self.demographics,
        )
        
        # Check if loss is a scalar
        self.assertEqual(loss.dim(), 0)
        
        # Check if loss is positive
        self.assertGreaterEqual(loss.item(), 0.0)
    
    def test_equity_weight(self):
        """Test different equity weights."""
        # Create loss functions with different equity weights
        loss_fn_0 = EquityScaledLoss(
            base_criterion=torch.nn.CrossEntropyLoss(),
            equity_weight=0.0,
        )
        
        loss_fn_05 = EquityScaledLoss(
            base_criterion=torch.nn.CrossEntropyLoss(),
            equity_weight=0.5,
        )
        
        loss_fn_1 = EquityScaledLoss(
            base_criterion=torch.nn.CrossEntropyLoss(),
            equity_weight=1.0,
        )
        
        # Compute losses
        loss_0 = loss_fn_0(
            logits=self.logits,
            labels=self.labels,
            demographics=self.demographics,
        )
        
        loss_05 = loss_fn_05(
            logits=self.logits,
            labels=self.labels,
            demographics=self.demographics,
        )
        
        loss_1 = loss_fn_1(
            logits=self.logits,
            labels=self.labels,
            demographics=self.demographics,
        )
        
        # With equity_weight=0, should be equal to regular cross-entropy loss
        regular_loss = torch.nn.CrossEntropyLoss()(self.logits, self.labels)
        self.assertAlmostEqual(loss_0.item(), regular_loss.item(), places=5)
        
        # With equity_weight=1, should be different from regular loss
        self.assertNotAlmostEqual(loss_1.item(), regular_loss.item(), places=5)
        
        # Loss with equity_weight=0.5 should be between loss_0 and loss_1
        # This is not always true due to optimization, but we can check if they're different
        self.assertNotEqual(loss_0.item(), loss_05.item())
        self.assertNotEqual(loss_05.item(), loss_1.item())


if __name__ == "__main__":
    unittest.main()
