#!/usr/bin/env python
"""
Unit tests for LoRA implementation.
"""

import os
import sys
import unittest
import torch
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fairtune.models.lora import LoRAConfig, LoRALayer, LoRAModel, prepare_lora_config


class MockLinear(torch.nn.Module):
    """Mock linear layer for testing."""

    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(out_features, in_features))
        self.bias = torch.nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        return torch.nn.functional.linear(x, self.weight, self.bias)


class MockModel(torch.nn.Module):
    """Mock model for testing."""

    def __init__(self):
        super().__init__()
        self.linear1 = MockLinear(10, 20)
        self.linear2 = MockLinear(20, 5)

    def forward(self, x):
        x = self.linear1(x)
        x = torch.nn.functional.relu(x)
        x = self.linear2(x)
        return x


class TestLoRAConfig(unittest.TestCase):
    """Test LoRAConfig class."""

    def test_init(self):
        """Test initialization."""
        config = LoRAConfig(
            r=8,
            alpha=16.0,
            dropout=0.1,
            demographic_groups=["group1", "group2"],
        )

        self.assertEqual(config.r, 8)
        self.assertEqual(config.alpha, 16.0)
        self.assertEqual(config.dropout, 0.1)
        self.assertEqual(config.demographic_groups, ["group1", "group2"])

    def test_prepare_lora_config(self):
        """Test prepare_lora_config function."""
        config_dict = {
            "r": 16,
            "alpha": 32.0,
            "dropout": 0.2,
            "demographic_groups": ["group1", "group2", "group3"],
        }

        config = prepare_lora_config(config_dict)

        self.assertEqual(config.r, 16)
        self.assertEqual(config.alpha, 32.0)
        self.assertEqual(config.dropout, 0.2)
        self.assertEqual(config.demographic_groups, ["group1", "group2", "group3"])


class TestLoRALayer(unittest.TestCase):
    """Test LoRALayer class."""

    def setUp(self):
        """Set up test case."""
        self.in_features = 10
        self.out_features = 20
        self.r = 4
        self.demographic_groups = ["group1", "group2"]
        self.layer = LoRALayer(
            in_features=self.in_features,
            out_features=self.out_features,
            r=self.r,
            demographic_groups=self.demographic_groups,
        )

    def test_init(self):
        """Test initialization."""
        self.assertEqual(self.layer.in_features, self.in_features)
        self.assertEqual(self.layer.out_features, self.out_features)
        self.assertEqual(self.layer.r, self.r)
        self.assertEqual(self.layer.demographic_groups, self.demographic_groups)

        # Check parameter shapes
        self.assertEqual(self.layer.lora_A.shape, (self.r, self.in_features))
        self.assertEqual(self.layer.lora_B.shape, (self.out_features, self.r))

        # Check scaling matrices
        for group in self.demographic_groups:
            self.assertEqual(
                getattr(self.layer, f"lora_S_{group}").shape, (self.r, self.r)
            )

    def test_forward(self):
        """Test forward pass."""
        batch_size = 5
        x = torch.randn(batch_size, self.in_features)
        demographics = ["group1"] * batch_size

        # Forward pass
        output = self.layer(x, demographics)

        # Check output shape
        self.assertEqual(output.shape, (batch_size, self.out_features))

    def test_forward_mixed_demographics(self):
        """Test forward pass with mixed demographics."""
        batch_size = 6
        x = torch.randn(batch_size, self.in_features)
        demographics = ["group1", "group2", "group1", "group2", "group1", "group2"]

        # Forward pass
        output = self.layer(x, demographics)

        # Check output shape
        self.assertEqual(output.shape, (batch_size, self.out_features))

    def test_forward_unknown_demographic(self):
        """Test forward pass with unknown demographic."""
        batch_size = 3
        x = torch.randn(batch_size, self.in_features)
        demographics = ["group3"] * batch_size  # Unknown group

        # Should raise ValueError
        with self.assertRaises(ValueError):
            self.layer(x, demographics)


class TestLoRAModel(unittest.TestCase):
    """Test LoRAModel class."""

    def setUp(self):
        """Set up test case."""
        self.base_model = MockModel()
        self.demographic_groups = ["group1", "group2"]
        self.config = LoRAConfig(
            r=4,
            alpha=8.0,
            dropout=0.1,
            demographic_groups=self.demographic_groups,
            target_modules=["linear1", "linear2"],
        )

        self.model = LoRAModel(
            base_model=self.base_model,
            config=self.config,
        )

    def test_init(self):
        """Test initialization."""
        # Check if LoRA layers were added
        self.assertTrue(hasattr(self.model.base_model.linear1, "lora_layer"))
        self.assertTrue(hasattr(self.model.base_model.linear2, "lora_layer"))

        # Check LoRA layer configurations
        self.assertEqual(self.model.base_model.linear1.lora_layer.r, self.config.r)
        self.assertEqual(
            self.model.base_model.linear1.lora_layer.demographic_groups,
            self.demographic_groups,
        )

    def test_forward(self):
        """Test forward pass."""
        batch_size = 4
        x = torch.randn(batch_size, 10)
        demographics = ["group1"] * batch_size

        # Forward pass
        output = self.model(x, demographics)

        # Check output shape
        self.assertEqual(output.shape, (batch_size, 5))

    def test_forward_mixed_demographics(self):
        """Test forward pass with mixed demographics."""
        batch_size = 4
        x = torch.randn(batch_size, 10)
        demographics = ["group1", "group2", "group1", "group2"]

        # Forward pass
        output = self.model(x, demographics)

        # Check output shape
        self.assertEqual(output.shape, (batch_size, 5))

    def test_get_lora_parameters(self):
        """Test get_lora_parameters method."""
        # Get LoRA parameters
        lora_params = list(self.model.get_lora_parameters())

        # Check if parameters are returned
        self.assertTrue(len(lora_params) > 0)

        # Check if all parameters are from LoRA layers
        for param in lora_params:
            self.assertTrue("lora_layer" in param[0])


if __name__ == "__main__":
    unittest.main()
