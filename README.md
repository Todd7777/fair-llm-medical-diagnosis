# Fair-LLM-Medical-Diagnosis

## Evaluating and Enhancing Large Language Models (LLMs) for Fair and Accurate Medical Image Diagnosis

## Overview

This research project aims to address the growing concern of using general-purpose Large Language Models (LLMs) for medical image diagnosis without proper validation or regulatory approval. The project introduces a comprehensive framework to:

1. **Compare** the diagnostic performance of general-purpose LLMs with specialized medical AI models
2. **Fine-tune** these LLMs via Low-Rank Adaptation (LoRA) to narrow performance gaps
3. **Mitigate bias** across diverse patient populations through our novel Fair LoRA (FairTune) methodology

## Key Features

- **Comparative Benchmarking**. Rigorous evaluation of general LLMs vs. specialized medical AI models
- **Bias Detection**. Comprehensive analysis of model performance across demographic groups
- **Fair LoRA Implementation**. Novel approach to parameter-efficient fine-tuning that addresses demographic bias
- **Equity-Focused Metrics**. Implementation of equity-scaled evaluation metrics
- **Adversarial Fairness**. Innovative adversarial training approach to remove demographic information from representations
- **Counterfactual Fairness**. Causal intervention techniques to ensure invariant predictions across demographic groups
- **Multi-Modal Contrastive Learning**. Integration of clinical text and image data with fairness-aware contrastive objectives

## Project Structure

```
fair-llm-medical-diagnosis/
├── configs/                  # Configuration files for experiments
│   ├── chexray_evaluation.yaml     # CheXRay dataset evaluation config
│   ├── fair_lora_training.yaml     # Fair LoRA training config
│   └── integrated_fairness.yaml    # Integrated fairness approaches config
├── data/
│   ├── loaders/              # Dataset loading utilities
│   └── processors/           # Data preprocessing pipelines
├── docs/                     # Documentation and research notes
├── experiments/              # Experiment scripts and notebooks
│   ├── train_fair_lora.py          # Fair LoRA training script
│   └── integrated_fairness_training.py  # Unified fairness training pipeline
├── fairtune/                 # Core Fair LoRA implementation
│   ├── models/               # Model architectures and adapters
│   │   ├── fair_lora.py            # Base Fair LoRA implementation
│   │   ├── adversarial_fairness.py # Adversarial fairness approach
│   │   ├── counterfactual_fair_lora.py # Counterfactual fairness approach
│   │   └── multimodal_fair_lora.py # Multi-modal contrastive fairness
│   ├── metrics/              # Performance and fairness metrics
│   │   └── fairness.py             # Equity-scaled metrics implementation
│   └── training/             # Training utilities
│       └── trainer.py              # Trainer implementations
├── notebooks/                # Jupyter notebooks for analysis
├── results/                  # Experiment results and visualizations
├── scripts/                  # Utility scripts
│   ├── prepare_datasets.py         # Dataset preparation script
│   └── benchmark_models.py         # Model benchmarking script
└── tests/                    # Unit tests
    ├── test_fair_lora.py           # Tests for Fair LoRA implementation
    └── test_fairness_metrics.py    # Tests for fairness metrics
```

## Installation

```bash
# Clone the repository
git clone https://github.com/Todd7777/fair-llm-medical-diagnosis/
cd fair-llm-medical-diagnosis

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Getting Started

### Data Preparation

```bash
# Download and prepare datasets
python scripts/prepare_datasets.py --dataset chexray
```

### Running Experiments

```bash
# Evaluate baseline models
python experiments/evaluate_baseline.py --config configs/baseline_eval.yaml

# Train with Fair LoRA
python experiments/train_fair_lora.py --config configs/fair_lora_training.yaml

# Train with integrated fairness approaches
python experiments/integrated_fairness_training.py --config configs/integrated_fairness.yaml
```

## Datasets

This project utilizes three primary datasets:

1. **CheXRay Dataset (NOTE: subject to change)**: Chest X-rays with labels for conditions like pneumonia, pleural effusion, and atelectasis
2. **Pathology Image Dataset**: Histopathological slides for malignant vs. benign tissue recognition
3. **Retinal Image Dataset**: Retinal photographs for detecting conditions like diabetic retinopathy

## Fairness Approaches

This project implements several innovative fairness approaches:

1. Fair LoRA. Our base approach that uses demographic-specific scaling matrices with shared low-rank matrices to adapt LLMs for fairness.

2. Adversarial Fairness. Implements a gradient reversal layer and demographic adversary to remove demographic information from learned representations.

3. Counterfactual Fairness. Generates counterfactual examples by modifying demographic-specific features in medical images, ensuring predictions remain invariant under counterfactual interventions.

4. Multi-Modal Contrastive Fairness. Integrates clinical text with medical images using contrastive learning with fairness constraints, ensuring fair representations across modalities.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

This research is supported by Harvard University.
