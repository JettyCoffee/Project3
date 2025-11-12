# CIFAR-10 Image Classification

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.9.0-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> A high-performance deep learning project for CIFAR-10 image classification using modern CNN architectures with advanced training techniques.

This project implements a comprehensive image classification system on the CIFAR-10 dataset, achieving **96.45% test accuracy** using Wide ResNet with advanced regularization and optimization techniques. The project includes multiple state-of-the-art model architectures, extensive training utilities, and comprehensive evaluation and visualization tools.

## Table of Contents

- [Background](#background)
- [Install](#install)
- [Usage](#usage)
  - [Quick Start](#quick-start)
  - [Training](#training)
  - [Evaluation](#evaluation)
  - [Visualization](#visualization)
  - [Full Pipeline](#full-pipeline)
- [Project Structure](#project-structure)
- [Models](#models)
- [Configuration](#configuration)
- [Results](#results)
- [Advanced Features](#advanced-features)
- [Contributing](#contributing)
- [License](#license)

## Background

CIFAR-10 is a widely-used benchmark dataset in computer vision, containing 60,000 32×32 color images across 10 classes. This project explores modern deep learning techniques for image classification, including:

- **Multiple CNN Architectures**: ResNet (18/34/50), Wide ResNet, DLA-34, Vision Transformer (ViT)
- **Advanced Data Augmentation**: Random Crop, Random Horizontal Flip, Cutout
- **Regularization Techniques**: Batch Normalization, Dropout, Weight Decay, Label Smoothing
- **Optimization Strategies**: SGD with Momentum, Cosine Annealing Learning Rate Schedule, Early Stopping
- **Model Interpretability**: Grad-CAM visualization for understanding model decisions

The project is designed with modularity and reproducibility in mind, making it easy to experiment with different configurations and extend with new models or techniques.

## Install

### Requirements

- Python 3.8 or higher
- CUDA-compatible GPU (recommended for training)
- 8GB+ RAM
- 2GB+ disk space

### Setup

1. Clone the repository:
```bash
git clone https://github.com/JettyCoffee/Project3.git
cd Project3
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

The required packages include:
- `torch==2.9.0` - PyTorch deep learning framework
- `torchvision==0.24.0` - Computer vision utilities and datasets
- `numpy==2.3.4` - Numerical computing
- `matplotlib==3.10.7` - Plotting and visualization
- `seaborn==0.13.2` - Statistical data visualization
- `scikit-learn==1.7.2` - Machine learning metrics
- `tqdm==4.67.1` - Progress bars

## Usage

### Quick Start

Train a Wide ResNet model with default settings and evaluate on test set:

```bash
python main.py --mode full --model wide_resnet --epochs 100 --batch-size 256
```

This will:
1. Download CIFAR-10 dataset automatically
2. Train the model for 100 epochs
3. Evaluate on test set
4. Generate visualizations and analysis report

### Training

Train a model with custom hyperparameters:

```bash
# Train ResNet-50 with custom learning rate
python main.py --mode train --model resnet50 --epochs 100 --batch-size 256 --lr 0.1

# Resume training from checkpoint
python main.py --mode train --resume

# Train on specific GPU
python main.py --mode train --gpu-id 0
```

Available models: `resnet18`, `resnet34`, `resnet50`, `wide_resnet`, `dla34`, `vit`

### Evaluation

Evaluate a trained model on test set:

```bash
# Evaluate with checkpoint path
python main.py --mode eval --checkpoint checkpoints/best_model_wide_resnet_1112_082255.pth

# Evaluate and generate visualizations
python main.py --mode eval --checkpoint checkpoints/best_model.pth --visualize
```

### Visualization

Generate visualizations for a trained model:

```bash
python main.py --mode visualize --checkpoint checkpoints/best_model.pth
```

This generates:
- Training curves (loss and accuracy)
- Confusion matrix
- Per-class accuracy bar chart
- Misclassified samples analysis
- Grad-CAM heatmaps

### Full Pipeline

Run the complete training, evaluation, and visualization pipeline:

```bash
python main.py --mode full --model wide_resnet --epochs 100 --batch-size 256 --lr 0.1
```

## Project Structure

```
Project3/
├── main.py                 # Main entry point with argument parsing
├── config.py              # Configuration and hyperparameters
├── models.py              # Model architectures (ResNet, Wide ResNet, DLA, ViT)
├── data_loader.py         # Data loading and augmentation
├── train.py               # Training loop and early stopping
├── evaluate.py            # Model evaluation and metrics
├── visualize.py           # Visualization utilities
├── grad_cam.py            # Grad-CAM implementation
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── TODO.md               # Project tasks and requirements
├── data/                 # CIFAR-10 dataset (auto-downloaded)
├── checkpoints/          # Saved model checkpoints (gitignored)
├── logs/                 # Training logs (gitignored)
├── results/              # Evaluation results and visualizations (gitignored)
│   ├── wideresnet/      # Wide ResNet results (96.45% accuracy)
│   ├── resnet50/        # ResNet-50 results
│   ├── dla34/           # DLA-34 results
│   └── ...
└── report/              # Experiment report
    └── Project3.md      # Detailed analysis and findings
```

## Models

### Implemented Architectures

| Model | Parameters | Test Accuracy | Training Time |
|-------|-----------|---------------|---------------|
| Wide ResNet-28-10 | ~36.5M | **96.45%** | ~8h (100 epochs) |
| ResNet-50 | ~23.5M | 94.23% | ~6h (100 epochs) |
| DLA-34 | ~15.7M | 93.52% | ~5h (100 epochs) |
| ResNet-34 | ~21.3M | 93.18% | ~5h (100 epochs) |
| ResNet-18 | ~11.2M | 92.74% | ~4h (100 epochs) |
| ViT-Tiny | ~5.7M | 89.31% | ~10h (100 epochs) |

### Wide ResNet Architecture

The best-performing model is Wide ResNet-28-10 with:
- Depth: 28 layers
- Widening factor: 10
- Dropout rate: 0.3
- Batch Normalization after each convolution
- Shortcut connections for gradient flow

## Configuration

Key hyperparameters can be modified in `config.py`:

```python
# Training hyperparameters
BATCH_SIZE = 256           # Batch size for training
NUM_EPOCHS = 100          # Number of training epochs
LEARNING_RATE = 0.1       # Initial learning rate
MOMENTUM = 0.9            # SGD momentum
WEIGHT_DECAY = 0.0005     # L2 regularization coefficient

# Data augmentation
RANDOM_CROP = True        # Random crop with padding
RANDOM_HORIZONTAL_FLIP = True  # Random horizontal flip
CUTOUT = True             # Cutout regularization
CUTOUT_LENGTH = 16        # Size of cutout region

# Regularization
DROPOUT_RATE = 0.5        # Dropout probability
LABEL_SMOOTHING = 0.1     # Label smoothing factor

# Early stopping
EARLY_STOPPING = True     # Enable early stopping
PATIENCE = 25             # Patience for early stopping
MIN_DELTA = 0.001         # Minimum improvement threshold
```

## Results

### Best Model Performance (Wide ResNet-28-10)

- **Test Accuracy**: 96.45%
- **Top-3 Accuracy**: 99.55%
- **Top-5 Accuracy**: 99.75%

### Per-Class Accuracy

| Class | Accuracy |
|-------|----------|
| Frog | 98.5% |
| Ship | 98.1% |
| Automobile | 97.9% |
| Horse | 97.9% |
| Truck | 97.5% |
| Deer | 97.2% |
| Airplane | 96.7% |
| Bird | 95.3% |
| Cat | 93.7% |
| Dog | 91.7% |

### Common Misclassifications

1. **Dog ↔ Cat** (6.3% and 3.4%): Similar fur textures and poses
2. **Airplane ↔ Ship** (1.6%): Similar geometric shapes in small resolution
3. **Truck ↔ Automobile** (1.6%): Similar vehicle structures

All results including confusion matrices, training curves, and Grad-CAM visualizations are saved in the `results/` directory after evaluation.

## Advanced Features

### 1. Unified Timestamp Management
The project uses a `TimestampManager` class to ensure consistent naming across training, evaluation, and visualization phases.

### 2. Comprehensive Logging
- Training history saved in JSON format
- Hyperparameters logged for reproducibility
- Real-time progress bars with tqdm
- Tensorboard-compatible logging structure

### 3. Automatic Checkpointing
- Best model saved based on validation accuracy
- Last checkpoint for resuming training
- Model state includes optimizer and scheduler states

### 4. Early Stopping
Prevents overfitting by monitoring validation accuracy with configurable patience.

### 5. Model Interpretability
Grad-CAM visualizations show which regions the model focuses on for predictions.

### 6. Reproducibility
- Fixed random seeds for PyTorch, NumPy, and Python
- Deterministic CUDA operations
- Complete hyperparameter logging

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

### Development Setup

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Author**: JettyCoffee  
**Date**: November 2025  
**Course**: Modern AI Technology - Project 3
