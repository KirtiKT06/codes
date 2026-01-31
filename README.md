# **Codes — ML, Simulations, and Scientific Computing**

This repository contains my structured collection of code for machine learning, Monte Carlo simulations, neural networks, and scientific computing experiments. 
It is organized to support reproducible experiments, modular development, and clean separation between models, datasets, training scripts, and utilities.

## **Repository Structure**

## Repository Structure

```
codes/
├── ml/                — Machine learning framework
│   ├── models/        — Neural network architectures
│   ├── training/      — Training entry scripts
│   ├── datasets/      — Dataset loaders
│   ├── evaluation/    — Evaluation code
│   ├── utils/         — Logging, config loaders, helpers
│   ├── configs/       — YAML experiment configs
│   └── experiments/   — Experiment scripts

├── simulations/       — Monte Carlo & physics simulations
├── c_cpp/             — C/C++ implementations
├── notebooks/         — Jupyter notebooks
├── theory/            — Notes and theoretical work
├── results/           — Generated outputs (ignored by git)
└── requirements.txt
```


## **Machine Learning Module (ml/)**

The ML module is designed like a lightweight research framework.

Features:

- Modular neural network implementations
- YAML-based configuration
- Dataset abstraction
- Experiment logging support
- VS Code run-button compatible
- Package-style imports (ml.models, ml.datasets, etc.)

## **Running Training Scripts (VS Code Recommended)**

Use the VS Code Run button with configured launch targets.

Available run targets:

- Run MNIST training
- Run XOR training
- Run OR training

Example entry modules:

ml.training.run_mnist
ml.training.train_xor
ml.training.train_or

## **Environment Setup**

Create virtual environment:

python3 -m venv .venv
source .venv/bin/activate

Install dependencies:
pip install -r requirements.txt

If using YAML configs:
pip install pyyaml

## **Datasets**

MNIST dataset file expected at:
ml/datasets/data/mnist.pkl.gz

If missing, download with:
wget http://deeplearning.net/data/mnist/mnist.pkl.gz

## **Simulations**

The simulations/ directory contains:
Monte Carlo sampling
Lennard-Jones potential studies
RNG experiments
Statistical sampling methods

Each simulation module contains its own README and result folders.

## **Experiment Configs**

Experiment parameters are stored as YAML files such as:
ml/configs/mnist_baseline.yaml

Loaded in code using:
from ml.utils.config_loader import load_config
cfg = load_config("mnist_baseline.yaml")

## **Results Handling**

Generated outputs go to:
results/
This folder is git-ignored to avoid committing large files.

## **Goals of This Repository**

Reproducible ML experiments
Clean modular architecture
Research-ready structure
Expandable simulation framework
Multi-language scientific coding

## **Author**

Kirti Kashyap 'Taijas'
Research focus: Machine Learning, Simulations, Scientific Computing, AI
