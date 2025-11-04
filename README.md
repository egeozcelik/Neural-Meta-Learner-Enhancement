# Neural Meta Learner Enhancement

> A sophisticated ensemble stacking framework that enhances pre-trained models through meta-learning and deep neural architectures.

**Continuation of:** [Comparative-ML-Model-Evaluation](https://github.com/egeozcelik/Comparative-ML-Model-Evaluation)

---

## Overview

This project implements advanced **ensemble stacking** techniques by leveraging a pre-trained CatBoost model as the base learner and training secondary meta-learners to capture higher-order patterns. The framework systematically compares three meta-learning approaches: **Logistic Regression**, **LightGBM**, and **Shallow Neural Networks**.

**Core Objective:** Enhance predictive performance of production-ready models through layered learning architectures without retraining from scratch.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Pre-trained CatBoost                    │
│                    (Frozen Base Model)                      │
└──────────────────────────┬──────────────────────────────────┘
                           │ Probability Predictions
                           ▼
┌─────────────────────────────────────────────────────────────┐
│                    Meta-Feature Layer                       │
│           [P(class_0), P(class_1)] + Original Features      │
└──────────────────────────┬──────────────────────────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        ▼                  ▼                  ▼
  ┌──────────┐      ┌──────────┐      ┌──────────┐
  │ Logistic │      │ LightGBM │      │  Neural  │
  │   Reg.   │      │          │      │  Network │
  └──────────┘      └──────────┘      └──────────┘
        │                  │                  │
        └──────────────────┴──────────────────┘
                           ▼
                 Enhanced Predictions
```

---

## Methodology

## Methodology

**Phase 1: Base Model Evaluation**  
Load pre-trained CatBoost model and establish baseline metrics on holdout test set.

**Phase 2: Meta-Feature Engineering**  
Extract prediction probabilities from base model to construct meta-feature space.

**Phase 3: Meta-Learner Training**  
Train three distinct meta-learners in parallel using cross-validation for robust performance estimation.

**Phase 4: Comparative Analysis**  
Quantify performance gains and identify optimal meta-learning architecture.

---

## Installation

### Prerequisites
- Python 3.8+
- Part 1 trained artifacts (CatBoost model, scaler, feature schema)

### Setup

```bash
# Clone repository
git clone https://github.com/username/Deep-Ensemble-Meta-Learning.git
cd Deep-Ensemble-Meta-Learning

# Create isolated environment
python -m venv venv

# Activate environment
source venv/bin/activate          # macOS/Linux
# venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## Project Status

**Current Stage:** Initial Development  
This repository is under active development as a direct continuation of the base model training pipeline.

---

## Usage

```bash
python main.py
```


---

## Results

All outputs are automatically saved:
- `results/metrics/` - Performance metrics in CSV format
- `results/plots/` - Comparative visualizations

---

## Technical Stack

| Component | Technology |
|-----------|------------|
| Base Model | CatBoost |
| Meta-Learners | Logistic Regression, LightGBM, PyTorch |
| Validation | Stratified K-Fold CV |
| Evaluation | Scikit-learn Metrics |

---

## Performance Expectations

| Meta-Learner | Computational Cost | Expected Gain |
|--------------|-------------------|---------------|
| Logistic Regression | Minimal | +1-3% |
| LightGBM | Low | +2-5% |
| Shallow Neural Network | Moderate | +2-4% |

*Optimized for MacBook Air M1/M2 environments*

---


---

## Acknowledgments

Built upon the foundation established in [Comparative-ML-Model-Evaluation](https://github.com/egeozcelik/Comparative-ML-Model-Evaluation)
