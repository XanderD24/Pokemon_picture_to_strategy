# Vision-Guided Pokémon Team Recommendation Using CNNs and Transformers

A multimodal machine learning project that combines **computer vision** and **transformer-based strategic reasoning** to analyze Pokémon teams from images and recommend optimal team compositions.

This system connects two traditionally separate machine learning tasks:

1. **Visual understanding** — identifying Pokémon from images using convolutional neural networks (CNNs)
2. **Strategic reasoning** — modeling team composition and recommending the best 6th team member using transformer architectures and learned feature representations

---

# Project Overview

The system takes Pokémon images as input, identifies the Pokémon present in the team, converts them into strategic vector representations, and predicts optimal team completions based on competitive team structure, role balance, and matchup coverage.

## Full Pipeline

```text
Pokémon Images
       ↓
EfficientNet-B0 CNN Classifier
       ↓
Predicted Pokémon Classes
       ↓
Feature / Embedding Generation
       ↓
Transformer Team Representation
       ↓
6th Pokémon Recommendation
```

---

# Repository Structure

```text
Pokemon_picture_to_strategy/
│
├── Gen-I Pokemon/
│   ├── pokemon_team_builder_v2.ipynb
│   ├── team_recommender.ipynb
│   ├── best_model_v2.pth
│   ├── best_model_v3.pth
│   ├── CNN_BEST_PRACTICES.md
│   └── README.md
│
├── webapp/
│   ├── app.py
│   ├── model_utils.py
│   ├── static/
│   ├── requirements.txt
│   └── README.md
│
├── prepare_raw_vectors.ipynb
├── train_masked_team_transformer.ipynb
└── README.md
```

---

# System Components

## 1. CNN-Based Pokémon Recognition

The computer vision component uses an **EfficientNet-B0** architecture pretrained on ImageNet and fine-tuned on Pokémon images.

### Dataset
- ~6,800 labeled Pokémon images
- 150 Gen-I Pokémon classes
- Stratified train/validation/test split
- Data augmentation and transfer learning pipeline

### Training Strategy

The CNN was trained in two phases:

| Phase | Description |
|---|---|
| Phase 1 | Train classifier head only |
| Phase 2 | Full fine-tuning of all layers |

### Final Performance

| Metric | Score |
|---|---|
| Top-1 Accuracy | **95.61%** |
| Top-5 Accuracy | **99%+** |

The CNN component also includes:
- per-class evaluation
- confusion matrix analysis
- augmentation experiments
- calibration improvements
- transfer learning optimization

Additional implementation details are documented inside:

```text
Gen-I Pokemon/README.md
```

---

## 2. Transformer-Based Team Recommendation

The strategic reasoning component models Pokémon teams using transformer architectures.

Instead of operating directly on images, the transformer receives:
- Pokémon embeddings
- move vectors
- item vectors
- type interactions
- weaknesses/resistances
- team composition vectors

The system learns patterns from real competitive Pokémon teams and predicts strategically coherent team completions.

### Core Idea

Given:

```text
5 Pokémon team members
```

Predict:

```text
the best 6th team member
```

based on:
- type synergy
- role balance
- matchup coverage
- learned team-building structure

---

# Data Processing Pipeline

The repository includes several preprocessing stages that convert raw Pokémon/game data into machine-learning-ready representations.

## Game Information Dictionaries

Contains structured Pokémon knowledge:
- Pokémon metadata
- abilities
- moves
- items
- weaknesses
- resistances
- immunities

## Transformer Ready Vectors

Transforms game knowledge into embedding/vector formats:
- Pokémon vectors
- move vectors
- item vectors

## Scraped Pokémon Teams

Competitive team datasets converted into:
- team vectors
- battle-ready representations
- transformer training inputs

---

# Web Application Demo

The repository includes a local FastAPI application that allows users to:
- upload Pokémon images
- classify Pokémon in real time
- visualize prediction confidence
- interact with the trained model through a browser interface

The demo loads the trained CNN checkpoint directly and performs inference locally.

### Technologies

- FastAPI
- PyTorch
- EfficientNet-B0
- Vanilla HTML/CSS/JS

Additional setup instructions are documented inside:

```text
webapp/README.md
```

---

# Methodology Summary

This project combines several machine learning concepts into a unified pipeline:

| Area | Techniques Used |
|---|---|
| Computer Vision | CNNs, Transfer Learning, EfficientNet |
| Deep Learning | PyTorch, Fine-Tuning, Data Augmentation |
| Representation Learning | Feature Embeddings, Vectorization |
| Sequence Modeling | Transformers |
| Software Engineering | FastAPI, Modular Repository Design |
| Evaluation | Top-1/Top-5 Accuracy, Per-Class Metrics |

---

# Future Work

Potential future extensions include:
- expanding from Gen-I to all Pokémon generations
- multi-Pokémon image detection (YOLO/DETR)
- battle simulation integration
- multimodal end-to-end architectures
- reinforcement learning for competitive optimization
- transformer-conditioned visual reasoning

---

# Contributors

* **[Eloi Bernier](https://github.com/eloibernier)**
* **[Eduardo Tovilla](https://github.com/Eduardo-Tovilla)**
* **[Xander Deanhardt](https://github.com/XanderD24?tab=overview&from=2026-05-01&to=2026-05-23)**
