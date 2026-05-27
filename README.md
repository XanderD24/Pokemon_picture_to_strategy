# Vision-Guided Pokémon Team Recommendation Using CNNs and Transformers

A multimodal machine learning project that combines **computer vision** and **transformer-based strategic reasoning** to analyze Pokémon teams from images and recommend optimal team compositions.

This system connects two traditionally separate machine learning tasks:

1. **Visual understanding** — identifying Pokémon from images using convolutional neural networks (CNNs)
2. **Strategic reasoning** — modeling team composition and recommending the best 6th team member using transformer architectures and learned feature representations

---

## Files

### `pokemon_team_builder_v2.ipynb` — Training Notebook

The main notebook. It trains a neural network to recognize 150 Gen-I Pokémon from images and includes a basic team recommender.

**Dataset**
- 6,820 images across 150 Pokémon classes (avg. ~45 images per class)
- Exploratory data analysis: class balance, image size distribution, visual samples

**Data preparation**
- Stratified 70/15/15 train/val/test split — every class is proportionally represented in all three sets
- Training augmentations: horizontal flip, rotation ±15°, color jitter
- All images resized to 224×224

**Model — EfficientNet-B0 with transfer learning**

Training runs in two phases:

| Phase | What trains | Epochs | Learning rate |
|-------|-------------|--------|---------------|
| 1 | Classifier head only (backbone frozen) | 5 | 1e-3 |
| 2 | All layers (full fine-tuning) | up to 20 (early stopping, patience=7) | 1e-4 |

**Results achieved**

| Metric | Score |
|--------|-------|
| Top-1 accuracy (test set) | **94.04%** |
| Top-5 accuracy (test set) | **99.22%** |

The notebook also includes:
- Training curve plots (loss & accuracy by epoch, with phase boundary)
- Per-class accuracy bar chart
- Sample correct and incorrect predictions
- Confusion matrix heatmap focused on the 20 hardest classes
- Side-by-side visualization of the most confused Pokémon pairs
- A basic 6th-member recommender (type coverage only)

---

### `best_model_v2.pth` — Trained Model Weights

The saved PyTorch checkpoint from `pokemon_team_builder_v2.ipynb`. It stores the EfficientNet-B0 weights after the best validation epoch.

- **Architecture:** EfficientNet-B0 (pretrained on ImageNet, fine-tuned for 150 Pokémon classes)
- **Output:** probability distribution over 150 Pokémon
- Loaded by `team_recommender.ipynb` — no retraining needed

---

### `team_recommender.ipynb` — Inference & Recommendation Notebook

A standalone notebook for using the trained model in practice. Loads `best_model_v2.pth` and runs in seconds.

**Data enrichment**
- Fetches HP / Attack / Defense / Sp.Atk / Sp.Def / Speed for all 150 Pokémon from the PokéAPI
- Results cached locally to `pokemon_stats.json` so internet is only needed once

**Composite recommendation algorithm**

The 6th member is scored with a weighted formula:

```
score = w_type × new_types_added
      + w_weak × weakness_coverage
      + w_role × fills_missing_role
```

| Criterion | What it measures |
|-----------|-----------------|
| **Type coverage** | How many new types the candidate brings to the team |
| **Weakness mitigation** | How well the candidate handles the team's shared weaknesses (immune = 2 pts, resists = 1.5 pts, neutral = 1 pt) |
| **Role balance** | Whether the candidate fills a missing role (sweeper / tank / balanced), derived from base stats |

**Smart filtering**
- Pre-evolutions excluded (BST < 500)
- Legendaries excluded (Articuno, Zapdos, Moltres, Mewtwo, Mew)
- Only the best candidate per evolution line is shown (no recommending both Machop and Machamp)

**What you can do with it**

1. **Random team** — picks 5 random Pokémon images, classifies them with the model, and recommends the best 6th
2. **Compare multiple teams** — evaluate several preset team compositions side by side
3. **Experiment with weights** — adjust `w_type`, `w_weak`, `w_role` to prioritize different strategies and see how the ranking changes

**Example output (all-sweeper team)**

```
Team: [Jolteon, Alakazam, Gengar, Charizard, Aerodactyl]
Shared weaknesses: Dark, Electric, Ghost, Ground, Ice, Rock, Water
Missing roles: balanced, tank

Balanced composite strategy → Top pick:
  Poliwrath  score=15.0 | new types=[Fighting, Water] | resists=[Dark, Ice, Rock, Water] | role=balanced | BST=510
```

---

## How everything connects

```
PokemonData/          ← 6,820 labeled images (150 Gen-I Pokémon)
        ↓
pokemon_team_builder_v2.ipynb   ← trains EfficientNet-B0
        ↓
best_model_v2.pth     ← saved model weights
        ↓
team_recommender.ipynb          ← loads model, classifies images, recommends 6th member
```
---

# Contributors

* **[Eloi Bernier](https://github.com/eloibernier)**
* **[Eduardo Tovilla](https://github.com/Eduardo-Tovilla)**
* **[Xander Deanhardt](https://github.com/XanderD24?tab=overview&from=2026-05-01&to=2026-05-23)**
