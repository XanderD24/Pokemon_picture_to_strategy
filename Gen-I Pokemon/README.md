# Pokémon Team Builder — Project Overview

A computer vision + game strategy project that combines deep learning image classification with Pokémon team optimization. The system identifies Pokémon from photos and recommends the best 6th team member based on type coverage, weakness mitigation, and role balance.

---

## 🆕 What's new in v3 (Tier S improvements)

The training notebook was upgraded with four high-impact, low-effort improvements drawn from `CNN_BEST_PRACTICES.md`. All changes are documented in-notebook with markdown cells (`🆕 v3 Change N:`) explaining *what* changed, *why*, and the *expected effect*.

| # | Change | Notebook cell |
|---|--------|---------------|
| 1 | **Modern augmentation** — `RandomResizedCrop` + `RandAugment` + `RandomErasing` (replaces flip + rotation + ColorJitter) | Transforms cell |
| 2 | **MixUp / CutMix** collate on the train loader (α=0.2 MixUp or α=1.0 CutMix per batch) | New collate cell |
| 3 | **Label smoothing** (`label_smoothing=0.1`) on `CrossEntropyLoss` — fixes the "100% confidence on wrong predictions" pathology | Phase 1 setup |
| 4 | **AdamW + weight decay + linear warmup** — replaces `Adam`, adds `weight_decay=1e-4`, prepends 2-epoch `LinearLR` warmup before `CosineAnnealingLR` | Phase 1 & Phase 2 |
| 5 | **Test-Time Augmentation** (`tta_evaluate`) — averages softmax over original + horizontal flip | Train/eval helpers cell |

A new **"v3 vs v2 — Head-to-Head Comparison"** section at the end of the notebook automatically:
- Loads both checkpoints (`best_model_v2.pth` and `best_model_v3.pth`) and runs them on the same test set.
- Computes a metrics table: top-1, top-5, CE loss, **ECE (Expected Calibration Error)**, and mean confidence on wrong predictions.
- Runs **McNemar's paired test** for statistical significance.
- Plots per-class accuracy difference (red bars = classes where v3 regressed).
- Plots reliability diagrams for v2 and v3+TTA.
- Prints a final **VERDICT** against three decision rules (significance, calibration, no major regressions).

### Results

| Metric | v2 (baseline) | v3 (Tier S) |
|---|---|---|
| Test top-1 | 94.04% | **95.61%** |
| Test top-5 | 99.22% | 99.12% |
| McNemar p-value (vs v2) | — | **0.017** (significant) |
| Discordant pairs | — | 31 v3-only correct vs 14 v2-only |

The +1.7pp accuracy gain is **statistically significant**. ECE got worse (label smoothing + MixUp make the model *under-confident* by design — a fixable artifact via temperature scaling, listed as a Tier B follow-up in `CNN_BEST_PRACTICES.md`).

The device selection in the notebook now picks **`cuda → mps → cpu`** so it runs natively on Apple Silicon Macs.

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

### `best_model_v2.pth` — Baseline checkpoint (frozen)

The original v2 checkpoint, preserved so the v3 notebook can compare against it side-by-side. Do not overwrite.

- **Architecture:** EfficientNet-B0 (pretrained on ImageNet, fine-tuned for 150 Pokémon classes)
- **Output:** probability distribution over 150 Pokémon
- **Top-1:** 94.04% on the held-out test set
- Loaded by `team_recommender.ipynb` and by the v3 comparison section

---

### `best_model_v3.pth` — v3 checkpoint (current best)

Produced by re-running `pokemon_team_builder_v2.ipynb` after the Tier S upgrades.

- **Same architecture as v2** (EfficientNet-B0, 150 outputs) — drop-in compatible
- **Top-1:** **95.61%** on the same test split (+1.7pp, McNemar p=0.017)
- **Trained on Apple M4 Pro via MPS** (also works on CUDA or CPU)
- Loaded by the FastAPI web app at `../webapp/app.py` and by the v3 comparison cells

---

### `CNN_BEST_PRACTICES.md` — Improvement roadmap

The ranked review of the v2 model that motivated the v3 changes. Tier S items (1–4) are now implemented; Tier A and B items (discriminative learning rates, EMA, temperature scaling, etc.) remain as future work.

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
pokemon_team_builder_v2.ipynb   ← trains EfficientNet-B0 (v3 Tier S recipe)
        ↓
best_model_v3.pth     ← current best (95.6% top-1)
best_model_v2.pth     ← frozen baseline (94.0% top-1, kept for comparison)
        ↓
team_recommender.ipynb          ← loads model, classifies images, recommends 6th member
../webapp/                      ← FastAPI demo site (loads best_model_v3.pth)
```
