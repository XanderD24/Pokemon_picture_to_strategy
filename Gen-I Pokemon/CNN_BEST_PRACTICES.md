# Pokémon CNN — Best-Practice Review & Improvement Plan

Current pipeline: `EfficientNet-B0` (ImageNet pretrained) → 2-phase fine-tune (head-only 5 epochs @ 1e-3, then full @ 1e-4 with cosine LR + early stopping). 150 classes, ~6.8k images, 70/15/15 stratified split, light augmentation, top-1 = **94.0%**, top-5 = **99.2%**.

The model is already strong. Below are improvements ranked by **impact / effort**, then a discussion of tuning and alternative methods.

---

## Ranked Improvements

### Tier S — High impact, low effort (do these first)

#### 1. Stronger, modern augmentation
Your current augmentation (flip, ±15° rotation, ColorJitter) is conservative. Add:
- **RandAugment** or **TrivialAugmentWide** (`torchvision.transforms`) — one-line drop-in, consistently +1–3% on small datasets.
- **RandomResizedCrop(224, scale=(0.7, 1.0))** instead of fixed `Resize((224,224))` — teaches scale invariance and is the standard ImageNet recipe.
- **RandomErasing(p=0.25)** after `ToTensor` — cheap regularizer.
- **MixUp / CutMix** (`torchvision.transforms.v2.MixUp`) with α≈0.2 — strong regularizer for small per-class counts (you have only ~45 images/class).

**Effort:** ~10 lines. **Expected gain:** +1–3% top-1, better calibration.

#### 2. Label smoothing + better loss
Switch `CrossEntropyLoss()` → `CrossEntropyLoss(label_smoothing=0.1)`. Free regularization, reduces overconfidence (your softmax outputs are already pinned at 100%, a red flag for calibration).

#### 3. AdamW + weight decay + warmup
- Replace `Adam` with **`AdamW(..., weight_decay=1e-4)`** — Adam's L2 is broken; AdamW fixes it.
- Add **linear warmup for 1–2 epochs** before cosine annealing. Critical when unfreezing the backbone — prevents the large initial gradients from wrecking pretrained weights.
- Use `CosineAnnealingLR` over **the full Phase-2 length actually run**, not the max.

#### 4. Test-Time Augmentation (TTA)
Average predictions over `{original, hflip, 5 crops}`. ~5 lines, +0.5–1.5% top-1, zero retraining.

---

### Tier A — Medium impact, medium effort

#### 5. Discriminative / layer-wise learning rates
Backbone should learn slower than the new head. Use parameter groups:
```python
optim.AdamW([
    {'params': model.features.parameters(),  'lr': 1e-5},
    {'params': model.classifier.parameters(), 'lr': 1e-3},
], weight_decay=1e-4)
```
Often eliminates the need for a separate Phase 1.

#### 6. Bigger backbone (only if compute allows)
EfficientNet-B0 = 5M params. Try **EfficientNet-B3**, **ConvNeXt-Tiny**, or **ViT-B/16** pretrained. ConvNeXt-Tiny is the modern sweet spot (28M params, ~98% achievable on this kind of dataset).

#### 7. Stratified **K-Fold CV** for the final number
With only ~7 test images/class, your reported 94.04% has a 95% CI of roughly ±1.5%. Run 5-fold CV once at the end to get a trustworthy estimate.

#### 8. Mixed precision training (`torch.cuda.amp`)
2× faster, ~½ memory. Free win on any modern GPU:
```python
scaler = torch.cuda.amp.GradScaler()
with torch.cuda.amp.autocast(): ...
```

#### 9. Save by validation **loss** + restore best weights
You save by val-acc, which is noisier than val-loss on small val sets. Either monitor loss, or — better — keep `best_state = copy.deepcopy(model.state_dict())` in RAM and restore at the end.

---

### Tier B — Worth knowing, more effort

#### 10. EMA of weights
Maintain an exponential moving average of model weights (`torch.optim.swa_utils.AveragedModel` with `avg_fn` for EMA). Common in SOTA recipes, +0.3–1%.

#### 11. Class-balanced sampling / loss weighting
Imbalance is mild here (26–66 per class), but a `WeightedRandomSampler` with weights = `1/√count` removes the bias against rare classes (Alakazam, etc.).

#### 12. Proper experiment tracking
Use **Weights & Biases** or **TensorBoard** instead of `print`. Log per-class accuracy, confusion matrix, LR, gradient norms. Reproducibility and ablation become trivial.

#### 13. Calibration
Add **temperature scaling** on the validation set after training. Your top-1 probabilities are 100.0% even when wrong — a calibrated model is much more useful downstream (e.g., for the team-builder confidence).

#### 14. Hyperparameter search
Use **Optuna** with a small budget (20–30 trials) over: LR (1e-5 → 3e-3, log), weight_decay (1e-5 → 1e-3, log), label_smoothing (0.0–0.2), MixUp α (0.0–0.4), augmentation strength.

---

### Tier C — Low ROI for this project (skip unless required)

- Training from scratch (you do not have enough data).
- Custom CNN architecture (no upside vs pretrained).
- Self-supervised pretraining on Pokémon images — academically interesting, but ImageNet weights already transfer well.

---

## Hyperparameters that matter (and sensible search ranges)

| Hyperparameter | Current | Recommended range | Notes |
|---|---|---|---|
| Backbone LR (Phase 2) | 1e-4 | 1e-5 – 3e-4 | Most impactful; too high destroys pretrained features |
| Head LR (Phase 1) | 1e-3 | 5e-4 – 3e-3 | Less critical |
| Weight decay | 0 | 1e-5 – 1e-3 | Use AdamW |
| Batch size | 32 | 32 – 128 | Scale LR linearly if you raise it |
| Label smoothing | 0 | 0.05 – 0.15 | 0.1 is a safe default |
| MixUp α | – | 0.1 – 0.4 | 0.2 is standard |
| Dropout in head | 0 | 0.2 – 0.5 | EfficientNet head already has 0.2 |
| Image size | 224 | 224 or 256 | 256 + RandomResizedCrop(224) helps |
| Warmup epochs | 0 | 1 – 3 | Critical when unfreezing |
| Early stopping patience | 7 | 5 – 10 | Current value fine |

---

## Alternative methods and when they're interesting

| Method | When it shines | Trade-off |
|---|---|---|
| **Vision Transformer (ViT-B/16, DINOv2 features)** | When you want SOTA accuracy and have ≥medium GPU | Heavier; needs strong augmentation to avoid overfit |
| **ConvNeXt-Tiny / V2** | Modern CNN that often beats ViT at this scale | Slightly more params than B0 |
| **CLIP zero-shot / linear probe** | Quick baseline; works without training | Lower accuracy than fine-tuning |
| **DINOv2 frozen features + linear head** | Excellent for small datasets, ~3 min training | Backbone is large at inference |
| **k-NN over embeddings** (image retrieval) | Interpretable, easy to add new Pokémon without retraining | Slower inference; needs index |
| **Metric learning (ArcFace / triplet)** | Open-set: recognize Pokémon not in train set | More complex training loop |
| **Knowledge distillation** | Compress to small student for deployment | Needs a strong teacher first |
| **Ensembling (3–5 models, different seeds/backbones)** | Squeezing the last 1–2% for a final submission | 3–5× inference cost |
| **YOLO / DETR detector** | Multi-Pokémon images, bounding boxes (real "team" input) | Needs box annotations |
| **Segment Anything + classifier per crop** | When pictures contain multiple Pokémon | No retraining, but two-stage |

For your "5 Pokémon → recommend 6th" use case, the most natural extension is **detection** (one photo of a team rather than 5 separate crops). That's the biggest *product* improvement, separate from accuracy gains.

---

## Suggested Minimal Diff (highest ROI in <30 lines)

1. `train_transforms`: swap to `RandomResizedCrop(224) + RandAugment() + ToTensor + Normalize + RandomErasing(0.25)`.
2. `criterion = nn.CrossEntropyLoss(label_smoothing=0.1)`.
3. Optimizer: `AdamW` with parameter groups (head LR 1e-3, backbone LR 1e-5) + `weight_decay=1e-4`.
4. Scheduler: `SequentialLR(LinearLR warmup 2 epochs, CosineAnnealingLR)`.
5. Wrap training step in `torch.cuda.amp.autocast` + `GradScaler`.
6. Add TTA (hflip average) in `evaluate`.
7. Track everything in W&B.

Expected outcome: **96–98% top-1**, better-calibrated probabilities, ~2× faster training. Code complexity barely increases.
