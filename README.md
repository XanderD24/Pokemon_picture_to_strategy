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

# Component 2 — Masked Team Transformer

A transformer model trained to help players build competitively viable teams for the newest regulation of Pokemon: **Regulation M-A of Pokemon Champions**. Every Pokemon, ability, move, and item is treated as a token; the model is trained as a *masked Pokemon model* — given some of a 6-Pokemon team, it predicts the remaining slots (species, held item, ability, and moveset of 4 moves).

## Table of Contents

- [Overview](#overview)
- [Repository Layout](#repository-layout)
- [Data](#data)
- [Architecture](#architecture)
- [Decoding-Time Constraints](#decoding-time-constraints)
- [Name Normalization Pipeline](#name-normalization-pipeline)
- [Training](#training)
- [Per-Slot Ensemble](#per-slot-ensemble)
- [Inference Utilities](#inference-utilities)
- [Saved Checkpoints](#saved-checkpoints)
- [Running the Notebooks](#running-the-notebooks)

## Overview

Given 1–5 visible Pokemon on a team, predict the rest. The transformer treats a team as an unordered set (no positional encoding); a learned mask token replaces the input vectors at masked positions, and the encoder output at each masked position feeds four prediction heads (species / ability / item / moves). Hard rules of legal team construction are enforced at decode time, not learned.

## Repository Layout

| File | Role |
| --- | --- |
| `prepare_raw_vectors.ipynb` | Builds the three raw-data-vector dictionaries (`pokemon_vectors.pkl`, `item_vectors.pkl`, `move_vectors.pkl`) from the scraped CSVs and reference dicts. |
| `train_masked_team_transformer.ipynb` | Single-slot training notebook (local file paths). |
| `train_masked_team_transformer_colab.ipynb` | Single-slot notebook adapted for Google Colab — mounts Drive and routes I/O through shared-drive prefixes. |
| `train_masked_team_transformer_per_slot_colab.ipynb` | Colab variant that trains a 5-model ensemble, one transformer per number of visible Pokemon. |
| `scrape_pokepastes.ipynb`, `scrape_vgenc.ipynb` | BeautifulSoup4 scrapers for the team-archive sources. |

## Data

### Scraped team pools

| Source | Teams | File |
| --- | --- | --- |
| [vgcpastes](https://twitter.com/vgcpastes) | 769 | `team_vectors.pkl` |
| [VGenC](https://victoryroad.pro) | 2,611 | `vgenc_team_vectors.pkl` |
| [LimitlessTCG VGC](https://limitlesstcg.com) (Champions Grand Festival) | 6,109 | `limitless_team_vectors.pkl` |

Each pool file is a `list[list[list]]`: a list of teams, each team a list of 6 Pokemon, each Pokemon a 7-element list of strings — `[species, ability, item, move1, move2, move3, move4]`. Every site was scraped with a custom BeautifulSoup4 scraper.

### Game reference data

Scraped from [serebii.net](https://serebii.net) and used to constrain decoding (the model can only predict abilities/moves that the chosen species can actually learn):

| File | Contents |
| --- | --- |
| `pokemon_dict.pkl` | Per-species typing, base stats, legal abilities, and legal moves |
| `ability_dict.pkl` | Text description of every ability |
| `moves_dict.pkl` | Type, power, priority, target, secondary effect of every move |
| `items_dict.pkl` | Text description of every item |

### Raw Data Vectors (RDVs)

Hand-engineered feature vectors that supplement each token's learned embedding. Produced by `prepare_raw_vectors.ipynb` and saved as three pickled dictionaries mapping `name → vector`.

**Pokemon RDV — 42 dim**
- 18-dim type one-hot
- 18-dim typechart defensive matchup. Built from `weaknesses.pkl`, `resistances.pkl`, `immunities.pkl`: each entry starts at 1.0, then multiplied by ×2 for every weakness, ×0.5 for every resistance, and set to 0 for any immunity. Finally divided by 2 for normalization.
- 6-dim z-scored base stats (HP, Attack, Defense, Sp. Atk, Sp. Def, Speed)

**Item RDV — 6 dim**

6 dummy-coded categories (each item was classified into one or more categories via an OpenAI API pass):

| Category | Meaning |
| --- | --- |
| `durability` | Helps a Pokemon endure hits it otherwise couldn't |
| `offense` | Helps it move faster or deal more damage |
| `hp_restore` | Restores HP mid-battle |
| `type_specific` | Only works for Pokemon or moves of a given type |
| `mega_stone` | Triggers Mega Evolution |
| `status_restore` | Heals a major or minor status condition |

**Move RDV — 44 dim**
- 18-dim type one-hot
- 3-dim category one-hot (Physical / Special / Status)
- 1-dim z-scored base power
- 1-dim priority (raw / 5)
- 5-dim target group one-hot (Selected Target / Self / Opponent-side AoE / Field / Ally)
- 16-dim effect-category multi-hot from `moves_classified_full.csv`:

| Effect | Meaning |
| --- | --- |
| `Damage` | Deals direct damage |
| `Redirection` | Redirects an opponent's attack from an ally to the user |
| `Buff` | Buffs the target's stats |
| `Offense` | Alters the target's offensive stats (Atk, Sp. Atk) |
| `Defense` | Alters the target's defensive stats (Def, Sp. Def) |
| `Speed` | Alters the target's Speed stat |
| `Debuff` | Inflicts stat debuffs on the target |
| `Heal` | Restores HP to the target |
| `Weather` | Sets Sun / Rain / Sandstorm / Snow |
| `Terrain` | Sets Electric / Psychic / Misty / Grassy Terrain |
| `Speed Control` | Alters speed for more than one Pokemon at a time |
| `Status` | Inflicts a major status condition |
| `Minor status` | Inflicts a minor / volatile status |
| `Protection` | Makes the user immune to damage or status for the turn |
| `Recharge` | Forces the user to recharge next turn |
| `Hazards` | Sets entry hazards on the opponent's side |

## Architecture

Each Pokemon's input vector concatenates learned embeddings with the corresponding RDVs:

```
[ Species Emb | Species RDV | Ability Emb | Item Emb | Item RDV | Moveset Emb | Moveset RDV ]
```

- Ability has only a learned embedding (no RDV was prepared).
- The Moveset embedding/RDV are the **mean** across the Pokemon's 4 moves. An optional multi-head attention layer over the 4 moves (`USE_MOVE_ATTENTION`) runs before averaging when enabled.
- The 6 per-Pokemon vectors are linearly projected to `D_MODEL` and passed through a stack of `nn.TransformerEncoder` layers. No positional encoding — a team is a set.
- A learned `mask_token` of size `input_dim` replaces the input at masked positions; the encoder output at each masked position feeds four prediction heads:
  - **Species** — softmax over the species vocab
  - **Ability** — softmax over the ability vocab
  - **Item** — softmax over the item vocab
  - **Moves** — multi-label sigmoid over the move vocab (predicts the unordered 4-move set)

### Embedding-size presets

| Preset | Species | Ability | Item | Move |
| --- | --- | --- | --- | --- |
| `small` | 16 | 8 | 8 | 16 |
| `medium` (default) | 32 | 16 | 16 | 32 |
| `large` | 64 | 32 | 32 | 64 |

### Transformer defaults

| Setting | Default |
| --- | --- |
| `D_MODEL` | 128 |
| `N_HEADS` | 4 |
| `N_LAYERS` | 2 |
| `DIM_FEEDFORWARD` | 256 |
| `DROPOUT` | 0.1 |

## Decoding-Time Constraints

`model.predict()` enforces every hard rule of legal team construction **before** sampling/argmax over the species head, so illegal predictions are impossible by construction:

1. **No duplicates (base ↔ Mega aware)** — a species sharing its family (base name) with any visible Pokemon is forbidden. Predicting Charizard-Mega-Y is blocked if base Charizard is already on the team, and vice versa.
2. **At most two Megas** — if ≥2 visible Pokemon are Mega forms, all Mega-form species are masked out of the species distribution.
3. **Mega → forced stone** — if the chosen species is a Mega form, the item prediction is overridden to the Mega Stone that enables that form.
4. **Mega Stone Assertion** — a Mega Stone is legal only for its base species *and* its Mega form. Mimikyu cannot be predicted holding Blastoisinite; no non-corresponding Mega Stone can be predicted for any species.
5. **Ability / move legality** (toggle: `ENFORCE_LEGALITY`) — when on, ability and move logits are masked to the chosen species' legal set per `pokemon_dict.pkl`, so the model self-corrects when it pairs a species with an ability or move it cannot actually have (e.g. predicting Incineroar then trying Drought).

### Sampling

The species head supports both top-k and nucleus (top-p) sampling with temperature:

| Setting | Behavior |
| --- | --- |
| `TOP_K = 0` | top-k disabled |
| `TOP_K ≥ 1` | keep the K highest-logit species, softmax over those, sample |
| `TOP_P = 0` (or ≥ 1) | nucleus disabled |
| `0 < TOP_P < 1` | keep the smallest set with cumulative probability ≥ P |
| `TEMP` | divides filtered logits before softmax |
| `SAMPLING_PHASE` | `"off"` / `"train"` / `"predict"` / `"both"` |

When sampling is off (or `n_predictions > 1`), decoding falls back to deterministic top-N from the constrained logits.

### Multi-prediction (best-of-n)

- `TRAIN_PREDICTIONS` / `TEST_PREDICTIONS` set how many candidates the model emits per slot.
- With `n > 1`, the loss becomes **best-of-n CE**: CE is zeroed on samples where the true label is among the model's top-n predictions, so weight updates only flow from cases where all n candidates were wrong.
- Evaluation accuracy is any-of-n: a slot counts as correct if the true label is among the n candidates.

## Name Normalization Pipeline

Scraped names are messy — nicknames, gender labels, prefix-style form names, inconsistent casing. Before any lookup, every species name passes through four layered passes, defined together in the Mega Stone / alternate-name handling cells:

1. **Parenthesized nickname extraction** — if the bare name isn't a real Pokemon, look for a parenthesized sub-string that is. `"Recto/Verso (Gardevoir)" → "Gardevoir"`, `"Warding Chime (Chimecho-Mega) (M)" → "Chimecho-Mega"`.
2. **Prefix-style alias map** — auto-built from `pokemon_dict.pkl`:
   - **Mega**: `"Mega Venusaur" → "Venusaur-Mega"`, `"Mega Charizard Y" → "Charizard-Mega-Y"`
   - **Alolan / Galarian / Hisuian / Paldean regionals**: `"Hisuian Zoroark" → "Zoroark-Hisui"`, `"Alolan Ninetales" → "Ninetales-Alola"`, `"Galarian Slowking" → "Slowking-Galar"`
   - **Paldean Tauros breeds**: `"Paldean Tauros" → "Tauros-Paldea-Combat"` (bare default), `"Paldean Tauros Blaze Breed" → "Tauros-Paldea-Blaze"`
   - **Rotom appliances**: `"Heat Rotom" → "Rotom-Heat"`, `"Wash Rotom" → "Rotom-Wash"`, etc.
   - **Lycanroc forms**: `"Midnight Lycanroc" → "Lycanroc-Midnight"`, `"Dusk Lycanroc" → "Lycanroc-Dusk"`
3. **Substring fallback** — used only when the name is still not canonical *and* the nickname parser already failed. Scans the string for any canonical name or alias appearing as a substring (longest match wins). Strips gender tags / annotations like `"Garchomp (M)" → "Garchomp"`, `"Wash Rotom (F)" → "Rotom-Wash"`. Pre-mapped aliases (`"Mega Venusaur"`) are detectable as substrings just like canonical names (`"Venusaur-Mega"`).
4. **Case-insensitive lookup** (toggle: `CASE_INSENSITIVE_NAMES`, default `True`) — all three passes accept any casing; the canonical (properly-cased) name is always what's returned. Recovers strays like `"Kommo-O" → "Kommo-o"`.

After the species name is canonical, **Mega Stone promotion** runs: if the Pokemon holds a Mega Stone, its species is renamed to the corresponding Mega form so it inherits the Mega's stats / typing RDV. The held item itself is preserved.

End-to-end this drops the UNK species rate from **5.6 % → 0.0 %** across all 56,934 Pokemon entries in the scraped pools.

## Training

### Loss (single-slot)

Standard cross-entropy on species/ability/item plus multi-label BCE on moves:

```
L = CE(species) + CE(ability) + CE(item) + BCE(moves)
```

With `n > 1` multi-prediction, the CE terms become best-of-n CE.

### Masking strategy

- Training: a uniformly-random slot is masked each `__getitem__` (data augmentation).
- Evaluation: deterministic — the last slot is masked, matching the stated task of predicting "the 6th member."

### Train / test split

The held-out test set is a random **20 %** of the LimitlessTCG teams (deterministic given `SEED`). Everything else — vgcpastes, VGenC, and the remaining 80 % of LimitlessTCG — is training. Edit `TEST_SOURCE` / `TEST_FRACTION` in the split cell to change this.

### Hyperparameter gridsearches

The single-slot notebooks ship two togglable gridsearches:

- **Architecture sweep** — `D_MODEL × N_HEADS × N_LAYERS × DIM_FEEDFORWARD`, with the other knobs pinned to vanilla defaults (1 prediction, no top-k/p, T = 1). Invalid combos (`D_MODEL % N_HEADS != 0`) are auto-skipped with a logged reason.
- **Behavior sweep** — `TRAIN_PREDICTIONS × TEST_PREDICTIONS × (TOP_K | TOP_P) × TEMP`. A shared sweep list routes `value ≥ 1` to `TOP_K` and `0 < value < 1` to `TOP_P` (the unused one is set to 0).

Both produce a per-configuration results table — final train loss, val loss, species / ability / item accuracy, move recall — and save it to CSV. If both toggles are off, the cell trains a single model as normal.

## Per-Slot Ensemble

The `*_per_slot_colab.ipynb` notebook trains five separate transformers — one per number of visible Pokemon — and saves each with a `_K` suffix where `K` is the number of slots given as input:

| Checkpoint | Visible | Predicts | Max-reward threshold |
| --- | --- | --- | --- |
| `masked_team_transformer_1.pt` | 1 | 5 | 3 of 5 species correct |
| `masked_team_transformer_2.pt` | 2 | 4 | 3 of 4 species correct |
| `masked_team_transformer_3.pt` | 3 | 3 | all 3 species correct |
| `masked_team_transformer_4.pt` | 4 | 2 | both species correct |
| `masked_team_transformer_5.pt` | 5 | 1 | the one species correct |

Each model emits 5 candidate predictions per masked slot. Its loss for a sample is **zero** when it meets the per-model threshold; otherwise the standard summed per-slot loss applies. Threshold = `min(3, K_remaining)`.

## Inference Utilities

### `model.predict(batch, n_predictions=N)`

Returns top-N candidate species / ability / item / moveset per masked slot, with **all** decoding constraints applied:

```python
pred = model.predict(batch, n_predictions=5)
pred["species"]   # [B, K, N] LongTensor
pred["ability"]   # [B, K, N]
pred["item"]      # [B, K, N] (Mega-stone-asserted; Mega forms get forced stone)
pred["moves"]     # list[B] of list[K] of list[N] LongTensor[k_moves]
pred["logits"]    # raw model output
```

### `model.get_embedding(name, kind)`

Returns the learned embedding for any vocabulary token. `kind ∈ {"species", "ability", "item", "move"}`. Useful for nearest-neighbour analysis, t-SNE / UMAP projections, or downstream usage of the learned token representations after training.

```python
emb = model.get_embedding("Garchomp", "species")    # tensor of shape [species_dim]
```

### Inspection cells

Each notebook has a final cell that randomly samples N validation teams and prints the masked slot's top-N predictions side-by-side with ground truth, marking species / ability / item correctness. The per-slot notebook additionally prints a `REACHED / missed` banner for each sample's max-reward threshold.

## Saved Checkpoints

`save_checkpoint(model, path)` bundles the trained weights together with the architecture config and all four vocabularies, so the model can be reloaded for inference without re-deriving anything:

```python
torch.save({
    "state_dict": model.state_dict(),
    "config":     {EMB, D_MODEL, N_HEADS, N_LAYERS, DIM_FEEDFORWARD, DROPOUT,
                   USE_MOVE_ATTENTION, MOVE_ATTENTION_HEADS, embed_config_name, ...},
    "vocabs":     {species, ability, item, move},
}, path)
```

`load_checkpoint(path)` rebuilds the model ready for `.eval()` use.

## Running the Notebooks

### Required input files

| Source | Files |
| --- | --- |
| `prepare_raw_vectors.ipynb` output | `pokemon_vectors.pkl`, `item_vectors.pkl`, `move_vectors.pkl` |
| Serebii scrape | `pokemon_dict.pkl`, `ability_dict.pkl`, `moves_dict.pkl`, `items_dict.pkl`, `weaknesses.pkl`, `resistances.pkl`, `immunities.pkl` |
| Team archives | `team_vectors.pkl`, `vgenc_team_vectors.pkl`, `limitless_team_vectors.pkl` |
| Classification CSVs | `items_dict_classified.csv`, `moves_classified_full.csv` |

### Local

```bash
pip install torch nbformat nbclient numpy pandas
```

Open `train_masked_team_transformer.ipynb` and run top-to-bottom. All input files must sit alongside the notebook.

### Google Colab

Open either Colab notebook in Colab. The first post-import cell mounts Google Drive; the next sets five directory prefixes:

```python
TEAM_DIR    = '/content/drive/Shared drives/.../Scraped Pokemon Teams/'
VECTOR_DIR  = '/content/drive/Shared drives/.../Transformer Ready Vectors/'
DICT_DIR    = '/content/drive/Shared drives/.../Game Info Dictionaries/'
MODEL_DIR   = '/content/drive/Shared drives/.../Models'

Edit those to match your Drive layout. Everything downstream reads through these prefixes, so adapting the notebook to a new environment is a single-cell change.
# Contributors

* **[Eloi Bernier](https://github.com/eloibernier)**
* **[Eduardo Tovilla](https://github.com/Eduardo-Tovilla)**
* **[Xander Deanhardt](https://github.com/XanderD24?tab=overview&from=2026-05-01&to=2026-05-23)**
