# Pokémon Classifier — Web App

A small **local-only** FastAPI site for the v3 EfficientNet-B0 model (95.6% top-1 on 150 Gen-I Pokémon). Drop in up to 5 images, get the predicted Pokémon for each with top-3 confidences.

![stack](https://img.shields.io/badge/stack-FastAPI%20+%20vanilla%20JS-ee1515) ![model](https://img.shields.io/badge/model-EfficientNet--B0-ffcb05?labelColor=1a1a2e)

---

## What's inside

```
webapp/
├── app.py             # FastAPI server (GET /, POST /predict, /docs)
├── model_utils.py     # Model load, eval transforms, top-3 prediction
├── static/
│   └── index.html     # Pokédex-themed UI (HTML + inline CSS + vanilla JS)
├── requirements.txt
└── README.md          # this file
```

**Backend** — loads `../Gen-I Pokemon/best_model_v3.pth` once at startup, exposes `POST /predict` that takes 1–5 image files and returns `{filename, top3: [{name, confidence}]}` for each.

**Frontend** — single-page UI with:
- Drag-and-drop or click-to-pick image upload (max 5).
- Per-image card with the thumbnail, a confidence-colored top-1 badge (green ≥80%, amber 50–80%, red <50%), and a top-3 bar chart.
- Clear button, loading spinner, friendly empty/error states.

---

## Setup (colleague-friendly)

### Prerequisites
- **Python 3.10+** (3.12 tested).
- The model weights `Gen-I Pokemon/best_model_v3.pth` sitting next to this folder.
- The `Gen-I Pokemon/PokemonData/` folder (used only to read the 150 class names at startup — no images sent over the network).

### 1. Clone / pull the repo
```bash
git clone <repo-url>
cd Pokemon_picture_to_strategy/webapp
```

### 2. (Recommended) Create a virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

> ⚠️ If you see `Form data requires "python-multipart" to be installed. It seems you installed "multipart" instead.` — uninstall the wrong package:
> ```bash
> pip uninstall -y multipart
> ```
> The two PyPI packages (`multipart` vs `python-multipart`) share an import name and conflict.

### 4. Run the server
```bash
uvicorn app:app --reload
```

Then open **<http://localhost:8000>** in your browser.

Auto-generated API docs are at **<http://localhost:8000/docs>**.

The first prediction takes ~1–2 seconds (model warmup); subsequent ones are under 200 ms on Apple Silicon (MPS) or CUDA, ~1 s on CPU.

---

## How it picks a device

`model_utils.pick_device()` chooses in this order:

1. **CUDA** if `torch.cuda.is_available()` — NVIDIA GPUs.
2. **MPS** if `torch.backends.mps.is_available()` — Apple Silicon (M1/M2/M3/M4).
3. **CPU** otherwise — works fine, just slower.

The chosen device is printed in the uvicorn log on startup.

---

## API reference

### `GET /`
Serves the UI (`static/index.html`).

### `POST /predict`
- **Body:** `multipart/form-data` with one or more `files` fields (max 5).
- **Returns:**
  ```json
  {
    "results": [
      {
        "filename": "pikachu.jpg",
        "top3": [
          {"name": "Pikachu",  "confidence": 0.952},
          {"name": "Raichu",   "confidence": 0.003},
          {"name": "Eevee",    "confidence": 0.002}
        ]
      }
    ]
  }
  ```
- **Errors:** `400` for no files, too many files, or unrecognized image format.

### Example with `curl`
```bash
curl -X POST http://localhost:8000/predict \
  -F "files=@path/to/pikachu.jpg" \
  -F "files=@path/to/charizard.jpg"
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `RuntimeError: Form data requires "python-multipart"...` | `pip uninstall -y multipart` (the wrong package was installed) |
| `FileNotFoundError: ... best_model_v3.pth` | Make sure `Gen-I Pokemon/best_model_v3.pth` exists. Either retrain via the notebook or ask for the file. |
| `FileNotFoundError: ... PokemonData` | The `PokemonData/` directory must be present so the server can read the class list. Restore from the dataset source. |
| Port 8000 already in use | `uvicorn app:app --reload --port 8001` |
| Predictions look random | You're loading the wrong checkpoint — verify the uvicorn startup line prints `best_model_v3.pth` and `150 classes`. |
| Slow on Mac | First request is always slow (warmup). Confirm the log says `device=mps`, not `cpu`. |

---

## What's intentionally out of scope

- **No authentication, no HTTPS, no deployment** — this is a localhost demo.
- **No persistence** — uploaded images live in memory only.
- **No team-builder / 6th-Pokémon recommendation** — that logic stays in `../Gen-I Pokemon/team_recommender.ipynb`.
- **No TTA at inference** — single forward pass for low latency; the v3 weights are strong enough on their own.

---

## Credits

- Model: EfficientNet-B0 fine-tuned per `CNN_BEST_PRACTICES.md` Tier S recipe (see `../Gen-I Pokemon/README.md`).
- Backend: FastAPI + Uvicorn.
- Frontend: Vanilla HTML/CSS/JS — no build step, no framework.
