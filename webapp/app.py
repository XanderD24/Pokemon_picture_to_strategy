import io
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Body, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from model_utils import get_class_names, load_model, pick_device, predict
from team_transformer import TeamPredictor, map_cnn_species_to_vocab

ROOT       = Path(__file__).resolve().parent
REPO       = ROOT.parent
DATA_DIR   = REPO / 'Gen-I Pokemon' / 'PokemonData'
WEIGHTS    = REPO / 'Gen-I Pokemon' / 'best_model_v3.pth'
STATIC_DIR = ROOT / 'static'

MAX_FILES = 5

state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    device = pick_device()
    class_names = get_class_names(DATA_DIR)
    model = load_model(WEIGHTS, num_classes=len(class_names), device=device)
    state['device'] = device
    state['class_names'] = class_names
    state['model'] = model
    print(f'Loaded {WEIGHTS.name} on device={device}  ({len(class_names)} classes)')

    # Team transformer is optional: the classifier must work even if its
    # artifacts (checkpoint / RDV vectors) are missing.
    try:
        state['predictor'] = TeamPredictor()
        print('Loaded masked-team transformer (recommend-6th enabled)')
    except Exception as exc:  # missing files, load error, etc.
        state['predictor'] = None
        print(f'Team transformer unavailable, recommend-6th disabled: {exc}')

    yield
    state.clear()


app = FastAPI(title='Pokémon Classifier', lifespan=lifespan)
app.mount('/static', StaticFiles(directory=STATIC_DIR), name='static')


@app.get('/')
def index():
    return FileResponse(STATIC_DIR / 'index.html')


@app.post('/predict')
async def predict_endpoint(files: list[UploadFile] = File(...)):
    if not files:
        raise HTTPException(400, 'No files uploaded.')
    if len(files) > MAX_FILES:
        raise HTTPException(400, f'Too many files (max {MAX_FILES}).')

    results = []
    for f in files:
        raw = await f.read()
        try:
            img = Image.open(io.BytesIO(raw))
        except UnidentifiedImageError:
            raise HTTPException(400, f'{f.filename}: not a valid image.')
        top3 = predict(state['model'], img, state['class_names'], state['device'], top_k=3)
        results.append({'filename': f.filename, 'top3': top3})

    return {'results': results}


@app.get('/vocab')
def vocab():
    predictor = state.get('predictor')
    if predictor is None:
        raise HTTPException(503, 'Team transformer not available.')
    return predictor.vocab_lists()


@app.get('/defaults')
def defaults():
    """Per-species default sets (most-common ability/item/moves) for UI pre-fill."""
    predictor = state.get('predictor')
    if predictor is None:
        raise HTTPException(503, 'Team transformer not available.')
    return {'defaults': predictor.species_defaults()}


@app.post('/map_species')
def map_species(names: list[str] = Body(..., embed=True)):
    """Map CNN species labels to transformer vocab keys (None if no match)."""
    predictor = state.get('predictor')
    if predictor is None:
        raise HTTPException(503, 'Team transformer not available.')
    species_vocab = predictor.vocabs['species']
    return {'mapped': {n: map_cnn_species_to_vocab(n, species_vocab) for n in names}}


@app.post('/recommend')
def recommend(team: list[dict] = Body(..., embed=True)):
    """team: 1-5 dicts {species, ability, item, moves:[...]} -> suggested 6th."""
    predictor = state.get('predictor')
    if predictor is None:
        raise HTTPException(503, 'Team transformer not available.')
    if not team:
        raise HTTPException(400, 'Provide at least one Pokémon.')
    if len(team) > 5:
        raise HTTPException(400, 'Provide at most 5 Pokémon.')
    candidates = predictor.predict_sixth(team, n_candidates=5)
    return {'candidates': candidates}
