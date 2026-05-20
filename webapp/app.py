import io
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from model_utils import get_class_names, load_model, pick_device, predict

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
