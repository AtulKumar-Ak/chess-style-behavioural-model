
# backend/main.py

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from predictor import ChessPredictor
import sys
sys.path.append(str(Path(__file__).parent.parent))


BASE_DIR          = Path(__file__).parent
CHECKPOINT_PATH   = BASE_DIR / "checkpoints" / "gpt_best.pt"
MOVE_VOCAB_PATH   = BASE_DIR / "vocab" / "move_vocab.json"
PLAYER_VOCAB_PATH = BASE_DIR / "vocab" / "player_vocab.json"


app = FastAPI(
    title="Chess Style Predictor",
    description="Predicts player-specific next moves using a behavioral transformer.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "http://localhost:5174"],
    allow_methods=["*"],
    allow_headers=["*"],
)


predictor: ChessPredictor = None

@app.on_event("startup")
def load_model():
    global predictor
    print("[startup] loading model...")
    predictor = ChessPredictor(
        checkpoint_path=str(CHECKPOINT_PATH),
        move_vocab_path=str(MOVE_VOCAB_PATH),
        player_vocab_path=str(PLAYER_VOCAB_PATH),
        temperature=0.7,
        top_k=5,
    )
    print("[startup] model ready")

# REQUEST / RESPONSE MODELS

class PredictRequest(BaseModel):
    moves:       list[str]   # UCI move strings e.g. ["e2e4", "e7e5"]
    player_name: str         # e.g. "MagnusCarlsen"
    top_k:       int = 5

class PredictResponse(BaseModel):
    predictions: list[dict]
    player_name: str
    move_count:  int

class PGNRequest(BaseModel):
    pgn: str

class PGNResponse(BaseModel):
    moves: list[str]   # UCI strings parsed from PGN

class PlayersResponse(BaseModel):
    players: list[str]


@app.get("/health")
def health():
    """Quick liveness check."""
    return {"status": "ok", "model_loaded": predictor is not None}


@app.get("/players", response_model=PlayersResponse)
def get_players():
    """Returns all available player names."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"players": predictor.get_available_players()}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    """
    Given a list of UCI moves and a player name,
    returns the top-k predicted next moves with probabilities.
    """
    print(f"\n[Request] Predict next move for player '{request.player_name}' after {len(request.moves)} moves (top_k={request.top_k})")
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        predictions = predictor.predict(
            moves=request.moves,
            player_name=request.player_name,
            top_k=request.top_k,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

    return {
        "predictions": predictions,
        "player_name": request.player_name,
        "move_count":  len(request.moves),
    }


@app.post("/parse-pgn", response_model=PGNResponse)
def parse_pgn(request: PGNRequest):
    """
    Parses a PGN string and returns a list of UCI move strings.
    Used by the frontend to load a recorded game.
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        moves = predictor.moves_from_pgn(request.pgn)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {"moves": moves}