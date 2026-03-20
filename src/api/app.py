import os, time
from contextlib import asynccontextmanager
from typing import List, Optional

import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.models.sentiment_model import AmazonLSTMSentiment
from src.utils.dataset import encode, load_vocab

CHECKPOINT = os.getenv("CHECKPOINT", "checkpoints/best_lstm.pt")
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN    = 256
IDX2LABEL  = {0: "positive", 1: "negative", 2: "neutral"}
EMOJI      = {0: "😊",       1: "😞",       2: "😐"}
STARS      = {0: "★★★★★",  1: "★☆☆☆☆",   2: "★★★☆☆"}

_model: Optional[AmazonLSTMSentiment] = None
_vocab: Optional[dict] = None
_meta:  dict  = {}
_start: float = time.time()
_stats: dict  = {"total": 0, "positive": 0, "negative": 0, "neutral": 0}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _vocab, _meta
    if not os.path.exists(CHECKPOINT):
        print(f"Checkpoint not found: {CHECKPOINT}")
        print(f"Run first:  python -m src.train")
    else:
        ckpt   = torch.load(CHECKPOINT, map_location=DEVICE)
        a      = ckpt.get("args", {})
        vpath  = ckpt.get("vocab_path", "data/processed/vocab.json")
        _vocab = load_vocab(vpath) if os.path.exists(vpath) else ckpt.get("vocab", {})

        _model = AmazonLSTMSentiment(
            vocab_size  = len(_vocab),
            embed_dim   = a.get("embed_dim",  128),
            hidden_dim  = a.get("hidden_dim", 256),
            num_classes = 3,
            num_layers  = a.get("num_layers", 2),
            dropout     = 0.0,
        ).to(DEVICE)
        _model.load_state_dict(ckpt["model_state"])
        _model.eval()

        _meta = {
            "arch":       "LSTM",
            "epoch":      ckpt.get("epoch"),
            "val_f1":     ckpt.get("val_f1"),
            "val_acc":    ckpt.get("val_acc"),
            "vocab_size": len(_vocab),
            "device":     str(DEVICE),
            "dataset":    "Amazon Polarity",
            "classes":    ["positive", "negative", "neutral"],
        }
        print(f"LSTM loaded  val_f1={ckpt.get('val_f1','?')}  vocab={len(_vocab):,}")

    yield
    print("Shutting down.")


app = FastAPI(
    title       = "Amazon LSTM Sentiment API",
    description = (
        "Bidirectional LSTM sentiment classifier trained on **Amazon Polarity** (3.6M reviews).\n\n"
        "Classes: `positive` · `negative` · `neutral`\n\n"
        "Architecture: `Embedding -> Bidir-LSTM (x2) -> Bahdanau Attention -> FC -> Softmax`"
    ),
    version  = "1.0.0",
    lifespan = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000,
                      example="Absolutely love this product — fast shipping and great quality!")

class BatchRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1, max_length=100,
                              example=["Great product!", "Terrible experience.", "It's okay I guess."])

class SentimentScores(BaseModel):
    positive: float = Field(example=0.9123)
    negative: float = Field(example=0.0512)
    neutral:  float = Field(example=0.0365)

class TokenWeight(BaseModel):
    token:  str   = Field(example="amazing")
    weight: float = Field(example=0.4231)

class PredictResponse(BaseModel):
    text:           str
    sentiment:      str             = Field(example="positive")
    emoji:          str             = Field(example="😊")
    stars:          str             = Field(example="★★★★★")
    confidence:     float           = Field(example=0.9123)
    scores:         SentimentScores
    top_tokens:     List[TokenWeight]
    full_attention: List[TokenWeight]

class BatchResponse(BaseModel):
    results: List[PredictResponse]
    count:   int
    arch:    str = "LSTM"

class HealthResponse(BaseModel):
    status:         str  = "ok"
    model_loaded:   bool
    arch:           str  = "LSTM"
    uptime_seconds: int

class StatsResponse(BaseModel):
    total:    int
    positive: dict
    negative: dict
    neutral:  dict


def _infer(text: str) -> PredictResponse:
    if _model is None or _vocab is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run: python -m src.train")

    ids = encode(text, _vocab, MAX_LEN)
    if not ids:
        raise HTTPException(status_code=422, detail="Text is empty after tokenization.")

    x       = torch.tensor([ids], dtype=torch.long).to(DEVICE)
    lengths = torch.tensor([len(ids)])

    with torch.no_grad():
        logits, attn = _model(x, lengths)
        probs = F.softmax(logits, dim=-1)[0].cpu().tolist()
        pred  = int(logits.argmax(-1).item())

    tokens    = text.lower().split()[:MAX_LEN]
    attn_vals = attn[0, :len(tokens)].cpu().tolist()
    label     = IDX2LABEL[pred]

    _stats["total"] += 1
    _stats[label]   += 1

    all_attn   = [TokenWeight(token=t, weight=round(w, 4)) for t, w in zip(tokens, attn_vals)]
    top_tokens = sorted(all_attn, key=lambda x: x.weight, reverse=True)[:10]

    return PredictResponse(
        text           = text[:200],
        sentiment      = label,
        emoji          = EMOJI[pred],
        stars          = STARS[pred],
        confidence     = round(max(probs), 4),
        scores         = SentimentScores(
                           positive=round(probs[0], 4),
                           negative=round(probs[1], 4),
                           neutral =round(probs[2], 4),
                         ),
        top_tokens     = top_tokens,
        full_attention = all_attn,
    )


@app.get("/api/health", response_model=HealthResponse, tags=["System"], summary="Health check")
async def health():
    return HealthResponse(
        status         = "ok",
        model_loaded   = _model is not None,
        arch           = "LSTM",
        uptime_seconds = round(time.time() - _start),
    )


@app.get("/api/model/info", tags=["System"], summary="Loaded model metadata")
async def model_info():
    if not _meta:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    return _meta


@app.get("/api/stats", response_model=StatsResponse, tags=["System"], summary="Session prediction statistics")
async def stats():
    total = max(_stats["total"], 1)
    return StatsResponse(
        total    = _stats["total"],
        positive = {"count": _stats["positive"], "pct": round(_stats["positive"] / total * 100, 1)},
        negative = {"count": _stats["negative"], "pct": round(_stats["negative"] / total * 100, 1)},
        neutral  = {"count": _stats["neutral"],  "pct": round(_stats["neutral"]  / total * 100, 1)},
    )


@app.post("/api/predict", response_model=PredictResponse, tags=["Inference"], summary="Analyze a single Amazon review")
async def predict(req: PredictRequest):
    return _infer(req.text.strip())


@app.post("/api/predict/batch", response_model=BatchResponse, tags=["Inference"], summary="Analyze up to 100 reviews")
async def predict_batch(req: BatchRequest):
    if len(req.texts) > 100:
        raise HTTPException(status_code=422, detail="Maximum 100 texts per batch.")
    results = [_infer(t.strip()) for t in req.texts if t.strip()]
    return BatchResponse(results=results, count=len(results))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.app:app", host="0.0.0.0", port=8000, reload=True, workers=1)
