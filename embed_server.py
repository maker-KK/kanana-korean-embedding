#!/usr/bin/env python3
"""
Kanana Korean Embedding Server
==============================
Lightweight FastAPI server for Kakao's Kanana-Nano-2.1b-Embedding model.

Provides:
  - /embed              — Native embedding endpoint
  - /api/embeddings     — Ollama-compatible (single prompt)
  - /api/embed          — Ollama-compatible (batch)
  - /api/tags           — Ollama-compatible model listing
  - /health             — Health check

Supports MPS (Apple Silicon), CUDA, and CPU backends.

Usage:
  python embed_server.py                          # default: localhost:11435
  MODEL_DIR=/path/to/model python embed_server.py # custom model path
"""

import os
import sys
import time
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Union, Optional

MODEL_DIR = os.environ.get("MODEL_DIR", os.path.dirname(os.path.abspath(__file__)))
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.path.insert(0, MODEL_DIR)

from transformers import AutoTokenizer, AutoModel
import torch

app = FastAPI(
    title="Kanana Korean Embedding API",
    description="Local Korean embedding server using Kakao Kanana-Nano-2.1b-Embedding",
    version="1.0.0",
)

_model = None
_tokenizer = None
_device = "cpu"

MODEL_NAME = "kanana-nano-2.1b-embedding"


# ── Request / Response schemas ───────────────────────────────────────

class EmbedRequest(BaseModel):
    input: List[str]
    instruction: str = ""


class EmbedResponse(BaseModel):
    model: str
    embeddings: List[List[float]]


class OllamaEmbedRequest(BaseModel):
    model: str = MODEL_NAME
    prompt: Optional[str] = None
    input: Optional[Union[str, List[str]]] = None
    instruction: str = ""


class OllamaEmbedResponse(BaseModel):
    model: str
    embedding: Optional[List[float]] = None
    embeddings: Optional[List[List[float]]] = None


# ── Core logic ───────────────────────────────────────────────────────

def load_model():
    global _model, _tokenizer, _device

    if torch.backends.mps.is_available():
        _device = "mps"
    elif torch.cuda.is_available():
        _device = "cuda"
    else:
        _device = "cpu"

    _tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    _model = AutoModel.from_pretrained(MODEL_DIR, trust_remote_code=True).to(_device)
    _model.eval()
    print(f"[INFO] Model loaded on {_device}, embedding dim = {_model.config.hidden_size}")


def get_embeddings(texts: List[str], instruction: str = "") -> np.ndarray:
    if instruction:
        formatted = [f"Instruct: {instruction}\nQuery: {t.strip()}" for t in texts]
    else:
        formatted = [t.strip() for t in texts]

    inputs = _tokenizer(
        formatted,
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    ).to(_device)

    with torch.no_grad():
        outputs = _model(**inputs)

    mask = inputs["attention_mask"].unsqueeze(-1).float()
    embeddings = (outputs.last_hidden_state * mask).sum(1) / mask.sum(1)
    return embeddings.cpu().numpy()


# ── Endpoints ────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    load_model()


@app.post("/embed")
async def embed(request: EmbedRequest) -> EmbedResponse:
    """Native embedding endpoint — batch input, batch output."""
    start = time.time()
    embs = get_embeddings(request.input, request.instruction)
    elapsed = time.time() - start
    print(f"[EMBED] {len(request.input)} texts, {elapsed:.3f}s")
    return EmbedResponse(
        model=MODEL_NAME,
        embeddings=[e.tolist() for e in embs],
    )


@app.post("/api/embeddings")
async def ollama_embeddings_v1(request: OllamaEmbedRequest):
    """Ollama /api/embeddings compatible — single prompt → single embedding."""
    text = request.prompt or (request.input if isinstance(request.input, str) else "")
    embs = get_embeddings([text], request.instruction)
    return OllamaEmbedResponse(
        model=MODEL_NAME,
        embedding=embs[0].tolist(),
    )


@app.post("/api/embed")
async def ollama_embed_v2(request: OllamaEmbedRequest):
    """Ollama /api/embed compatible — array input → array embeddings."""
    if isinstance(request.input, list):
        texts = request.input
    elif request.input:
        texts = [request.input]
    elif request.prompt:
        texts = [request.prompt]
    else:
        texts = [""]
    embs = get_embeddings(texts, request.instruction)
    return OllamaEmbedResponse(
        model=MODEL_NAME,
        embeddings=[e.tolist() for e in embs],
    )


@app.get("/api/tags")
async def ollama_tags():
    """Ollama /api/tags compatible — model listing for discovery."""
    return {
        "models": [
            {
                "name": MODEL_NAME,
                "model": MODEL_NAME,
                "size": 4_200_000_000,
                "details": {"family": "kanana", "parameter_size": "2.1B"},
            }
        ]
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "model": MODEL_NAME, "device": _device}


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 11435))
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="info")
