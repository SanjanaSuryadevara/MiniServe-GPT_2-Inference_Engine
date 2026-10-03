"""FastAPI server: text generation endpoint + a benchmark dashboard."""
from __future__ import annotations

import copy
import json
import os
import time
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .generate import generate
from .quant import model_size_bytes, quantize_model
from .weights import get_tokenizer, load_gpt2, random_gpt2

MODEL_NAME = os.environ.get("MINISERVE_MODEL", "gpt2")  # "gpt2" or "random" (offline demo)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_PATH = Path(__file__).resolve().parent.parent / "results" / "results.json"

app = FastAPI(title="MiniServe", description="A GPT-2 inference engine built from scratch.")
_state: dict = {}


@app.on_event("startup")
def load_models():
    torch.manual_seed(0)
    fp = random_gpt2(device=DEVICE) if MODEL_NAME == "random" else load_gpt2(MODEL_NAME, DEVICE)
    _state["fp32"] = fp
    _state["int8"] = quantize_model(copy.deepcopy(fp))
    _state["tok"] = get_tokenizer(MODEL_NAME)
    _state["eos_id"] = getattr(_state["tok"], "eos_token_id", None)


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=2000)
    max_new_tokens: int = Field(40, ge=1, le=200)
    mode: str = Field("kv_cache", pattern="^(naive|kv_cache|int8)$")
    temperature: float = Field(0.0, ge=0.0, le=2.0)


class GenerateResponse(BaseModel):
    text: str
    mode: str
    n_generated: int
    seconds: float
    tokens_per_sec: float
    ttft_ms: float


@app.post("/generate", response_model=GenerateResponse)
def generate_endpoint(req: GenerateRequest):
    if "fp32" not in _state:
        raise HTTPException(503, "Model still loading")
    model = _state["int8"] if req.mode == "int8" else _state["fp32"]
    prompt_ids = _state["tok"].encode(req.prompt)
    if not prompt_ids:
        raise HTTPException(400, "Prompt encoded to zero tokens")
    result = generate(model, [prompt_ids], req.max_new_tokens, use_cache=req.mode != "naive",
                       temperature=req.temperature, top_k=50 if req.temperature > 0 else None,
                       eos_id=_state["eos_id"])
    text = _state["tok"].decode(result.tokens[0])
    return GenerateResponse(text=text, mode=req.mode, n_generated=result.n_generated,
                            seconds=round(result.seconds, 4),
                            tokens_per_sec=round(result.tokens_per_sec, 1),
                            ttft_ms=round(result.ttft * 1000, 1))


@app.get("/benchmark")
def benchmark_results():
    if not RESULTS_PATH.exists():
        raise HTTPException(404, "No benchmark results yet — run `python benchmark.py` first")
    return json.loads(RESULTS_PATH.read_text())


@app.get("/health")
def health():
    return {"status": "ok" if "fp32" in _state else "loading", "model": MODEL_NAME, "device": DEVICE,
            "weights_mb": round(model_size_bytes(_state["fp32"]) / 2**20, 1) if "fp32" in _state else None}


static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/")
def index():
    return FileResponse(static_dir / "index.html")
