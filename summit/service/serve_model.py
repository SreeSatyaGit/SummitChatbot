#!/usr/bin/env python
"""FastAPI model worker — loads a HF pipeline and serves /generate.

- Auto-detects model family:
    * causal LM (e.g., GPT-2)  -> text-generation
    * encoder-decoder (e.g., T5)-> text2text-generation
- Defaults to your GPT-2 finetune path: fine_tuning/models/gpt2-onboarding
- Designed to pair with the onboarding service which sends short prompts.

Env:
- MODEL_PATH: optional local or hub path to load
"""

from __future__ import annotations

import os
import re
import argparse
import logging
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

try:
    from transformers import (
        AutoConfig,
        AutoTokenizer,
        pipeline,
        Pipeline,
    )
except Exception:
    AutoConfig = None
    AutoTokenizer = None
    pipeline = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("serve_model")

app = FastAPI(title="model-worker")

class GenerateRequest(BaseModel):
    prompt: str
    # modern controls (tuned for a single concise question)
    max_new_tokens: Optional[int] = 64
    temperature: Optional[float] = 0.8
    top_k: Optional[int] = 50
    top_p: Optional[float] = 0.9
    repetition_penalty: Optional[float] = 1.1
    no_repeat_ngram_size: Optional[int] = 3
    num_return_sequences: Optional[int] = 1
    do_sample: Optional[bool] = True  # default to a bit of variety
    # legacy compat
    max_length: Optional[int] = None
    task: Optional[str] = None  # ignored

class GenerateResponse(BaseModel):
    generated_texts: List[str]

MODEL_PIPELINE: Optional[Pipeline] = None
TOKENIZER = None
IS_ENCODER_DECODER = False
MODEL_NAME_OR_PATH: str = "fine_tuning/models/gpt2-onboarding"

# Stop sequences — ensure we don't run into the next user turn
STOP_SEQUENCES = ["\n<|user|>", "<|user|>", "\n\n"]
TAGY = re.compile(r"<[^>]*>")

def _choose_pipeline_and_tokenizer(model_path: str):
    """Detect architecture and return (pipeline_task, tokenizer)."""
    global IS_ENCODER_DECODER
    if AutoConfig is None or AutoTokenizer is None:
        raise RuntimeError("transformers is not available in the image")

    cfg = AutoConfig.from_pretrained(model_path)
    IS_ENCODER_DECODER = bool(getattr(cfg, "is_encoder_decoder", False))

    tok = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    # Ensure special tokens exist for chat style
    additional = []
    for sp in ["<|user|>", "<|assistant|>"]:
        if tok.convert_tokens_to_ids(sp) is None:
            additional.append(sp)
    if additional:
        tok.add_special_tokens({"additional_special_tokens": additional})

    # GPT-2 family often lacks pad token → map to eos
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token

    task = "text2text-generation" if IS_ENCODER_DECODER else "text-generation"
    return task, tok

def load_pipeline(model_path: Optional[str] = None) -> Pipeline:
    """Load once; prefer GPU with device_map='auto'; fall back to CPU."""
    global MODEL_PIPELINE, MODEL_NAME_OR_PATH, TOKENIZER

    if MODEL_PIPELINE is not None:
        return MODEL_PIPELINE

    if pipeline is None:
        raise RuntimeError("transformers is not available in the image")

    model_to_load = model_path or os.environ.get("MODEL_PATH") or MODEL_NAME_OR_PATH
    MODEL_NAME_OR_PATH = model_to_load

    logger.info("Loading model from: %s", model_to_load)
    task, tok = _choose_pipeline_and_tokenizer(model_to_load)

    try:
        MODEL_PIPELINE = pipeline(
            task,
            model=model_to_load,
            tokenizer=tok,
            device_map="auto",
        )
        TOKENIZER = tok
        logger.info("Pipeline loaded with device_map=auto (%s)", task)
        # If we added special tokens on a causal LM, resize embeddings
        if not IS_ENCODER_DECODER and hasattr(MODEL_PIPELINE.model, "resize_token_embeddings"):
            MODEL_PIPELINE.model.resize_token_embeddings(len(TOKENIZER))
        return MODEL_PIPELINE
    except Exception as exc:
        logger.error("device_map=auto failed; retrying on CPU: %s", exc)
        MODEL_PIPELINE = pipeline(
            task,
            model=model_to_load,
            tokenizer=tok,
            device=-1,
        )
        TOKENIZER = tok
        logger.info("Pipeline loaded on CPU (%s)", task)
        if not IS_ENCODER_DECODER and hasattr(MODEL_PIPELINE.model, "resize_token_embeddings"):
            MODEL_PIPELINE.model.resize_token_embeddings(len(TOKENIZER))
        return MODEL_PIPELINE

def _apply_stops(text: str) -> str:
    cut = len(text)
    for stop in STOP_SEQUENCES:
        idx = text.find(stop)
        if idx != -1:
            cut = min(cut, idx)
    return text[:cut].rstrip()

def _sanitize(text: str) -> str:
    t = TAGY.sub(" ", text).replace("\r", " ").replace("\n", " ").strip()
    t = re.sub(r"\s+", " ", t)
    return t

@app.on_event("startup")
def on_startup() -> None:
    model_path = os.environ.get("MODEL_PATH")
    try:
        load_pipeline(model_path)
    except Exception as e:
        logger.exception("Model failed to load on startup: %s", e)

@app.get("/", tags=["health"])
def health() -> dict:
    return {"status": "ok", "model": MODEL_NAME_OR_PATH, "encoder_decoder": IS_ENCODER_DECODER}

@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest) -> GenerateResponse:
    if MODEL_PIPELINE is None:
        try:
            load_pipeline(os.environ.get("MODEL_PATH"))
        except Exception as e:
            logger.exception("Model not available: %s", e)
            raise HTTPException(status_code=500, detail="Model not available")

    prompt = req.prompt
    if not isinstance(prompt, str) or not prompt.strip():
        raise HTTPException(status_code=422, detail="Empty prompt")

    # Prefer max_new_tokens; if only max_length is provided, pass through (Seq2Seq compatibility)
    gen_kwargs: Dict[str, Any] = dict(
        do_sample=bool(req.do_sample),
        temperature=float(req.temperature or 0.8),
        top_k=int(req.top_k or 50),
        top_p=float(req.top_p or 0.9),
        repetition_penalty=float(req.repetition_penalty or 1.1),
        no_repeat_ngram_size=int(req.no_repeat_ngram_size or 3),
        num_return_sequences=int(req.num_return_sequences or 1),
        pad_token_id=TOKENIZER.pad_token_id if TOKENIZER is not None else None,
        eos_token_id=TOKENIZER.eos_token_id if TOKENIZER is not None else None,
    )
    if req.max_length is not None:
        gen_kwargs["max_length"] = int(req.max_length)
    else:
        # keep outputs short—just a single question
        max_new = int(req.max_new_tokens or 64)
        gen_kwargs["max_new_tokens"] = max(16, min(64, max_new))

    if IS_ENCODER_DECODER:
        outputs = MODEL_PIPELINE(prompt, **gen_kwargs)
        texts = [ _sanitize(o.get("generated_text") or o.get("text") or str(o)) for o in outputs ]
    else:
        outputs = MODEL_PIPELINE(prompt, **gen_kwargs)
        texts = []
        for o in outputs:
            gen = o.get("generated_text") or o.get("text") or str(o)
            # If the model echoes the prompt, keep only the newly generated tail
            tail = gen[len(prompt):] if gen.startswith(prompt) else gen
            tail = _apply_stops(tail)
            texts.append(_sanitize(tail))

    return GenerateResponse(generated_texts=texts)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=os.environ.get("MODEL_PATH"))
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", default=8000, type=int)
    args = parser.parse_args()
    if args.model_path:
        os.environ["MODEL_PATH"] = args.model_path

    import uvicorn
    logger.info("Starting model-worker on %s:%s using model=%s", args.host, args.port, os.environ.get("MODEL_PATH") or MODEL_NAME_OR_PATH)
    uvicorn.run("serve_model:app", host=args.host, port=args.port, log_level="info")
