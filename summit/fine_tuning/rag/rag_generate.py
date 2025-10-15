#!/usr/bin/env python3
"""
RAG-augmented generation using your finetuned GPT-2 model (token-budgeted, SFT-aligned).

Example:
  MODEL=fine_tuning/models/gpt2-onboarding
  INDEX=fine_tuning/data/rag_index
  python3 fine_tuning/rag/rag_generate.py \
    --model-path "$MODEL" \
    --index-path "$INDEX" \
    --query "How do I update my graduation year and what happens next?" \
    --top-k 3 \
    --max-new-tokens 96 \
    --temperature 0.2 \
    --repetition-penalty 1.2
"""
import argparse, os, pickle, json
from typing import List, Dict, Tuple
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# This is embedded inside the <|user|> turn so it matches your SFT format.
DEFAULT_PREFACE = (
    "You are Summit, a friendly onboarding assistant. Use the notes to answer "
    "strictly about onboarding. If the question is out of scope, say so and offer onboarding help."
)

# ----------------- FAISS helpers -----------------

def load_index(index_path: str):
    index = faiss.read_index(os.path.join(index_path, "index.faiss"))
    with open(os.path.join(index_path, "meta.pkl"), "rb") as fh:
        meta = pickle.load(fh)
    with open(os.path.join(index_path, "stats.json"), "r") as fh:
        stats = json.load(fh)
    return index, meta, stats

def embed_query(encoder, text: str):
    v = encoder.encode([text], convert_to_numpy=True, normalize_embeddings=True)[0]
    return v.astype(np.float32)

def search(index, meta, encoder, query: str, top_k: int = 5) -> List[Dict]:
    D, I = index.search(np.expand_dims(embed_query(encoder, query), axis=0), top_k)
    out = []
    for pos, idx in enumerate(I[0]):
        if 0 <= idx < len(meta):
            item = dict(meta[idx])
            src = str(item.get("source", "")).lower()
            # keep only docs, not training JSON/JSONL
            if not (src.endswith(".md") or src.endswith(".markdown") or src.endswith(".txt")):
                continue
            item["_rank"] = pos
            out.append(item)
    return out

# ----------------- Token utils -----------------

def truncate_text_by_tokens(text: str, tokenizer, limit: int) -> str:
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(ids) <= limit:
        return text
    ids = ids[: max(0, limit)]
    return tokenizer.decode(ids, skip_special_tokens=True)

def build_prompt_with_budget(
    tokenizer,
    user_query: str,
    contexts: List[Dict],
    max_input_tokens: int,
    per_chunk_token_cap: int = 120,
    preface: str = DEFAULT_PREFACE,
) -> Tuple[str, List[Dict]]:
    """
    Compose a single <|user|> turn that embeds notes, followed by <|assistant|>.
    Trim context lines from the top until the prompt fits the input budget.
    """
    # Prepare (truncated) context lines
    ctx_lines_raw: List[str] = []
    for ctx in contexts:
        chunk_txt = truncate_text_by_tokens(ctx["text"], tokenizer, per_chunk_token_cap).strip()
        if chunk_txt:
            ctx_lines_raw.append(f"- {chunk_txt}")

    def make_template(start_idx: int) -> Tuple[str, List[int]]:
        notes = "\n".join(ctx_lines_raw[start_idx:]) if ctx_lines_raw[start_idx:] else "(no relevant notes)"
        user_block = f"{preface}\nNotes:\n{notes}\n\nQuestion: {user_query}"
        template = f"<|user|> {user_block}\n<|assistant|> "
        ids = tokenizer(template, add_special_tokens=True)["input_ids"]
        return template, ids

    k = 0
    prompt, ids = make_template(k)
    while len(ids) > max_input_tokens and k < len(ctx_lines_raw):
        k += 1
        prompt, ids = make_template(k)

    # figure out which contexts survived
    kept = contexts[k:] if k < len(contexts) else []
    return prompt, kept

# ----------------- Generate -----------------

def generate(
    model_path: str,
    index_path: str,
    query: str,
    top_k: int = 3,
    max_new_tokens: int = 128,
    temperature: float = 0.7,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
    no_repeat_ngram_size: int = 3,
    per_chunk_token_cap: int = 120,
    preface: str = DEFAULT_PREFACE,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    index, meta, _ = load_index(index_path)
    encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    model.eval()

    # Model context window (GPT-2 typically 1024)
    model_max = getattr(model.config, "n_positions", getattr(model.config, "max_position_embeddings", 1024))
    safety = 16
    max_input_tokens = max(64, model_max - int(max_new_tokens) - safety)

    # Retrieval (keep k small for GPT-2)
    ctx = search(index, meta, encoder, query, top_k=top_k)

    # Build SFT-aligned prompt within budget
    prompt, used_ctx = build_prompt_with_budget(
        tokenizer=tokenizer,
        user_query=query,
        contexts=ctx,
        max_input_tokens=max_input_tokens,
        per_chunk_token_cap=per_chunk_token_cap,
        preface=preface,
    )

    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(device)

    # Only sample when temperature > 0
    do_sample = float(temperature) > 0.0
    gen_kwargs = {
        "max_new_tokens": int(max_new_tokens),
        "repetition_penalty": float(repetition_penalty),
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        gen_kwargs.update(
            do_sample=True,
            temperature=float(temperature),
            top_p=float(top_p),
            no_repeat_ngram_size=int(no_repeat_ngram_size),
        )
    else:
        gen_kwargs.update(do_sample=False)

    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)

    # Decode and trim to just the assistant reply
    text = tokenizer.decode(out[0], skip_special_tokens=False)
    cut = text.rfind("<|assistant|>")
    answer = text[cut + len("<|assistant|>"):].strip() if cut != -1 else text
    # If the model loops back to a new user turn, cut it off
    answer = answer.split("<|user|>")[0].strip()

    meta = {
        "model_max": model_max,
        "max_input_tokens": max_input_tokens,
        "prompt_tokens": len(inputs["input_ids"][0]),
    }
    return answer, used_ctx, meta

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--index-path", required=True)
    ap.add_argument("--query", required=True)
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--repetition-penalty", type=float, default=1.1)
    ap.add_argument("--no-repeat-ngram-size", type=int, default=3)
    ap.add_argument("--per-chunk-token-cap", type=int, default=120)
    ap.add_argument("--preface", type=str, default=DEFAULT_PREFACE)
    args = ap.parse_args()

    answer, ctx, meta = generate(
        model_path=args.model_path,
        index_path=args.index_path,
        query=args.query,
        top_k=args.top_k,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        per_chunk_token_cap=args.per_chunk_token_cap,
        preface=args.preface,
    )
    print("=== ANSWER ===")
    print(answer)
    print("\n=== SOURCES ===")
    for i, c in enumerate(ctx, 1):
        print(f"[{i}] {c['source']}#chunk={c['chunk_id']}")
    print("\n=== META ===")
    print(meta)

if __name__ == "__main__":
    main()
