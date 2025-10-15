#!/usr/bin/env python3
import os, json, pickle
from typing import List, Dict, Optional, Tuple
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
try:
    from langdetect import detect as _ld_detect
except Exception:
    _ld_detect = None
try:
    from peft import PeftModel
except Exception:
    PeftModel = None
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
import torch
import logging

log = logging.getLogger("rag_server")

# Import enhanced RAG service
try:
    from .enhanced_rag_service import EnhancedRAGService, QueryContext
    ENHANCED_RAG_AVAILABLE = True
except ImportError:
    ENHANCED_RAG_AVAILABLE = False
    log.warning("Enhanced RAG service not available, using fallback")

# Safer HF downloads on small disks (disables xet backend)
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

app = FastAPI(title="Summit RAG Server")

# --------- Config ---------
MODEL_PATH = os.environ.get("MODEL_PATH", "Qwen/Qwen2.5-1.5B-Instruct")
RAG_INDEX_PATH = os.environ.get("RAG_INDEX_PATH", "fine_tuning/data/rag_index")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "intfloat/multilingual-e5-base")
QUANTIZE = os.environ.get("QUANTIZE", "").lower()  # "", "4bit", or "8bit"
PEFT_ADAPTER_PATH = os.environ.get("PEFT_ADAPTER_PATH", "")
FORCE_GENERATE = os.environ.get("FORCE_GENERATE", "").lower() in ("1", "true", "yes")
RAG_API_KEY = os.environ.get("RAG_API_KEY", "")
# If running with Docker secrets, read /run/secrets/rag_api_key
try:
    if os.path.exists("/run/secrets/rag_api_key"):
        with open("/run/secrets/rag_api_key", "r") as fh:
            secret_val = fh.read().strip()
            if secret_val:
                RAG_API_KEY = secret_val
except Exception:
    pass

# Allow multiple keys via comma-separated env var or secret for per-request keys / rotation
raw_keys = os.environ.get("RAG_API_KEYS", RAG_API_KEY)
# If a separate secret file exists for multiple keys, prefer it
try:
    if os.path.exists("/run/secrets/rag_api_keys"):
        with open("/run/secrets/rag_api_keys", "r") as fh:
            sk = fh.read().strip()
            if sk:
                raw_keys = sk
except Exception:
    pass

RAG_API_KEYS = [k.strip() for k in raw_keys.split(",") if k.strip()]

# Stronger default instructions to improve answer quality and encourage citations
DEFAULT_INSTRUCTIONS = (
    "You are Summit, a friendly, professional onboarding assistant. Only answer onboarding-related questions. "
    "Be concise and factual. Use the provided Context to support your answer. If you cite facts from context, append citations in the form [source:chunk_id] after the sentence. "
    "If information is missing or ambiguous, ask one concise clarifying question. Never hallucinate facts — say 'I don't know' if unsupported. "
    "Keep responses short and actionable (<= 4 short paragraphs). Use <|user|>/<|assistant|> chat markers when constructing prompts."
)

# Default generation kwargs — used when caller doesn't override via request body
DEFAULT_GEN_KWARGS = {
    "max_new_tokens": 220,
    "temperature": 0.2,
    "top_p": 0.9,
    "repetition_penalty": 1.1,
    "no_repeat_ngram_size": 3,
    "do_sample": True,
}

# Optional cross-encoder re-ranker model (improves retrieval quality). Set via env var CROSS_ENCODER_MODEL
CROSS_ENCODER_MODEL = os.environ.get("CROSS_ENCODER_MODEL", "")
_cross_tokenizer = None
_cross_model = None

# --------- Models & Index (loaded at startup) ---------
_device = "cuda" if torch.cuda.is_available() else "cpu"
_tokenizer = None
_model = None
_index = None
_meta = []
_encoder = None
_model_max = 4096  # will be read from model config
_enhanced_rag_service = None

# --- Intent prototypes (used for a simple embedding-based intent classifier)
_INTENT_PROTOS = {
    "suggestion": "Provide actionable suggestions, tips or recommendations based on the user's question.",
    "create_profile": "Help create or draft a user profile, resume or biography; ask clarifying questions when information is missing.",
}
_PROTO_EMBS = {}

def _load_index(path: str):
    idx_path = os.path.join(path, "index.faiss")
    meta_path = os.path.join(path, "meta.pkl")
    if not (os.path.exists(idx_path) and os.path.exists(meta_path)):
        return None, []
    idx = faiss.read_index(idx_path)
    with open(meta_path, "rb") as fh:
        meta = pickle.load(fh)
    return idx, meta

@app.on_event("startup")
def _startup() -> None:
    global _tokenizer, _model, _index, _meta, _encoder, _model_max
    global _cross_tokenizer, _cross_model

    # --- Tokenizer
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)
    if _tokenizer.pad_token is None and _tokenizer.eos_token is not None:
        _tokenizer.pad_token = _tokenizer.eos_token

    # --- Model (supports 4-bit/8-bit/full)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model_kwargs = dict(
        device_map="auto",
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )

    if QUANTIZE == "4bit":
        qconf = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model_kwargs["quantization_config"] = qconf
    elif QUANTIZE == "8bit":
        qconf = BitsAndBytesConfig(load_in_8bit=True)
        model_kwargs["quantization_config"] = qconf

    _model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, **model_kwargs)
    _model.eval()

    # If PEFT adapters are present, try to load them. Priority:
    # 1) PEFT_ADAPTER_PATH env var (explicit path), 2) adapters saved under MODEL_PATH.
    if PeftModel is not None:
        # 1) explicit adapter path
        if PEFT_ADAPTER_PATH:
            try:
                if os.path.exists(PEFT_ADAPTER_PATH):
                    _model = PeftModel.from_pretrained(_model, PEFT_ADAPTER_PATH)
                    log.info("Loaded PEFT adapters from PEFT_ADAPTER_PATH=%s", PEFT_ADAPTER_PATH)
                else:
                    log.warning("PEFT_ADAPTER_PATH set but path does not exist: %s", PEFT_ADAPTER_PATH)
            except Exception:
                log.exception("Failed to load PEFT adapters from PEFT_ADAPTER_PATH=%s", PEFT_ADAPTER_PATH)

        # 2) Attempt to load adapters directly from MODEL_PATH (works if adapters were saved to the same repo/dir)
        try:
            _model = PeftModel.from_pretrained(_model, MODEL_PATH)
            log.info("Loaded PEFT adapters from %s", MODEL_PATH)
        except Exception:
            # Try common subfolder names under MODEL_PATH
            for sub in ("peft", "adapter", "lora", "adapters"):
                cand = os.path.join(MODEL_PATH, sub)
                if os.path.exists(cand):
                    try:
                        _model = PeftModel.from_pretrained(_model, cand)
                        log.info("Loaded PEFT adapters from %s", cand)
                        break
                    except Exception:
                        continue

    # --- Model context length
    _model_max = int(
        getattr(_model.config, "max_position_embeddings",
        getattr(_model.config, "n_positions", 4096))
    )

    # --- RAG index (optional but recommended)
    _index, _meta = _load_index(RAG_INDEX_PATH)

    # --- Embedder
    _encoder = SentenceTransformer(EMBED_MODEL, device=_device)

    # --- Optional cross-encoder (for reranking)
    if CROSS_ENCODER_MODEL:
        try:
            from transformers import AutoTokenizer as _AT, AutoModelForSequenceClassification as _AM
            _cross_tokenizer = _AT.from_pretrained(CROSS_ENCODER_MODEL)
            _cross_model = _AM.from_pretrained(CROSS_ENCODER_MODEL)
            _cross_model.eval()
            log.info("Loaded cross-encoder model for reranking: %s", CROSS_ENCODER_MODEL)
        except Exception:
            log.exception("Failed to load cross-encoder model %s", CROSS_ENCODER_MODEL)

    # Initialize enhanced RAG service
    if ENHANCED_RAG_AVAILABLE:
        try:
            _enhanced_rag_service = EnhancedRAGService()
            log.info("Enhanced RAG service initialized successfully")
        except Exception:
            log.exception("Failed to initialize enhanced RAG service")
            _enhanced_rag_service = None

    # Verify index dimension matches embedder output (guard against mismatched embed models)
    try:
        if _index is not None and len(_meta) > 0:
            # faiss index dimension (d) vs embedder dimension
            idx_dim = getattr(_index, "d", None)
            try:
                enc_dim = _encoder.get_sentence_embedding_dimension()
            except Exception:
                # fallback: encode a tiny dummy to infer dimension
                enc_dim = len(_encoder.encode(["test"], convert_to_numpy=True)[0])
            if idx_dim is not None and enc_dim is not None and idx_dim != enc_dim:
                log.warning("RAG index dim (%s) != embedder dim (%s); disabling index to avoid errors.", idx_dim, enc_dim)
                _index = None
                _meta = []
    except Exception:
        # don't let the validation break startup
        pass

@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": _device,
        "model_max": _model_max,
        "quantize": QUANTIZE or "none",
        "has_index": bool(_index is not None and len(_meta) > 0),
        "embed_model": EMBED_MODEL,
        "enhanced_rag_available": ENHANCED_RAG_AVAILABLE and _enhanced_rag_service is not None,
    }

# --------- Retrieval helpers ---------
def _embed_query(text: str) -> np.ndarray:
    v = _encoder.encode([text], convert_to_numpy=True, normalize_embeddings=True)[0]
    return v.astype(np.float32)

def _search(query: str, top_k: int = 3) -> List[Dict]:
    if _index is None or not len(_meta):
        return []
    q = _embed_query(query)
    # retrieve a larger candidate pool for reranking (if available)
    fetch_k = max(top_k * 5, top_k)
    D, I = _index.search(np.expand_dims(q, axis=0), fetch_k)
    out = []
    # prefer results matching the detected language of the query (if available)
    try:
        q_lang = _ld_detect(query) if _ld_detect is not None else None
    except Exception:
        q_lang = None

    for pos, idx in enumerate(I[0]):
        if 0 <= idx < len(_meta):
            item = dict(_meta[idx])
            src = str(item.get("source", "")).lower()
            # Keep only plain docs
            if not (src.endswith(".md") or src.endswith(".markdown") or src.endswith(".txt")):
                continue
            # if both chunk and query have languages, prefer same-language by tagging rank
            chunk_lang = item.get("lang")
            if q_lang and chunk_lang and q_lang == chunk_lang:
                # move higher in the returned list by lowering _rank
                item["_rank"] = pos - 0.5
            else:
                item["_rank"] = pos
            item["_rank"] = pos
            out.append(item)
    # sort by _rank (so same-language chunks come first when detected)
    # If a cross-encoder is available, rerank candidates by cross-encoder score
    if _cross_model is not None and _cross_tokenizer is not None and out:
        try:
            from torch.nn.functional import softmax
            pairs = [(query, c["text"]) for c in out]
            # tokenize in batches
            inputs = _cross_tokenizer([q for q, t in pairs], [t for q, t in pairs], padding=True, truncation=True, return_tensors="pt")
            inputs = {k: v.to(next(_cross_model.parameters()).device) for k, v in inputs.items()}
            with torch.no_grad():
                logits = _cross_model(**inputs).logits
                # assume binary/regression style scoring; sum across logits for a single score
                scores = logits.squeeze(-1).cpu().numpy() if logits.ndim == 2 and logits.shape[-1] == 1 else logits[:, 1].cpu().numpy() if logits.ndim == 2 else logits.cpu().numpy()
            for c, s in zip(out, scores):
                c['_cross_score'] = float(s)
            out = sorted(out, key=lambda x: x.get('_cross_score', x.get('_rank', 999)), reverse=True)[:top_k]
        except Exception:
            log.exception("Cross-encoder rerank failed — falling back to vector ranks")
            out = sorted(out, key=lambda x: x.get("_rank", 999))[:top_k]
    else:
        # sort by _rank (so same-language chunks come first when detected)
        out = sorted(out, key=lambda x: x.get("_rank", 999))[:top_k]
    return out

def _tok_len(text: str) -> int:
    return len(_tokenizer(text, add_special_tokens=False)["input_ids"])

def _truncate_by_tokens(text: str, limit: int) -> str:
    ids = _tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(ids) <= limit:
        return text
    ids = ids[:max(0, limit)]
    return _tokenizer.decode(ids, skip_special_tokens=True)


def _detect_intent(text: str) -> str:
    """Lightweight intent detection using the embedder: returns 'suggestion' or 'create_profile'.

    Falls back to keyword heuristics when embedder isn't available.
    """
    if _encoder is None:
        # simple keyword fallback
        t = text.lower()
        if any(k in t for k in ("profile", "bio", "resume", "cv", "create my")):
            return "create_profile"
        return "suggestion"

    # compute prototype embeddings lazily
    if not _PROTO_EMBS:
        for k, s in _INTENT_PROTOS.items():
            _PROTO_EMBS[k] = _encoder.encode([s], convert_to_numpy=True, normalize_embeddings=True)[0].astype(np.float32)

    q_emb = _encoder.encode([text], convert_to_numpy=True, normalize_embeddings=True)[0].astype(np.float32)
    # cosine similarities
    best, best_score = None, -1.0
    for k, p in _PROTO_EMBS.items():
        score = float(np.dot(q_emb, p) / (np.linalg.norm(q_emb) * np.linalg.norm(p) + 1e-12))
        if score > best_score:
            best_score = score
            best = k

    # small threshold safety: if both low, fallback to keyword heuristic
    if best_score < 0.35:
        t = text.lower()
        if any(k in t for k in ("profile", "bio", "resume", "cv", "create my", "edit my profile")):
            return "create_profile"
        return "suggestion"
    return best

def _build_prompt_with_budget(
    user_query: str,
    contexts: List[Dict],
    instructions: str,
    max_input_tokens: int,
    per_chunk_cap: int = 120,
) -> Tuple[str, List[Dict]]:
    header = f"Instructions: {instructions}\n\nContext:\n"
    user_turn = f"\n\n<|user|> {user_query}\n<|assistant|> "
    used = len(_tokenizer(header, add_special_tokens=False)["input_ids"]) + \
           len(_tokenizer(user_turn, add_special_tokens=False)["input_ids"])

    chosen, parts = [], []
    for ctx in contexts:
        piece = _truncate_by_tokens(ctx.get("text", ""), per_chunk_cap).strip()
        if not piece:
            continue
        candidate = f"- {piece}\n"
        add_len = len(_tokenizer(candidate, add_special_tokens=False)["input_ids"])
        if used + add_len <= max_input_tokens:
            parts.append(candidate)
            chosen.append(ctx)
            used += add_len
        else:
            break
    context_block = "".join(parts) if parts else "- (no relevant context found)\n"
    return f"{header}{context_block}{user_turn}", chosen

def _strip_chat_markers(s: str) -> str:
    s = s.strip()
    if s.startswith("<|user|>"):
        s = s[len("<|user|>"):].strip()
    if s.endswith("<|assistant|>"):
        s = s[:-len("<|assistant|>")].strip()
    return s


def _one_sentence_truncate(s: str, max_words: int = 20) -> str:
    words = s.strip().split()
    if len(words) <= max_words:
        return s.strip()
    return " ".join(words[:max_words]).rstrip(' ,;:') + "..."


def _post_process_answer(raw: str, intent: str, used_ctx: List[Dict], user_query: str) -> str:
    """Clean up model output to enforce concise suggestions or a profile skeleton.

    - For 'suggestion' intent: return up to 4 numbered, one-sentence suggestions.
    - For 'create_profile' intent: return a short structured profile skeleton and one clarifying question.
    """
    # normalize lines
    lines = [l.strip() for l in raw.splitlines() if l.strip()]

    if intent == "suggestion":
        items = []
        for l in lines:
            # capture numbered or bullet lines
            if l[0].isdigit() or l.startswith("-") or l.startswith("•") or l.startswith("*"):
                # strip leading numbering/bullet
                tok = l.lstrip('-•* ').lstrip('0123456789. ')[:400]
                tok = _one_sentence_truncate(tok, max_words=20)
                items.append(tok)
            else:
                # fallback: treat whole line as one suggestion
                items.append(_one_sentence_truncate(l, max_words=20))
            if len(items) >= 4:
                break

        # if model returned fewer than 1 item, create simple heuristic suggestions
        if not items:
            items = [
                _one_sentence_truncate("Start with a one-line summary of your current role."),
                _one_sentence_truncate("List 3-5 key skills relevant to recruiting."),
                _one_sentence_truncate("Add one measurable achievement (numbers)."),
            ]

        out_lines = [f"{i+1}. {it}" for i, it in enumerate(items)]
        return "\n".join(out_lines)

    if intent == "create_profile":
        # Try to craft a small profile skeleton using available context hints
        # Pull a short summary from the top context if available
        summary = "TBD"
        if used_ctx and isinstance(used_ctx[0], dict):
            txt = used_ctx[0].get("text", "").strip()
            if txt:
                # take first 20 words
                summary = _one_sentence_truncate(txt, max_words=20)

        skeleton = []
        skeleton.append("Profile Draft (skeleton):")
        skeleton.append(f"Name: [your full name]")
        skeleton.append(f"Title: [current role/title]")
        skeleton.append(f"Summary: {summary}")
        skeleton.append("Experience: - Role / Company / Years / 1-line responsibility summary")
        skeleton.append("Skills: - skill1, skill2, skill3")
        skeleton.append("Achievements: - brief, quantifiable statements (e.g. 'led X to Y')")
        # Ask a single clarifying question
        clarq = "Do you want me to draft the full profile using the context I found?"
        skeleton.append("")
        skeleton.append(f"Clarifying question: {clarq}")
        return "\n".join(skeleton)

    # default: return the first 4 short lines
    short = []
    for l in lines:
        short.append(_one_sentence_truncate(l, max_words=25))
        if len(short) >= 4:
            break
    return "\n".join(short)

# --------- I/O schema ---------
class GenerateIn(BaseModel):
    query: Optional[str] = None
    prompt: Optional[str] = None  # legacy compatibility
    task: Optional[str] = None    # ignored
    top_k: int = 3
    max_new_tokens: int = 128
    temperature: float = 0.0
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    no_repeat_ngram_size: int = 3

class GenerateOut(BaseModel):
    text: str
    answer: str
    sources: List[Dict]
    meta: Dict

# --------- Endpoint ---------
@app.post("/generate", response_model=GenerateOut)
def generate(body: GenerateIn, request: Request):
    # Normalize input
    # Simple API key auth: require header Authorization: Bearer <RAG_API_KEY> when RAG_API_KEY set
    if RAG_API_KEYS:
        # Accept Authorization: Bearer <key> or X-API-KEY: <key>
        auth = request.headers.get("authorization") or request.headers.get("Authorization")
        xkey = request.headers.get("x-api-key") or request.headers.get("X-API-KEY")
        provided_keys = []
        if auth and auth.lower().startswith("bearer "):
            provided_keys.append(auth.split(None, 1)[1])
        if xkey:
            provided_keys.append(xkey)

        if not any(pk in RAG_API_KEYS for pk in provided_keys):
            raise HTTPException(status_code=401, detail="Missing or invalid API key")

    user_query = body.query or body.prompt
    if not user_query:
        raise HTTPException(status_code=400, detail="Missing 'query' or 'prompt'.")
    user_query = _strip_chat_markers(user_query)

    # Token budget
    safety = 16
    max_input_tokens = max(64, _model_max - int(body.max_new_tokens) - safety)

    # Use enhanced RAG service if available
    if _enhanced_rag_service:
        try:
            # Process query with enhanced RAG service
            enhanced_result = _enhanced_rag_service.process_query(
                user_message=user_query,
                session_state={},  # Could be passed from request body in future
                extracted_fields={}  # Could be passed from request body in future
            )
            
            # Use rewritten query for retrieval
            rewritten_query = enhanced_result["rewritten_query"]
            ctx = _search(rewritten_query, top_k=max(1, int(body.top_k)))
            
            # Use enhanced prompt
            instructions = enhanced_result["enhanced_prompt"]
            
        except Exception as e:
            log.exception("Enhanced RAG service failed, falling back to standard processing: %s", e)
            # Fall back to standard processing
            ctx = _search(user_query, top_k=max(1, int(body.top_k)))
            intent = _detect_intent(user_query)
            if intent == "create_profile":
                instructions = (
                    "You are Summit, a friendly onboarding assistant. The user wants to CREATE or EDIT a profile. "
                    "Produce a clear, structured profile draft or a short checklist of missing fields, and ask one concise clarifying question if needed. "
                    "Use the Context to fill missing info. Be polite and concise."
                )
            else:
                instructions = (
                    "You are Summit, a friendly onboarding assistant. The user is asking for SUGGESTIONS or HELP. "
                    "Provide up to 4 numbered, prioritized, actionable suggestions; each suggestion must be one short sentence (<=20 words). "
                    "Reference the Context briefly when relevant. Do not produce long paragraphs or lists greater than 4 items."
                )
    else:
        # Standard processing when enhanced RAG is not available
        ctx = _search(user_query, top_k=max(1, int(body.top_k)))
        intent = _detect_intent(user_query)
        if intent == "create_profile":
            instructions = (
                "You are Summit, a friendly onboarding assistant. The user wants to CREATE or EDIT a profile. "
                "Produce a clear, structured profile draft or a short checklist of missing fields, and ask one concise clarifying question if needed. "
                "Use the Context to fill missing info. Be polite and concise."
            )
        else:
            instructions = (
                "You are Summit, a friendly onboarding assistant. The user is asking for SUGGESTIONS or HELP. "
                "Provide up to 4 numbered, prioritized, actionable suggestions; each suggestion must be one short sentence (<=20 words). "
                "Reference the Context briefly when relevant. Do not produce long paragraphs or lists greater than 4 items."
            )

    # Prepend a simple 'Respond in {Language}.' instruction based on detected language of the user query
    try:
        q_lang = _ld_detect(user_query) if _ld_detect is not None else None
    except Exception:
        q_lang = None

    if q_lang:
        # map common short codes to readable names
        lang_map = {
            "en": "English",
            "es": "Spanish",
            "fr": "French",
            "pt": "Portuguese",
            "de": "German",
            "it": "Italian",
            "nl": "Dutch",
        }
        lang_name = lang_map.get(q_lang, None)
        if lang_name:
            instructions = f"Respond in {lang_name}. " + instructions

    # Build prompt
    prompt, used_ctx = _build_prompt_with_budget(
        user_query=user_query,
        contexts=ctx,
        instructions=instructions,
        max_input_tokens=max_input_tokens,
        per_chunk_cap=120,
    )

    # For targeted intents we can bypass the LLM and return deterministic, concise outputs.
    def _build_suggestions_from_ctx(used_ctx: List[Dict], user_query: str, max_items: int = 4) -> str:
        items = []
        # Prefer short sentences from the top-k contexts
        for c in used_ctx:
            txt = c.get("text", "").strip()
            if not txt:
                continue
            # split into candidate sentences
            sents = [s.strip() for s in txt.replace('\n', ' ').split('.') if s.strip()]
            for s in sents:
                s_short = _one_sentence_truncate(s, max_words=20)
                if s_short and s_short not in items:
                    items.append(s_short)
                if len(items) >= max_items:
                    break
            if len(items) >= max_items:
                break
        # fallback heuristics
        if not items:
            items = [
                _one_sentence_truncate("Start with a one-line summary of your current role."),
                _one_sentence_truncate("List 3-5 key skills relevant to the role."),
                _one_sentence_truncate("Add one measurable achievement (numbers)."),
            ]
        out_lines = [f"{i+1}. {it}" for i, it in enumerate(items[:max_items])]
        return "\n".join(out_lines)

    # For targeted intents, we can bypass the LLM and return deterministic, concise outputs.
    # But only if FORCE_GENERATE is disabled (default behavior for production efficiency)
    if not FORCE_GENERATE and intent in ("suggestion", "create_profile"):
        if intent == "suggestion":
            processed = _build_suggestions_from_ctx(used_ctx, user_query, max_items=4)
        else:
            # create_profile
            processed = _post_process_answer("", intent="create_profile", used_ctx=used_ctx, user_query=user_query)

        return GenerateOut(
            text=processed,
            answer=processed,
            sources=[{"source": c.get("source"), "chunk_id": c.get("chunk_id"), "rank": c.get("_rank", 0)} for c in used_ctx],
            meta={
                "model_max": _model_max,
                "max_input_tokens": max_input_tokens,
                "prompt_tokens": 0,
                "do_sample": False,
                "quantize": QUANTIZE or "none",
                "intent": intent,
                "generated_bypassed": True,
            },
        )

    # Tokenize and move tensors to the model's device to avoid cpu/cuda mismatch
    inputs = _tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    # Prefer the device of the input embeddings (works when model is sharded and embeddings live on CPU)
    try:
        emb = _model.get_input_embeddings()
        model_device = emb.weight.device
    except Exception:
        try:
            # Fall back to the device of some model parameter
            model_device = next(_model.parameters()).device
        except StopIteration:
            model_device = torch.device(_device)
    # move all input tensors to the model device
    inputs = {k: v.to(model_device) for k, v in inputs.items()}

    # Merge request-level generation params with defaults
    gen_cfg = DEFAULT_GEN_KWARGS.copy()
    # Allow request to override defaults
    try:
        if body.max_new_tokens:
            gen_cfg["max_new_tokens"] = int(body.max_new_tokens)
    except Exception:
        pass
    try:
        gen_cfg["temperature"] = float(body.temperature)
    except Exception:
        pass
    try:
        gen_cfg["top_p"] = float(body.top_p)
    except Exception:
        pass
    try:
        gen_cfg["repetition_penalty"] = float(body.repetition_penalty)
    except Exception:
        pass

    # Decide sampling vs deterministic
    do_sample = float(gen_cfg.get("temperature", 0.0)) > 0.0
    pad_id = _tokenizer.pad_token_id or _tokenizer.eos_token_id
    eos_id = _tokenizer.eos_token_id

    gen_kwargs = dict(
        max_new_tokens=int(gen_cfg.get("max_new_tokens", 220)),
        repetition_penalty=float(gen_cfg.get("repetition_penalty", 1.0)),
        pad_token_id=pad_id,
        eos_token_id=eos_id,
        do_sample=do_sample,
        no_repeat_ngram_size=int(body.no_repeat_ngram_size),
    )
    if do_sample:
        gen_kwargs.update(
            temperature=float(gen_cfg.get("temperature", 0.2)),
            top_p=float(gen_cfg.get("top_p", 0.9)),
        )
    else:
        # Use beam search for deterministic outputs when sampling disabled
        gen_kwargs.update({"num_beams": 3, "early_stopping": True})

    with torch.no_grad():
        out = _model.generate(**inputs, **gen_kwargs)

    txt = _tokenizer.decode(out[0], skip_special_tokens=False)
    cut = txt.rfind("<|assistant|>")
    raw_answer = txt[cut + len("<|assistant|>"):].strip() if cut != -1 else txt

    processed = _post_process_answer(raw_answer, intent=intent, used_ctx=used_ctx, user_query=user_query)
    # Log intent and processing for observability
    try:
        log.info("intent=%s prompt_tokens=%d raw_len=%d processed_len=%d", intent, int(inputs["input_ids"].shape[-1]), len(raw_answer), len(processed))
    except Exception:
        log.info("intent=%s processed_len=%d", intent, len(processed))

    return GenerateOut(
        text=processed,
        answer=processed,
        sources=[{"source": c.get("source"), "chunk_id": c.get("chunk_id"), "rank": c.get("_rank", 0)} for c in used_ctx],
        meta={
            "model_max": _model_max,
            "max_input_tokens": max_input_tokens,
            "prompt_tokens": int(inputs["input_ids"].shape[-1]),
            "do_sample": do_sample,
            "quantize": QUANTIZE or "none",
            "intent": intent,
        },
    )
