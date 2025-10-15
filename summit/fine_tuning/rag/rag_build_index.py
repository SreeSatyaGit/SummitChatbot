#!/usr/bin/env python3
"""
Build a FAISS index over onboarding docs for retrieval-augmented generation (RAG).

Usage:
  python3 fine_tuning/rag/rag_build_index.py \
    --docs-path path/to/onboarding_docs \
    --index-path fine_tuning/rag_index

Key features:
- Markdown-aware section chunking (keeps headings with their content)
- Sliding-window fallback for non-markdown/plain text
- Exact de-duplication across all chunks
- Stores per-chunk metadata (source path, heading, chunk_id)
"""
import argparse, os, json, pickle, re, hashlib
from pathlib import Path
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
# optional language detection (best-effort). If langdetect isn't installed we'll skip language tagging.
try:
    from langdetect import detect as _ld_detect
except Exception:
    _ld_detect = None
import faiss
import numpy as np

TEXT_EXTS = {".txt", ".md", ".markdown"}

# ---------- IO ----------

def read_text_file(p: Path) -> str:
    if p.suffix.lower() in {".json", ".jsonl"}:
        out = []
        with p.open("r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                line=line.strip()
                if not line:
                    continue
                try:
                    obj=json.loads(line)
                    if isinstance(obj, dict):
                        if "text" in obj and isinstance(obj["text"], str):
                            out.append(obj["text"])
                        else:
                            out.append(" ".join([str(v) for v in obj.values() if isinstance(v, str)]))
                    else:
                        out.append(str(obj))
                except Exception:
                    out.append(line)
        return "\n".join(out)
    return p.read_text(encoding="utf-8", errors="ignore")

# ---------- Chunking ----------

MD_HEAD_RE = re.compile(r"(?m)^\s{0,3}#{1,6}\s+(.*)$")

def _normalize_ws(s: str) -> str:
    s = s.replace("\r", "")
    s = re.sub(r"[ \t]+\n", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

def _split_markdown_sections(text: str) -> List[Tuple[str, int, int]]:
    """Return list of (section_text, start_idx, end_idx) keeping headings."""
    text = _normalize_ws(text)
    if not text:
        return []
    sections = []
    last = 0
    for m in MD_HEAD_RE.finditer(text):
        start = m.start()
        if start != 0:
            sec = text[last:start].strip()
            if sec:
                sections.append((sec, last, start))
        last = start
    # tail
    tail = text[last:].strip()
    if tail:
        sections.append((tail, last, last + len(tail)))
    # If no headings found, treat whole doc as one section
    if not sections:
        return [(text, 0, len(text))]
    return sections

def _window_chunks(s: str, chunk_size: int, overlap: int) -> List[str]:
    chunks, i = [], 0
    while i < len(s):
        j = min(i+chunk_size, len(s))
        piece = s[i:j].strip()
        if piece:
            chunks.append(piece)
        if j == len(s):
            break
        i = max(0, j - overlap)
        if i >= len(s):
            break
    return chunks

def chunk_markdown_first(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Prefer section-level chunks; window inside large sections."""
    out = []
    for sec, _, _ in _split_markdown_sections(text):
        if len(sec) <= chunk_size:
            out.append(sec)
        else:
            out.extend(_window_chunks(sec, chunk_size, overlap))
    # de-blank small fragments
    return [c for c in out if c and not re.fullmatch(r"\s*", c)]

# ---------- Collect docs ----------

def collect_docs(root: str, chunk_size: int, overlap: int) -> List[Dict]:
    rootp = Path(root)
    items = []
    for p in rootp.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in TEXT_EXTS:
            continue
        try:
            txt = read_text_file(p)
            txt = _normalize_ws(txt)
            # choose MD-aware chunking for md/markdown
            if p.suffix.lower() in {".md", ".markdown"}:
                pieces = chunk_markdown_first(txt, chunk_size, overlap)
            else:
                pieces = _window_chunks(txt, chunk_size, overlap)
            for idx, ch in enumerate(pieces):
                # include the first heading in metadata if any
                m = MD_HEAD_RE.search(ch)
                heading = m.group(1).strip() if m else ""
                # detect language if available (best-effort)
                lang = None
                try:
                    if _ld_detect is not None and ch and len(ch.strip()) > 10:
                        lang = _ld_detect(ch)
                except Exception:
                    lang = None

                items.append({"text": ch, "source": str(p), "chunk_id": idx, "heading": heading, "lang": lang})
        except Exception as e:
            print(f"[skip] {p}: {e}")
    return items

def exact_dedup(items: List[Dict]) -> List[Dict]:
    seen = set()
    out = []
    for it in items:
        h = hashlib.md5(it["text"].strip().encode("utf-8")).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        out.append(it)
    return out

# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs-path", required=True)
    ap.add_argument("--index-path", required=True)
    ap.add_argument("--embed-model", default="sentence-transformers/paraphrase-multilingual-mpnet-base-v2")
    ap.add_argument("--chunk-size", type=int, default=900)
    ap.add_argument("--chunk-overlap", type=int, default=150)
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    os.makedirs(args.index_path, exist_ok=True)
    print("Scanning docs...")
    docs = collect_docs(args.docs_path, args.chunk_size, args.chunk_overlap)
    docs = exact_dedup(docs)
    if not docs:
        raise SystemExit("No documents found to index.")
    print(f"Loaded {len(docs)} unique chunks. Encoding with {args.embed_model} …")

    st = SentenceTransformer(args.embed_model)
    corpus = [d["text"] for d in docs]
    embs = st.encode(
        corpus,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    dim = embs.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine similarity via normalized vectors
    index.add(embs)

    faiss.write_index(index, os.path.join(args.index_path, "index.faiss"))
    with open(os.path.join(args.index_path, "meta.pkl"), "wb") as fh:
        pickle.dump(docs, fh)
    with open(os.path.join(args.index_path, "stats.json"), "w") as fh:
        json.dump({"chunks": len(docs), "dim": dim}, fh, indent=2)

    print("Index built:", args.index_path, f"(chunks={len(docs)}, dim={dim})")
    print("Tip: for improved reranking you can create a cross-encoder model (e.g. 'cross-encoder/ms-marco-MiniLM-L-6-v2') and set CROSS_ENCODER_MODEL in the RAG server to enable re-ranking at query time.")

if __name__ == "__main__":
    main()
