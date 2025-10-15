"""
Lightweight wrapper to call rag_generate.generate from the rag package.
Keeps return shape stable and forwards new guard/threshold options.
"""
from typing import Any, Dict, Optional
from fine_tuning.rag.rag_generate import generate, DEFAULT_REFUSAL

def gen_once(
    model_path: str,
    index_path: str,
    text: str,
    top_k: int = 5,
    min_sim: float = 0.28,
    max_new_tokens: int = 140,
    temperature: float = 0.0,
    top_p: float = 1.0,
    system_file: str = "",
    refusal_text: Optional[str] = None,
) -> Dict[str, Any]:
    refusal = refusal_text or DEFAULT_REFUSAL
    answer, ctx, sims = generate(
        model_path=model_path,
        index_path=index_path,
        query=text,
        top_k=top_k,
        min_sim=min_sim,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        system_file=system_file,
        refusal_text=refusal,
    )
    return {
        "answer": answer,
        "sources": [
            {
                "source": c.get("source", ""),
                "chunk_id": c.get("chunk_id", -1),
                "heading": c.get("heading", ""),
                "similarity": sims[i] if i < len(sims) else None,
            }
            for i, c in enumerate(ctx)
        ],
    }
