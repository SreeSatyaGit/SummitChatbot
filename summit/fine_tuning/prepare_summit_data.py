#!/usr/bin/env python3
import argparse, json, random, re
from pathlib import Path

# strip things like " (variant 2)" at the end of completions
VARIANT_TAIL = re.compile(r"\s*\(variant\s*\d+\)\s*$", re.IGNORECASE)

def read_sft(path):
    """summit_sft.jsonl → already has {"text": "..."}"""
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            t = (obj.get("text") or "").strip()
            if t:
                out.append(t)
    return out

def read_onboarding(path):
    """onboarding_train.jsonl → {"prompt": "...", "completion": "..."} → join to one 'text' string"""
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            prompt = (obj.get("prompt") or "").strip()
            completion = (obj.get("completion") or "").strip()
            if not prompt or not completion:
                continue
            completion = VARIANT_TAIL.sub("", completion).strip()
            if not prompt.endswith(" "):
                prompt += " "
            text = (prompt + completion).strip()
            # keep only examples that contain our chat markers
            if "<|user|>" in text and "<|assistant|>" in text:
                out.append(text)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onboarding", required=True, help="onboarding_train.jsonl")
    ap.add_argument("--sft", required=True, help="summit_sft.jsonl")
    ap.add_argument("--out-train", default="data/summit_all.jsonl")
    ap.add_argument("--out-val", default="data/summit_val.jsonl")
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    Path(args.out_train).parent.mkdir(parents=True, exist_ok=True)

    sft = read_sft(args.sft)
    onb = read_onboarding(args.onboarding)

    # dedupe while preserving order
    merged = list(dict.fromkeys(sft + onb))

    random.Random(args.seed).shuffle(merged)
    cut = max(1, int(len(merged) * (1 - args.val_ratio)))
    train, val = merged[:cut], merged[cut:]

    with open(args.out_train, "w", encoding="utf-8") as w:
        for t in train:
            w.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")

    with open(args.out_val, "w", encoding="utf-8") as w:
        for t in val:
            w.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")

    print(f"Wrote {len(train)} train → {args.out_train}")
    print(f"Wrote {len(val)}   val  → {args.out_val}")

if __name__ == "__main__":
    main()
