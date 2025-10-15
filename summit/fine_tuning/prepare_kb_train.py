#!/usr/bin/env python3
"""
Prepare a training text file from all documents in fine_tuning/data/kb.
Each paragraph (separated by blank lines) becomes one line/example.
"""
import os
from pathlib import Path

KB_DIR = Path(__file__).parent / "data" / "kb"
OUT = Path(__file__).parent / "data" / "kb_train.txt"

def paragraphs(text: str):
    parts = [p.strip() for p in text.splitlines()]
    buf = []
    for line in parts:
        if not line:
            if buf:
                yield " ".join(buf).strip()
                buf = []
        else:
            buf.append(line)
    if buf:
        yield " ".join(buf).strip()

def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for p in sorted(KB_DIR.iterdir()):
        if p.is_file():
            try:
                text = p.read_text(encoding="utf-8")
            except Exception:
                text = p.read_text(encoding="latin-1")
            for para in paragraphs(text):
                # prefix with filename as light context marker
                records.append(para)

    # write one example per line
    with OUT.open("w", encoding="utf-8") as fh:
        for r in records:
            # sanitize newlines
            fh.write(r.replace("\n", " ").strip() + "\n")

    print(f"Wrote {len(records)} examples to {OUT}")

if __name__ == "__main__":
    main()
