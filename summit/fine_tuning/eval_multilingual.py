#!/usr/bin/env python3
"""Simple multilingual evaluation: sends queries in 4 languages to the running RAG server
and saves the responses for inspection.

Usage: python3 fine_tuning/eval_multilingual.py --url http://127.0.0.1:8001/generate
"""
import argparse
import requests
import json

SAMPLES = {
    "en": "How do I sign up for the summit?",
    "es": "¿Cómo me inscribo en la cumbre?",
    "fr": "Comment puis-je m'inscrire au sommet ?",
    "pt": "Como me inscrevo para a cúpula?",
}


def run_one(url, lang, text):
    payload = {"query": text, "top_k": 3, "max_new_tokens": 128}
    headers = {"Authorization": "Bearer testkey"}
    r = requests.post(url, json=payload, headers=headers, timeout=30)
    try:
        data = r.json()
    except Exception:
        return {"status": r.status_code, "raw": r.text}
    return data


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8001/generate")
    args = p.parse_args()

    results = {}
    for lang, txt in SAMPLES.items():
        print(f"Running {lang} -> {txt}")
        out = run_one(args.url, lang, txt)
        results[lang] = out
        print(json.dumps(out, ensure_ascii=False, indent=2))
        print("---\n")

    with open("/tmp/multilingual_eval.json", "w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2)
    print("Saved results to /tmp/multilingual_eval.json")


if __name__ == '__main__':
    main()
