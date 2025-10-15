from fastapi import FastAPI, Request
import requests
import os

app = FastAPI()
RAG_URL = os.environ.get("RAG_URL", "http://rag_server:8000/generate")
RAG_API_KEY = os.environ.get("RAG_API_KEY", "")
# Prefer Docker secret file if present
try:
    if os.path.exists("/run/secrets/rag_api_key"):
        with open("/run/secrets/rag_api_key", "r") as fh:
            sk = fh.read().strip()
            if sk:
                RAG_API_KEY = sk
except Exception:
    pass
# Optional: a default adapter path that this action server will signal to RAG via header
ACTION_PEFT_ADAPTER = os.environ.get("ACTION_PEFT_ADAPTER", "")


@app.post("/webhook")
async def rasa_action_webhook(req: Request):
    payload = await req.json()
    # Expect minimal Rasa action format: { "next_action": "action_call_rag", "tracker": {"latest_message": {"text": ".."}} }
    try:
        text = payload.get("tracker", {}).get("latest_message", {}).get("text")
    except Exception:
        text = None
    if not text:
        return {"events": [], "responses": [{"text": "I didn't get your message."}]}

    # Build RAG request
    # Allow Rasa to set a `language` slot in tracker.latest_message.intent_ranking or slots (simple extraction)
    # For backwards compatibility we use the language slot if available in tracker.latest_message.get("intent", {})
    body = {"query": text, "top_k": 3, "max_new_tokens": 128, "temperature": 0.0}
    headers = {}
    if RAG_API_KEY:
        headers["Authorization"] = f"Bearer {RAG_API_KEY}"
    # If the tracker provides a slot 'language', forward it as a header to RAG
    try:
        slots = payload.get("tracker", {}).get("slots", {}) or {}
        lang = slots.get("language")
        if not lang:
            # also check latest_message metadata
            lang = payload.get("tracker", {}).get("latest_message", {}).get("metadata", {}).get("language")
        if lang:
            headers["X-User-Language"] = str(lang)
    except Exception:
        pass
    if ACTION_PEFT_ADAPTER:
        headers["X-PEFT-ADAPTER"] = ACTION_PEFT_ADAPTER
    try:
        r = requests.post(RAG_URL, json=body, headers=headers, timeout=20)
        r.raise_for_status()
        j = r.json()
        txt = j.get("text") or j.get("answer") or ""
    except Exception as e:
        txt = f"Error contacting RAG: {e}"

    return {"events": [], "responses": [{"text": txt}]}


@app.post("/test")
async def test_call(payload: dict):
    # direct test endpoint: {"text":"..."}
    text = payload.get("text")
    body = {"query": text, "top_k": 3, "max_new_tokens": 128, "temperature": 0.0}
    headers = {}
    if RAG_API_KEY:
        headers["Authorization"] = f"Bearer {RAG_API_KEY}"
    if ACTION_PEFT_ADAPTER:
        headers["X-PEFT-ADAPTER"] = ACTION_PEFT_ADAPTER
    r = requests.post(RAG_URL, json=body, headers=headers, timeout=20)
    r.raise_for_status()
    return r.json()
