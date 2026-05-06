"""Probe what raw content a model returns for a tiny extraction prompt."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

model = sys.argv[1] if len(sys.argv) > 1 else "bjoernb/gemma4-31b-think:latest"

payload = {
    "model": model,
    "messages": [
        {"role": "system", "content": "You are a JSON-only assistant."},
        {"role": "user", "content": 'Reply with: {"hello": "world"}'},
    ],
    "format": "json",
    "stream": False,
    "options": {"temperature": 0.1, "num_ctx": 4096},
}

print(f"Probing {model}...")
t0 = time.time()
with httpx.Client(timeout=300.0) as c:
    resp = c.post("http://localhost:11434/api/chat", json=payload)
print(f"HTTP {resp.status_code}  in {time.time()-t0:.1f}s")
print("--- raw body ---")
print(resp.text[:2000])
