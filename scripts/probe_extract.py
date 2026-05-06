"""Probe the raw extraction output for a real PDF — see why JSON parsing failed."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from ai_cv.parser import parse_cv  # noqa: E402
from ai_cv.prompts import EXTRACTION_SYSTEM, build_extraction_user_prompt  # noqa: E402

model = sys.argv[2] if len(sys.argv) > 2 else "bjoernb/gemma4-31b-think:latest"
pdf = Path(sys.argv[1])

cv_text = parse_cv(pdf)
print(f"CV chars: {len(cv_text)}")

payload = {
    "model": model,
    "messages": [
        {"role": "system", "content": EXTRACTION_SYSTEM},
        {"role": "user", "content": build_extraction_user_prompt(cv_text)},
    ],
    "format": "json",
    "stream": False,
    "options": {"temperature": 0.1, "num_ctx": 16384},
}

print(f"Calling {model}...")
t0 = time.time()
with httpx.Client(timeout=600.0) as c:
    resp = c.post("http://localhost:11434/api/chat", json=payload)
elapsed = time.time() - t0
print(f"HTTP {resp.status_code}  in {elapsed:.1f}s")
data = resp.json()
content = data["message"]["content"]
print(f"Output length: {len(content)} chars")
print("--- first 1500 chars of content ---")
print(content[:1500])
print("--- last 500 chars ---")
print(content[-500:])
