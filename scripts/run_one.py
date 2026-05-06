"""One-off: run the full scoring pipeline on a PDF with a generous LLM timeout."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from ai_cv.llm import LLMClient  # noqa: E402
from ai_cv.profile_loader import load_profile  # noqa: E402
from ai_cv.report import print_score  # noqa: E402
from ai_cv.scorer import CVScorer  # noqa: E402

if len(sys.argv) < 2:
    print("usage: run_one.py <pdf> [model]")
    sys.exit(2)

pdf = Path(sys.argv[1])
model = sys.argv[2] if len(sys.argv) > 2 else "gemma4:26b-a4b-it-q4_K_M"

profile = load_profile(ROOT / "profiles" / "embedded_engineer.yaml")
llm = LLMClient(model=model, timeout=900.0, temperature=0.1, num_ctx=16384)

print(f"Scoring {pdf.name} with {model} (this can take several minutes)...")
scorer = CVScorer(profile, llm)
result = scorer.score_cv(pdf)
print_score(result)
