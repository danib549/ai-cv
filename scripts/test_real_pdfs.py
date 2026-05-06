"""Inspect PDF parser output and keyword grounding on real-world CV PDFs.

This is a diagnostic — it does NOT call the LLM. It shows what the extraction
LLM would actually see when handed each PDF, plus which rubric keywords get
detected by the deterministic grounding pre-pass.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from ai_cv.grounding import detect_domain, ground_keywords  # noqa: E402
from ai_cv.parser import parse_cv  # noqa: E402
from ai_cv.profile_loader import load_profile  # noqa: E402

PROFILE = load_profile(ROOT / "profiles" / "embedded_engineer.yaml")
SAMPLES_DIR = ROOT / "cvs" / "real_world_samples"


def inspect(path: Path) -> None:
    print("=" * 78)
    print(f"FILE: {path.name}  ({path.stat().st_size:,} bytes)")
    print("=" * 78)

    try:
        text = parse_cv(path)
    except Exception as e:
        print(f"  PARSE FAILED: {e}")
        return

    n_chars = len(text)
    n_lines = text.count("\n")
    n_words = len(text.split())
    print(f"Extracted:  {n_chars:,} chars,  {n_lines:,} lines,  {n_words:,} words")

    # Sample of the raw text — first 600 chars, with line breaks visible
    print("\n--- Raw text head (first 600 chars) ---")
    head = text[:600].replace("\t", "  ")
    print(head)
    print("--- end head ---\n")

    # Look for symptoms of parser problems
    issues = []
    if n_chars < 200:
        issues.append("Very little text extracted — possibly image-based PDF (needs OCR)")
    if "(cid:" in text:
        issues.append("Encoded characters present — broken font CMap")
    # Repeated header detection
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if lines:
        from collections import Counter
        counts = Counter(lines)
        repeated = [(ln, c) for ln, c in counts.items() if c >= 3 and len(ln) < 80]
        if repeated:
            issues.append(f"Repeated lines (likely header/footer): {repeated[:2]}")

    if issues:
        print("Symptoms:")
        for s in issues:
            print(f"  - {s}")
    else:
        print("Symptoms: none flagged")

    # Domain detection + keyword grounding against the embedded rubric
    domain = detect_domain(text)
    print(f"\nDetected domain (embedded rubric): {domain or '(none)'}")

    groundings = ground_keywords(text, PROFILE)
    if not groundings:
        print("Keyword hits: NONE — this CV doesn't match any embedded-engineer keywords")
    else:
        print(f"Keyword hits across {len(groundings)} skills:")
        for g in sorted(groundings, key=lambda x: -x.total)[:10]:
            kws = ", ".join(f"{h.keyword}×{h.count}" for h in g.hits[:5])
            more = f" (+{len(g.hits)-5} more)" if len(g.hits) > 5 else ""
            print(f"  - {g.skill}: {kws}{more}")


def main() -> None:
    pdfs = sorted(SAMPLES_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {SAMPLES_DIR}")
        sys.exit(1)
    for pdf in pdfs:
        inspect(pdf)
        print()


if __name__ == "__main__":
    main()
