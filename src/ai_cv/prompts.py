"""Prompt templates for two-pass LLM-based CV scoring."""

from __future__ import annotations

import json

from ai_cv.models import ExtractedCV, ProfessionProfile


# ---------------------------------------------------------------------------
# Pass 1 — extraction
# ---------------------------------------------------------------------------

EXTRACTION_SYSTEM = """You are an expert CV/resume parser.

Your job is to extract STRUCTURED FACTS from a CV. You do not score, judge, or
infer. You only record what the candidate explicitly stated. If something is
not in the CV, leave the field empty — never invent.

For each work experience, copy the original bullets verbatim (do not summarize).
For skills, list the technologies/tools/languages exactly as they appear; if the
candidate stated explicit years for a skill (e.g. "C (8yr)") record it, otherwise
leave years null.

Detected_domain should be one of: automotive, aerospace, medical, industrial,
iot, consumer, telecom, generic — based on the dominant industry of the recent
roles. Use "generic" if unclear.

Respond with valid JSON matching this schema:

{
  "candidate_name": "string",
  "contact": {"email": "...", "linkedin": "...", "github": "...", "location": "..."},
  "summary": "the candidate's own summary text, verbatim",
  "total_years_experience": number_or_null,
  "skills": [
    {"name": "string", "years": number_or_null, "context": "short quote where it appeared"}
  ],
  "experiences": [
    {
      "company": "string",
      "title": "string",
      "start": "string",
      "end": "string",
      "duration_years": number_or_null,
      "domain": "automotive|aerospace|medical|industrial|iot|consumer|telecom|generic",
      "bullets": ["verbatim bullet 1", "verbatim bullet 2"]
    }
  ],
  "education": [
    {"degree": "BS|MS|PhD|...", "field": "string", "institution": "string", "year": "string"}
  ],
  "certifications": ["string"],
  "shipped_products": ["evidence sentences mentioning units shipped, production, launch"],
  "detected_domain": "string"
}
"""


def build_extraction_user_prompt(cv_text: str) -> str:
    return f"""Extract structured facts from this CV.

---BEGIN CV---
{cv_text}
---END CV---

Respond with JSON only. Do not score or evaluate — only extract."""


# ---------------------------------------------------------------------------
# Pass 2 — scoring
# ---------------------------------------------------------------------------


def build_scoring_system_prompt(profile: ProfessionProfile) -> str:
    """Build the scoring system prompt from a profession profile."""
    cats = ""
    for cat in profile.categories:
        cats += f"\n### {cat.name} (max {cat.max_points} pts)\n"
        cats += f"{cat.description}\n"
        for skill in cat.skills:
            cats += (
                f"  - **{skill.name}** (max {skill.max_points} pts): "
                f"{skill.description.strip()}\n"
            )

    flags = ""
    for rf in profile.red_flags:
        flags += f"  - {rf.name} (penalty: {rf.penalty}): {rf.description}\n"

    tiers = ""
    for tier in profile.experience_tiers:
        yrs = f"{tier.min_years}-{tier.max_years}" if tier.max_years else f"{tier.min_years}+"
        tiers += f"  - **{tier.level}** ({yrs} yrs): {', '.join(tier.expected)}\n"

    return f"""You are an expert evaluator for the role of **{profile.profession}**.

You will receive (1) STRUCTURED FACTS already extracted from a candidate's CV
and (2) DETERMINISTIC KEYWORD HITS that confirm which technologies were
literally mentioned. Use both as your evidence base. Do NOT invent skills the
candidate did not mention. Vague mentions without context get partial credit
at best.

## Scoring Rubric (Total: {profile.total_points} pts)
{cats}

## Red Flags (apply as negative penalties)
{flags}

## Experience Level Reference
{tiers}

## Output Format

Respond with valid JSON matching this schema:

{{
  "experience_level": "junior | mid | senior | staff",
  "categories": [
    {{
      "category": "category name exactly as listed",
      "score": number,
      "max_points": number,
      "skills": [
        {{
          "skill": "skill name exactly as listed",
          "score": number,
          "max_points": number,
          "evidence": "brief quote or rationale grounded in the extracted facts"
        }}
      ]
    }}
  ],
  "red_flags": [
    {{"flag": "name", "penalty": negative_number, "reason": "why"}}
  ],
  "summary": "2-3 sentence overall assessment"
}}

Rules:
- Each skill score must be between 0 and its max_points.
- Only include red_flags that actually apply.
- Cite evidence from the extracted facts or keyword hits — not from your priors.
"""


def build_scoring_user_prompt(extracted: ExtractedCV, grounding_block: str) -> str:
    facts_json = extracted.model_dump_json(indent=2, exclude_none=True)
    return f"""## Extracted Facts (from pass 1)

```json
{facts_json}
```

## Keyword Hits (deterministic — these terms literally appear in the CV)

{grounding_block}

## Task

Score this candidate against the rubric. Respond with JSON only.
"""


# ---------------------------------------------------------------------------
# Backwards-compatible single-pass helpers (kept so older callers / tests work)
# ---------------------------------------------------------------------------


def build_system_prompt(profile: ProfessionProfile) -> str:  # legacy
    return build_scoring_system_prompt(profile)


def build_user_prompt(cv_text: str) -> str:  # legacy
    return f"""Analyze and score the following CV/resume:

---BEGIN CV---
{cv_text}
---END CV---

Score this candidate according to the rubric. Respond with JSON only."""


# small helper used by tests to ensure the prompts contain valid JSON examples
def _example_payload_is_valid_json(s: str) -> bool:
    try:
        json.loads(s)
        return True
    except Exception:
        return False
