"""Keyword grounding — deterministic anchors fed to the LLM to reduce hallucination."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ai_cv.models import ProfessionProfile


@dataclass
class KeywordHit:
    keyword: str
    count: int
    sample: str = ""  # short surrounding snippet from the CV


@dataclass
class SkillGrounding:
    skill: str
    hits: list[KeywordHit] = field(default_factory=list)

    @property
    def total(self) -> int:
        return sum(h.count for h in self.hits)


def _word_pattern(keyword: str) -> re.Pattern[str]:
    """Build a case-insensitive regex with word boundaries.

    Special chars in keywords (./+-#) are escaped. Word boundaries are skipped
    when the keyword starts/ends with a non-word char (e.g. "C++" or ".NET")
    because Python's \\b would never match there.
    """
    esc = re.escape(keyword)
    left = r"(?<![A-Za-z0-9_])" if keyword[:1].isalnum() or keyword[:1] == "_" else ""
    right = r"(?![A-Za-z0-9_])" if keyword[-1:].isalnum() or keyword[-1:] == "_" else ""
    return re.compile(f"{left}{esc}{right}", re.IGNORECASE)


def ground_keywords(
    cv_text: str,
    profile: ProfessionProfile,
    snippet_window: int = 60,
) -> list[SkillGrounding]:
    """Match every profile keyword against the CV and collect hits per skill.

    The result is a deterministic, citable inventory: which keywords appeared,
    how often, and a sample context. We pass this to the scoring LLM so it
    treats keywords as evidence anchors rather than guessing.
    """
    out: list[SkillGrounding] = []
    for cat in profile.categories:
        for skill in cat.skills:
            grounding = SkillGrounding(skill=skill.name)
            for kw in skill.keywords:
                pat = _word_pattern(kw)
                matches = list(pat.finditer(cv_text))
                if not matches:
                    continue
                first = matches[0]
                start = max(0, first.start() - snippet_window)
                end = min(len(cv_text), first.end() + snippet_window)
                sample = cv_text[start:end].replace("\n", " ").strip()
                grounding.hits.append(
                    KeywordHit(keyword=kw, count=len(matches), sample=sample)
                )
            if grounding.hits:
                out.append(grounding)
    return out


def format_grounding_block(groundings: list[SkillGrounding]) -> str:
    """Render groundings as a compact block for the scoring prompt."""
    if not groundings:
        return "(no keyword matches found)"
    lines = []
    for g in groundings:
        kw_summary = ", ".join(f"{h.keyword}×{h.count}" for h in g.hits)
        lines.append(f"- {g.skill}: {kw_summary}")
        # one representative snippet keeps the prompt small but anchored
        if g.hits and g.hits[0].sample:
            snippet = g.hits[0].sample
            if len(snippet) > 140:
                snippet = snippet[:140] + "..."
            lines.append(f"    e.g. \"{snippet}\"")
    return "\n".join(lines)


_DOMAIN_HINTS: dict[str, list[str]] = {
    "automotive": ["automotive", "AUTOSAR", "ASIL", "ISO 26262", "CAN-FD", "ECU", "ADAS"],
    "aerospace": ["aerospace", "DO-178", "DAL ", "avionics", "defense"],
    "medical": ["medical device", "IEC 62304", "FDA 510", "clinical"],
    "industrial": ["industrial", "PLC", "Modbus", "EtherCAT", "PROFINET", "SCADA"],
    "iot": ["IoT", "smart home", "MQTT", "AWS IoT", "Azure IoT", "BLE mesh", "LoRaWAN"],
}


def detect_domain(cv_text: str) -> str:
    """Heuristically infer the candidate's primary domain from raw CV text.

    Used to apply ProfessionProfile.domain_variants without an extra LLM call.
    Returns an empty string when no domain dominates.
    """
    scores: dict[str, int] = {}
    for domain, hints in _DOMAIN_HINTS.items():
        total = 0
        for h in hints:
            total += len(_word_pattern(h).findall(cv_text))
        if total > 0:
            scores[domain] = total
    if not scores:
        return ""
    best, best_count = max(scores.items(), key=lambda kv: kv[1])
    # Require at least 2 hits OR a clear lead (2× the runner-up) to avoid noise.
    runner_up = max((v for k, v in scores.items() if k != best), default=0)
    if best_count < 2 and best_count <= runner_up:
        return ""
    return best
