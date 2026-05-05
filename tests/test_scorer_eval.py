"""End-to-end eval harness — drives the full scorer with a mocked LLM and
asserts expected score ranges, red flags, and experience-level mapping per CV.

Each fixture CV gets a hand-crafted FakeLLM script that mirrors what a
well-behaved model would produce. The point of these tests is NOT to evaluate
the LLM — it's to evaluate the scoring pipeline (extraction → grounding →
domain reweighting → clamping → totals) is wired correctly and stable across
prompt/profile changes.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ai_cv.scorer import CVScorer
from tests.conftest import CVS_DIR, make_fake_llm


# Minimal, schema-valid extraction payloads keyed by CV file name.
EXTRACTION_FIXTURES: dict[str, dict] = {
    "senior_embedded_strong.txt": {
        "candidate_name": "Ahmed Khalil",
        "summary": "Senior embedded engineer, automotive, ASIL-D.",
        "total_years_experience": 8,
        "skills": [
            {"name": "C", "years": 8, "context": "C (expert, 8yr)"},
            {"name": "C++", "years": 5, "context": "C++ (advanced, 5yr)"},
            {"name": "FreeRTOS", "context": "FreeRTOS (expert)"},
            {"name": "Zephyr", "context": "Zephyr (advanced)"},
        ],
        "experiences": [
            {"company": "Bosch", "title": "Lead Firmware", "domain": "automotive",
             "duration_years": 5, "bullets": []},
            {"company": "Siemens", "title": "Senior Embedded SWE", "domain": "industrial",
             "duration_years": 2.8, "bullets": []},
            {"company": "Continental", "title": "Embedded SWE", "domain": "automotive",
             "duration_years": 1.8, "bullets": []},
        ],
        "education": [{"degree": "MS", "field": "EE", "institution": "TUM", "year": "2016"}],
        "certifications": ["TUV Functional Safety"],
        "shipped_products": ["1.2M units shipped"],
        "detected_domain": "automotive",
    },
    "mid_level_iot.txt": {
        "candidate_name": "Priya Sharma",
        "summary": "Mid-level IoT firmware engineer.",
        "total_years_experience": 3.5,
        "skills": [
            {"name": "C", "years": 3.5, "context": "C primary"},
            {"name": "ESP32", "context": "ESP32-S3, ESP32-C3"},
            {"name": "FreeRTOS", "context": "FreeRTOS"},
        ],
        "experiences": [
            {"company": "Xiaomi", "title": "Embedded SWE", "domain": "iot",
             "duration_years": 2, "bullets": []},
            {"company": "Wipro", "title": "Firmware Dev", "domain": "iot",
             "duration_years": 1.5, "bullets": []},
        ],
        "education": [{"degree": "BTech", "field": "ECE", "institution": "VIT", "year": "2020"}],
        "certifications": [],
        "shipped_products": ["sold 200 units on Tindie"],
        "detected_domain": "iot",
    },
    "junior_fresh_grad.txt": {
        "candidate_name": "Carlos Mendez",
        "summary": "Recent ECE graduate.",
        "total_years_experience": 0,
        "skills": [
            {"name": "C", "context": "C primary"},
            {"name": "STM32", "context": "STM32F407 capstone"},
            {"name": "FreeRTOS", "context": "FreeRTOS coursework"},
        ],
        "experiences": [
            {"company": "Texas Instruments", "title": "Intern", "domain": "consumer",
             "duration_years": 0.3, "bullets": []},
        ],
        "education": [{"degree": "BS", "field": "ECE", "institution": "UT Austin", "year": "2025"}],
        "certifications": [],
        "shipped_products": [],
        "detected_domain": "generic",
    },
    "weak_mismatch.txt": {
        "candidate_name": "Jason Turner",
        "summary": "Web dev wanting to switch to embedded.",
        "total_years_experience": 6,
        "skills": [
            {"name": "Go", "years": 6, "context": "Go primary"},
            {"name": "Arduino", "context": "Arduino projects"},
        ],
        "experiences": [
            {"company": "Stripe", "title": "Senior SWE", "domain": "generic",
             "duration_years": 3, "bullets": []},
        ],
        "education": [{"degree": "BS", "field": "CS", "institution": "Berkeley", "year": "2018"}],
        "certifications": [],
        "shipped_products": [],
        "detected_domain": "generic",
    },
    "buzzword_red_flags.txt": {
        "candidate_name": "Alex Johnson",
        "summary": "Claims expert in 20 things.",
        "total_years_experience": 4,
        "skills": [
            {"name": "C", "context": "Expert level"},
            {"name": "Arduino", "context": "Arduino + Raspberry Pi"},
        ],
        "experiences": [
            {"company": "TechStartup", "title": "IoT Dev", "domain": "iot",
             "duration_years": 2, "bullets": []},
        ],
        "education": [{"degree": "BS", "field": "IT", "institution": "NYU", "year": "2020"}],
        "certifications": ["AWS SA Associate"],
        "shipped_products": [],
        "detected_domain": "iot",
    },
}


def _scoring_payload(
    profile,
    skill_scores: dict[str, float],
    *,
    level: str,
    flags: list[tuple[str, float]] | None = None,
    summary: str = "",
) -> dict:
    """Build a schema-valid scoring response from per-skill score targets.

    Skills not listed get 0. Category totals are recomputed by the pipeline so
    we don't have to sum them here.
    """
    cats = []
    for cat in profile.categories:
        cat_skills = []
        for skill in cat.skills:
            cat_skills.append({
                "skill": skill.name,
                "score": float(skill_scores.get(skill.name, 0.0)),
                "max_points": skill.max_points,
                "evidence": "",
            })
        cats.append({
            "category": cat.name,
            "score": 0.0,  # will be recomputed
            "max_points": cat.max_points,
            "skills": cat_skills,
        })
    return {
        "experience_level": level,
        "categories": cats,
        "red_flags": [{"flag": n, "penalty": p, "reason": "test"} for n, p in (flags or [])],
        "summary": summary,
    }


# Each entry: (cv file, scoring scenario, expected (min, max) final score, expected level, expected flag count).
EVAL_CASES = [
    pytest.param(
        "senior_embedded_strong.txt",
        # Strong scoring: most categories near max.
        {
            "Programming Languages": 14, "Microcontroller Families": 7,
            "RTOS Knowledge": 6, "Bare-Metal Programming": 3,
            "Hardware Interfaces": 4, "Memory Management": 2, "Interrupts & Real-Time": 1,
            "Debugging Tools & Methodology": 5, "Build Systems & Toolchains": 3,
            "Version Control": 2, "Linux Kernel & Driver Development": 4,
            "Bootloader Development": 3, "Power Management": 3,
            "Functional Safety": 3, "Embedded Security": 3, "CI/CD for Embedded": 2,
            "Industry Domain Experience": 6, "Education": 4, "Certifications": 2,
            "Experience Level Match": 5, "Shipped Products & Impact": 3,
        },
        (75, 105),  # automotive variant raises Hardware Interfaces ceiling
        "senior",
        0,
        id="senior_embedded_strong",
    ),
    pytest.param(
        "mid_level_iot.txt",
        {
            "Programming Languages": 10, "Microcontroller Families": 5,
            "RTOS Knowledge": 4, "Hardware Interfaces": 3, "Memory Management": 1,
            "Interrupts & Real-Time": 1, "Debugging Tools & Methodology": 3,
            "Build Systems & Toolchains": 2, "Version Control": 2,
            "Power Management": 4, "Wireless Protocols": 2,
            "Cloud & Connectivity": 1, "Industry Domain Experience": 4,
            "Education": 3, "Experience Level Match": 4, "Shipped Products & Impact": 2,
        },
        (40, 75),
        "mid",
        0,
        id="mid_level_iot",
    ),
    pytest.param(
        "junior_fresh_grad.txt",
        {
            "Programming Languages": 8, "Microcontroller Families": 4,
            "RTOS Knowledge": 3, "Bare-Metal Programming": 2,
            "Hardware Interfaces": 2, "Debugging Tools & Methodology": 2,
            "Version Control": 2, "Education": 3,
            "Experience Level Match": 3,
        },
        (20, 50),
        "junior",
        0,
        id="junior_fresh_grad",
    ),
    pytest.param(
        "weak_mismatch.txt",
        {
            "Programming Languages": 3, "Microcontroller Families": 1,
            "Version Control": 2, "Education": 2,
        },
        (0, 20),
        "junior",
        2,  # arduino-only + only-high-level-langs (or no-C)
        id="weak_mismatch",
    ),
    pytest.param(
        "buzzword_red_flags.txt",
        {
            "Programming Languages": 4, "Microcontroller Families": 2,
            "RTOS Knowledge": 1, "Education": 2,
            "Experience Level Match": 1,
        },
        (0, 35),
        "mid",
        2,  # buzzword stuffing + claims-expert-in-everything
        id="buzzword_red_flags",
    ),
]


@pytest.mark.parametrize("cv_file,skill_scores,score_range,expected_level,min_flags", EVAL_CASES)
def test_scorer_eval(profile, cv_file, skill_scores, score_range, expected_level, min_flags):
    extraction = EXTRACTION_FIXTURES[cv_file]
    domain = extraction.get("detected_domain", "")
    scoring_profile = profile.with_domain(domain)

    flags = []
    if "weak_mismatch" in cv_file:
        flags = [("Arduino-only experience", -8), ("Only high-level languages", -10)]
    elif "buzzword" in cv_file:
        flags = [("Buzzword stuffing", -5), ("Claims expert in everything", -4)]

    scoring = _scoring_payload(
        scoring_profile,
        skill_scores,
        level=expected_level,
        flags=flags,
        summary=f"eval: {cv_file}",
    )

    llm = make_fake_llm(extraction=extraction, scoring=scoring)
    scorer = CVScorer(profile, llm)
    result = scorer.score_cv(CVS_DIR / cv_file)

    lo, hi = score_range
    assert lo <= result.final_score <= hi, (
        f"{cv_file}: expected {lo}..{hi}, got {result.final_score}"
    )
    assert result.experience_level == expected_level
    assert len(result.red_flags) == min_flags
    # Category totals must match summed skills (defensive recompute).
    for cat in result.categories:
        assert cat.score == pytest.approx(sum(s.score for s in cat.skills), abs=0.01)


def test_clamping_protects_against_runaway_llm(profile):
    """If the LLM returns a skill score above the cap, the pipeline must clamp it."""
    extraction = EXTRACTION_FIXTURES["junior_fresh_grad.txt"]
    bad_scoring = _scoring_payload(
        profile,
        {"Programming Languages": 9999.0},  # absurd
        level="junior",
    )
    llm = make_fake_llm(extraction=extraction, scoring=bad_scoring)
    scorer = CVScorer(profile, llm)
    result = scorer.score_cv(CVS_DIR / "junior_fresh_grad.txt")
    pl = next(s for c in result.categories for s in c.skills if s.skill == "Programming Languages")
    assert pl.score == pl.max_points  # clamped
    assert result.final_score <= profile.total_points


def test_domain_variants_applied_for_iot(profile):
    """Mid-level IoT CV should be scored against the IoT-adjusted profile."""
    extraction = EXTRACTION_FIXTURES["mid_level_iot.txt"]
    iot_profile = profile.with_domain("iot")
    scoring = _scoring_payload(iot_profile, {}, level="mid")
    llm = make_fake_llm(extraction=extraction, scoring=scoring)
    scorer = CVScorer(profile, llm)
    result = scorer.score_cv(CVS_DIR / "mid_level_iot.txt")
    assert result.detected_domain == "iot"
    # The IoT variant lifts Wireless Protocols / Cloud / Power Management caps.
    iot_max_total = sum(c.max_points for c in iot_profile.categories)
    assert iot_max_total >= profile.total_points
