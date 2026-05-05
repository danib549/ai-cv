"""Tests for the deterministic keyword-grounding pre-pass."""

from __future__ import annotations

from ai_cv.grounding import detect_domain, ground_keywords


def test_keyword_hits_collected_per_skill(profile, cv_text):
    text = cv_text("senior_embedded_strong.txt")
    groundings = ground_keywords(text, profile)
    by_skill = {g.skill: g for g in groundings}

    # Strong embedded CV must surface programming languages and microcontroller hits.
    assert "Programming Languages" in by_skill
    assert by_skill["Programming Languages"].total >= 3
    assert any(h.keyword == "C" for h in by_skill["Programming Languages"].hits)

    assert "Microcontroller Families" in by_skill
    keywords = {h.keyword for h in by_skill["Microcontroller Families"].hits}
    # CV mentions ARM Cortex-M, NXP, Nordic, ESP32 as full tokens.
    assert "ARM Cortex-M" in keywords or "Cortex-M7" in keywords
    assert any(k in keywords for k in ("ESP32", "NXP", "Nordic nRF"))


def test_word_boundary_avoids_false_positives(profile):
    """A C in 'Cortex-M' must not count as a hit for the 'C' language keyword
    once we strip out cortex mentions; we still want the ones in 'C (expert)'.
    """
    # synthetic example: 'CMake' contains 'C' but should not match the 'C' keyword.
    text = "I use CMake and Python. I have not written C."
    groundings = ground_keywords(text, profile)
    by_skill = {g.skill: g for g in groundings}
    pl = by_skill.get("Programming Languages")
    assert pl is not None
    c_hits = [h for h in pl.hits if h.keyword == "C"]
    assert len(c_hits) == 1
    assert c_hits[0].count == 1


def test_special_chars_in_keyword_match(profile):
    text = "Wrote firmware in C++ and embedded C for STM32 boards."
    groundings = ground_keywords(text, profile)
    pl = next(g for g in groundings if g.skill == "Programming Languages")
    keywords = {h.keyword for h in pl.hits}
    assert "C++" in keywords
    assert "C" in keywords


def test_detect_domain_automotive(cv_text):
    text = cv_text("senior_embedded_strong.txt")
    assert detect_domain(text) == "automotive"


def test_detect_domain_iot(cv_text):
    text = cv_text("mid_level_iot.txt")
    assert detect_domain(text) == "iot"


def test_detect_domain_generic_returns_empty():
    assert detect_domain("I make sandwiches and play guitar.") == ""
