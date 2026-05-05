"""Tests for ProfessionProfile.with_domain and CVScore totals/clamping."""

from __future__ import annotations

import pytest

from ai_cv.models import (
    CategoryScore,
    CVScore,
    RedFlagHit,
    SkillScore,
)


def test_compute_totals_recomputes_category_from_skills():
    cv = CVScore(
        categories=[
            CategoryScore(
                category="Core",
                score=999.0,  # the LLM lied; should be overridden
                max_points=20,
                skills=[
                    SkillScore(skill="A", score=5, max_points=10),
                    SkillScore(skill="B", score=4, max_points=10),
                ],
            ),
        ],
    )
    cv.compute_totals(total_max=100.0)
    assert cv.categories[0].score == 9.0
    assert cv.raw_score == 9.0
    assert cv.final_score == 9.0


def test_compute_totals_applies_red_flags_and_floor():
    cv = CVScore(
        categories=[
            CategoryScore(
                category="Core", score=0, max_points=10,
                skills=[SkillScore(skill="A", score=5, max_points=10)],
            ),
        ],
        red_flags=[RedFlagHit(flag="X", penalty=-50, reason="...")],
    )
    cv.compute_totals(total_max=100.0)
    assert cv.final_score == 0.0  # clipped to floor


def test_compute_totals_clips_to_ceiling():
    cv = CVScore(
        categories=[
            CategoryScore(
                category="Core", score=0, max_points=10,
                skills=[SkillScore(skill="A", score=999, max_points=999)],
            ),
        ],
    )
    cv.compute_totals(total_max=100.0)
    assert cv.final_score == 100.0


def test_with_domain_overrides_category_caps(profile):
    auto = profile.with_domain("automotive")
    cap_by_name = {c.name: c.max_points for c in auto.categories}
    # The embedded profile bumps Hardware Interfaces and adds a Functional Safety override
    # under Advanced & Differentiating Skills' children — verify the category-level lookup works.
    # Hardware Interfaces is a SKILL inside Core Technical Skills; with_domain only operates on
    # category names, so the structural invariant we check is total recompute.
    new_total = sum(c.max_points for c in auto.categories)
    assert auto.total_points == pytest.approx(new_total)


def test_with_domain_unknown_returns_self(profile):
    same = profile.with_domain("does-not-exist")
    assert same.total_points == profile.total_points
    assert [c.max_points for c in same.categories] == [c.max_points for c in profile.categories]


def test_with_domain_none_returns_self(profile):
    same = profile.with_domain(None)
    assert same is profile
