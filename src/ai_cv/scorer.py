"""Scoring engine — two-pass pipeline (extract → ground → score)."""

from __future__ import annotations

import asyncio
from pathlib import Path

from ai_cv.grounding import (
    SkillGrounding,
    detect_domain,
    format_grounding_block,
    ground_keywords,
)
from ai_cv.llm import LLMClient
from ai_cv.models import (
    CategoryScore,
    CVScore,
    ExtractedCV,
    ExtractionResponse,
    ProfessionProfile,
    RedFlagHit,
    ScoringResponse,
    SkillScore,
)
from ai_cv.parser import parse_cv
from ai_cv.prompts import (
    EXTRACTION_SYSTEM,
    build_extraction_user_prompt,
    build_scoring_system_prompt,
    build_scoring_user_prompt,
)


class CVScorer:
    """Two-pass CV scorer: structured extraction → rubric scoring."""

    def __init__(self, profile: ProfessionProfile, llm: LLMClient):
        self.profile = profile
        self.llm = llm

    # --- Pass 1 ---
    def extract(self, cv_text: str) -> ExtractedCV:
        resp = self.llm.generate_validated(
            EXTRACTION_SYSTEM,
            build_extraction_user_prompt(cv_text),
            ExtractionResponse,
        )
        return ExtractedCV(**resp.model_dump())

    # --- Pass 2 ---
    def score_extracted(
        self,
        extracted: ExtractedCV,
        groundings: list[SkillGrounding],
        profile: ProfessionProfile,
    ) -> ScoringResponse:
        system = build_scoring_system_prompt(profile)
        user = build_scoring_user_prompt(extracted, format_grounding_block(groundings))
        return self.llm.generate_validated(system, user, ScoringResponse)

    # --- Public API ---
    def score_cv(self, cv_path: str | Path) -> CVScore:
        cv_path = Path(cv_path)
        cv_text = parse_cv(cv_path)
        if not cv_text.strip():
            raise ValueError(f"CV file is empty: {cv_path}")

        extracted = self.extract(cv_text)

        # Domain detection: prefer the LLM's call, fall back to heuristic on raw text.
        domain = (extracted.detected_domain or detect_domain(cv_text) or "").lower()
        scoring_profile = self.profile.with_domain(domain)

        groundings = ground_keywords(cv_text, scoring_profile)
        scoring = self.score_extracted(extracted, groundings, scoring_profile)

        return self._assemble(
            extracted=extracted,
            scoring=scoring,
            scoring_profile=scoring_profile,
            domain=domain,
            file_path=str(cv_path),
        )

    def score_batch(self, cv_paths: list[str | Path]) -> list[CVScore]:
        results = []
        for path in cv_paths:
            try:
                results.append(self.score_cv(path))
            except Exception as e:
                results.append(self._error_result(path, e))
        return results

    async def score_batch_async(self, cv_paths: list[str | Path]) -> list[CVScore]:
        loop = asyncio.get_event_loop()
        # Preserve order via gather (return_exceptions for robust error handling).
        coros = [loop.run_in_executor(None, self.score_cv, p) for p in cv_paths]
        raw = await asyncio.gather(*coros, return_exceptions=True)
        results: list[CVScore] = []
        for path, item in zip(cv_paths, raw):
            if isinstance(item, BaseException):
                results.append(self._error_result(path, item))
            else:
                results.append(item)
        return results

    # --- Helpers ---
    def _assemble(
        self,
        extracted: ExtractedCV,
        scoring: ScoringResponse,
        scoring_profile: ProfessionProfile,
        domain: str,
        file_path: str,
    ) -> CVScore:
        # Build a name → max_points lookup so we can clamp scores defensively.
        skill_caps: dict[tuple[str, str], float] = {}
        for cat in scoring_profile.categories:
            for skill in cat.skills:
                skill_caps[(cat.name, skill.name)] = skill.max_points

        categories: list[CategoryScore] = []
        for cat in scoring.categories:
            clamped_skills: list[SkillScore] = []
            for s in cat.skills:
                cap = skill_caps.get((cat.category, s.skill), s.max_points)
                clamped_skills.append(
                    SkillScore(
                        skill=s.skill,
                        score=max(0.0, min(s.score, cap)),
                        max_points=cap,
                        evidence=s.evidence,
                    )
                )
            categories.append(
                CategoryScore(
                    category=cat.category,
                    score=cat.score,  # recomputed from skills below
                    max_points=cat.max_points,
                    skills=clamped_skills,
                )
            )

        result = CVScore(
            candidate_name=extracted.candidate_name or "Unknown",
            file=file_path,
            profession=self.profile.profession,
            detected_domain=domain,
            categories=categories,
            red_flags=list(scoring.red_flags),
            summary=scoring.summary,
            experience_level=scoring.experience_level,
        )
        result.compute_totals(total_max=scoring_profile.total_points)
        return result

    def _error_result(self, path: str | Path, err: BaseException) -> CVScore:
        return CVScore(
            candidate_name="ERROR",
            file=str(path),
            profession=self.profile.profession,
            categories=[],
            summary=f"Failed to score: {err}",
        )
