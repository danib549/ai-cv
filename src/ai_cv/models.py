"""Pydantic models for profession profiles, extraction, and scoring results."""

from __future__ import annotations

from pydantic import BaseModel, Field


class SkillCriterion(BaseModel):
    """A single scorable skill within a category."""

    name: str
    max_points: float
    description: str = ""
    keywords: list[str] = Field(default_factory=list)


class ScoringCategory(BaseModel):
    """A group of related skills (e.g. 'Core Technical Skills')."""

    name: str
    max_points: float
    description: str = ""
    skills: list[SkillCriterion]


class RedFlag(BaseModel):
    """A pattern that should reduce the candidate's score."""

    name: str
    penalty: float  # negative number
    description: str = ""


class ExperienceTier(BaseModel):
    """Expected competencies at a given experience level."""

    level: str  # junior, mid, senior, staff
    min_years: int
    max_years: int | None = None
    expected: list[str] = Field(default_factory=list)


class ProfessionProfile(BaseModel):
    """Complete scoring rubric for a profession."""

    profession: str
    version: str = "1.0"
    total_points: float = 100.0
    description: str = ""
    categories: list[ScoringCategory]
    red_flags: list[RedFlag] = Field(default_factory=list)
    experience_tiers: list[ExperienceTier] = Field(default_factory=list)
    domain_variants: dict[str, dict[str, float]] = Field(default_factory=dict)

    def category_names(self) -> list[str]:
        return [c.name for c in self.categories]

    def with_domain(self, domain: str | None) -> "ProfessionProfile":
        """Return a copy of the profile with category max_points overridden by domain_variants.

        If domain is None or unknown, returns the original profile unchanged.
        Variants apply at the category level — total_points is recomputed from the new
        category caps so the rubric stays internally consistent.
        """
        if not domain:
            return self
        overrides = self.domain_variants.get(domain.lower())
        if not overrides:
            return self

        new_categories: list[ScoringCategory] = []
        for cat in self.categories:
            if cat.name in overrides:
                new_max = float(overrides[cat.name])
                # Scale child skill max_points proportionally so they still sum to category max.
                old_max = cat.max_points
                if old_max > 0 and new_max != old_max:
                    factor = new_max / old_max
                    new_skills = [
                        s.model_copy(update={"max_points": round(s.max_points * factor, 2)})
                        for s in cat.skills
                    ]
                else:
                    new_skills = list(cat.skills)
                new_categories.append(
                    cat.model_copy(update={"max_points": new_max, "skills": new_skills})
                )
            else:
                new_categories.append(cat)

        new_total = sum(c.max_points for c in new_categories)
        return self.model_copy(
            update={"categories": new_categories, "total_points": new_total}
        )


# --- Extraction (pass 1) models ---


class ExtractedSkillMention(BaseModel):
    """A skill or technology mentioned in the CV with surrounding context."""

    name: str
    years: float | None = None  # if explicitly stated
    context: str = ""  # short quote / where it appeared


class ExtractedExperience(BaseModel):
    """A single role from the work history."""

    company: str = ""
    title: str = ""
    start: str = ""
    end: str = ""
    duration_years: float | None = None
    domain: str = ""  # automotive, medical, iot, etc.
    bullets: list[str] = Field(default_factory=list)


class ExtractedEducation(BaseModel):
    """An education entry."""

    degree: str = ""
    field: str = ""
    institution: str = ""
    year: str = ""


class ExtractedCV(BaseModel):
    """Structured facts pulled from a raw CV (pass 1 of the pipeline)."""

    candidate_name: str = "Unknown"
    contact: dict[str, str] = Field(default_factory=dict)
    summary: str = ""
    total_years_experience: float | None = None
    skills: list[ExtractedSkillMention] = Field(default_factory=list)
    experiences: list[ExtractedExperience] = Field(default_factory=list)
    education: list[ExtractedEducation] = Field(default_factory=list)
    certifications: list[str] = Field(default_factory=list)
    shipped_products: list[str] = Field(default_factory=list)
    detected_domain: str = ""  # primary domain inferred from experience


# --- Scoring output models ---


class SkillScore(BaseModel):
    """LLM-assigned score for a single skill."""

    skill: str
    score: float
    max_points: float
    evidence: str = ""  # quote or rationale from the CV


class CategoryScore(BaseModel):
    """Aggregated score for a category."""

    category: str
    score: float
    max_points: float
    skills: list[SkillScore]


class RedFlagHit(BaseModel):
    """A red flag detected in the CV."""

    flag: str
    penalty: float
    reason: str = ""


class CVScore(BaseModel):
    """Complete scoring result for one CV."""

    candidate_name: str = "Unknown"
    file: str = ""
    profession: str = ""
    detected_domain: str = ""
    categories: list[CategoryScore]
    red_flags: list[RedFlagHit] = Field(default_factory=list)
    raw_score: float = 0.0
    penalty_total: float = 0.0
    final_score: float = 0.0
    summary: str = ""
    experience_level: str = ""

    def compute_totals(self, total_max: float | None = None) -> None:
        # Category totals are recomputed from skill scores — never trust the LLM's sum.
        for cat in self.categories:
            cat.score = round(sum(s.score for s in cat.skills), 2)
        self.raw_score = round(sum(c.score for c in self.categories), 2)
        self.penalty_total = round(sum(r.penalty for r in self.red_flags), 2)
        ceiling = total_max if total_max is not None else float("inf")
        self.final_score = max(0.0, min(self.raw_score + self.penalty_total, ceiling))


# --- Wire formats for the two LLM passes (used with generate_validated) ---


class ExtractionResponse(BaseModel):
    """Pass-1 LLM response wire format."""

    candidate_name: str = "Unknown"
    contact: dict[str, str] = Field(default_factory=dict)
    summary: str = ""
    total_years_experience: float | None = None
    skills: list[ExtractedSkillMention] = Field(default_factory=list)
    experiences: list[ExtractedExperience] = Field(default_factory=list)
    education: list[ExtractedEducation] = Field(default_factory=list)
    certifications: list[str] = Field(default_factory=list)
    shipped_products: list[str] = Field(default_factory=list)
    detected_domain: str = ""


class ScoringResponse(BaseModel):
    """Pass-2 LLM response wire format."""

    experience_level: str = ""
    categories: list[CategoryScore]
    red_flags: list[RedFlagHit] = Field(default_factory=list)
    summary: str = ""
