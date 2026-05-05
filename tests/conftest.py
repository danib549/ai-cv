"""Shared fixtures for the ai-cv test suite."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Type, TypeVar

import pytest
from pydantic import BaseModel

from ai_cv.llm import LLMClient
from ai_cv.profile_loader import load_profile

REPO_ROOT = Path(__file__).resolve().parent.parent
PROFILE_PATH = REPO_ROOT / "profiles" / "embedded_engineer.yaml"
CVS_DIR = REPO_ROOT / "cvs"


T = TypeVar("T", bound=BaseModel)


class FakeLLM(LLMClient):
    """LLM stub driven by a route function that returns canned JSON per prompt.

    The route is `(system_prompt, user_prompt) -> dict`. Every test that uses the
    scorer wires up a route that mimics what the real model would return for the
    given CV. This lets us pin down the deterministic behaviour (clamping,
    keyword grounding, domain variants, totals) without hitting Ollama.
    """

    def __init__(self, route: Callable[[str, str], dict[str, Any]]):
        super().__init__(base_url="http://fake")
        self._route = route

    def _chat(self, system_prompt: str, user_prompt: str) -> str:  # type: ignore[override]
        return json.dumps(self._route(system_prompt, user_prompt))

    def check_health(self) -> bool:  # type: ignore[override]
        return True


@pytest.fixture(scope="session")
def profile():
    return load_profile(PROFILE_PATH)


@pytest.fixture(scope="session")
def cv_text():
    def _read(name: str) -> str:
        return (CVS_DIR / name).read_text(encoding="utf-8")
    return _read


def make_fake_llm(
    extraction: dict[str, Any] | None = None,
    scoring: dict[str, Any] | None = None,
) -> FakeLLM:
    """Build a FakeLLM that returns `extraction` for the parser prompt and
    `scoring` for the rubric prompt. Routing is by system-prompt content."""

    def route(system: str, user: str) -> dict[str, Any]:
        if "expert CV/resume parser" in system:
            return extraction or {}
        return scoring or {}

    return FakeLLM(route)
