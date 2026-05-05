"""Tests for the LLMClient validation/retry behaviour."""

from __future__ import annotations

import json

import pytest
from pydantic import BaseModel

from ai_cv.llm import LLMClient, LLMError


class _Schema(BaseModel):
    score: int
    name: str


class _ScriptedLLM(LLMClient):
    """LLM stub that returns a predetermined sequence of raw chat responses."""

    def __init__(self, responses: list[str], max_retries: int = 1):
        super().__init__(base_url="http://fake", max_retries=max_retries)
        self._responses = list(responses)
        self.calls: list[str] = []

    def _chat(self, system_prompt: str, user_prompt: str) -> str:  # type: ignore[override]
        self.calls.append(user_prompt)
        return self._responses.pop(0)


def test_validates_and_returns_model():
    llm = _ScriptedLLM([json.dumps({"score": 7, "name": "x"})])
    out = llm.generate_validated("sys", "user", _Schema)
    assert out.score == 7 and out.name == "x"


def test_retries_on_invalid_json_then_succeeds():
    llm = _ScriptedLLM(
        ["this is not json", json.dumps({"score": 1, "name": "ok"})],
        max_retries=1,
    )
    out = llm.generate_validated("sys", "user", _Schema)
    assert out.name == "ok"
    assert len(llm.calls) == 2
    # The retry prompt must include the prior error.
    assert "previous response failed" in llm.calls[1]


def test_retries_on_validation_error_then_succeeds():
    llm = _ScriptedLLM(
        [json.dumps({"score": "not-an-int", "name": "x"}), json.dumps({"score": 3, "name": "y"})],
        max_retries=1,
    )
    out = llm.generate_validated("sys", "user", _Schema)
    assert out.score == 3


def test_raises_after_exhausting_retries():
    llm = _ScriptedLLM(["bad", "still bad"], max_retries=1)
    with pytest.raises(LLMError):
        llm.generate_validated("sys", "user", _Schema)


def test_generate_json_returns_dict():
    llm = _ScriptedLLM([json.dumps({"a": 1})])
    out = llm.generate_json("sys", "user")
    assert out == {"a": 1}
