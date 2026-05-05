"""LLM client — communicates with Ollama (or compatible OpenAI-style API)."""

from __future__ import annotations

import json
from typing import Any, Type, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    """Raised when the LLM call fails after retries."""


class LLMClient:
    """Thin wrapper around the Ollama HTTP API with validation + retry."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "gemma3:27b",
        temperature: float = 0.1,
        num_ctx: int = 16384,
        timeout: float = 300.0,
        max_retries: int = 1,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.num_ctx = num_ctx
        self.timeout = timeout
        self.max_retries = max_retries

    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> dict[str, Any]:
        """Send a chat completion and return parsed JSON. Retries on parse failure."""
        last_error: str | None = None
        for attempt in range(self.max_retries + 1):
            prompt = user_prompt
            if last_error is not None:
                prompt = (
                    f"{user_prompt}\n\n"
                    f"Your previous response failed to parse with this error:\n"
                    f"{last_error}\n\n"
                    f"Respond again with valid JSON only — no prose, no markdown fences."
                )
            content = self._chat(system_prompt, prompt)
            try:
                return json.loads(content)
            except json.JSONDecodeError as e:
                last_error = f"JSONDecodeError: {e.msg} at line {e.lineno} col {e.colno}"
                if attempt >= self.max_retries:
                    raise LLMError(
                        f"LLM returned invalid JSON after {attempt + 1} attempt(s):\n"
                        f"{content[:500]}"
                    ) from e
        raise LLMError("unreachable")

    def generate_validated(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: Type[T],
    ) -> T:
        """Send a chat completion and parse into a Pydantic model. Retries on validation failure."""
        last_error: str | None = None
        for attempt in range(self.max_retries + 1):
            prompt = user_prompt
            if last_error is not None:
                prompt = (
                    f"{user_prompt}\n\n"
                    f"Your previous response failed validation with this error:\n"
                    f"{last_error}\n\n"
                    f"Fix it and respond again with valid JSON matching the schema."
                )
            content = self._chat(system_prompt, prompt)
            try:
                raw = json.loads(content)
                return schema.model_validate(raw)
            except json.JSONDecodeError as e:
                last_error = f"JSONDecodeError: {e.msg} at line {e.lineno} col {e.colno}"
            except ValidationError as e:
                last_error = f"ValidationError:\n{e}"
            if attempt >= self.max_retries:
                raise LLMError(
                    f"LLM output failed validation after {attempt + 1} attempt(s): {last_error}"
                )
        raise LLMError("unreachable")

    def generate(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        """Backwards-compatible alias for generate_json."""
        return self.generate_json(system_prompt, user_prompt)

    def _chat(self, system_prompt: str, user_prompt: str) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "format": "json",
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_ctx": self.num_ctx,
            },
        }
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(f"{self.base_url}/api/chat", json=payload)
            resp.raise_for_status()
        data = resp.json()
        return data["message"]["content"]

    def check_health(self) -> bool:
        """Check if the Ollama server is reachable."""
        try:
            with httpx.Client(timeout=5.0) as client:
                resp = client.get(f"{self.base_url}/api/tags")
                return resp.status_code == 200
        except httpx.ConnectError:
            return False

    def list_models(self) -> list[str]:
        """List available models on the Ollama server."""
        with httpx.Client(timeout=10.0) as client:
            resp = client.get(f"{self.base_url}/api/tags")
            resp.raise_for_status()
        data = resp.json()
        return [m["name"] for m in data.get("models", [])]
