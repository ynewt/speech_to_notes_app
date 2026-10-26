"""Minimal client for a local LLM served by Ollama (https://ollama.com).

Uses only the standard library so there is nothing magic between you and the
HTTP API: one POST to /api/chat per call.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Optional

from .transcript import log


class LLMError(RuntimeError):
    pass


@dataclass
class LLMResponse:
    content: str
    prompt_tokens: int
    output_tokens: int
    seconds: float


class OllamaClient:
    def __init__(
        self,
        model: str = "llama3.2:3b",
        host: str = "http://localhost:11434",
        num_ctx: int = 4096,
        temperature: float = 0.1,
        timeout: float = 900,
    ):
        self.model = model
        self.host = host.rstrip("/")
        self.num_ctx = num_ctx
        self.temperature = temperature
        self.timeout = timeout

    def _request(self, path: str, payload: Optional[dict] = None) -> dict:
        data = json.dumps(payload).encode() if payload is not None else None
        req = urllib.request.Request(
            self.host + path, data=data, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            raise LLMError(f"Ollama returned HTTP {e.code}: {e.read().decode(errors='replace')}") from e
        except urllib.error.URLError as e:
            raise LLMError(
                f"Could not reach Ollama at {self.host} ({e.reason}). Is it running? Try `ollama serve`."
            ) from e

    def check(self) -> None:
        """Fail early with a helpful message if the server or model is missing."""
        names = {m["name"] for m in self._request("/api/tags").get("models", [])}
        if self.model not in names and f"{self.model}:latest" not in names:
            raise LLMError(f"Model '{self.model}' is not pulled. Run: ollama pull {self.model}")

    def chat(self, system: str, user: str, schema: Optional[dict] = None) -> LLMResponse:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            # num_ctx is the context window. Ollama's default is small, and anything
            # beyond it is silently dropped from the *start* of the prompt.
            "options": {"num_ctx": self.num_ctx, "temperature": self.temperature},
        }
        if schema is not None:
            # Structured outputs: Ollama constrains generation to match this JSON schema.
            payload["format"] = schema

        data = self._request("/api/chat", payload)
        resp = LLMResponse(
            content=data["message"]["content"],
            prompt_tokens=data.get("prompt_eval_count", 0),
            output_tokens=data.get("eval_count", 0),
            seconds=data.get("total_duration", 0) / 1e9,
        )
        if resp.prompt_tokens >= self.num_ctx:
            log(
                f"  WARNING: prompt used {resp.prompt_tokens} tokens, filling the whole {self.num_ctx}-token "
                "context; input was probably truncated. Lower --chunk-tokens or raise --num-ctx."
            )
        return resp
