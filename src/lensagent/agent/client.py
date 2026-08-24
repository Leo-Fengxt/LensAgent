"""OpenAI-compatible multimodal chat-completions transport."""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any

import requests

from lensagent.config import LLMConfig

log = logging.getLogger(__name__)


class CallBudgetExhausted(RuntimeError):
    pass


class ContextLengthExceeded(RuntimeError):
    pass


class ChatCompletionsError(RuntimeError):
    def __init__(self, status_code: int, body: str):
        self.status_code = status_code
        self.body = body
        super().__init__(f"HTTP {status_code}: {body[:500]}")


class ChatCompletionsClient:
    def __init__(
        self,
        api_key: str,
        config: LLMConfig,
        *,
        model: str | None = None,
        timeout_seconds: int = 600,
    ):
        if not api_key:
            raise ValueError("an API key is required")
        self.api_key = api_key
        self.config = config
        self.model = model or config.primary_model
        self.timeout_seconds = timeout_seconds
        self.maximum_counted_calls: int | None = None
        self.counted_calls = 0
        self.request_count = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.cost = 0.0
        self._lock = threading.Lock()
        self._trace_path: Path | None = None

    def set_trace(self, path: str | Path, *, append: bool = False) -> None:
        self._trace_path = Path(path)
        self._trace_path.parent.mkdir(parents=True, exist_ok=True)
        if append:
            self._trace_path.touch(exist_ok=True)
        else:
            self._trace_path.write_text("", encoding="utf-8")

    @property
    def calls_remaining(self) -> int | None:
        if self.maximum_counted_calls is None:
            return None
        return max(0, self.maximum_counted_calls - self.counted_calls)

    def start_call_budget(self, maximum_calls: int) -> None:
        if maximum_calls <= 0:
            raise ValueError("maximum_calls must be positive")
        with self._lock:
            self.maximum_counted_calls = maximum_calls
            self.counted_calls = 0

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stop: list[str] | None = None,
        count_toward_budget: bool = True,
    ) -> str:
        with self._lock:
            if (
                count_toward_budget
                and self.maximum_counted_calls is not None
                and self.counted_calls >= self.maximum_counted_calls
            ):
                raise CallBudgetExhausted(
                    f"model call budget exhausted ({self.maximum_counted_calls})"
                )
            self.request_count += 1
            request_id = self.request_count
            if count_toward_budget:
                self.counted_calls += 1

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.config.temperature
            if temperature is None
            else temperature,
            "max_tokens": self.config.max_output_tokens
            if max_tokens is None
            else max_tokens,
            "top_p": self.config.top_p,
            "reasoning": {
                "effort": self.config.reasoning_effort,
                "exclude": True,
            },
        }
        if stop:
            payload["stop"] = stop
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "X-Title": "LensAgent",
        }

        response = None
        elapsed = 0.0
        for attempt in range(5):
            started = time.monotonic()
            try:
                response = requests.post(
                    self.config.api_base_url,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout_seconds,
                )
            except (requests.Timeout, requests.ConnectionError):
                if attempt == 4:
                    raise
                time.sleep(8 * 2**attempt)
                continue
            elapsed = time.monotonic() - started
            if response.status_code == 200:
                break
            body = response.text[:2000]
            if "context" in body.lower() and "length" in body.lower():
                raise ContextLengthExceeded(body)
            if (
                response.status_code in {429, 500, 502, 503, 520, 522, 524}
                and attempt < 4
            ):
                time.sleep(8 * 2**attempt)
                continue
            raise ChatCompletionsError(response.status_code, body)

        if response is None:
            raise ChatCompletionsError(0, "request produced no response")
        try:
            data = response.json()
        except ValueError as exc:
            raise ChatCompletionsError(
                response.status_code, response.text[:500]
            ) from exc
        if not data.get("choices"):
            message = data.get("error", {}).get("message") or str(data)[:800]
            raise ChatCompletionsError(response.status_code, message)

        choice = data["choices"][0]
        content = choice.get("message", {}).get("content") or ""
        if choice.get("finish_reason") == "context_length_exceeded":
            raise ContextLengthExceeded(content)
        usage = data.get("usage", {})
        with self._lock:
            self.prompt_tokens += int(usage.get("prompt_tokens") or 0)
            self.completion_tokens += int(usage.get("completion_tokens") or 0)
            self.cost += float(usage.get("cost") or 0.0)
        self._record(request_id, messages, content, usage, elapsed)
        return content

    def _record(
        self,
        request_id: int,
        messages: list[dict[str, Any]],
        response: str,
        usage: dict[str, Any],
        elapsed: float,
    ) -> None:
        if self._trace_path is None:
            return
        record = {
            "request_id": request_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "model": self.model,
            "elapsed_seconds": round(elapsed, 2),
            "usage": usage,
            "messages": strip_image_data(messages),
            "response": response,
        }
        try:
            with self._lock, self._trace_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        except OSError as exc:
            log.warning("could not append model trace: %s", exc)


def strip_image_data(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            result.append(dict(message))
            continue
        parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "image_url":
                url = part.get("image_url", {}).get("url", "")
                if isinstance(url, str) and url.startswith("data:"):
                    parts.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": f"[image data: {len(url)} chars]"},
                        }
                    )
                    continue
            parts.append(part)
        result.append({**message, "content": parts})
    return result
