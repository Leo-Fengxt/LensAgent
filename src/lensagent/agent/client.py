"""Multimodal model transport for LensAgent."""

from __future__ import annotations

import json
import logging
import threading
import time
from copy import deepcopy
from pathlib import Path
from typing import Any
from urllib.parse import quote

import requests

from lensagent.config import LLMConfig, LLMProvider

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


class ChatResponse(str):
    """Text response carrying native Gemini parts for the next chat turn."""

    gemini_parts: list[dict[str, Any]] | None

    def __new__(
        cls,
        value: str,
        *,
        gemini_parts: list[dict[str, Any]] | None = None,
    ) -> ChatResponse:
        instance = super().__new__(cls, value)
        instance.gemini_parts = deepcopy(gemini_parts)
        return instance


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

        request_url, headers, payload = self._request(
            messages,
            temperature=self.config.temperature
            if temperature is None
            else temperature,
            max_tokens=self.config.max_output_tokens
            if max_tokens is None
            else max_tokens,
            stop=stop,
        )

        response = None
        elapsed = 0.0
        for attempt in range(5):
            started = time.monotonic()
            try:
                response = requests.post(
                    request_url,
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
            if _is_context_error(body):
                raise ContextLengthExceeded(body)
            if (
                response.status_code in {429, 500, 502, 503, 504, 520, 522, 524}
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

        if self.config.provider is LLMProvider.GEMINI:
            content, usage = self._parse_gemini_response(data, response.status_code)
        else:
            content, usage = self._parse_chat_completions_response(
                data, response.status_code
            )

        with self._lock:
            self.prompt_tokens += int(usage.get("prompt_tokens") or 0)
            self.completion_tokens += int(usage.get("completion_tokens") or 0)
            self.cost += float(usage.get("cost") or 0.0)
        self._record(request_id, messages, content, usage, elapsed)
        return content

    def _request(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float,
        max_tokens: int,
        stop: list[str] | None,
    ) -> tuple[str, dict[str, str], dict[str, Any]]:
        if self.config.provider is LLMProvider.GEMINI:
            model = self.model.removeprefix("models/")
            if not model or any(character in model for character in "/:?#"):
                raise ValueError(f"invalid native Gemini model name: {self.model}")
            url = (
                f"{self.config.api_base_url.rstrip('/')}/models/"
                f"{quote(model, safe='._-')}:generateContent"
            )
            headers = {
                "x-goog-api-key": self.api_key,
                "Content-Type": "application/json",
            }
            payload = _gemini_payload(
                messages,
                model=model,
                temperature=temperature,
                top_p=self.config.top_p,
                max_tokens=max_tokens,
                reasoning_effort=self.config.reasoning_effort,
                stop=stop,
            )
            return url, headers, payload

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
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
        }
        return self.config.api_base_url, headers, payload

    @staticmethod
    def _parse_chat_completions_response(
        data: dict[str, Any], status_code: int
    ) -> tuple[str, dict[str, Any]]:
        if not data.get("choices"):
            message = data.get("error", {}).get("message") or str(data)[:800]
            raise ChatCompletionsError(status_code, message)

        choice = data["choices"][0]
        content = choice.get("message", {}).get("content") or ""
        if choice.get("finish_reason") == "context_length_exceeded":
            raise ContextLengthExceeded(content)
        return str(content), data.get("usage", {})

    @staticmethod
    def _parse_gemini_response(
        data: dict[str, Any], status_code: int
    ) -> tuple[ChatResponse, dict[str, Any]]:
        candidates = data.get("candidates") or []
        if not candidates:
            error = data.get("error", {}).get("message")
            feedback = data.get("promptFeedback", {}).get("blockReason")
            raise ChatCompletionsError(
                status_code, str(error or feedback or data)[:800]
            )
        candidate = candidates[0]
        parts = candidate.get("content", {}).get("parts") or []
        text = "".join(
            str(part.get("text") or "")
            for part in parts
            if not part.get("thought")
        )
        if not text:
            reason = candidate.get("finishReason") or "empty response"
            raise ChatCompletionsError(status_code, str(reason))

        metadata = data.get("usageMetadata") or {}
        response_tokens = int(metadata.get("candidatesTokenCount") or 0)
        thought_tokens = int(metadata.get("thoughtsTokenCount") or 0)
        usage = {
            "prompt_tokens": int(metadata.get("promptTokenCount") or 0),
            "completion_tokens": response_tokens + thought_tokens,
            "response_tokens": response_tokens,
            "reasoning_tokens": thought_tokens,
            "total_tokens": int(metadata.get("totalTokenCount") or 0),
            "cost": 0.0,
        }
        return ChatResponse(text, gemini_parts=parts), usage

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


def _is_context_error(body: str) -> bool:
    lowered = body.lower()
    return (
        "context" in lowered and ("length" in lowered or "window" in lowered)
    ) or ("token count" in lowered and "exceed" in lowered)


def _gemini_payload(
    messages: list[dict[str, Any]],
    *,
    model: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    reasoning_effort: str,
    stop: list[str] | None,
) -> dict[str, Any]:
    system_parts: list[dict[str, Any]] = []
    contents: list[dict[str, Any]] = []
    for message in messages:
        role = str(message.get("role") or "")
        if role == "system":
            system_parts.extend(_gemini_parts(message.get("content"), role=role))
            continue
        if role not in {"user", "assistant"}:
            raise ValueError(f"unsupported Gemini message role: {role}")
        contents.append(
            {
                "role": "model" if role == "assistant" else "user",
                "parts": _gemini_parts(message.get("content"), role=role),
            }
        )
    if not contents:
        raise ValueError("Gemini requests require at least one user message")

    generation_config: dict[str, Any] = {
        "temperature": temperature,
        "topP": top_p,
        "maxOutputTokens": max_tokens,
    }
    if model.startswith("gemini-3"):
        generation_config["thinkingConfig"] = {
            "thinkingLevel": reasoning_effort.lower(),
            "includeThoughts": False,
        }
    if stop:
        generation_config["stopSequences"] = stop
    payload: dict[str, Any] = {
        "contents": contents,
        "generationConfig": generation_config,
    }
    if system_parts:
        payload["systemInstruction"] = {"parts": system_parts}
    return payload


def _gemini_parts(content: Any, *, role: str) -> list[dict[str, Any]]:
    if role == "assistant" and isinstance(content, ChatResponse):
        if content.gemini_parts:
            return deepcopy(content.gemini_parts)
    if isinstance(content, str):
        return [{"text": content}]
    if not isinstance(content, list):
        raise ValueError(f"unsupported Gemini message content: {type(content).__name__}")

    parts = []
    for part in content:
        if not isinstance(part, dict):
            raise ValueError("Gemini message parts must be objects")
        part_type = part.get("type")
        if part_type == "text":
            parts.append({"text": str(part.get("text") or "")})
        elif part_type == "image_url" and role != "system":
            image = part.get("image_url") or {}
            parts.append(_gemini_image_part(str(image.get("url") or "")))
        else:
            raise ValueError(f"unsupported Gemini message part: {part_type}")
    return parts


def _gemini_image_part(url: str) -> dict[str, Any]:
    header, separator, data = url.partition(",")
    if not separator or not header.startswith("data:") or ";base64" not in header:
        raise ValueError("native Gemini image inputs must be base64 data URLs")
    mime_type = header[5:].split(";", 1)[0]
    if not mime_type.startswith("image/") or not data:
        raise ValueError("invalid image data URL")
    return {"inlineData": {"mimeType": mime_type, "data": data}}
