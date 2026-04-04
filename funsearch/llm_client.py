import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

log = logging.getLogger(__name__)


class OpenRouterClient:
    """Synchronous LLM chat-completions client with multimodal support.

    Sends requests directly via the ``requests`` library.  All sampling
    parameters are tunneled through to the API.  Reasoning tokens are
    configured via the ``reasoning`` extra-body field (effort: high,
    exclude: true by default -- no ``<think>`` tokens in the response).

    Supports both OpenRouter and Requesty backends via BASE_URL.
    """

    BASE_URL = "https://router.requesty.ai/v1/chat/completions"

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "vertex/google/gemini-3.1-pro-preview",
        temperature: float = 0.7,
        top_p: Optional[float] = None,
        max_tokens: int = 16384,
        timeout_s: int = 600,
        reasoning_effort: str = "high",
        reasoning_exclude: bool = True,
        app_title: str = "lensing-funsearch",
    ):
        if not api_key:
            raise ValueError("api_key is required")
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.timeout_s = timeout_s
        self.reasoning_effort = reasoning_effort
        self.reasoning_exclude = reasoning_exclude
        self.app_title = app_title

        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_cost = 0.0
        self._call_count = 0
        self.max_llm_calls: Optional[int] = None
        self._log_path: Optional[Path] = None

    def set_log_path(self, path: str) -> None:
        """Set the JSONL file for progressive request/response logging.

        The file is truncated on first call (new run overwrites previous).
        """
        self._log_path = Path(path)
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_path.write_text("")

    def _build_headers(self) -> Dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "X-Title": self.app_title,
            "HTTP-Referer": "https://lensing-funsearch.app",
        }
        return headers

    def _build_payload(
        self,
        messages: List[Dict[str, Any]],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.max_tokens,
        }
        if self.top_p is not None:
            payload["top_p"] = self.top_p
        if stop:
            payload["stop"] = stop

        if self.reasoning_effort and self.reasoning_effort.lower() != "none":
            payload["reasoning"] = {
                "effort": self.reasoning_effort,
                "exclude": self.reasoning_exclude,
            }

        return payload

    def chat(
        self,
        messages: List[Dict[str, Any]],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
        count_toward_budget: bool = True,
    ) -> str:
        """Send a chat completion request and return the content string.

        ``messages`` follows the OpenAI format.  Each message may contain
        multimodal content blocks (text + image_url with base64 data URLs).
        Set ``count_toward_budget=False`` for auxiliary calls (e.g. image
        descriptions) that should not drain the agent's exploration budget.
        """
        if count_toward_budget and self.max_llm_calls is not None and self._call_count >= self.max_llm_calls:
            raise BudgetExhausted(
                f"LLM call budget exhausted ({self.max_llm_calls} calls)")

        payload = self._build_payload(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop,
        )
        headers = self._build_headers()
        if count_toward_budget:
            self._call_count += 1
        call_id = self._call_count

        payload_size = len(json.dumps(payload, ensure_ascii=False))
        n_images = sum(1 for m in messages for c in (m.get("content", []) if isinstance(m.get("content"), list) else [])
                       if isinstance(c, dict) and c.get("type") == "image_url")
        log.debug("LLM call #%d  model=%s  payload=%.1fKB  images=%d  n_messages=%d",
                  call_id, self.model, payload_size / 1024, n_images, len(messages))

        max_retries = 5
        base_wait = 8
        for attempt in range(1, max_retries + 1):
            log.debug("LLM call #%d (attempt %d)  model=%s",
                      call_id, attempt, self.model)

            t0 = time.time()
            try:
                resp = requests.post(
                    self.BASE_URL,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout_s,
                )
            except (requests.Timeout, requests.ConnectionError) as e:
                if attempt < max_retries:
                    wait = base_wait * (2 ** (attempt - 1))
                    log.warning("LLM call #%d attempt %d failed (%s), "
                                "retrying in %ds", call_id, attempt, e, wait)
                    time.sleep(wait)
                    continue
                raise
            elapsed = time.time() - t0

            if resp.status_code == 200:
                break

            body = resp.text[:2000]
            if "context" in body.lower() and "length" in body.lower():
                raise ContextLengthExceeded(body)

            if resp.status_code in (429, 500, 502, 503, 520, 522, 524) and attempt < max_retries:
                wait = base_wait * (2 ** (attempt - 1))
                log.warning("LLM call #%d attempt %d HTTP %d, retrying in %ds",
                            call_id, attempt, resp.status_code, wait)
                time.sleep(wait)
                continue

            raise OpenRouterError(resp.status_code, body)

        try:
            data = resp.json()
        except Exception:
            raise OpenRouterError(resp.status_code,
                                  f"Non-JSON response: {resp.text[:500]}")

        if "choices" not in data or not data["choices"]:
            error_msg = data.get("error", {}).get("message", "")
            error_code = data.get("error", {}).get("code", "")
            full_resp = str(data)[:800]
            log.warning("No choices in response. code=%s msg=%s full=%s",
                        error_code, error_msg, full_resp)
            raise OpenRouterError(resp.status_code,
                                  f"No choices in response: {error_msg or full_resp}")

        choice = data["choices"][0]
        content = choice.get("message", {}).get("content", "") or ""
        finish_reason = choice.get("finish_reason", "")

        usage = data.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens

        log.debug(
            "LLM call #%d  done in %.1fs  finish=%s  "
            "prompt_tokens=%d  completion_tokens=%d  content_len=%d",
            call_id, elapsed, finish_reason,
            prompt_tokens, completion_tokens, len(content),
        )

        if finish_reason == "context_length_exceeded":
            raise ContextLengthExceeded(content)

        self._log_exchange(call_id, messages, content, usage, elapsed)
        return content

    def _log_exchange(
        self, call_id: int,
        messages: List[Dict[str, Any]],
        response: str,
        usage: Dict[str, Any],
        elapsed: float,
    ) -> None:
        if not self._log_path:
            return
        try:
            record = {
                "call_id": call_id,
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "model": self.model,
                "elapsed_s": round(elapsed, 2),
                "usage": usage,
                "messages": self._strip_images(messages),
                "response": response,
            }
            with open(self._log_path, "a") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            log.debug("Failed to log LLM exchange: %s", e)

    @staticmethod
    def _strip_images(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Replace base64 image data with a placeholder for logging."""
        out = []
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, list):
                parts = []
                for part in content:
                    if (isinstance(part, dict)
                            and part.get("type") == "image_url"):
                        url = (part.get("image_url", {}).get("url", "") or "")
                        if url.startswith("data:"):
                            parts.append({"type": "image_url",
                                          "image_url": {"url": f"[base64 {len(url)} chars]"}})
                            continue
                    parts.append(part)
                out.append({**msg, "content": parts})
            else:
                out.append(msg)
        return out


    @property
    def calls_remaining(self) -> Optional[int]:
        if self.max_llm_calls is None:
            return None
        return max(0, self.max_llm_calls - self._call_count)

    def budget_summary(self) -> str:
        budget = (f"{self.max_llm_calls} max" if self.max_llm_calls
                  else "unlimited")
        return (f"LLM calls: {self._call_count} used ({budget})  "
                f"tokens: {self.total_prompt_tokens} prompt + "
                f"{self.total_completion_tokens} completion")


class BudgetExhausted(Exception):
    pass


class ContextLengthExceeded(Exception):
    pass


class OpenRouterError(Exception):
    def __init__(self, status_code: int, body: str):
        self.status_code = status_code
        self.body = body
        super().__init__(f"HTTP {status_code}: {body[:500]}")
