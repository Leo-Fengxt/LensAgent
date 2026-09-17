from __future__ import annotations

import json
from typing import Any

from lensagent.agent.client import ChatCompletionsClient, ChatResponse
from lensagent.config import LLMProvider, llm_config_for


class FakeResponse:
    def __init__(self, payload: dict[str, Any], status_code: int = 200):
        self.payload = payload
        self.status_code = status_code
        self.text = json.dumps(payload)

    def json(self) -> dict[str, Any]:
        return self.payload


def test_provider_configuration_defaults():
    requesty = llm_config_for(LLMProvider.REQUESTY)
    assert requesty.api_base_url == "https://router.requesty.ai/v1/chat/completions"
    assert requesty.api_key_environment == "LENSAGENT_API_KEY"
    assert requesty.primary_model == "vertex/google/gemini-3.1-pro-preview"

    openrouter = llm_config_for("openrouter")
    assert openrouter.api_base_url == "https://openrouter.ai/api/v1/chat/completions"
    assert openrouter.api_key_environment == "OPENROUTER_API_KEY"
    assert openrouter.primary_model == "google/gemini-3.1-pro-preview"

    gemini = llm_config_for("gemini", auxiliary_model="gemini-test")
    assert gemini.api_base_url == "https://generativelanguage.googleapis.com/v1beta"
    assert gemini.api_key_environment == "GEMINI_API_KEY"
    assert gemini.auxiliary_model == "gemini-test"


def test_openrouter_uses_chat_completions_api(monkeypatch, tmp_path):
    captured = {}

    def post(url, **kwargs):
        captured.update(url=url, **kwargs)
        return FakeResponse(
            {
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "result"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 12,
                    "completion_tokens": 4,
                    "cost": 0.01,
                },
            }
        )

    monkeypatch.setattr("lensagent.agent.client.requests.post", post)
    client = ChatCompletionsClient("openrouter-key", llm_config_for("openrouter"))
    trace = tmp_path / "trace.jsonl"
    client.set_trace(trace)
    result = client.chat([{"role": "user", "content": "fit this lens"}])

    assert result == "result"
    assert captured["url"] == "https://openrouter.ai/api/v1/chat/completions"
    assert captured["headers"]["Authorization"] == "Bearer openrouter-key"
    assert captured["json"]["model"] == "google/gemini-3.1-pro-preview"
    assert captured["json"]["messages"][0]["content"] == "fit this lens"
    assert client.prompt_tokens == 12
    assert client.completion_tokens == 4
    assert client.cost == 0.01
    assert "openrouter-key" not in trace.read_text()


def test_native_gemini_converts_images_and_preserves_thought_signatures(monkeypatch):
    calls = []
    responses = iter(
        [
            FakeResponse(
                {
                    "candidates": [
                        {
                            "content": {
                                "role": "model",
                                "parts": [
                                    {
                                        "text": "first result",
                                        "thoughtSignature": "signature-1",
                                    }
                                ],
                            },
                            "finishReason": "STOP",
                        }
                    ],
                    "usageMetadata": {
                        "promptTokenCount": 20,
                        "candidatesTokenCount": 5,
                        "thoughtsTokenCount": 7,
                        "totalTokenCount": 32,
                    },
                }
            ),
            FakeResponse(
                {
                    "candidates": [
                        {
                            "content": {
                                "role": "model",
                                "parts": [{"text": "second result"}],
                            },
                            "finishReason": "STOP",
                        }
                    ],
                    "usageMetadata": {
                        "promptTokenCount": 30,
                        "candidatesTokenCount": 4,
                        "totalTokenCount": 34,
                    },
                }
            ),
        ]
    )

    def post(url, **kwargs):
        calls.append({"url": url, **kwargs})
        return next(responses)

    monkeypatch.setattr("lensagent.agent.client.requests.post", post)
    client = ChatCompletionsClient("gemini-key", llm_config_for("gemini"))
    messages = [
        {"role": "system", "content": "Fit the physical model."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Observed image"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,aW1hZ2U="},
                },
            ],
        },
    ]
    first = client.chat(messages)
    assert isinstance(first, ChatResponse)
    assert first == "first result"

    second = client.chat(
        messages
        + [
            {"role": "assistant", "content": first},
            {"role": "user", "content": "Refine the parameters."},
        ]
    )
    assert second == "second result"

    first_request = calls[0]
    assert first_request["url"].endswith(
        "/models/gemini-3.1-pro-preview:generateContent"
    )
    assert first_request["headers"]["x-goog-api-key"] == "gemini-key"
    assert first_request["json"]["systemInstruction"] == {
        "parts": [{"text": "Fit the physical model."}]
    }
    assert first_request["json"]["contents"][0]["parts"][1] == {
        "inlineData": {"mimeType": "image/png", "data": "aW1hZ2U="}
    }
    assert first_request["json"]["generationConfig"]["thinkingConfig"] == {
        "thinkingLevel": "high",
        "includeThoughts": False,
    }

    model_turn = calls[1]["json"]["contents"][1]
    assert model_turn["role"] == "model"
    assert model_turn["parts"] == [
        {"text": "first result", "thoughtSignature": "signature-1"}
    ]
    assert client.prompt_tokens == 50
    assert client.completion_tokens == 16
