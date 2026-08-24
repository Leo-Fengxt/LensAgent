"""One LensAgent proposal and evaluation episode."""

from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from lensagent.agent.client import (
    CallBudgetExhausted,
    ChatCompletionsClient,
    ContextLengthExceeded,
)
from lensagent.agent.database import ProposalRecord
from lensagent.agent.prompts import (
    VISUAL_ANALYSIS_SYSTEM_PROMPT,
    build_system_prompt,
    build_user_prompt,
    format_evaluation,
)
from lensagent.data.observation import Observation
from lensagent.modeling.parameters import ParameterSpace
from lensagent.modeling.safe_evaluate import safe_evaluate
from lensagent.modeling.scoring import ScoringPolicy, residual_randomness
from lensagent.output.figures import render_evaluation_images, render_observation_images

log = logging.getLogger(__name__)

ACTION_PATTERN = re.compile(
    r"<action>\s*tool:\s*(?P<tool>\w+)\s*(?P<body>.*?)</action>", re.DOTALL
)
SOLUTION_PATTERN = re.compile(
    r"<solution_\d+>\s*(?P<json>\{.*?\})\s*</solution_\d+>", re.DOTALL
)

Evaluator = Callable[[dict[str, Any]], tuple[dict[str, Any] | None, str | None]]
EvaluationFormatter = Callable[[dict[str, Any], Observation], str]
ProposalNormalizer = Callable[[dict[str, Any]], dict[str, Any]]


@dataclass
class EpisodeStep:
    number: int
    reasoning: str
    tool: str | None
    tool_input: Any
    observation: str = ""
    elapsed_seconds: float = 0.0
    evaluated_proposals: list[dict[str, Any]] = field(default_factory=list)


class LensAgentEpisode:
    def __init__(
        self,
        client: ChatCompletionsClient,
        observation: Observation,
        parameter_space: ParameterSpace,
        scoring: ScoringPolicy,
        *,
        maximum_steps: int,
        auxiliary_client: ChatCompletionsClient | None = None,
        evaluator: Evaluator | None = None,
        system_prompt: str | None = None,
        evaluation_formatter: EvaluationFormatter = format_evaluation,
        proposal_normalizer: ProposalNormalizer | None = None,
        maximum_context_characters: int = 400_000,
        evaluation_timeout_seconds: int = 60,
    ):
        self.client = client
        self.auxiliary_client = auxiliary_client or client
        self.observation = observation
        self.parameter_space = parameter_space
        self.scoring = scoring
        self.maximum_steps = maximum_steps
        self.maximum_context_characters = maximum_context_characters
        self.evaluation_timeout_seconds = evaluation_timeout_seconds
        self.system_prompt = system_prompt or build_system_prompt(
            parameter_space, scoring
        )
        self.evaluation_formatter = evaluation_formatter
        self.proposal_normalizer = proposal_normalizer
        self.evaluator = evaluator or self._evaluate
        self.candidate_results: list[dict[str, Any]] = []

    def run(
        self, references: Sequence[ProposalRecord]
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None, list[EpisodeStep]]:
        messages = self._initial_messages(references)
        steps: list[EpisodeStep] = []
        best_proposal = None
        best_evaluation = None
        best_chi_squared = float("inf")
        self.candidate_results = []

        for number in range(1, self.maximum_steps + 1):
            messages = self._trim(messages)
            started = time.monotonic()
            try:
                response = self.client.chat(messages)
            except CallBudgetExhausted:
                raise
            except ContextLengthExceeded:
                messages = self._trim(messages, force=True)
                try:
                    response = self.client.chat(messages)
                except ContextLengthExceeded:
                    break
            except Exception as first_error:
                response = None
                error = first_error
                for attempt in range(4):
                    time.sleep(10 * 2**attempt)
                    try:
                        response = self.client.chat(messages)
                        break
                    except CallBudgetExhausted:
                        raise
                    except Exception as retry_error:
                        error = retry_error
                if response is None:
                    log.warning("model call failed after retries: %s", error)
                    continue

            reasoning, tool, tool_input = parse_action(response)
            step = EpisodeStep(
                number=number,
                reasoning=reasoning,
                tool=tool,
                tool_input=tool_input,
                elapsed_seconds=time.monotonic() - started,
            )
            messages.append({"role": "assistant", "content": response})

            if tool not in {"evaluate", "finish"}:
                messages.append(
                    {
                        "role": "user",
                        "content": "Submit an <action> using either evaluate or finish and include three solution blocks.",
                    }
                )
                steps.append(step)
                continue
            proposals = normalize_proposals(tool_input)
            if not proposals:
                messages.append(
                    {
                        "role": "user",
                        "content": "No valid proposals were parsed. Submit three JSON proposals in solution blocks.",
                    }
                )
                steps.append(step)
                continue

            observations = []
            images_by_proposal = []
            best_images: dict[str, str] = {}
            for index, raw_proposal in enumerate(proposals):
                try:
                    normalized = (
                        self.proposal_normalizer(raw_proposal)
                        if self.proposal_normalizer is not None
                        else raw_proposal
                    )
                    proposal = self.scoring.inject_fixed(normalized)
                except ValueError as exc:
                    observations.append(f"### Proposal {index + 1}: rejected: {exc}")
                    images_by_proposal.append({})
                    continue
                evaluation, error = self.evaluator(proposal)
                if evaluation is None:
                    observations.append(
                        f"### Proposal {index + 1}: evaluation failed: {error or 'unknown error'}"
                    )
                    images_by_proposal.append({})
                    continue
                evaluation["residual_randomness"] = residual_randomness(evaluation)
                record = {
                    "step": number,
                    "tool": tool,
                    "proposal_index": index,
                    "proposal": proposal,
                    "evaluation": evaluation,
                }
                self.candidate_results.append(record)
                step.evaluated_proposals.append(record)
                images = render_evaluation_images(self.observation, evaluation)
                images_by_proposal.append(images)
                chi_squared = float(
                    evaluation.get("reduced_image_chi_squared", float("inf"))
                )
                if chi_squared < best_chi_squared:
                    best_chi_squared = chi_squared
                    best_proposal = proposal
                    best_evaluation = evaluation
                    best_images = images
                observations.append(
                    f"### Proposal {index + 1}\n"
                    + self.evaluation_formatter(evaluation, self.observation)
                )

            combined = "\n\n".join(observations)
            step.observation = combined
            steps.append(step)
            if tool == "finish":
                return best_proposal, best_evaluation, steps

            descriptions = self._describe(images_by_proposal)
            if descriptions:
                combined += "\n\n## Image Analysis"
                for index, description in descriptions.items():
                    combined += f"\n\n### Proposal {index + 1}\n{description}"
            messages.append(self._evaluation_message(combined, best_images))

        return best_proposal, best_evaluation, steps

    def _evaluate(
        self, proposal: dict[str, Any]
    ) -> tuple[dict[str, Any] | None, str | None]:
        return safe_evaluate(
            proposal,
            self.observation,
            self.parameter_space,
            timeout_seconds=self.evaluation_timeout_seconds,
        )

    def _initial_messages(
        self, references: Sequence[ProposalRecord]
    ) -> list[dict[str, Any]]:
        blocks: list[dict[str, Any]] = [
            {"type": "text", "text": "## Observed Image"},
            {
                "type": "image_url",
                "image_url": {
                    "url": render_observation_images(self.observation)["observed"]
                },
            },
        ]
        for index, record in enumerate(references):
            evaluation = record.evaluation
            if "model_image" not in evaluation:
                evaluation, _ = self.evaluator(record.proposal)
            if evaluation is None:
                continue
            images = render_evaluation_images(self.observation, evaluation)
            if "comparison" in images:
                blocks.extend(
                    [
                        {"type": "text", "text": f"## Reference {index + 1} Images"},
                        {
                            "type": "image_url",
                            "image_url": {"url": images["comparison"]},
                        },
                    ]
                )
        blocks.append(
            {
                "type": "text",
                "text": build_user_prompt(
                    references, self.observation, self.parameter_space
                ),
            }
        )
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": blocks},
        ]

    def _describe(self, images: Sequence[dict[str, str]]) -> dict[int, str]:
        descriptions = {}
        for index, group in enumerate(images):
            comparison = group.get("comparison")
            if not comparison:
                continue
            try:
                result = self.auxiliary_client.chat(
                    [
                        {"role": "system", "content": VISUAL_ANALYSIS_SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {"url": comparison},
                                },
                                {
                                    "type": "text",
                                    "text": "Compare the observed arcs, model arcs, and normalized residual.",
                                },
                            ],
                        },
                    ],
                    count_toward_budget=False,
                )
            except Exception as exc:
                log.warning("image analysis failed: %s", exc)
                continue
            if result.strip():
                descriptions[index] = result.strip()
        return descriptions

    @staticmethod
    def _evaluation_message(text: str, images: dict[str, str]) -> dict[str, Any]:
        blocks: list[dict[str, Any]] = [{"type": "text", "text": text}]
        for name in ("comparison", "residual"):
            if name in images:
                blocks.append({"type": "image_url", "image_url": {"url": images[name]}})
        return {"role": "user", "content": blocks}

    def _trim(
        self, messages: list[dict[str, Any]], *, force: bool = False
    ) -> list[dict[str, Any]]:
        limit = 0 if force else self.maximum_context_characters
        if not force and sum(message_length(message) for message in messages) <= limit:
            return messages
        tail = 4 if force else 10
        return (
            messages if len(messages) <= 2 + tail else messages[:2] + messages[-tail:]
        )


def parse_action(text: str) -> tuple[str, str | None, list[dict[str, Any]] | None]:
    matches = list(ACTION_PATTERN.finditer(text))
    if not matches:
        return text.strip(), None, None
    match = matches[-1]
    proposals = []
    for solution in SOLUTION_PATTERN.finditer(match.group("body")):
        try:
            value = json.loads(solution.group("json"))
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            proposals.append(value)
    return (
        text[: match.start()].strip(),
        match.group("tool").strip().lower(),
        proposals or None,
    )


def normalize_proposals(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [
        proposal
        for proposal in value
        if isinstance(proposal, dict) and "kwargs_lens" in proposal
    ][:3]


def message_length(message: dict[str, Any]) -> int:
    content = message.get("content", "")
    if isinstance(content, str):
        return len(content)
    if isinstance(content, list):
        return sum(
            len(item.get("text", "")) if item.get("type") == "text" else 200
            for item in content
            if isinstance(item, dict)
        )
    return 0
