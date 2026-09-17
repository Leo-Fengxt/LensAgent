"""Island population search driven by LensAgent episodes."""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from lensagent.agent.client import CallBudgetExhausted, ChatCompletionsClient
from lensagent.agent.database import ProposalDatabase, ProposalRecord
from lensagent.agent.episode import (
    EvaluationFormatter,
    LensAgentEpisode,
    ProposalNormalizer,
)
from lensagent.config import AgentBudget
from lensagent.data.observation import Observation
from lensagent.modeling.parameters import ParameterSpace
from lensagent.modeling.scoring import ScoringPolicy, admission_reason
from lensagent.output.artifacts import NumpyEncoder

log = logging.getLogger(__name__)

Evaluator = Callable[[dict[str, Any]], tuple[dict[str, Any] | None, str | None]]


@dataclass(frozen=True)
class EpisodeOutcome:
    iteration: int
    island: int
    admitted: tuple[str, ...]
    rejected: tuple[dict[str, Any], ...]
    elapsed_seconds: float
    exhausted_budget: bool = False
    error: str | None = None


class IslandSearch:
    def __init__(
        self,
        observation: Observation,
        parameter_space: ParameterSpace,
        scoring: ScoringPolicy,
        database: ProposalDatabase,
        client: ChatCompletionsClient,
        auxiliary_client: ChatCompletionsClient,
        budget: AgentBudget,
        *,
        event_log: str | Path,
        evaluator: Evaluator | None = None,
        system_prompt: str | None = None,
        evaluation_formatter: EvaluationFormatter | None = None,
        proposal_normalizer: ProposalNormalizer | None = None,
        evaluation_timeout_seconds: int = 60,
        island_capacity: int = 30,
        random_seed: int | None = None,
    ):
        self.observation = observation
        self.parameter_space = parameter_space
        self.scoring = scoring
        self.database = database
        self.client = client
        self.auxiliary_client = auxiliary_client
        self.budget = budget
        self.event_log = Path(event_log)
        self.evaluator = evaluator
        self.system_prompt = system_prompt
        self.evaluation_formatter = evaluation_formatter
        self.proposal_normalizer = proposal_normalizer
        self.evaluation_timeout_seconds = evaluation_timeout_seconds
        self.island_capacity = island_capacity
        self.rng = np.random.default_rng(random_seed)
        self._database_lock = threading.Lock()
        self._event_lock = threading.Lock()
        if observation.hst:
            from lensagent.agent.refinement import RefinementPolicy

            self.database.refinement = RefinementPolicy(observation, scoring)

    def run_episode(self, iteration: int) -> EpisodeOutcome:
        started = time.monotonic()
        island = int(self.rng.integers(0, self.budget.islands))
        with self._database_lock:
            references = self.database.sample(
                self.budget.context_entries, self.rng, island=island
            )
        episode_kwargs: dict[str, Any] = {
            "maximum_steps": self.budget.inner_steps,
            "auxiliary_client": self.auxiliary_client,
            "evaluator": self.evaluator,
            "system_prompt": self.system_prompt,
            "proposal_normalizer": self.proposal_normalizer,
            "evaluation_timeout_seconds": self.evaluation_timeout_seconds,
        }
        if self.evaluation_formatter is not None:
            episode_kwargs["evaluation_formatter"] = self.evaluation_formatter
        episode = LensAgentEpisode(
            self.client,
            self.observation,
            self.parameter_space,
            self.scoring,
            **episode_kwargs,
        )
        exhausted = False
        try:
            episode.run(references)
        except CallBudgetExhausted:
            exhausted = True
            if not self.observation.hst:
                return EpisodeOutcome(iteration, island, (), (), time.monotonic() - started,
                                      exhausted_budget=True)
        except Exception as exc:
            log.exception("LensAgent episode %d failed", iteration)
            return EpisodeOutcome(
                iteration=iteration,
                island=island,
                admitted=(),
                rejected=(),
                elapsed_seconds=time.monotonic() - started,
                error=f"{type(exc).__name__}: {exc}",
            )

        admitted: list[str] = []
        rejected: list[dict[str, Any]] = []
        with self._database_lock:
            for candidate in episode.candidate_results:
                proposal = candidate["proposal"]
                evaluation = candidate["evaluation"]
                if evaluation.get("subhalo_mass_limit_ok") is False:
                    rejected.append(
                        {
                            "proposal_index": candidate["proposal_index"],
                            "reason": "mass_limit",
                        }
                    )
                    continue
                if (self.database.refinement is None
                        and self.scoring.is_duplicate(proposal, self.database.proposals())):
                    rejected.append(
                        {
                            "proposal_index": candidate["proposal_index"],
                            "reason": "duplicate",
                        }
                    )
                    continue

                record = self.database.create(proposal, evaluation, island=island)
                island_records = self.database.records_in_island(island)
                reason = admission_reason(
                    record.quality,
                    record.diversity,
                    [item.quality for item in island_records],
                    [item.diversity for item in island_records],
                )
                if self.database.refinement is not None:
                    decision = self.database.refinement.decide(record, self.database.records, reason)
                    if decision["outcome"] != "admitted":
                        rejected.append({"proposal_index": candidate["proposal_index"], **decision})
                        continue
                    if decision.get("replaced_id"):
                        self.database.remove(decision["replaced_id"])
                    reason = decision["admission_reason"]
                if reason == "dominated":
                    rejected.append(
                        {
                            "proposal_index": candidate["proposal_index"],
                            "reason": reason,
                            "quality": record.quality,
                            "diversity": record.diversity,
                        }
                    )
                    continue
                self.database.add(record)
                self.database.trim_island(island, self.island_capacity)
                admitted.append(record.id)

        return EpisodeOutcome(
            iteration=iteration,
            island=island,
            admitted=tuple(admitted),
            rejected=tuple(rejected),
            elapsed_seconds=time.monotonic() - started,
            exhausted_budget=exhausted,
        )

    def run(
        self,
        *,
        iterations: int | None = None,
        stop: Callable[[int], bool] | None = None,
        on_complete: Callable[[EpisodeOutcome], None] | None = None,
    ) -> list[EpisodeOutcome]:
        maximum_iterations = (
            self.budget.iterations if iterations is None else iterations
        )
        self.client.maximum_counted_calls = self.budget.max_calls
        outcomes: list[EpisodeOutcome] = []
        next_iteration = 1
        pending: dict[Future[EpisodeOutcome], int] = {}

        with ThreadPoolExecutor(max_workers=self.budget.parallel_workers) as executor:
            while (
                next_iteration <= maximum_iterations
                and len(pending) < self.budget.parallel_workers
            ):
                pending[executor.submit(self.run_episode, next_iteration)] = (
                    next_iteration
                )
                next_iteration += 1

            while pending:
                completed, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in completed:
                    iteration = pending.pop(future)
                    try:
                        outcome = future.result()
                    except Exception as exc:
                        outcome = EpisodeOutcome(
                            iteration=iteration,
                            island=-1,
                            admitted=(),
                            rejected=(),
                            elapsed_seconds=0.0,
                            error=f"{type(exc).__name__}: {exc}",
                        )
                    outcomes.append(outcome)
                    self.write_event(outcome)
                    if on_complete is not None:
                        on_complete(outcome)

                calls_remaining = self.client.calls_remaining
                should_stop = (
                    any(
                        outcome.exhausted_budget
                        for outcome in outcomes[-len(completed) :]
                    )
                    or calls_remaining == 0
                    or (stop is not None and stop(len(outcomes)))
                )
                if should_stop:
                    for future in pending:
                        future.cancel()
                    continue
                while (
                    next_iteration <= maximum_iterations
                    and len(pending) < self.budget.parallel_workers
                ):
                    pending[executor.submit(self.run_episode, next_iteration)] = (
                        next_iteration
                    )
                    next_iteration += 1
        return outcomes

    def write_event(self, outcome: EpisodeOutcome) -> None:
        record = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            **outcome.__dict__,
        }
        self.event_log.parent.mkdir(parents=True, exist_ok=True)
        with self._event_lock, self.event_log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, cls=NumpyEncoder) + "\n")


def eligible_records(
    database: ProposalDatabase, observation: Observation
) -> list[ProposalRecord]:
    eligible = []
    for record in database.records:
        evaluation = record.evaluation
        predicted = evaluation.get("sigma_predicted")
        if predicted is None or evaluation.get("is_physical") is not True:
            continue
        if abs(float(predicted) - observation.sigma_obs) > observation.sigma_obs_err:
            continue
        chi_squared = evaluation.get("reduced_image_chi_squared")
        try:
            value = float(chi_squared)
        except (TypeError, ValueError):
            continue
        if np.isfinite(value) and value > 0:
            eligible.append(record)
    return eligible


def target_distance(record: ProposalRecord) -> float:
    value = float(record.evaluation["reduced_image_chi_squared"])
    return abs(float(np.log(value)))


def target_ranked_records(
    database: ProposalDatabase, observation: Observation
) -> list[ProposalRecord]:
    return sorted(
        eligible_records(database, observation),
        key=lambda item: (target_distance(item), item.id),
    )
