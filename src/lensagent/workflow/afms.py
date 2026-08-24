"""Adaptive fit and model selection across physical lens families."""

from __future__ import annotations

import json
import logging
import math
import os
import threading
from collections.abc import Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from lensagent.agent.client import ChatCompletionsClient
from lensagent.agent.database import ProposalDatabase, ProposalRecord
from lensagent.agent.evolution import IslandSearch, eligible_records, target_distance
from lensagent.config import AFMSConfig
from lensagent.data.observation import Observation
from lensagent.modeling.families import ModelFamily
from lensagent.modeling.parameters import ParameterSpace
from lensagent.modeling.scoring import ScoringPolicy
from lensagent.output.artifacts import write_json
from lensagent.workflow.pso import scout_families, seed_database

log = logging.getLogger(__name__)


@dataclass
class FamilyState:
    family: ModelFamily
    observation: Observation
    parameter_space: ParameterSpace
    scoring: ScoringPolicy
    database: ProposalDatabase
    search: IslandSearch
    pulls: int = 0
    active: bool = True
    stale_pulls: int = 0
    early_best_quality: float | None = None
    early_best_distance: float | None = None

    def physical_records(self) -> list[ProposalRecord]:
        return [
            record
            for record in self.database.records
            if record.evaluation.get("is_physical") is True
        ]

    def best_any(self) -> ProposalRecord | None:
        return max(
            self.physical_records(), key=lambda record: record.quality, default=None
        )

    def best_valid(self) -> ProposalRecord | None:
        return max(
            eligible_records(self.database, self.observation),
            key=lambda record: (
                record.quality,
                -abs(float(record.evaluation["reduced_image_chi_squared"]) - 1.0),
            ),
            default=None,
        )


@dataclass(frozen=True)
class AFMSResult:
    family: str
    family_label: str
    record: ProposalRecord
    states: tuple[FamilyState, ...]
    completed_pulls: int


def scoring_policy(
    parameter_space: ParameterSpace, config: AFMSConfig
) -> ScoringPolicy:
    return ScoringPolicy(
        parameter_space=parameter_space,
        quality_config=config.quality,
        residual_weight=config.quality.residual_weight,
        diversity_weight=config.quality.diversity_weight,
    )


def _select_state(
    states: Sequence[FamilyState],
    total_pulls: int,
    config: AFMSConfig,
    rng: np.random.Generator,
) -> FamilyState | None:
    active = [state for state in states if state.active]
    if not active:
        return None
    untried = [state for state in active if state.pulls == 0]
    if untried:
        return untried[int(rng.integers(0, len(untried)))]

    valid = sorted(
        (state for state in active if state.best_valid() is not None),
        key=lambda state: state.best_valid().quality,
        reverse=True,
    )
    valid_rank = {
        state.family.slug: 1.0 + (len(valid) - rank) / max(len(valid), 1)
        for rank, state in enumerate(valid, start=1)
    }
    invalid = sorted(
        (
            state
            for state in active
            if state.best_valid() is None and state.best_any() is not None
        ),
        key=lambda state: state.best_any().quality,
        reverse=True,
    )
    invalid_rank = {
        state.family.slug: (len(invalid) - rank + 1) / max(len(invalid), 1)
        for rank, state in enumerate(invalid, start=1)
    }

    scores = []
    for state in active:
        exploitation = valid_rank.get(
            state.family.slug,
            config.no_valid_family_score
            + config.no_valid_quality_bonus * invalid_rank.get(state.family.slug, 0.0),
        )
        exploration = config.scheduler_exploration * math.sqrt(
            math.log(max(total_pulls, 1)) / state.pulls
        )
        scores.append(exploitation + exploration)
    probabilities = np.asarray(scores, dtype=float)
    probabilities /= probabilities.sum()
    return active[int(rng.choice(len(active), p=probabilities))]


def _check_family_stop(state: FamilyState, config: AFMSConfig) -> None:
    if config.budget.early_stop_patience <= 0:
        return
    best = state.best_valid()
    if best is None:
        return
    quality = best.quality
    distance = abs(float(best.evaluation["reduced_image_chi_squared"]) - 1.0)
    improved = (
        state.early_best_quality is None
        or quality > state.early_best_quality + config.budget.early_stop_delta
        or state.early_best_distance is None
        or distance < state.early_best_distance - config.budget.early_stop_delta
    )
    if improved:
        state.early_best_quality = quality
        state.early_best_distance = distance
        state.stale_pulls = 0
    else:
        state.stale_pulls += 1
    if state.stale_pulls >= config.budget.early_stop_patience:
        state.active = False


def _global_winner(states: Sequence[FamilyState]) -> FamilyState | None:
    candidates = [state for state in states if state.best_valid() is not None]
    return max(
        candidates,
        key=lambda state: (
            state.best_valid().quality,
            -abs(
                float(state.best_valid().evaluation["reduced_image_chi_squared"]) - 1.0
            ),
        ),
        default=None,
    )


def _write_summary(path: Path, states: Sequence[FamilyState], completed: int) -> None:
    rows = []
    for state in states:
        valid = state.best_valid()
        any_record = state.best_any()
        rows.append(
            {
                "family": state.family.slug,
                "label": state.family.label,
                "pulls": state.pulls,
                "active": state.active,
                "proposal_count": state.database.size,
                "best_valid_record": valid.id if valid else None,
                "best_valid_quality": valid.quality if valid else None,
                "best_valid_reduced_image_chi_squared": (
                    valid.evaluation.get("reduced_image_chi_squared") if valid else None
                ),
                "best_any_record": any_record.id if any_record else None,
                "best_any_quality": any_record.quality if any_record else None,
            }
        )
    write_json(path, {"completed_pulls": completed, "families": rows})


def run_afms(
    observation: Observation,
    families: Sequence[ModelFamily],
    client: ChatCompletionsClient,
    auxiliary_client: ChatCompletionsClient,
    config: AFMSConfig,
    output_directory: str | Path,
    *,
    random_seed: int | None = None,
) -> AFMSResult:
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(random_seed)
    spaces = [ParameterSpace.from_family(family) for family in families]
    scout = scout_families(
        observation,
        spaces,
        config.pso,
        workers=min(8, max(1, os.cpu_count() or 1)),
        cache_path=output / "pso_scout.json",
    )
    selected_scout = scout[: config.scout_family_limit]
    family_by_slug = {family.slug: family for family in families}

    states: list[FamilyState] = []
    for result in selected_scout:
        family = family_by_slug[result.family]
        space = ParameterSpace.from_family(family)
        family_observation = observation.with_model(space.model)
        scoring = scoring_policy(space, config)
        family_directory = output / "families" / family.slug
        database = ProposalDatabase(family_directory / "proposals.json", scoring)
        database.set_scoring(scoring)
        if database.size == 0:
            seed_database(
                database,
                family_observation,
                space,
                result.fits,
                seed_count=config.pso.seeds,
                island_count=config.budget.islands,
                rng=rng,
            )
        search = IslandSearch(
            family_observation,
            space,
            scoring,
            database,
            client,
            auxiliary_client,
            config.budget,
            event_log=family_directory / "events.jsonl",
            random_seed=int(rng.integers(0, 2**32 - 1)),
        )
        states.append(
            FamilyState(
                family=family,
                observation=family_observation,
                parameter_space=space,
                scoring=scoring,
                database=database,
                search=search,
            )
        )

    if not states:
        raise RuntimeError("PSO did not produce a usable model family")
    selected_path = output / "selected_model.json"
    if selected_path.exists():
        saved = json.loads(selected_path.read_text(encoding="utf-8"))
        selected_state = next(
            state for state in states if state.family.slug == saved["family"]
        )
        selected_record = next(
            record
            for record in selected_state.database.records
            if record.id == saved["record_id"]
        )
        completed_pulls = 0
        summary_path = output / "summary.json"
        if summary_path.exists():
            completed_pulls = int(
                json.loads(summary_path.read_text(encoding="utf-8")).get(
                    "completed_pulls", 0
                )
            )
        return AFMSResult(
            family=selected_state.family.slug,
            family_label=selected_state.family.label,
            record=selected_record,
            states=tuple(states),
            completed_pulls=completed_pulls,
        )
    client.start_call_budget(config.budget.max_calls)
    total_submitted = 0
    completed = 0
    scoreboard_ticks = 0
    winner_slug: str | None = None
    stale_ticks = 0
    stop_requested = False
    pending: dict[Future[Any], FamilyState] = {}
    state_lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=config.budget.parallel_workers) as executor:
        while len(pending) < config.budget.parallel_workers:
            state = _select_state(states, total_submitted, config, rng)
            if state is None:
                break
            total_submitted += 1
            state.pulls += 1
            pending[executor.submit(state.search.run_episode, total_submitted)] = state

        while pending:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                state = pending.pop(future)
                completed += 1
                try:
                    outcome = future.result()
                except Exception:
                    log.exception("AFMS pull failed for %s", state.family.slug)
                else:
                    state.search.write_event(outcome)
                with state_lock:
                    _check_family_stop(state, config)

            if completed // 5 > scoreboard_ticks:
                scoreboard_ticks = completed // 5
                _write_summary(output / "progress.json", states, completed)
                if all(
                    not state.active or state.pulls >= config.minimum_family_pulls
                    for state in states
                ):
                    winner = _global_winner(states)
                    current_slug = winner.family.slug if winner else None
                    if current_slug is not None and current_slug == winner_slug:
                        stale_ticks += 1
                    else:
                        winner_slug = current_slug
                        stale_ticks = 0
                    if stale_ticks >= config.global_patience:
                        stop_requested = True

            if (
                client.calls_remaining == 0
                or total_submitted >= config.budget.iterations
            ):
                stop_requested = True
            if stop_requested:
                for future in pending:
                    future.cancel()
                continue
            while len(pending) < config.budget.parallel_workers:
                state = _select_state(states, total_submitted, config, rng)
                if state is None:
                    stop_requested = True
                    break
                total_submitted += 1
                state.pulls += 1
                pending[executor.submit(state.search.run_episode, total_submitted)] = (
                    state
                )

    _write_summary(output / "summary.json", states, completed)
    candidates = [
        (target_distance(record), state.family.slug, record.id, state, record)
        for state in states
        for record in eligible_records(state.database, state.observation)
    ]
    if not candidates:
        raise RuntimeError(
            "AFMS found no physical model within the velocity-dispersion uncertainty"
        )
    _, _, _, selected_state, selected_record = min(candidates)
    result = AFMSResult(
        family=selected_state.family.slug,
        family_label=selected_state.family.label,
        record=selected_record,
        states=tuple(states),
        completed_pulls=completed,
    )
    write_json(
        selected_path,
        {
            "family": result.family,
            "label": result.family_label,
            "record_id": result.record.id,
            "quality": result.record.quality,
            "evaluation": result.record.evaluation,
            "proposal": result.record.proposal,
        },
    )
    return result
