"""Precision refinement of the model selected by AFMS."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from lensagent.agent.client import ChatCompletionsClient
from lensagent.agent.database import ProposalRecord
from lensagent.agent.evolution import (
    IslandSearch,
    target_distance,
    target_ranked_records,
)
from lensagent.agent.prompts import build_system_prompt
from lensagent.config import PRLConfig
from lensagent.modeling.scoring import ScoringPolicy
from lensagent.output.artifacts import write_json
from lensagent.workflow.afms import AFMSResult, FamilyState


@dataclass(frozen=True)
class PRLResult:
    family: str
    family_label: str
    record: ProposalRecord
    handoff_source: str
    afms_reference_record: str
    completed_episodes: int


def scoring_policy(state: FamilyState, config: PRLConfig) -> ScoringPolicy:
    return ScoringPolicy(
        parameter_space=state.parameter_space,
        quality_config=config.quality,
        residual_weight=config.quality.residual_weight,
        diversity_weight=config.quality.diversity_weight,
    )


def run_prl(
    afms: AFMSResult,
    client: ChatCompletionsClient,
    auxiliary_client: ChatCompletionsClient,
    config: PRLConfig,
    output_directory: str | Path,
    *,
    random_seed: int | None = None,
) -> PRLResult:
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    state = next(item for item in afms.states if item.family.slug == afms.family)
    reference_id = afms.record.id
    reference_distance = target_distance(afms.record)
    scoring = scoring_policy(state, config)
    state.database.set_scoring(scoring)
    selected_path = output / "selected_model.json"
    if selected_path.exists():
        saved = json.loads(selected_path.read_text(encoding="utf-8"))
        selected = next(
            record
            for record in state.database.records
            if record.id == saved["record_id"]
        )
        events_path = output / "events.jsonl"
        completed = 0
        if events_path.exists():
            with events_path.open(encoding="utf-8") as handle:
                completed = sum(1 for line in handle if line.strip())
        return PRLResult(
            family=state.family.slug,
            family_label=state.family.label,
            record=selected,
            handoff_source=saved["handoff_source"],
            afms_reference_record=saved["afms_reference_record"],
            completed_episodes=completed,
        )

    base_prompt = build_system_prompt(state.parameter_space, scoring)
    system_prompt = (
        base_prompt
        + """

## Precision

The model is already well fitted. Use five or six decimal places for continuous parameters. Small parameter differences matter in this refinement.
"""
    )
    search = IslandSearch(
        state.observation,
        state.parameter_space,
        scoring,
        state.database,
        client,
        auxiliary_client,
        config.budget,
        event_log=output / "events.jsonl",
        system_prompt=system_prompt,
        random_seed=random_seed,
    )
    client.start_call_budget(config.budget.max_calls)
    best_distance = reference_distance
    stale = 0

    def on_complete(_outcome) -> None:
        nonlocal best_distance, stale
        ranked = target_ranked_records(state.database, state.observation)
        if not ranked:
            stale += 1
            return
        distance = target_distance(ranked[0])
        if distance < best_distance - config.budget.early_stop_delta:
            best_distance = distance
            stale = 0
        else:
            stale += 1

    def stop(_completed: int) -> bool:
        return (
            config.budget.early_stop_patience > 0
            and stale >= config.budget.early_stop_patience
        )

    outcomes = search.run(stop=stop, on_complete=on_complete)
    ranked = target_ranked_records(state.database, state.observation)
    selected = ranked[0] if ranked else afms.record
    if target_distance(selected) > reference_distance:
        selected = next(
            record for record in state.database.records if record.id == reference_id
        )
    source = "afms_fallback" if selected.id == reference_id else "prl_refinement"
    result = PRLResult(
        family=state.family.slug,
        family_label=state.family.label,
        record=selected,
        handoff_source=source,
        afms_reference_record=reference_id,
        completed_episodes=len(outcomes),
    )
    write_json(
        selected_path,
        {
            "family": result.family,
            "label": result.family_label,
            "record_id": result.record.id,
            "handoff_source": result.handoff_source,
            "afms_reference_record": result.afms_reference_record,
            "quality": result.record.quality,
            "evaluation": result.record.evaluation,
            "proposal": result.record.proposal,
        },
    )
    return result
