"""Frozen-macro single-subhalo RSI for SDSS observations."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from lensagent.agent.client import ChatCompletionsClient
from lensagent.agent.database import ProposalDatabase, ProposalRecord
from lensagent.agent.evolution import IslandSearch
from lensagent.config import SingleSubhaloRSIConfig
from lensagent.modeling.safe_evaluate import safe_evaluate
from lensagent.output.artifacts import write_json
from lensagent.output.figures import save_fit_figure
from lensagent.rsi.common import (
    PullCandidate,
    annotate_subhalos,
    detect_blob_candidates,
    evidence_scoring_policy,
    independent_nfw_space,
    lens_centered_candidates,
    pull_map,
)
from lensagent.workflow.afms import FamilyState
from lensagent.workflow.prl import PRLResult
from lensagent.workflow.pso import (
    PSOFit,
    load_scout_results,
    run_family_pso,
    save_scout_results,
)

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class SingleCandidateResult:
    candidate: PullCandidate
    record: ProposalRecord
    proposal: dict[str, Any]
    evaluation: dict[str, Any]
    directory: str


@dataclass(frozen=True)
class SingleRSIResult:
    system_id: str
    status: str
    detected: bool
    candidate_count: int
    searched_count: int
    selected: SingleCandidateResult | None
    candidates: tuple[SingleCandidateResult, ...]


def _selected_state(result: PRLResult, states: tuple[FamilyState, ...]) -> FamilyState:
    return next(state for state in states if state.family.slug == result.family)


def _system_prompt(candidate: PullCandidate) -> str:
    bounds = candidate.center_bounds or {
        "center_x": (candidate.ra - 0.1, candidate.ra + 0.1),
        "center_y": (candidate.dec - 0.1, candidate.dec + 0.1),
    }
    example = (
        '{"kwargs_lens": [{"Rs": 0.05, "alpha_Rs": 0.01, '
        f'"center_x": {candidate.ra:.6f}, "center_y": {candidate.dec:.6f}'
        "}]}"
    )
    return f"""You refine one NFW subhalo in an otherwise fixed strong-lens model.

The NFW parameters are Rs, alpha_Rs, center_x, and center_y. Use Rs in [0.001, 0.5], alpha_Rs in [0.0001, 0.5], center_x in [{bounds["center_x"][0]}, {bounds["center_x"][1]}], and center_y in [{bounds["center_y"][0]}, {bounds["center_y"][1]}]. The smooth lens, lens light, and source light are fixed.

Higher delta-BIC is better. Reduced image chi-squared has a target of 1.0. The derived subhalo mass must not exceed the stated limit. Examine the observed, model, lens-subtracted, and normalized-residual images when moving the subhalo.

Each action must contain three different proposals. Supply only the NFW component in kwargs_lens.

<action>
tool: evaluate
<solution_1>{example}</solution_1>
<solution_2>{example}</solution_2>
<solution_3>{example}</solution_3>
</action>

Use tool `finish` in the same format when the search is complete.
"""


def _format_evaluation(evaluation: dict[str, Any], _observation) -> str:
    masses = evaluation.get("masses_msun") or []
    return "\n".join(
        [
            f"Delta-BIC: {float(evaluation.get('delta_bic', float('-inf'))):.3f}",
            (
                "Reduced image chi-squared: "
                f"{float(evaluation.get('reduced_image_chi_squared', float('inf'))):.6f}"
            ),
            "Subhalo mass: "
            + (f"{float(masses[0]):.3e} Msun" if masses else "unavailable"),
            "Mass limit: "
            + ("satisfied" if evaluation.get("subhalo_mass_limit_ok") else "violated"),
            "Center bounds: "
            + (
                "satisfied"
                if evaluation.get("subhalo_center_bounds_ok")
                else "violated"
            ),
        ]
    )


def _compact_single_proposal(
    proposal: dict[str, Any], base_proposal: dict[str, Any]
) -> dict[str, Any]:
    supplied = proposal.get("kwargs_lens", [])
    if len(supplied) != 1:
        return proposal
    return {
        "kwargs_lens": list(base_proposal["kwargs_lens"]) + [dict(supplied[0])],
        "kwargs_lens_light": list(base_proposal["kwargs_lens_light"]),
        "kwargs_source": list(base_proposal["kwargs_source"]),
    }


def _seed_database(
    database: ProposalDatabase,
    fits: tuple[PSOFit, ...],
    evaluator,
    *,
    seed_count: int,
    island_count: int,
    rng: np.random.Generator,
) -> None:
    proposals = [fit.proposal for fit in fits]
    random_count = max(seed_count - len(proposals), 5)
    proposals.extend(database.scoring.random_proposal(rng) for _ in range(random_count))
    for index, proposal in enumerate(proposals):
        evaluation, error = evaluator(proposal)
        if evaluation is None:
            log.warning("RSI seed evaluation failed: %s", error)
            continue
        database.add(database.create(proposal, evaluation, island=index % island_count))
    for island in range(island_count):
        database.trim_island(island, 20)


def _save_pull_map(
    directory: Path,
    residual_pull: np.ndarray,
    candidates: list[PullCandidate],
) -> None:
    np.savez_compressed(
        directory / "pull_map.npz",
        pull_map=residual_pull,
        candidates=np.asarray(
            [[candidate.ra, candidate.dec, candidate.pull] for candidate in candidates],
            dtype=float,
        ),
    )
    figure, axis = plt.subplots(figsize=(7, 7), dpi=120)
    axis.imshow(residual_pull, origin="lower", cmap="RdBu_r", vmin=-6, vmax=6)
    for candidate in candidates:
        axis.add_patch(
            plt.Circle(
                (candidate.pixel_x, candidate.pixel_y),
                2.0,
                fill=False,
                color="black",
                linewidth=0.8,
            )
        )
        axis.text(
            candidate.pixel_x + 1.5,
            candidate.pixel_y,
            str(candidate.rank),
            color="black",
            fontsize=7,
        )
    axis.set_title("Pull map")
    axis.set_axis_off()
    figure.tight_layout()
    figure.savefig(directory / "pull_map.png", bbox_inches="tight")
    plt.close(figure)


def run_single_rsi(
    prl: PRLResult,
    afms_states: tuple[FamilyState, ...],
    client: ChatCompletionsClient,
    auxiliary_client: ChatCompletionsClient,
    config: SingleSubhaloRSIConfig,
    output_directory: str | Path,
    *,
    random_seed: int | None = None,
) -> SingleRSIResult:
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    state = _selected_state(prl, afms_states)
    base_proposal = state.scoring.inject_fixed(prl.record.proposal)
    residual_pull, base_evaluation = pull_map(
        base_proposal, state.observation, state.parameter_space
    )
    candidates = detect_blob_candidates(
        residual_pull,
        state.observation,
        threshold=config.candidate_threshold,
    )
    search_region = None
    if config.lens_search_radius_einstein is not None:
        candidates, search_region = lens_centered_candidates(
            candidates,
            base_proposal,
            state.parameter_space,
            config.lens_search_radius_einstein,
        )
    candidates = [
        PullCandidate(**{**candidate.__dict__, "rank": index})
        for index, candidate in enumerate(candidates)
    ]
    _save_pull_map(output, residual_pull, candidates)
    write_json(
        output / "candidates.json",
        {
            "threshold": config.candidate_threshold,
            "search_region": search_region,
            "candidates": [candidate.as_dict() for candidate in candidates],
        },
    )
    if not candidates:
        result = SingleRSIResult(
            system_id=state.observation.system_id,
            status="no_candidates",
            detected=False,
            candidate_count=0,
            searched_count=0,
            selected=None,
            candidates=(),
        )
        _write_result(output, result, config.significant_delta_bic)
        return result

    rng = np.random.default_rng(random_seed)
    child_results: list[SingleCandidateResult] = []
    base_lens_count = len(base_proposal["kwargs_lens"])
    for candidate in candidates[: config.candidate_limit]:
        child_directory = output / "candidates" / f"candidate_{candidate.rank:02d}"
        child_directory.mkdir(parents=True, exist_ok=True)
        candidate_dict = candidate.as_dict()
        space = independent_nfw_space(
            state.parameter_space,
            base_proposal,
            [candidate_dict],
            freeze_smooth_model=config.freeze_smooth_model,
            center_half_width=0.1,
        )
        observation = state.observation.with_model(space.model)
        scoring = evidence_scoring_policy(
            space,
            config.kinematic_weight,
            chi_squared_tiebreak=True,
        )

        def evaluator(
            proposal,
            *,
            _space=space,
            _observation=observation,
            _candidate=candidate_dict,
            _scoring=scoring,
        ):
            full = _compact_single_proposal(proposal, base_proposal)
            full = _scoring.inject_fixed(full)
            evaluation, error = safe_evaluate(
                full, _observation, _space, timeout_seconds=60
            )
            if evaluation is None:
                return None, error
            evaluation = annotate_subhalos(
                evaluation,
                full,
                _observation,
                base_evaluation,
                base_lens_count=base_lens_count,
                candidates=[_candidate],
                maximum_mass_msun=config.maximum_mass_msun,
                raw_evidence=False,
            )
            evaluation["is_significant"] = bool(
                evaluation["delta_bic"] > config.significant_delta_bic
                and evaluation["subhalo_mass_limit_ok"]
                and evaluation["subhalo_center_bounds_ok"]
            )
            return evaluation, None

        database = ProposalDatabase(child_directory / "proposals.json", scoring)
        database.set_scoring(scoring)
        selected_path = child_directory / "selected_model.json"
        if selected_path.exists():
            saved = json.loads(selected_path.read_text(encoding="utf-8"))
            record = next(
                item for item in database.records if item.id == saved["record_id"]
            )
            proposal = scoring.inject_fixed(record.proposal)
            evaluation, error = evaluator(proposal)
            if evaluation is None:
                raise RuntimeError(
                    f"saved RSI candidate {candidate.rank} failed evaluation: {error}"
                )
            child_results.append(
                SingleCandidateResult(
                    candidate=candidate,
                    record=record,
                    proposal=proposal,
                    evaluation=evaluation,
                    directory=str(child_directory.relative_to(output)),
                )
            )
            continue

        pso_path = child_directory / "pso.json"
        if pso_path.exists():
            pso_result = load_scout_results(pso_path)[0]
        else:
            pso_result = run_family_pso(observation, space, config.pso)
            save_scout_results(pso_path, [pso_result])
        if database.size == 0:
            _seed_database(
                database,
                pso_result.fits,
                evaluator,
                seed_count=config.pso.seeds,
                island_count=config.budget.islands,
                rng=rng,
            )
        client.start_call_budget(config.budget.max_calls)
        search = IslandSearch(
            observation,
            space,
            scoring,
            database,
            client,
            auxiliary_client,
            config.budget,
            event_log=child_directory / "events.jsonl",
            evaluator=evaluator,
            system_prompt=_system_prompt(candidate),
            evaluation_formatter=_format_evaluation,
            proposal_normalizer=lambda proposal, _base=base_proposal: (
                _compact_single_proposal(proposal, _base)
            ),
            evaluation_timeout_seconds=60,
            random_seed=int(rng.integers(0, 2**32 - 1)),
        )
        search.run()
        eligible = [
            record
            for record in database.records
            if record.evaluation.get("subhalo_mass_limit_ok") is not False
            and record.evaluation.get("subhalo_center_bounds_ok") is not False
        ]
        if not eligible:
            continue
        record = max(eligible, key=lambda item: item.quality)
        proposal = scoring.inject_fixed(record.proposal)
        evaluation, error = evaluator(proposal)
        if evaluation is None:
            log.warning(
                "final RSI evaluation failed for candidate %d: %s",
                candidate.rank,
                error,
            )
            continue
        save_fit_figure(
            child_directory / "best_fit.png",
            observation,
            evaluation,
            title=f"Candidate {candidate.rank}",
        )
        child = SingleCandidateResult(
            candidate=candidate,
            record=record,
            proposal=proposal,
            evaluation=evaluation,
            directory=str(child_directory.relative_to(output)),
        )
        child_results.append(child)
        write_json(selected_path, _candidate_payload(child))

    selected = max(
        child_results,
        key=lambda item: float(item.evaluation["delta_bic"]),
        default=None,
    )
    detected = bool(selected and selected.evaluation.get("is_significant"))
    result = SingleRSIResult(
        system_id=state.observation.system_id,
        status="selected" if selected else "no_valid_candidate_fit",
        detected=detected,
        candidate_count=len(candidates),
        searched_count=min(len(candidates), config.candidate_limit),
        selected=selected,
        candidates=tuple(child_results),
    )
    _write_result(output, result, config.significant_delta_bic)
    return result


def _candidate_payload(result: SingleCandidateResult) -> dict[str, Any]:
    return {
        "candidate": result.candidate.as_dict(),
        "record_id": result.record.id,
        "quality": result.record.quality,
        "proposal": result.proposal,
        "evaluation": {
            name: value
            for name, value in result.evaluation.items()
            if not isinstance(value, np.ndarray)
        },
        "directory": result.directory,
    }


def _write_result(
    directory: Path, result: SingleRSIResult, significant_delta_bic: float
) -> None:
    payload = {
        "system_id": result.system_id,
        "status": result.status,
        "detected": result.detected,
        "candidate_count": result.candidate_count,
        "searched_count": result.searched_count,
        "significant_delta_bic": significant_delta_bic,
        "selected": _candidate_payload(result.selected) if result.selected else None,
        "candidates": [_candidate_payload(item) for item in result.candidates],
    }
    write_json(directory / "result.json", payload)
