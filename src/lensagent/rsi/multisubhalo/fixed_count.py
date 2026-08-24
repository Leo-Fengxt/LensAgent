"""Fixed-count multisubhalo RSI."""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from lensagent.agent.client import ChatCompletionsClient
from lensagent.agent.database import ProposalDatabase, ProposalRecord
from lensagent.agent.evolution import IslandSearch
from lensagent.config import FixedCountRSIConfig
from lensagent.modeling.safe_evaluate import safe_evaluate
from lensagent.output.artifacts import write_json
from lensagent.output.figures import save_fit_figure
from lensagent.rsi.common import (
    annotate_subhalos,
    configure_tied_mass_space,
    evidence_scoring_policy,
    pull_map,
    tied_mass_nfw_space,
)
from lensagent.rsi.multisubhalo.candidates import (
    CandidateIdentification,
    identify_candidates,
)
from lensagent.rsi.multisubhalo.polish import MassPolishResult, polish_masses
from lensagent.rsi.multisubhalo.support import ExactCountSelection, search_exact_count
from lensagent.workflow.afms import FamilyState
from lensagent.workflow.prl import PRLResult


@dataclass(frozen=True)
class FixedCountRSIResult:
    system_id: str
    known_count: int
    detected: bool
    candidate_identification: CandidateIdentification
    support_selection: ExactCountSelection
    mass_polish: MassPolishResult
    record: ProposalRecord
    proposal: dict[str, Any]
    evaluation: dict[str, Any]


def _selected_state(result: PRLResult, states: tuple[FamilyState, ...]) -> FamilyState:
    return next(state for state in states if state.family.slug == result.family)


def _cutout_bounds(observation) -> dict[str, tuple[float, float]]:
    rows, columns = observation.image_data.shape
    edges = np.asarray(
        [
            (-0.5, -0.5),
            (columns - 0.5, -0.5),
            (-0.5, rows - 0.5),
            (columns - 0.5, rows - 0.5),
        ]
    )
    angles = edges @ observation.transform_pix2angle.T
    angles[:, 0] += observation.ra_at_xy_0
    angles[:, 1] += observation.dec_at_xy_0
    return {
        "center_x": (float(np.min(angles[:, 0])), float(np.max(angles[:, 0]))),
        "center_y": (float(np.min(angles[:, 1])), float(np.max(angles[:, 1]))),
    }


def _agent_candidates(polish: MassPolishResult, observation) -> list[dict[str, Any]]:
    bounds = _cutout_bounds(observation)
    return [
        {
            **copy.deepcopy(subhalo),
            "ra": float(subhalo["center_x"]),
            "dec": float(subhalo["center_y"]),
            "center_bounds": copy.deepcopy(bounds),
        }
        for subhalo in polish.selected.fitted_subhalos
    ]


def _compact_proposal(
    proposal: dict[str, Any],
    base_proposal: dict[str, Any],
    known_count: int,
    macro_names: set[str],
) -> dict[str, Any]:
    supplied = [dict(component) for component in proposal.get("kwargs_lens", [])]
    base_count = len(base_proposal["kwargs_lens"])
    if len(supplied) == base_count + known_count:
        return proposal
    if len(supplied) == known_count:
        primary_update = {}
        subhalos = supplied
    elif len(supplied) == known_count + 1:
        primary_update = supplied[0]
        subhalos = supplied[1:]
    else:
        raise ValueError(
            f"expected {known_count} subhalos with an optional primary macro object"
        )
    disallowed = set(primary_update) - macro_names
    if disallowed:
        raise ValueError(
            "primary macro object contains frozen parameters: "
            + ", ".join(sorted(disallowed))
        )
    base_lens = copy.deepcopy(base_proposal["kwargs_lens"])
    base_lens[0].update(primary_update)
    return {
        "kwargs_lens": base_lens + subhalos,
        "kwargs_lens_light": copy.deepcopy(base_proposal["kwargs_lens_light"]),
        "kwargs_source": copy.deepcopy(base_proposal["kwargs_source"]),
    }


def _validate_proposal(
    proposal: dict[str, Any],
    parameter_space,
    base_lens_count: int,
    known_count: int,
    minimum_separation: float,
) -> None:
    lens = proposal.get("kwargs_lens")
    if not isinstance(lens, list) or len(lens) != base_lens_count + known_count:
        raise ValueError("proposal does not contain the externally fixed subhalo count")
    primary_bounds = parameter_space.bounds_lens[0]
    for name, interval in primary_bounds.items():
        value = float(lens[0][name])
        if not float(interval[0]) <= value <= float(interval[1]):
            raise ValueError(f"primary macro parameter {name} is outside its range")
    positions = []
    for local_index, component in enumerate(lens[base_lens_count:]):
        bounds = parameter_space.bounds_lens[base_lens_count + local_index]
        unknown = set(component) - {"logM", "center_x", "center_y"}
        if unknown:
            raise ValueError(
                f"subhalo {local_index} contains unsupported parameters: "
                + ", ".join(sorted(unknown))
            )
        for name in ("logM", "center_x", "center_y"):
            if name not in component:
                raise ValueError(f"subhalo {local_index} is missing {name}")
            value = float(component[name])
            low, high = bounds[name]
            if not math.isfinite(value) or not float(low) <= value <= float(high):
                raise ValueError(f"subhalo {local_index} {name} is outside its range")
        positions.append((float(component["center_x"]), float(component["center_y"])))
    for left, first in enumerate(positions):
        for right, second in enumerate(positions[left + 1 :], start=left + 1):
            if (
                math.hypot(first[0] - second[0], first[1] - second[1])
                < minimum_separation
            ):
                raise ValueError(
                    f"subhalos {left} and {right} violate the minimum separation"
                )


def _system_prompt(candidates: Sequence[dict[str, Any]], parameter_space) -> str:
    macro_bounds = parameter_space.bounds_lens[0]
    macro_text = ", ".join(
        f"{name}=[{float(bounds[0]):.6g}, {float(bounds[1]):.6g}]"
        for name, bounds in macro_bounds.items()
    )
    anchor_text = "\n".join(
        f"- Subhalo {index}: ({candidate['ra']:.5f}, {candidate['dec']:.5f}) arcsec"
        for index, candidate in enumerate(candidates)
    )
    component_examples = [
        {
            "logM": float(candidate["logM"]),
            "center_x": float(candidate["ra"]),
            "center_y": float(candidate["dec"]),
        }
        for candidate in candidates
    ]
    example = json.dumps(
        {
            "kwargs_lens": [
                {
                    name: float(parameter_space.centers_lens[0][name])
                    for name in macro_bounds
                },
                *component_examples,
            ]
        }
    )
    return f"""You refine a strong-lens model with exactly {len(candidates)} NFW subhalos.

Every subhalo has three free parameters: logM, center_x, and center_y. Its mass is M200 = 10**logM Msun. Concentration, Rs, and alpha_Rs are tied to logM and must not be supplied. Keep every subhalo active and keep their centers at least 0.1 arcsec apart.

The inherited joint fit supplies these anchors:
{anchor_text}

Most centers should remain within 1.0 arcsec of their anchors. A larger displacement is allowed only when the image residuals support it. The subhalo bounds shown in the reference parameters span the image cutout.

The only smooth-lens parameters that may change are {macro_text}. All other smooth-lens, lens-light, and source parameters are fixed. In kwargs_lens, supply one object containing only these smooth-lens parameters followed by exactly {len(candidates)} subhalo objects. Do not supply kwargs_lens_light or kwargs_source.

Higher raw joint delta-BIC is better. Reduced image chi-squared is also lower-is-better here because the subhalo count is fixed. A negative delta-BIC does not authorize dropping a component.

Each action must contain three different proposals:
<action>
tool: evaluate
<solution_1>{example}</solution_1>
<solution_2>{example}</solution_2>
<solution_3>{example}</solution_3>
</action>

Use tool `finish` in the same format when the search is complete.
"""


def _format_evaluation(evaluation: dict[str, Any], _observation) -> str:
    lines = [
        f"Raw joint delta-BIC: {float(evaluation['delta_bic']):.3f}",
        (
            "Reduced image chi-squared: "
            f"{float(evaluation['reduced_image_chi_squared']):.6f}"
        ),
        f"Fit gain: {float(evaluation['fit_gain']):.3f}",
        f"Fixed parameter penalty: {float(evaluation['parameter_penalty']):.3f}",
    ]
    for index, (subhalo, mass, offset) in enumerate(
        zip(
            evaluation["subhalo_parameters"],
            evaluation["masses_msun"],
            evaluation["subhalo_seed_offsets_arcsec"],
        )
    ):
        lines.append(
            f"Subhalo {index}: center=({float(subhalo['center_x']):.5f}, "
            f"{float(subhalo['center_y']):.5f}), logM={float(subhalo['logM']):.6f}, "
            f"M200={float(mass):.3e} Msun, anchor offset={float(offset):.4f} arcsec"
        )
    lines.append(
        "Mass cap: "
        + ("satisfied" if evaluation["subhalo_mass_limit_ok"] else "violated")
    )
    lines.append(
        "Minimum separation: "
        + ("satisfied" if evaluation["subhalo_separation_ok"] else "violated")
    )
    return "\n".join(lines)


def _scalar_evaluation(evaluation: dict[str, Any]) -> dict[str, Any]:
    return {
        name: value
        for name, value in evaluation.items()
        if not isinstance(value, np.ndarray)
    }


def run_fixed_count_agent(
    polish: MassPolishResult,
    base_proposal: dict[str, Any],
    base_evaluation: dict[str, Any],
    observation,
    parameter_space,
    client: ChatCompletionsClient,
    auxiliary_client: ChatCompletionsClient,
    config: FixedCountRSIConfig,
    output_directory: str | Path,
    *,
    random_seed: int | None = None,
) -> tuple[ProposalRecord, dict[str, Any], dict[str, Any]]:
    """Run LensAgent directly from the polished fixed-count handoff."""
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    candidates = _agent_candidates(polish, observation)
    space = tied_mass_nfw_space(
        parameter_space,
        base_proposal,
        candidates,
        center_half_width=config.mass_polish.position_half_width_arcsec,
        macro_thaw={
            "theta_E": config.mass_polish.theta_e_half_width,
            "gamma": config.mass_polish.slope_half_width,
            "e1": config.mass_polish.ellipticity_half_width,
            "e2": config.mass_polish.ellipticity_half_width,
        },
        warm_base_lens=polish.selected.fitted_base_lens,
    )
    observation = observation.with_model(space.model)
    configure_tied_mass_space(space, observation)
    scoring = evidence_scoring_policy(
        space, config.kinematic_weight, chi_squared_tiebreak=False
    )
    base_lens_count = len(base_proposal["kwargs_lens"])
    known_count = polish.known_count
    macro_names = set(space.bounds_lens[0])

    def normalize(proposal):
        return _compact_proposal(proposal, base_proposal, known_count, macro_names)

    def evaluator(proposal):
        try:
            _validate_proposal(
                proposal,
                space,
                base_lens_count,
                known_count,
                config.mass_polish.minimum_separation_arcsec,
            )
        except (KeyError, TypeError, ValueError) as exc:
            return None, str(exc)
        evaluation, error = safe_evaluate(
            proposal, observation, space, timeout_seconds=60
        )
        if evaluation is None:
            return None, error
        evaluation = annotate_subhalos(
            evaluation,
            proposal,
            observation,
            base_evaluation,
            base_lens_count=base_lens_count,
            candidates=candidates,
            maximum_mass_msun=config.maximum_mass_msun,
            raw_evidence=True,
            minimum_separation=config.mass_polish.minimum_separation_arcsec,
        )
        evaluation["subhalo_seed_offsets_arcsec"] = [
            math.hypot(
                float(component["center_x"]) - float(candidate["ra"]),
                float(component["center_y"]) - float(candidate["dec"]),
            )
            for component, candidate in zip(
                evaluation["subhalo_parameters"], candidates
            )
        ]
        evaluation["is_significant"] = bool(
            evaluation["delta_bic"] > config.significant_delta_bic
            and evaluation["subhalo_mass_limit_ok"]
            and evaluation["subhalo_separation_ok"]
        )
        return evaluation, None

    initial = scoring.inject_fixed(copy.deepcopy(polish.selected.proposal))
    initial_evaluation, error = evaluator(initial)
    if initial_evaluation is None:
        raise RuntimeError(f"polished handoff evaluation failed: {error}")
    if not initial_evaluation["subhalo_mass_limit_ok"]:
        raise RuntimeError("polished handoff exceeds the subhalo mass cap")

    database = ProposalDatabase(output / "proposals.json", scoring)
    database.set_scoring(scoring)
    result_path = output / "result.json"
    if result_path.exists():
        saved = json.loads(result_path.read_text(encoding="utf-8"))
        record = next(
            item for item in database.records if item.id == saved["record_id"]
        )
        proposal = scoring.inject_fixed(record.proposal)
        evaluation, error = evaluator(proposal)
        if evaluation is None:
            raise RuntimeError(f"saved fixed-count proposal failed evaluation: {error}")
        return record, proposal, evaluation
    if database.size == 0:
        for island in range(config.budget.islands):
            database.add(
                database.create(
                    copy.deepcopy(initial),
                    copy.deepcopy(initial_evaluation),
                    island=island,
                )
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
        event_log=output / "events.jsonl",
        evaluator=evaluator,
        system_prompt=_system_prompt(candidates, space),
        evaluation_formatter=_format_evaluation,
        proposal_normalizer=normalize,
        evaluation_timeout_seconds=60,
        random_seed=random_seed,
    )
    search.run()
    eligible = [
        record
        for record in database.records
        if record.evaluation.get("subhalo_mass_limit_ok") is not False
        and record.evaluation.get("subhalo_center_bounds_ok") is not False
        and record.evaluation.get("subhalo_separation_ok") is not False
    ]
    if not eligible:
        raise RuntimeError("fixed-count RSI produced no valid proposal")
    record = max(
        eligible,
        key=lambda item: float(item.evaluation.get("delta_bic", float("-inf"))),
    )
    proposal = scoring.inject_fixed(record.proposal)
    evaluation, error = evaluator(proposal)
    if evaluation is None:
        raise RuntimeError(f"selected fixed-count proposal failed evaluation: {error}")
    save_fit_figure(
        output / "best_fit.png",
        observation,
        evaluation,
        title=f"{observation.system_id}: fixed-count RSI",
    )
    write_json(
        result_path,
        {
            "system_id": observation.system_id,
            "known_count": known_count,
            "record_id": record.id,
            "quality": record.quality,
            "proposal": proposal,
            "evaluation": _scalar_evaluation(evaluation),
        },
    )
    return record, proposal, evaluation


def _save_pull_map(
    output: Path, pull: np.ndarray, identification: CandidateIdentification
) -> None:
    np.savez_compressed(
        output / "pull_map.npz",
        pull_map=pull,
        candidates=np.asarray(
            [
                [candidate["ra"], candidate["dec"], candidate["peak_pull"]]
                for candidate in identification.ranked
            ],
            dtype=float,
        ),
    )
    figure, axis = plt.subplots(figsize=(7, 7), dpi=120)
    axis.imshow(pull, origin="lower", cmap="RdBu_r", vmin=-6, vmax=6)
    for candidate in identification.ranked:
        members = candidate.get("members") or []
        if not members:
            continue
        member = max(members, key=lambda item: item["absolute_pull"])
        axis.add_patch(
            plt.Circle(
                (member["pixel_x"], member["pixel_y"]),
                2.0,
                fill=False,
                color="black",
                linewidth=0.8,
            )
        )
        axis.text(
            member["pixel_x"] + 1.5,
            member["pixel_y"],
            str(candidate["source_rank"]),
            color="black",
            fontsize=7,
        )
    axis.set_title("Pull map")
    axis.set_axis_off()
    figure.tight_layout()
    figure.savefig(output / "pull_map.png", bbox_inches="tight")
    plt.close(figure)


def run_fixed_count_rsi(
    prl: PRLResult,
    afms_states: tuple[FamilyState, ...],
    client: ChatCompletionsClient,
    auxiliary_client: ChatCompletionsClient,
    config: FixedCountRSIConfig,
    output_directory: str | Path,
    *,
    random_seed: int | None = None,
) -> FixedCountRSIResult:
    """Run candidate preparation, joint PSO, mass polish, and LensAgent."""
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    state = _selected_state(prl, afms_states)
    initial_base = state.scoring.inject_fixed(prl.record.proposal)
    residual_pull, base_evaluation = pull_map(
        initial_base, state.observation, state.parameter_space
    )
    base_proposal = base_evaluation["proposal"]
    identification = identify_candidates(
        residual_pull,
        base_evaluation,
        state.observation,
        state.parameter_space,
        config.candidates,
    )
    _save_pull_map(output, residual_pull, identification)
    write_json(output / "candidate_identification.json", asdict(identification))
    if not identification.detected:
        raise RuntimeError(
            f"population gate failed with arc RMS {identification.arc_rms:.3f}"
        )
    known_count = config.count_for(state.observation.system_id)
    support = search_exact_count(
        identification.candidate_pool,
        known_count,
        base_proposal,
        state.observation,
        state.parameter_space,
        config.support_search,
        output / "support_search",
    )
    polish = polish_masses(
        support,
        base_proposal,
        state.observation,
        state.parameter_space,
        config.mass_polish,
        output / "mass_polish",
    )
    record, proposal, evaluation = run_fixed_count_agent(
        polish,
        base_proposal,
        base_evaluation,
        state.observation,
        state.parameter_space,
        client,
        auxiliary_client,
        config,
        output / "lensagent",
        random_seed=random_seed,
    )
    result = FixedCountRSIResult(
        system_id=state.observation.system_id,
        known_count=known_count,
        detected=True,
        candidate_identification=identification,
        support_selection=support,
        mass_polish=polish,
        record=record,
        proposal=proposal,
        evaluation=evaluation,
    )
    write_json(
        output / "result.json",
        {
            "system_id": result.system_id,
            "known_count": result.known_count,
            "detected": result.detected,
            "candidate_identification": asdict(result.candidate_identification),
            "support_selection": asdict(result.support_selection),
            "mass_polish": asdict(result.mass_polish),
            "record_id": result.record.id,
            "proposal": result.proposal,
            "evaluation": _scalar_evaluation(result.evaluation),
        },
    )
    return result
