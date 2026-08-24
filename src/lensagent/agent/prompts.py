"""Prompts for LensAgent parameter-refinement episodes."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from lensagent.agent.database import ProposalRecord
from lensagent.data.observation import Observation
from lensagent.modeling.parameters import ParameterSpace
from lensagent.modeling.scoring import ScoringPolicy


def component_label(group: str, model_name: str) -> str:
    if group == "kwargs_lens":
        if model_name == "SHEAR":
            return "External shear"
        if model_name == "MULTIPOLE":
            return "Multipole"
        return f"Lens mass ({model_name})"
    if group == "kwargs_lens_light":
        return f"Lens light ({model_name})"
    return f"Source light ({model_name})"


def visible_parameters(
    group: str, model_name: str, parameters: dict[str, Any]
) -> dict[str, Any]:
    result = {
        name: value
        for name, value in parameters.items()
        if name not in {"ra_0", "dec_0"}
    }
    if "SHAPELETS" in model_name:
        result.pop("n_max", None)
        if group == "kwargs_source":
            result.pop("center_x", None)
            result.pop("center_y", None)
    return result


def _model_names(space: ParameterSpace, group: str) -> list[str]:
    key = {
        "kwargs_lens": "lens_model_list",
        "kwargs_lens_light": "lens_light_model_list",
        "kwargs_source": "source_light_model_list",
    }[group]
    return space.model[key]


def format_proposal(proposal: dict[str, Any], space: ParameterSpace) -> str:
    lines = []
    for group in ("kwargs_lens", "kwargs_lens_light", "kwargs_source"):
        names = _model_names(space, group)
        for index, component in enumerate(proposal.get(group, [])):
            model_name = names[index] if index < len(names) else f"component {index}"
            lines.append(
                f"  [{component_label(group, model_name)}]: "
                f"{visible_parameters(group, model_name, component)}"
            )
    return "\n".join(lines)


def build_system_prompt(space: ParameterSpace, scoring: ScoringPolicy) -> str:
    components = []
    number = 1
    bounds_by_group = space.bounds
    for group in ("kwargs_lens", "kwargs_lens_light", "kwargs_source"):
        names = _model_names(space, group)
        for index, bounds in enumerate(bounds_by_group[group]):
            model_name = names[index] if index < len(names) else "component"
            visible = visible_parameters(group, model_name, bounds)
            intervals = ", ".join(
                f"{name} [{interval[0]},{interval[1]}]"
                for name, interval in visible.items()
            )
            components.append(
                f"{number}. **{component_label(group, model_name)}**: {intervals}"
            )
            number += 1

    counts = [
        len(space.bounds_lens),
        len(space.bounds_lens_light),
        len(space.bounds_source),
    ]
    example = (
        '{"kwargs_lens": [' + ", ".join("{...}" for _ in range(counts[0])) + "], "
        '"kwargs_lens_light": [' + ", ".join("{...}" for _ in range(counts[1])) + "], "
        '"kwargs_source": [' + ", ".join("{...}" for _ in range(counts[2])) + "]}"
    )
    quality = scoring.quality_config
    formula = (
        f"Q = -{quality.image_weight} * |log(reduced image chi-squared)| "
        f"- {scoring.residual_weight} * residual randomness "
        f"- {quality.kinematic_weight} * kinematic chi-squared "
        f"- {quality.boundary_weight} * boundary penalty "
        f"+ {scoring.diversity_weight * quality.diversity_scale} * parameter-space diversity"
    )
    return f"""You optimize continuous parameters for a strong gravitational lens model.

## Model Components

{chr(10).join(components)}

Light amplitudes are solved analytically. Velocity dispersion is calculated from the mass model.

## Scoring

{formula}

Higher Q is better. Reduced image chi-squared has a target of 1.0. Values above 1 indicate missing image structure. Residual randomness measures spatial structure; lower is better. A velocity-dispersion prediction within the measurement uncertainty is adequate and should not be improved at the expense of the image fit. Solutions that fail the physicality checks are excluded from final selection.

## Images

The comparison contains: Observed | Observed - Lens | Model | Model - Lens | Normalized Residual. The second panel shows the lensed source in the data and the fourth shows the model prediction. In the residual, blue means the model is too bright, red means it is too faint, and white indicates agreement.

## Tools

Each action contains three different proposals.

### evaluate
<action>
tool: evaluate
<solution_1>
{example}
</solution_1>
<solution_2>
{example}
</solution_2>
<solution_3>
{example}
</solution_3>
</action>

### finish

Use the same format with `tool: finish`. The best evaluated proposal is submitted.

## Instructions

- Study the reference proposals, numerical scores, and images.
- Use exactly the listed components and parameter names.
- Keep every value within its stated bounds.
- Submit three meaningfully different proposals in each action.
- Use precise continuous values.
- When the residual is structured, compare the observed and modeled arcs before adjusting parameters.
- You may finish at any time.
"""


def build_user_prompt(
    records: Sequence[ProposalRecord], observation: Observation, space: ParameterSpace
) -> str:
    sections = [
        "## Observation",
        (
            f"Observed velocity dispersion: {observation.sigma_obs:.1f} +/- "
            f"{observation.sigma_obs_err:.1f} km/s"
        ),
        "The observed image is attached above.",
        "",
        "## References",
    ]
    for rank, record in enumerate(records, 1):
        evaluation = record.evaluation
        sections.extend(
            [
                "",
                f"### Reference {rank}",
                f"Quality: {record.quality:+.3f}",
                (
                    "Reduced image chi-squared: "
                    f"{evaluation.get('reduced_image_chi_squared', 'N/A')}"
                ),
                (
                    "Kinematic chi-squared: "
                    f"{evaluation.get('kinematic_chi_squared', 'N/A')}"
                ),
                f"Predicted velocity dispersion: {evaluation.get('sigma_predicted', 'N/A')} km/s",
                "Parameters:",
                format_proposal(record.proposal, space),
                f"[Reference {rank} comparison image attached above]",
            ]
        )
    sections.extend(
        [
            "",
            "## Task",
            "Propose parameters that improve on the references and explore a different part of the allowed parameter space.",
        ]
    )
    return "\n".join(sections)


def format_evaluation(evaluation: dict[str, Any], observation: Observation) -> str:
    lines = ["## Evaluation Results"]
    chi_squared = evaluation.get("reduced_image_chi_squared")
    if chi_squared is None:
        lines.append("Reduced image chi-squared: N/A")
    else:
        lines.append(
            f"Reduced image chi-squared: {chi_squared:.6f} "
            f"(target: 1.0, gap: {chi_squared - 1.0:+.6f})"
        )

    predicted = evaluation.get("sigma_predicted")
    observed = evaluation.get("sigma_observed", observation.sigma_obs)
    uncertainty = evaluation.get("sigma_observed_err", observation.sigma_obs_err)
    if predicted is None:
        lines.append(
            f"Velocity dispersion: failed (observed={observed:.1f} +/- {uncertainty:.1f})"
        )
    else:
        delta = predicted - observed
        standard_deviations = abs(delta) / uncertainty
        lines.append(
            f"Velocity dispersion: predicted={predicted:.1f}, observed={observed:.1f} "
            f"+/- {uncertainty:.1f} (delta={delta:+.1f}, {standard_deviations:.1f} sigma)"
        )
    if evaluation.get("kinematic_chi_squared") is not None:
        lines.append(
            f"Kinematic chi-squared: {evaluation['kinematic_chi_squared']:.3f}"
        )
    if evaluation.get("residual_randomness") is not None:
        lines.append(
            f"Residual randomness: {evaluation['residual_randomness']:.4f} (lower is better)"
        )
    if evaluation.get("poisson_rmse") is not None:
        status = "physical" if evaluation.get("is_physical") else "unphysical"
        lines.append(
            f"Physicality: {status}; Poisson RMSE={evaluation['poisson_rmse']:.4f}; "
            f"minimum convergence={evaluation.get('minimum_convergence', float('nan')):.3f}; "
            f"negative mass fraction={evaluation.get('negative_mass_fraction', float('nan')):.3f}"
        )
        if not evaluation.get("is_physical"):
            lines.append("This proposal will be excluded from final selection.")
    masses = evaluation.get("masses_msun")
    if masses:
        lines.append(
            "Subhalo masses: " + ", ".join(f"{float(mass):.2e} Msun" for mass in masses)
        )
    if evaluation.get("subhalo_mass_limit_ok") is False:
        lines.append(
            "A subhalo exceeds the allowed mass and this proposal is excluded."
        )
    if evaluation.get("subhalo_center_bounds_enforced") and not evaluation.get(
        "subhalo_center_bounds_ok", False
    ):
        lines.append(
            "A subhalo lies outside its allowed center bounds and this proposal is excluded."
        )
    return "\n".join(lines)


VISUAL_ANALYSIS_SYSTEM_PROMPT = """Analyze a five-panel strong-lens comparison: Observed, Observed - Lens, Model, Model - Lens, and Normalized Residual. Describe the lensed arc geometry in the second panel, compare it with the model arcs in the fourth panel, and identify structured residuals. Blue residuals mean the model is too bright and red residuals mean it is too faint. Be concrete and brief."""
