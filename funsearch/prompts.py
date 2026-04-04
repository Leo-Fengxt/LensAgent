"""System prompt, user prompt templates, and tool descriptions."""

from typing import Any, Dict, List, Optional


def _component_prompt_label(key: str, model_name: str) -> str:
    if key == "kwargs_lens":
        if model_name == "SHEAR":
            return "External shear"
        if model_name == "MULTIPOLE":
            return "Multipole"
        return f"Lens mass ({model_name})"
    if key == "kwargs_lens_light":
        return f"Lens light ({model_name})"
    return f"Source light ({model_name})"


def _visible_bounds(key: str, model_name: str, bounds: Dict[str, Any]) -> Dict[str, Any]:
    visible = dict(bounds)
    if key == "kwargs_source" and "SHAPELETS" in model_name:
        visible.pop("center_x", None)
        visible.pop("center_y", None)
    return visible


def _sanitize_component_display(
    key: str,
    model_name: str,
    component: Dict[str, Any],
) -> Dict[str, Any]:
    filtered = {k: v for k, v in component.items() if k not in ("ra_0", "dec_0")}
    if "SHAPELETS" in model_name:
        filtered.pop("n_max", None)
        if key == "kwargs_source":
            filtered.pop("center_x", None)
            filtered.pop("center_y", None)
    return filtered


def _format_weight(value: float) -> str:
    """Format prompt weights without losing small non-integer values."""
    return repr(float(value))


def _normalize_available_tools(
    available_tools: Optional[List[str]],
) -> List[str]:
    """Normalize prompt tool selection while keeping finish mandatory."""
    if available_tools is None:
        return ["evaluate", "finish"]

    normalized: List[str] = []
    for tool in available_tools:
        name = str(tool).strip().lower()
        if name in ("evaluate", "finish") and name not in normalized:
            normalized.append(name)

    if "finish" not in normalized:
        raise ValueError("finish must remain available in the prompt tool schema")
    return normalized


def _tool_section_text(example_json: str, available_tools: List[str]) -> str:
    lines = [
        "## Tools",
        "",
        "Each tool call contains 3 DIFFERENT proposals inside <solution_1>, <solution_2>, <solution_3> tags.",
        "",
    ]

    if "evaluate" in available_tools:
        lines.extend([
            "### evaluate",
            "<action>",
            "tool: evaluate",
            "<solution_1>",
            example_json,
            "</solution_1>",
            "<solution_2>",
            example_json,
            "</solution_2>",
            "<solution_3>",
            example_json,
            "</solution_3>",
            "</action>",
            "",
        ])

    if "finish" in available_tools:
        if "evaluate" in available_tools:
            lines.extend([
                "### finish",
                "Same format as evaluate. The best of the 3 is submitted.",
            ])
        else:
            lines.extend([
                "### finish",
                "<action>",
                "tool: finish",
                "<solution_1>",
                example_json,
                "</solution_1>",
                "<solution_2>",
                example_json,
                "</solution_2>",
                "<solution_3>",
                example_json,
                "</solution_3>",
                "</action>",
                "",
                "Submit 3 final candidates. The best of the 3 is submitted.",
            ])

    return "\n".join(lines)


def _active_scoring_weights() -> Dict[str, float]:
    """Return the currently active scoring weights used by this run."""
    from . import scoring as _S

    if _S.QUALITY_FN is _S.compute_quality_pass15:
        return {
            "alpha": _S.ALPHA_P15,
            "beta": _S.BETA_P15,
            "gamma": _S.GAMMA_P15,
            "delta": _S.DELTA_P15,
            "epsilon": _S.EPSILON,
        }
    return {
        "alpha": _S.ALPHA,
        "beta": _S.BETA,
        "gamma": _S.GAMMA,
        "delta": _S.DELTA,
        "epsilon": _S.EPSILON,
    }


def _scoring_formula_text(*, blind_mode: bool, penalty_mode: str, physicality_mode: str) -> str:
    """Render the live scoring formula shown in the prompt."""
    weights = _active_scoring_weights()
    alpha = _format_weight(weights["alpha"])
    beta = _format_weight(weights["beta"])
    gamma = _format_weight(weights["gamma"])
    delta = _format_weight(weights["delta"])
    epsilon = _format_weight(weights["epsilon"])
    chi2_term = "|log(chi2_image)|" if penalty_mode == "log" else "|chi2_image - 1|"

    if blind_mode:
        return f"Q = -{alpha} * {chi2_term} - {gamma} * boundary_penalty"

    formula = (
        f"Q = -{alpha} * {chi2_term} - {delta} * residual_randomness "
        f"- {beta} * chi2_kinematic - {gamma} * boundary_penalty"
    )
    if physicality_mode == "active":
        formula += f" - {epsilon} * max(0, rmse_poisson - 0.05)"
    return formula


def build_system_prompt(
    *,
    available_tools: Optional[List[str]] = None,
    image_feedback_enabled: bool = True,
) -> str:
    """Generate the system prompt dynamically from the active model combo."""
    from .scoring import PRIOR_BOUNDS, FIXED_PARAMS, ACTIVE_COMBO, MODEL_COMBOS, BLIND_MODE, KIN_SOFT, PHYSICALITY_MODE, CHI2_PENALTY, SUBTRACTED_CHI2

    prompt_tools = _normalize_available_tools(available_tools)

    comp_lines = []
    idx = 1

    for key in ("kwargs_lens", "kwargs_lens_light", "kwargs_source"):
        bounds_list = PRIOR_BOUNDS[key]
        fixed_list = FIXED_PARAMS.get(key, [{}] * len(bounds_list))
        combo = MODEL_COMBOS[ACTIVE_COMBO]
        if key == "kwargs_lens":
            model_list = combo["kwargs_model"]["lens_model_list"]
        elif key == "kwargs_lens_light":
            model_list = combo["kwargs_model"]["lens_light_model_list"]
        else:
            model_list = combo["kwargs_model"]["source_light_model_list"]

        for ci, bounds in enumerate(bounds_list):
            model_name = model_list[ci] if ci < len(model_list) else "?"
            fixed = fixed_list[ci] if ci < len(fixed_list) else {}
            label = _component_prompt_label(key, model_name)

            param_strs = []
            for pname, (lo, hi) in _visible_bounds(key, model_name, bounds).items():
                param_strs.append(f"{pname} [{lo},{hi}]")
            params_text = ", ".join(param_strs)

            comp_lines.append(f"{idx}. **{label}**: {params_text}")
            idx += 1

    n_lens = len(PRIOR_BOUNDS["kwargs_lens"])
    n_ll = len(PRIOR_BOUNDS["kwargs_lens_light"])
    n_src = len(PRIOR_BOUNDS["kwargs_source"])
    lens_dots = ", ".join(["{...}"] * n_lens)
    ll_dots = ", ".join(["{...}"] * n_ll)
    src_dots = ", ".join(["{...}"] * n_src)
    example_json = (f'{{"kwargs_lens": [{lens_dots}], '
                    f'"kwargs_lens_light": [{ll_dots}], '
                    f'"kwargs_source": [{src_dots}]}}')

    ll_models = MODEL_COMBOS[ACTIVE_COMBO]["kwargs_model"]["lens_light_model_list"]
    src_models = MODEL_COMBOS[ACTIVE_COMBO]["kwargs_model"]["source_light_model_list"]

    ll_desc_parts = []
    for m in ll_models:
        if m == "SERSIC_ELLIPSE":
            ll_desc_parts.append("Sersic")
        elif m == "HERNQUIST":
            ll_desc_parts.append("Hernquist")
        elif m == "SERSIC":
            ll_desc_parts.append("Sersic (spherical)")
        elif "SHAPELETS" in m:
            ll_desc_parts.append("shapelets (amplitudes solved automatically)")
    ll_desc = " + ".join(ll_desc_parts)

    src_desc_parts = []
    for m in src_models:
        if m == "SERSIC_ELLIPSE":
            src_desc_parts.append("Sersic")
        elif "SHAPELETS" in m:
            src_desc_parts.append("shapelets (beta free, centers tied to the source Sersic, amplitudes solved automatically)")
        elif "SLIT_STARLETS" in m:
            src_desc_parts.append("starlet wavelets")
    src_desc = " + ".join(src_desc_parts)

    return f"""\
You are a parameter optimization agent. Your task is to propose continuous parameters that minimize the mismatch between a simulated gravitational lens image and a real observation.

## Model Components

The model has {idx-1} components. You set parameters for each:

{chr(10).join(comp_lines)}

Light amplitudes are solved automatically.{
    "" if BLIND_MODE else " Velocity dispersion (sigma_v) is calculated from the mass model."
}

## Scoring

{_scoring_formula_text(
    blind_mode=BLIND_MODE,
    penalty_mode=CHI2_PENALTY,
    physicality_mode=PHYSICALITY_MODE,
)}

Higher Q is better.

- **chi2_image_reduced**: target is 1.0 (noise-limited). chi2=1.0 is the best possible fit. chi2 > 1 means the model is missing features.{
    chr(10) + "- Once chi2 is already within 1% of 1.0, the scoring strongly favors the solution closest to 1.0. In that narrow band, prioritize image fit over secondary tie-breakers."
    + chr(10) + "- Inside an ultra-tiny inner band around 1.0, this preference becomes even stronger, so very small chi2 improvements near 1.0 still matter."
}{
    "" if BLIND_MODE else chr(10) +
    "- **residual_randomness** (0-1): measures spatial structure in residuals. Lower = more random = better. Higher = structured patterns = model is missing features." + chr(10) +
    "- **chi2_kinematic**: velocity dispersion match. " + (
        "Predictions within the measurement uncertainty get ZERO penalty — they are all equally good. Penalty only applies outside the error bar. Do NOT over-optimize sigma."
        if KIN_SOFT else
        "Any prediction within the measurement uncertainty is fine — do NOT over-optimize sigma at the expense of chi2_image."
    ) + (
        chr(10) + "- **rmse_poisson**: measures how well the mass model satisfies the Poisson equation (0.5 * laplacian(psi) = kappa). "
        "Values below 0.05 are physical and incur no penalty. Values above 0.05 are penalized — the mass distribution is unphysical. "
        "Avoid extreme or discontinuous mass parameters."
        if PHYSICALITY_MODE == "active" else ""
    )
}

## Images

{
    (
        "5-panel comparison: GT (Observed) | GT-Lens (real arcs in data) | Render (Full Model) | Render_Sub (model's predicted arcs) | Residual_Sub (source-only residual, blue=bright, red=faint). "
        "GT-Lens = observed data minus model lens light (real arcs). Render_Sub = full model minus model lens light (predicted arcs). Compare them to see what the model misses. "
        "Blue = model too bright. Red = model too faint. White = good fit."
        if SUBTRACTED_CHI2 else
        "6-panel comparison: GT (Observed) | GT-Lens (real arcs in data) | Render (Full Model) | Residual (blue=bright, red=faint) | Render_Sub (model's predicted arcs) | Residual_Sub (Source Only). "
        "GT-Lens = observed data minus model lens light (real arcs). Render_Sub = full model minus model lens light (predicted arcs). Compare them to see what the model misses. "
        "Blue = model too bright. Red = model too faint. White = good fit."
    )
    if image_feedback_enabled else
    "This ablation disables image feedback. No observation, reference, residual, or comparison images will be attached. "
    "Rely on the numerical evaluation text and the reference parameter summaries only."
}

{_tool_section_text(example_json, prompt_tools)}

## What is Strong Gravitational Lensing?

There are TWO galaxies along the line of sight:
- **Lens galaxy** (foreground): a massive galaxy whose gravity bends light. It emits its own light (modeled by lens light component).
- **Source galaxy** (background): a more distant galaxy whose light passes through the lens galaxy's gravitational field and gets distorted into arcs, rings, or multiple images (modeled by source light component, transformed by the mass model).

The observed image = lens galaxy light + gravitationally distorted source galaxy light, both convolved with the telescope PSF.

The **mass model** (EPL + shear + multipole) determines HOW the source light is distorted.

The **lens light** uses {ll_desc}. The **source light** uses {src_desc}.

All light amplitudes are solved analytically. theta_E (Einstein radius) sets the angular scale where lensing is strongest. Structured residuals mean the model is missing real features.

## Instructions

- Study the reference proposals and their scores.
- Any parameter value within the bounds is valid.
- Explore the FULL range of each parameter. Extreme values can be optimal.
{
    "- If residuals show structure: read the residual analysis and adjust accordingly."
    if image_feedback_enabled else
    "- In this ablation, use the numeric diagnostics instead of image feedback when deciding how to refine the proposals."
}
- Each call must contain 3 DIFFERENT proposals -- vary them meaningfully.
- Use precise continuous values (e.g., 0.1847 not 0.2, 3.716 not 4.0).
- You may call finish at any time.
"""


TASK_INSTRUCTION = """\
## Your Task

Propose parameters that improve on the references.
Try parameter combinations that are DIFFERENT from the references.
Trust the chi2 score over physical intuition.
"""


def _format_proposal_text(proposal: Dict[str, Any], kwargs_model: Optional[Dict] = None) -> str:
    """Format a proposal dict into readable text."""
    if kwargs_model is None:
        from .scoring import ACTIVE_COMBO, MODEL_COMBOS
        kwargs_model = MODEL_COMBOS[ACTIVE_COMBO]["kwargs_model"]
    component_models = {
        "kwargs_lens": list(kwargs_model.get("lens_model_list", [])),
        "kwargs_lens_light": list(kwargs_model.get("lens_light_model_list", [])),
        "kwargs_source": list(kwargs_model.get("source_light_model_list", [])),
    }
    lines = []
    for key in ("kwargs_lens", "kwargs_lens_light", "kwargs_source"):
        components = proposal.get(key, [])
        models = component_models.get(key, [])
        for i, comp in enumerate(components):
            model_name = models[i] if i < len(models) else f"component {i}"
            filtered = _sanitize_component_display(key, model_name, comp)
            label = _component_prompt_label(key, model_name)
            lines.append(f"  [{label}]: {filtered}")
    return "\n".join(lines)


def _format_entry_text(entry, rank: int, kwargs_model: Optional[Dict] = None) -> str:
    """Format a database entry for the user prompt."""
    from .scoring import BLIND_MODE
    er = entry.eval_results
    lines = [
        f"### Reference {rank}",
        f"Quality: {entry.quality:+.3f}",
        f"Image chi2 reduced: {er.get('image_chi2_reduced', 'N/A')}",
    ]
    if not BLIND_MODE:
        lines.append(f"Kinematic chi2: {er.get('kin_chi2', 'N/A')}")
        lines.append(f"Predicted sigma_v: {er.get('sigma_predicted', 'N/A')} km/s")
    lines += [
        "Parameters:",
        _format_proposal_text(entry.proposal, kwargs_model=kwargs_model),
    ]
    return "\n".join(lines)


def build_user_prompt(
    context_entries: List,
    obs_sigma: float = 0.0,
    obs_sigma_err: float = 0.0,
    z_lens: float = 0.0,
    z_source: float = 0.0,
    kwargs_model: Optional[Dict] = None,
    image_feedback_enabled: bool = True,
) -> str:
    """Build the initial user prompt text (images are injected separately)."""
    from .scoring import BLIND_MODE
    parts = ["## Observation"]
    if not BLIND_MODE:
        parts.append(f"Observed velocity dispersion: {obs_sigma:.1f} +/- {obs_sigma_err:.1f} km/s")
    parts += [
        "",
        (
            "The observed image is attached above."
            if image_feedback_enabled else
            "This ablation does not attach the observed image. Use the reference summaries and numeric evaluation feedback."
        ),
        "",
        "## References",
        "",
    ]
    for i, entry in enumerate(context_entries, 1):
        parts.append(_format_entry_text(entry, i, kwargs_model=kwargs_model))
        parts.append("")
        parts.append(
            (
                f"[Images for reference {i} attached above]"
                if image_feedback_enabled else
                f"[Reference {i} images are disabled in this ablation]"
            )
        )
        parts.append("")

    parts.append(TASK_INSTRUCTION)
    return "\n".join(parts)


def build_subhalo_system_prompt(
    n_subhalos: int = 1,
    candidates: list = None,
    kwargs_model: dict = None,
    max_subhalo_mass_msun: float = 1.0e10,
    freeze_non_subhalo_params: bool = False,
) -> str:
    """System prompt for the pass-2 subhalo detection agent.

    Same detailed component listing as pass1's build_system_prompt, with NFW
    subhalo components clearly marked as the primary target. Optionally marks
    all non-subhalo components as frozen to the pass1.5 reference solution.
    """
    from .scoring import (PRIOR_BOUNDS, FIXED_PARAMS, ACTIVE_COMBO,
                          MODEL_COMBOS, BLIND_MODE, KIN_SOFT, CHI2_PENALTY,
                          SUBTRACTED_CHI2, PHYSICALITY_MODE)

    combo = MODEL_COMBOS[ACTIVE_COMBO]
    km = kwargs_model or combo["kwargs_model"]
    lens_model_list = km.get("lens_model_list", [])
    n_base_lens = len(lens_model_list) - n_subhalos

    cand_desc = ""
    if candidates:
        cand_lines = []
        for i, c in enumerate(candidates[:n_subhalos]):
            cand_lines.append(
                f"  Candidate {i}: ra={c['ra']:.4f}, dec={c['dec']:.4f}, "
                f"pull={c['pull']:.1f} sigma")
        cand_desc = "\n".join(cand_lines)

    comp_lines = []
    idx = 1
    for key in ("kwargs_lens", "kwargs_lens_light", "kwargs_source"):
        bounds_list = PRIOR_BOUNDS.get(key, [])
        fixed_list = FIXED_PARAMS.get(key, [{}] * len(bounds_list))
        if key == "kwargs_lens":
            model_list = km.get("lens_model_list", [])
        elif key == "kwargs_lens_light":
            model_list = km.get("lens_light_model_list", [])
        else:
            model_list = km.get("source_light_model_list", [])

        for ci, bounds in enumerate(bounds_list):
            model_name = model_list[ci] if ci < len(model_list) else "?"
            fixed = fixed_list[ci] if ci < len(fixed_list) else {}
            fixed_str = ", ".join(f"{k}={v}" for k, v in fixed.items()) if fixed else ""

            param_strs = [
                f"{pname} [{lo},{hi}]"
                for pname, (lo, hi) in _visible_bounds(key, model_name, bounds).items()
            ]
            params_text = ", ".join(param_strs)

            is_sub = (key == "kwargs_lens" and ci >= n_base_lens)
            tag = ""
            if is_sub:
                tag = " **[NFW SUBHALO — PRIMARY TARGET]**"
            elif freeze_non_subhalo_params:
                tag = " **[FROZEN TO PASS1.5]**"
            # Do not surface fixed shapelet config such as n_max in the prompt.
            fixed_note = ""
            if fixed_str and "SHAPELETS" not in model_name:
                fixed_note = f"  (fixed: {fixed_str})"
            label = _component_prompt_label(key, model_name)
            comp_lines.append(f"{idx}. **{label}**{tag}: {params_text}{fixed_note}")
            idx += 1

    n_lens = len(PRIOR_BOUNDS.get("kwargs_lens", []))
    n_ll = len(PRIOR_BOUNDS.get("kwargs_lens_light", []))
    n_src = len(PRIOR_BOUNDS.get("kwargs_source", []))
    lens_dots = ", ".join(["{...}"] * n_lens)
    ll_dots = ", ".join(["{...}"] * max(n_ll, 1))
    src_dots = ", ".join(["{...}"] * max(n_src, 1))
    example_json = (f'{{"kwargs_lens": [{lens_dots}], '
                    f'"kwargs_lens_light": [{ll_dots}], '
                    f'"kwargs_source": [{src_dots}]}}')

    sub_range_str = f"components {n_base_lens + 1} through {n_base_lens + n_subhalos}" if n_subhalos > 1 else f"component {n_base_lens + 1}"
    sub_plural = "subhalos" if n_subhalos > 1 else "subhalo"
    added_verb = "has" if n_subhalos == 1 else "have"
    fit_sentence = (
        "The subhalo is fit on top of the fixed/reference base model."
        if n_subhalos == 1 else
        f"All {n_subhalos} subhalos are fit simultaneously on top of the base model."
    )

    return f"""\
You are a parameter optimization agent for gravitational lens subhalo detection. \
A first-pass model is already well-fitted. {n_subhalos} NFW {sub_plural} {added_verb} been added to detect \
dark matter substructure in the residuals. {fit_sentence}

## Model Components

The model has {idx-1} components. {
    "Only the NFW subhalo component(s) are free; all other components are frozen to the pass1.5 reference values. Still return the full JSON structure for every component."
    if freeze_non_subhalo_params else
    "You set ALL parameters for EVERY component:"
}

{chr(10).join(comp_lines)}

Light amplitudes are solved automatically.{
    "" if BLIND_MODE else " Velocity dispersion (sigma_v) is calculated from the mass model."
}

## Subhalo Candidate Locations

{cand_desc}

## Scoring

{_scoring_formula_text(
    blind_mode=BLIND_MODE,
    penalty_mode=CHI2_PENALTY,
    physicality_mode=PHYSICALITY_MODE,
)}

Higher Q is better.

- **chi2_image_reduced**: target is 1.0 (noise-limited). chi2=1.0 is the best possible fit. chi2 > 1 means the model is missing features.{
    chr(10) + "- Once chi2 is already within 1% of 1.0, the scoring strongly favors the solution closest to 1.0. In that narrow band, prioritize image fit over secondary tie-breakers."
    + chr(10) + "- Inside an ultra-tiny inner band around 1.0, this preference becomes even stronger, so very small chi2 improvements near 1.0 still matter."
}{
    "" if BLIND_MODE else chr(10) +
    "- **residual_randomness** (0-1): measures spatial structure in residuals. Lower = more random = better." + chr(10) +
    "- **chi2_kinematic**: velocity dispersion match. " + (
        "Predictions within measurement uncertainty get ZERO penalty."
        if KIN_SOFT else
        "Any prediction within measurement uncertainty is fine."
    )
}

Delta-BIC is reported for subhalo significance (delta-BIC > 10 = significant detection).

- Hard pass-2 constraint: each NFW subhalo must have derived M200 <= {max_subhalo_mass_msun:.2e} Msun. Any proposal above this cap is invalid, heavily penalized, and excluded from final ranking.

## Images

{
    "5-panel comparison: GT (Observed) | GT-Lens (real arcs in data) | Render (Full Model) | Render_Sub (model's predicted arcs) | Residual_Sub (source-only residual, blue=bright, red=faint). "
    "GT-Lens = observed data minus model lens light (real arcs). Render_Sub = full model minus model lens light (predicted arcs). Compare them to see what the model misses. "
    "Blue = model too bright. Red = model too faint. White = good fit."
    if SUBTRACTED_CHI2 else
    "6-panel comparison: GT (Observed) | GT-Lens (real arcs in data) | Render (Full Model) | Residual (blue=bright, red=faint) | Render_Sub (model's predicted arcs) | Residual_Sub (Source Only). "
    "GT-Lens = observed data minus model lens light (real arcs). Render_Sub = full model minus model lens light (predicted arcs). Compare them to see what the model misses. "
    "Blue = model too bright. Red = model too faint. White = good fit."
}

## Tools

Each tool call contains 3 DIFFERENT proposals inside <solution_1>, <solution_2>, <solution_3> tags.

### evaluate
<action>
tool: evaluate
<solution_1>
{example_json}
</solution_1>
<solution_2>
{example_json}
</solution_2>
<solution_3>
{example_json}
</solution_3>
</action>

### finish
Same format as evaluate. The best of the 3 is submitted.

## Instructions

- {
    "Non-subhalo components are frozen to the pass1.5 reference values. Do not try to re-optimize them; any changes will be ignored/overwritten."
    if freeze_non_subhalo_params else
    "Propose ALL parameters for ALL components in every call."
}
- {
    f"The base model (components 1-{n_base_lens}) is frozen. Focus entirely on the NFW subhalo parameter(s), but still emit the full JSON structure."
    if freeze_non_subhalo_params else
    f"The base model (components 1-{n_base_lens}) is well-fitted — keep base params close to the references. Only fine-tune base params at high precision for small chi2 gains."
}
- Your primary goal is to optimize the {n_subhalos} NFW {sub_plural} ({sub_range_str}): Rs, alpha_Rs, center_x, center_y for each.
- For each NFW subhalo, vary Rs (scale radius, 0.001-0.5) and alpha_Rs (deflection at Rs, 0.0001-0.5) across orders of magnitude. Adjust centers within ~0.1 arcsec of the candidate locations.
- Keep every subhalo physically modest: the derived NFW mass (M200 computed from Rs and alpha_Rs) must stay at or below {max_subhalo_mass_msun:.2e} Msun. If you hit the cap, reduce the subhalo strength rather than pushing to extreme masses.
- Each call must contain 3 DIFFERENT proposals — vary the subhalo params meaningfully across solutions.
- Use HIGH PRECISION values with 5-6 decimal places (e.g., 1.062478 not 1.06, 0.01473 not 0.01). Small parameter differences matter at this refinement stage.
- If delta-BIC stays negative after several attempts, the subhalos are not real. Call finish.
"""
