"""Particle-swarm initialization and model-family scouting."""

from __future__ import annotations

import inspect
import json
import logging
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.Workflow.fitting_sequence import FittingSequence

from lensagent.agent.database import ProposalDatabase
from lensagent.config import PSOConfig
from lensagent.data.observation import Observation
from lensagent.modeling.evaluate import evaluate_proposal
from lensagent.modeling.parameters import ParameterSpace
from lensagent.output.artifacts import write_json

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class PSOFit:
    family: str
    bic: float
    log_likelihood: float
    proposal: dict[str, Any]


@dataclass(frozen=True)
class FamilyScoutResult:
    family: str
    label: str
    bic: float
    log_likelihood: float
    fits: tuple[PSOFit, ...]


def _pack_indexed(component: dict[str, Any]) -> dict[str, Any]:
    if "amp" in component or "sigma" in component:
        return dict(component)
    result: dict[str, Any] = {}
    amplitudes: list[Any] = []
    sigmas: list[Any] = []
    for name, value in component.items():
        if name.startswith("amp_"):
            index = int(name.removeprefix("amp_"))
            amplitudes.extend(0.0 for _ in range(index + 1 - len(amplitudes)))
            amplitudes[index] = value
        elif name.startswith("sigma_"):
            index = int(name.removeprefix("sigma_"))
            sigmas.extend(0.1 for _ in range(index + 1 - len(sigmas)))
            sigmas[index] = value
        else:
            result[name] = value
    if amplitudes:
        result["amp"] = amplitudes
    if sigmas:
        result["sigma"] = sigmas
    return result


def _indexed_multi_gaussian_block(
    bounds: dict[str, Any],
    center: dict[str, Any],
    fixed: dict[str, Any],
    sigma: dict[str, Any],
    extra_parameters: Sequence[tuple[str, float]],
) -> tuple[dict[str, Any], ...] | None:
    indices = sorted(
        {
            int(name.split("_", 1)[1])
            for name in bounds
            if name.startswith(("amp_", "sigma_"))
        }
    )
    if not indices:
        return None

    initial: dict[str, Any] = {}
    step: dict[str, Any] = {}
    frozen = dict(fixed)
    lower: dict[str, Any] = {}
    upper: dict[str, Any] = {}
    center_amplitudes = center.get("amp", [])
    center_sigmas = center.get("sigma", [])
    for prefix, defaults, center_values in (
        ("amp", (0.001, 50.0), center_amplitudes),
        ("sigma", (0.01, 20.0), center_sigmas),
    ):
        starts, steps, lows, highs = [], [], [], []
        for index in indices:
            low, high = bounds.get(f"{prefix}_{index}", defaults)
            if isinstance(center_values, (list, tuple, np.ndarray)) and index < len(
                center_values
            ):
                midpoint = center_values[index]
            else:
                midpoint = center.get(f"{prefix}_{index}", (low + high) / 2.0)
            starts.append(float(midpoint))
            steps.append(sigma.get(f"{prefix}_{index}", (high - low) * 0.1))
            lows.append(low)
            highs.append(high)
        initial[prefix] = starts
        step[prefix] = steps
        lower[prefix] = lows
        upper[prefix] = highs

    for name, default in extra_parameters:
        if name in frozen:
            continue
        if name in bounds:
            low, high = bounds[name]
            initial[name] = float(center.get(name, (low + high) / 2.0))
            step[name] = sigma.get(name, (high - low) * 0.1)
            lower[name] = low
            upper[name] = high
        else:
            initial[name] = float(center.get(name, default))
            width = max(abs(float(default)) * 5.0, 1.0)
            step[name] = max(abs(float(default)) * 0.3, 0.1)
            lower[name] = float(default) - width
            upper[name] = float(default) + width
    return initial, step, frozen, lower, upper


def _generic_component_block(
    function: Any,
    bounds: dict[str, Any],
    center: dict[str, Any],
    fixed: dict[str, Any],
    sigma: dict[str, Any],
) -> tuple[dict[str, Any], ...]:
    if fixed and not bounds:
        return dict(fixed), {}, dict(fixed), {}, {}

    initial: dict[str, Any] = {}
    step: dict[str, Any] = {}
    frozen: dict[str, Any] = {}
    lower: dict[str, Any] = {}
    upper: dict[str, Any] = {}
    signature = inspect.signature(function.function)
    for name in function.param_names:
        if name in fixed:
            frozen[name] = fixed[name]
        elif name in bounds:
            low, high = bounds[name]
            initial[name] = float(center.get(name, (low + high) / 2.0))
            step[name] = sigma.get(name, (high - low) * 0.1)
            lower[name] = low
            upper[name] = high
        else:
            parameter = signature.parameters.get(name)
            default = (
                parameter.default
                if parameter is not None
                and parameter.default is not inspect.Parameter.empty
                else 0.0
            )
            if isinstance(default, (int, float)):
                initial[name] = float(center.get(name, default))
                width = max(abs(float(default)) * 5.0, 1.0)
                step[name] = max(abs(float(default)) * 0.3, 0.1)
                lower[name] = float(default) - width
                upper[name] = float(default) + width
            else:
                frozen[name] = default
    if "ra_0" in initial:
        frozen.update({"ra_0": 0.0, "dec_0": 0.0})
        for block in (initial, step, lower, upper):
            block.pop("ra_0", None)
            block.pop("dec_0", None)
    return initial, step, frozen, lower, upper


def _lens_parameter_block(
    parameter_space: ParameterSpace,
    fitting_model: dict[str, list[str]],
) -> list[list[dict[str, Any]]]:
    lens_model = LensModel(fitting_model["lens_model_list"])
    result = [[], [], [], [], []]
    proxy_index = 0
    for index, function in enumerate(lens_model.lens_model.func_list):
        bounds = (
            dict(parameter_space.bounds_lens[index])
            if index < len(parameter_space.bounds_lens)
            else {}
        )
        center = (
            dict(parameter_space.centers_lens[index])
            if index < len(parameter_space.centers_lens)
            else {}
        )
        fixed = (
            dict(parameter_space.fixed_lens[index])
            if index < len(parameter_space.fixed_lens)
            else {}
        )
        sigma = (
            dict(parameter_space.sigmas_lens[index])
            if index < len(parameter_space.sigmas_lens)
            else {}
        )
        model_name = fitting_model["lens_model_list"][index]

        if parameter_space.pso_proxy_lens_models and model_name == "GAUSSIAN":
            bounds = dict(parameter_space.bounds_lens[0])
            center = dict(parameter_space.centers_lens[0])
            fixed = {}
            sigma = dict(parameter_space.sigmas_lens[0])
            gaussian_index = proxy_index
            proxy_index += 1
            bounds = {
                "amp": bounds.get(f"amp_{gaussian_index}", (0.001, 50.0)),
                "sigma": bounds.get(f"sigma_{gaussian_index}", (0.01, 20.0)),
                **{
                    name: bounds[name]
                    for name in ("center_x", "center_y")
                    if name in bounds
                },
            }
            center = {
                "amp": center.get(f"amp_{gaussian_index}", 1.0),
                "sigma": center.get(f"sigma_{gaussian_index}", 1.0),
                "center_x": center.get("center_x", 0.0),
                "center_y": center.get("center_y", 0.0),
            }
            sigma = {
                "amp": sigma.get(f"amp_{gaussian_index}", 0.1 * np.ptp(bounds["amp"])),
                "sigma": sigma.get(
                    f"sigma_{gaussian_index}", 0.1 * np.ptp(bounds["sigma"])
                ),
                "center_x": sigma.get("center_x", 0.05),
                "center_y": sigma.get("center_y", 0.05),
            }

        if model_name == "MULTI_GAUSSIAN":
            block = _indexed_multi_gaussian_block(
                bounds,
                _pack_indexed(center),
                _pack_indexed(fixed),
                sigma,
                (("center_x", 0.0), ("center_y", 0.0)),
            )
            if block is None:
                block = _generic_component_block(function, bounds, center, fixed, sigma)
        else:
            block = _generic_component_block(function, bounds, center, fixed, sigma)
        for destination, values in zip(result, block):
            destination.append(values)
    return result


def _light_parameter_block(
    model_names: Sequence[str],
    bounds_list: Sequence[dict[str, Any]],
    centers: Sequence[dict[str, Any]],
    fixed_list: Sequence[dict[str, Any]],
    sigmas: Sequence[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    result = [[], [], [], [], []]
    for index, bounds_value in enumerate(bounds_list):
        bounds = dict(bounds_value)
        center = dict(centers[index]) if index < len(centers) else {}
        fixed = dict(fixed_list[index]) if index < len(fixed_list) else {}
        sigma = dict(sigmas[index]) if index < len(sigmas) else {}
        model_name = model_names[index] if index < len(model_names) else ""
        if model_name == "MULTI_GAUSSIAN":
            block = _indexed_multi_gaussian_block(
                _pack_indexed(bounds),
                _pack_indexed(center),
                _pack_indexed(fixed),
                sigma,
                (("center_x", 0.0), ("center_y", 0.0)),
            )
            if block is not None:
                for destination, values in zip(result, block):
                    destination.append(values)
                continue

        initial: dict[str, Any] = {}
        step: dict[str, Any] = {}
        frozen = dict(fixed)
        lower: dict[str, Any] = {}
        upper: dict[str, Any] = {}
        for name, interval in bounds.items():
            if name in frozen:
                continue
            low, high = interval
            initial[name] = float(center.get(name, (low + high) / 2.0))
            step[name] = sigma.get(name, (high - low) * 0.1)
            lower[name] = low
            upper[name] = high
        if "SHAPELETS" in model_name:
            for name in ("center_x", "center_y"):
                if name not in frozen:
                    initial.setdefault(name, 0.0)
                    step.setdefault(name, 0.05)
                    lower.setdefault(name, -10.0)
                    upper.setdefault(name, 10.0)
        elif "amp" not in initial and "amp" not in frozen:
            initial["amp"] = 1.0
            step["amp"] = 10.0
            lower["amp"] = 0.001
            upper["amp"] = 100_000.0
        for destination, values in zip(result, (initial, step, frozen, lower, upper)):
            destination.append(values)
    return result


def fitting_parameters(
    parameter_space: ParameterSpace,
    fitting_model: dict[str, list[str]] | None = None,
) -> dict[str, list[list[dict[str, Any]]]]:
    model = fitting_model or parameter_space.model
    return {
        "lens_model": _lens_parameter_block(parameter_space, model),
        "source_model": _light_parameter_block(
            model["source_light_model_list"],
            parameter_space.bounds_source,
            parameter_space.centers_source,
            parameter_space.fixed_source,
            parameter_space.sigmas_source,
        ),
        "lens_light_model": _light_parameter_block(
            model["lens_light_model_list"],
            parameter_space.bounds_lens_light,
            parameter_space.centers_lens_light,
            parameter_space.fixed_lens_light,
            parameter_space.sigmas_lens_light,
        ),
    }


def _bic_and_log_likelihood(sequence: FittingSequence) -> tuple[float, float]:
    log_likelihood = float(sequence.best_fit_likelihood())
    nonlinear, _ = sequence.param_class.num_param()
    parameter_count = int(nonlinear + sequence.param_class.num_param_linear())
    likelihood = getattr(sequence, "likelihood_class", None)
    if likelihood is None:
        likelihood = sequence.likelihoodModule
    data_count = likelihood.num_data
    if callable(data_count):
        data_count = data_count()
    return float(
        np.log(float(data_count)) * parameter_count - 2.0 * log_likelihood
    ), log_likelihood


def _merge_proxy_lenses(
    components: Sequence[dict[str, Any]],
    proxy_models: Sequence[str],
    target_models: Sequence[str],
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    proxy_index = 0
    for model_name in target_models:
        if model_name != "MULTI_GAUSSIAN":
            merged.append(dict(components[proxy_index]))
            proxy_index += 1
            continue
        gaussian_count = sum(name == "GAUSSIAN" for name in proxy_models[proxy_index:])
        gaussian_components = components[proxy_index : proxy_index + gaussian_count]
        merged.append(
            {
                "amp": [
                    float(component.get("amp", 0.0))
                    for component in gaussian_components
                ],
                "sigma": [
                    float(component.get("sigma", 0.1))
                    for component in gaussian_components
                ],
                "center_x": float(gaussian_components[-1].get("center_x", 0.0)),
                "center_y": float(gaussian_components[-1].get("center_y", 0.0)),
                "scale_factor": 1.0,
            }
        )
        proxy_index += gaussian_count
    return merged


def _proposal_from_best_fit(
    best_fit: dict[str, Any], parameter_space: ParameterSpace
) -> dict[str, Any]:
    proposal = {
        "kwargs_lens": best_fit.get("kwargs_lens", []),
        "kwargs_lens_light": best_fit.get("kwargs_lens_light", []),
        "kwargs_source": best_fit.get("kwargs_source", []),
    }
    if parameter_space.pso_proxy_lens_models:
        proposal["kwargs_lens"] = _merge_proxy_lenses(
            proposal["kwargs_lens"],
            parameter_space.pso_proxy_lens_models,
            parameter_space.model["lens_model_list"],
        )
    return proposal


def run_family_pso(
    observation: Observation,
    parameter_space: ParameterSpace,
    config: PSOConfig,
) -> FamilyScoutResult:
    native_space = parameter_space
    if observation.hst:
        from lensagent.workflow.optimizer import optimizer_space

        parameter_space = optimizer_space(parameter_space)
    fitting_model = dict(parameter_space.model)
    if parameter_space.pso_proxy_lens_models:
        fitting_model["lens_model_list"] = list(parameter_space.pso_proxy_lens_models)
    fitting_observation = observation.with_model(fitting_model)
    likelihood: dict[str, Any] = {"check_bounds": True}
    if observation.likelihood_mask is not None:
        likelihood["image_likelihood_mask_list"] = [observation.likelihood_mask]
    parameters = fitting_parameters(parameter_space, fitting_model)
    if observation.hst:
        from lensagent.modeling.constraints import MacroFloor

        floor = MacroFloor.from_model(fitting_model)
        parameters = floor.project_start(parameters)
        likelihood["custom_logL_addition"] = lambda kwargs_lens, **kwargs: (
            0.0 if floor.accepts(kwargs_lens) else -1e30)

    fits: list[PSOFit] = []
    for _ in range(config.runs):
        sequence = FittingSequence(
            fitting_observation.kwargs_data_joint,
            fitting_model,
            parameter_space.lenstronomy_constraints,
            likelihood,
            parameters,
        )
        sequence.fit_sequence(
            [
                [
                    "PSO",
                    {
                        "sigma_scale": config.sigma_scale,
                        "n_particles": config.particles,
                        "n_iterations": config.iterations,
                    },
                ]
            ]
        )
        bic, log_likelihood = _bic_and_log_likelihood(sequence)
        fits.append(
            PSOFit(
                family=parameter_space.family,
                bic=bic,
                log_likelihood=log_likelihood,
                proposal=_proposal_from_best_fit(sequence.best_fit(), native_space),
            )
        )
    fits.sort(key=lambda fit: fit.bic)
    best = fits[0]
    return FamilyScoutResult(
        family=parameter_space.family,
        label=parameter_space.family.replace("_", " ").title(),
        bic=best.bic,
        log_likelihood=best.log_likelihood,
        fits=tuple(fits),
    )


def scout_families(
    observation: Observation,
    spaces: Sequence[ParameterSpace],
    config: PSOConfig,
    *,
    workers: int,
    cache_path: str | Path,
) -> list[FamilyScoutResult]:
    destination = Path(cache_path)
    if destination.exists():
        return load_scout_results(destination)

    results: list[FamilyScoutResult] = []
    if observation.hst:
        from lensagent.workflow.processes import process_pool

        executor_context = process_pool(max(1, workers))
    else:
        executor_context = ThreadPoolExecutor(max_workers=max(1, workers))
    with executor_context as executor:
        futures = {
            executor.submit(
                run_family_pso, observation.with_model(space.model), space, config
            ): space.family
            for space in spaces
        }
        for future in as_completed(futures):
            family = futures[future]
            try:
                result = future.result()
            except Exception:
                log.exception("PSO failed for model family %s", family)
                continue
            results.append(result)
            log.info("PSO %-28s BIC=%10.3f", family, result.bic)
    results.sort(key=lambda result: (result.bic, result.family))
    save_scout_results(destination, results)
    return results


def save_scout_results(path: str | Path, results: Sequence[FamilyScoutResult]) -> None:
    write_json(path, [asdict(result) for result in results])


def load_scout_results(path: str | Path) -> list[FamilyScoutResult]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        FamilyScoutResult(
            family=item["family"],
            label=item["label"],
            bic=float(item["bic"]),
            log_likelihood=float(item["log_likelihood"]),
            fits=tuple(PSOFit(**fit) for fit in item["fits"]),
        )
        for item in data
    ]


def seed_database(
    database: ProposalDatabase,
    observation: Observation,
    parameter_space: ParameterSpace,
    fits: Sequence[PSOFit],
    *,
    seed_count: int,
    island_count: int,
    rng: np.random.Generator,
) -> None:
    proposals = [fit.proposal for fit in fits]
    minimum_random = min(5, seed_count)
    random_count = max(seed_count - len(proposals), minimum_random)
    proposals.extend(database.scoring.random_proposal(rng) for _ in range(random_count))
    for index, proposal in enumerate(proposals[:seed_count]):
        if observation.hst:
            from lensagent.modeling.safe_evaluate import safe_evaluate

            evaluation, error = safe_evaluate(proposal, observation.with_model(parameter_space.model),
                                               parameter_space, timeout_seconds=60)
            if evaluation is None:
                log.warning("PSO reference evaluation failed: %s", error)
                continue
        else:
            evaluation = evaluate_proposal(
                proposal,
                observation.with_model(parameter_space.model),
                parameter_space,
            )
        database.add(database.create(proposal, evaluation, island=index % island_count))
