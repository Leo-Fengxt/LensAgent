"""Shared residual, NFW, and evidence calculations for RSI."""

from __future__ import annotations

import copy
import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from astropy.cosmology import FlatLambdaCDM
from lenstronomy.Cosmo.lens_cosmo import LensCosmo
from scipy.optimize import brentq

from lensagent.config import QualityConfig
from lensagent.data.observation import Observation
from lensagent.modeling.evaluate import evaluate_proposal
from lensagent.modeling.parameters import ParameterComponents, ParameterSpace
from lensagent.modeling.profiles import register_mass_concentration_nfw
from lensagent.modeling.scoring import (
    ScoringPolicy,
    _number,
    chi_squared_priority_penalty,
)

COSMOLOGY = FlatLambdaCDM(H0=70, Om0=0.3, Ob0=0.05)


@dataclass(frozen=True)
class PullCandidate:
    ra: float
    dec: float
    pull: float
    absolute_pull: float
    pixel_x: float
    pixel_y: float
    blob_sigma: float = 0.0
    rank: int = 0
    center_bounds: dict[str, tuple[float, float]] | None = None

    def as_dict(self) -> dict[str, Any]:
        result = {
            "ra": self.ra,
            "dec": self.dec,
            "pull": self.pull,
            "absolute_pull": self.absolute_pull,
            "pixel_x": self.pixel_x,
            "pixel_y": self.pixel_y,
            "blob_sigma": self.blob_sigma,
            "rank": self.rank,
        }
        if self.center_bounds is not None:
            result["center_bounds"] = copy.deepcopy(self.center_bounds)
        return result


def pull_map(
    proposal: dict[str, Any],
    observation: Observation,
    parameter_space: ParameterSpace,
) -> tuple[np.ndarray, dict[str, Any]]:
    evaluation = evaluate_proposal(
        proposal,
        observation.with_model(parameter_space.model),
        parameter_space,
    )
    return np.asarray(evaluation["residual_map"], dtype=float), evaluation


def detect_blob_candidates(
    residual_pull: np.ndarray,
    observation: Observation,
    *,
    threshold: float,
) -> list[PullCandidate]:
    from skimage.feature import blob_log

    blobs = blob_log(
        np.abs(residual_pull),
        min_sigma=0.8,
        max_sigma=3.0,
        threshold=threshold / 10.0,
    )
    candidates = []
    transform = observation.transform_pix2angle
    for row, column, sigma in blobs:
        ra = observation.ra_at_xy_0 + transform[0, 0] * column + transform[0, 1] * row
        dec = observation.dec_at_xy_0 + transform[1, 0] * column + transform[1, 1] * row
        row_index = int(row)
        column_index = int(column)
        value = (
            float(residual_pull[row_index, column_index])
            if 0 <= row_index < residual_pull.shape[0]
            and 0 <= column_index < residual_pull.shape[1]
            else 0.0
        )
        candidates.append(
            PullCandidate(
                ra=float(ra),
                dec=float(dec),
                pull=value,
                absolute_pull=abs(value),
                pixel_x=float(column),
                pixel_y=float(row),
                blob_sigma=float(sigma),
            )
        )
    candidates.sort(key=lambda candidate: candidate.absolute_pull, reverse=True)
    return [
        replace(candidate, rank=index) for index, candidate in enumerate(candidates)
    ]


def effective_einstein_radius(
    proposal: dict[str, Any], parameter_space: ParameterSpace
) -> tuple[float, int]:
    for index, component in enumerate(proposal.get("kwargs_lens", [])):
        value = component.get("theta_E")
        if value is not None and np.isfinite(float(value)) and float(value) > 0:
            return float(value), index

    geometry = parameter_space.lens_centered_geometry or {}
    primary_index = int(geometry.get("center_component_index", 0))
    component_indices = geometry.get("einstein_radius_component_indices")
    if component_indices is None:
        raise ValueError("model family does not define an Einstein-radius geometry")
    from lenstronomy.Analysis.lens_profile import LensProfileAnalysis
    from lenstronomy.LensModel.lens_model import LensModel

    lens_components = proposal["kwargs_lens"]
    primary = lens_components[primary_index]
    enabled = [index in component_indices for index in range(len(lens_components))]
    analysis = LensProfileAnalysis(
        LensModel(lens_model_list=parameter_space.model["lens_model_list"])
    )
    radius = analysis.effective_einstein_radius_grid(
        lens_components,
        center_x=float(primary.get("center_x", 0.0)),
        center_y=float(primary.get("center_y", 0.0)),
        model_bool_list=enabled,
        grid_num=200,
        grid_spacing=0.05,
        get_precision=False,
        verbose=False,
    )
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("model has no finite positive Einstein radius")
    return float(radius), primary_index


def lens_centered_candidates(
    candidates: Sequence[PullCandidate],
    proposal: dict[str, Any],
    parameter_space: ParameterSpace,
    radius_in_einstein_units: float,
) -> tuple[list[PullCandidate], dict[str, Any]]:
    radius, primary_index = effective_einstein_radius(proposal, parameter_space)
    primary = proposal["kwargs_lens"][primary_index]
    center_x = float(primary.get("center_x", 0.0))
    center_y = float(primary.get("center_y", 0.0))
    search_radius = radius_in_einstein_units * radius
    bounds = {
        "center_x": (center_x - search_radius, center_x + search_radius),
        "center_y": (center_y - search_radius, center_y + search_radius),
    }
    selected = [
        replace(candidate, center_bounds=copy.deepcopy(bounds))
        for candidate in candidates
        if math.hypot(candidate.ra - center_x, candidate.dec - center_y)
        <= search_radius
    ]
    return selected, {
        "center_x": center_x,
        "center_y": center_y,
        "einstein_radius": radius,
        "radius_in_einstein_units": radius_in_einstein_units,
        "radius_arcsec": search_radius,
        "center_bounds": bounds,
    }


def local_candidate_bounds(
    candidate: dict[str, Any], half_width: float
) -> dict[str, tuple[float, float]]:
    custom = candidate.get("center_bounds")
    if custom:
        return {
            "center_x": tuple(float(value) for value in custom["center_x"]),
            "center_y": tuple(float(value) for value in custom["center_y"]),
        }
    return {
        "center_x": (
            float(candidate["ra"]) - half_width,
            float(candidate["ra"]) + half_width,
        ),
        "center_y": (
            float(candidate["dec"]) - half_width,
            float(candidate["dec"]) + half_width,
        ),
    }


def _freeze_components(
    proposal_components: Sequence[dict[str, Any]],
    original_fixed: ParameterComponents,
) -> ParameterComponents:
    result = []
    for index, component in enumerate(proposal_components):
        frozen = dict(component)
        if index < len(original_fixed):
            frozen.update(original_fixed[index])
        result.append(frozen)
    return tuple(result)


def independent_nfw_space(
    base: ParameterSpace,
    base_proposal: dict[str, Any],
    candidates: Sequence[dict[str, Any]],
    *,
    freeze_smooth_model: bool,
    center_half_width: float,
    macro_thaw: dict[str, float] | None = None,
    warm_base_lens: Sequence[dict[str, Any]] | None = None,
) -> ParameterSpace:
    model = copy.deepcopy(base.model)
    model["lens_model_list"] = list(model["lens_model_list"]) + [
        "NFW" for _ in candidates
    ]
    bounds_lens = [copy.deepcopy(value) for value in base.bounds_lens]
    centers_lens = [copy.deepcopy(value) for value in base_proposal["kwargs_lens"]]
    sigmas_lens = [copy.deepcopy(value) for value in base.sigmas_lens]
    if freeze_smooth_model:
        fixed_lens = list(
            _freeze_components(base_proposal["kwargs_lens"], base.fixed_lens)
        )
        bounds_lens = [{} for _ in bounds_lens]
        sigmas_lens = [{} for _ in sigmas_lens]
    else:
        fixed_lens = [copy.deepcopy(value) for value in base.fixed_lens]

    if macro_thaw:
        primary = base_proposal["kwargs_lens"][0]
        original_bounds = base.bounds_lens[0]
        thawed_bounds = {}
        thawed_center = dict(centers_lens[0])
        warmed = warm_base_lens[0] if warm_base_lens else primary
        for name, half_width in macro_thaw.items():
            if name not in primary or name not in original_bounds:
                continue
            low = max(
                float(original_bounds[name][0]), float(primary[name]) - half_width
            )
            high = min(
                float(original_bounds[name][1]), float(primary[name]) + half_width
            )
            if low >= high:
                raise ValueError(f"empty macro interval for {name}")
            thawed_bounds[name] = (low, high)
            thawed_center[name] = float(
                np.clip(warmed.get(name, primary[name]), low, high)
            )
            fixed_lens[0].pop(name, None)
        bounds_lens[0] = thawed_bounds
        centers_lens[0] = thawed_center
        sigmas_lens[0] = {
            name: 0.1 * (high - low) for name, (low, high) in thawed_bounds.items()
        }
        for index in range(1, len(base.bounds_lens)):
            bounds_lens[index] = {}

    for candidate in candidates:
        bounds = local_candidate_bounds(candidate, center_half_width)
        bounds_lens.append(
            {
                "Rs": (0.001, 0.5),
                "alpha_Rs": (0.0001, 0.5),
                **bounds,
            }
        )
        centers_lens.append(
            {
                "Rs": float(candidate.get("Rs") or 0.05),
                "alpha_Rs": float(candidate.get("alpha_Rs") or 0.01),
                "center_x": float(candidate["ra"]),
                "center_y": float(candidate["dec"]),
            }
        )
        sigmas_lens.append(
            {
                "Rs": 0.02,
                "alpha_Rs": 0.01,
                "center_x": 0.02,
                "center_y": 0.02,
            }
        )
        fixed_lens.append({})

    fixed_light = (
        _freeze_components(base_proposal["kwargs_lens_light"], base.fixed_lens_light)
        if freeze_smooth_model
        else base.fixed_lens_light
    )
    fixed_source = (
        _freeze_components(base_proposal["kwargs_source"], base.fixed_source)
        if freeze_smooth_model
        else base.fixed_source
    )
    bounds_light = (
        tuple({} for _ in base.bounds_lens_light)
        if freeze_smooth_model
        else base.bounds_lens_light
    )
    bounds_source = (
        tuple({} for _ in base.bounds_source)
        if freeze_smooth_model
        else base.bounds_source
    )
    return replace(
        base,
        family=f"{base.family}_single_rsi"
        if len(candidates) == 1
        else f"{base.family}_joint_rsi",
        model=model,
        bounds_lens=tuple(bounds_lens),
        centers_lens=tuple(centers_lens),
        sigmas_lens=tuple(sigmas_lens),
        fixed_lens=tuple(fixed_lens),
        bounds_lens_light=bounds_light,
        bounds_source=bounds_source,
        fixed_lens_light=tuple(fixed_light),
        fixed_source=tuple(fixed_source),
        counting_fixed_lens=tuple(base.fixed_lens) + tuple({} for _ in candidates),
        counting_fixed_lens_light=base.fixed_lens_light,
        counting_fixed_source=base.fixed_source,
        pso_proxy_lens_models=(),
    )


def tied_mass_nfw_space(
    base: ParameterSpace,
    base_proposal: dict[str, Any],
    candidates: Sequence[dict[str, Any]],
    *,
    center_half_width: float,
    macro_thaw: dict[str, float],
    warm_base_lens: Sequence[dict[str, Any]] | None = None,
) -> ParameterSpace:
    independent = independent_nfw_space(
        base,
        base_proposal,
        candidates,
        freeze_smooth_model=True,
        center_half_width=center_half_width,
        macro_thaw=macro_thaw,
        warm_base_lens=warm_base_lens,
    )
    model = copy.deepcopy(independent.model)
    model["lens_model_list"] = list(base.model["lens_model_list"]) + [
        "BLANK_PLANE" for _ in candidates
    ]
    bounds = list(independent.bounds_lens[: len(base.bounds_lens)])
    centers = list(independent.centers_lens[: len(base.bounds_lens)])
    sigmas = list(independent.sigmas_lens[: len(base.bounds_lens)])
    for candidate in candidates:
        center_bounds = local_candidate_bounds(candidate, center_half_width)
        mass = candidate.get("logM")
        if mass is None and candidate.get("mass_msun"):
            mass = math.log10(float(candidate["mass_msun"]))
        bounds.append({"logM": (7.0, 11.0), **center_bounds})
        centers.append(
            {
                "logM": float(np.clip(mass if mass is not None else 9.0, 7.0, 11.0)),
                "center_x": float(candidate["ra"]),
                "center_y": float(candidate["dec"]),
            }
        )
        sigmas.append({"logM": 0.2, "center_x": 0.02, "center_y": 0.02})
    return replace(
        independent,
        family=f"{base.family}_fixed_count_rsi",
        model=model,
        bounds_lens=tuple(bounds),
        centers_lens=tuple(centers),
        sigmas_lens=tuple(sigmas),
    )


def configure_tied_mass_space(space: ParameterSpace, observation: Observation) -> None:
    if "BLANK_PLANE" in space.model["lens_model_list"]:
        register_mass_concentration_nfw(observation.z_lens, observation.z_source)


def nfw_mass_msun(
    scale_radius: float,
    deflection_at_scale_radius: float,
    z_lens: float,
    z_source: float,
) -> float:
    lens_cosmo = LensCosmo(z_lens, z_source, cosmo=COSMOLOGY)
    from lenstronomy.Util import constants

    scale_radius = float(scale_radius)
    deflection = float(deflection_at_scale_radius)
    if (
        not np.isfinite(scale_radius)
        or not np.isfinite(deflection)
        or scale_radius <= 0
        or deflection <= 0
    ):
        return float("nan")
    physical_radius = scale_radius * constants.arcsec * lens_cosmo.dd
    scaled_deflection = (
        deflection * lens_cosmo.sigma_crit * lens_cosmo.dd * constants.arcsec
    )
    density = scaled_deflection / (4.0 * physical_radius**2 * (1.0 + np.log(0.5)))
    comoving_density = float(density / lens_cosmo.h**2)
    if not np.isfinite(comoving_density) or comoving_density <= 0:
        return float("nan")

    def residual(concentration: float) -> float:
        return float(
            lens_cosmo.nfw_param.rho0_c(concentration, lens_cosmo.z_lens)
            - comoving_density
        )

    upper = 100.0
    if residual(1.0e-12) > 0:
        return float("nan")
    while residual(upper) < 0:
        upper *= 2.0
        if upper > 1.0e10:
            return float("nan")
    concentration = brentq(
        residual, 1.0e-12, upper, xtol=1.0e-11, rtol=1.0e-11, maxiter=200
    )
    radius_200 = concentration * physical_radius
    return float(
        lens_cosmo.nfw_param.M_r200(radius_200 * lens_cosmo.h, z_lens) / lens_cosmo.h
    )


def _effective_reduced_chi_squared(value: float, raw: bool) -> float:
    value = float(value)
    return value if raw or value >= 1.0 else 2.0 - value


def evidence_metrics(
    base_evaluation: dict[str, Any],
    candidate_evaluation: dict[str, Any],
    *,
    raw: bool,
) -> dict[str, Any]:
    data_count = int(base_evaluation["fitted_pixels"])
    base_parameters = int(base_evaluation["parameter_count"])
    candidate_parameters = int(candidate_evaluation["parameter_count"])
    new_parameters = candidate_parameters - base_parameters
    if new_parameters < 0:
        raise ValueError("candidate model has fewer image parameters than its base")
    base_scale = int(base_evaluation["degrees_of_freedom"])
    candidate_scale = int(candidate_evaluation["degrees_of_freedom"])
    base_score = (
        _effective_reduced_chi_squared(
            base_evaluation["reduced_image_chi_squared"], raw
        )
        * base_scale
    )
    candidate_score = (
        _effective_reduced_chi_squared(
            candidate_evaluation["reduced_image_chi_squared"], raw
        )
        * candidate_scale
    )
    fit_gain = float(base_score - candidate_score)
    penalty = float(new_parameters * np.log(data_count))
    return {
        "delta_bic": fit_gain - penalty,
        "fit_gain": fit_gain,
        "parameter_penalty": penalty,
        "base_effective_image_chi_squared": float(base_score),
        "candidate_effective_image_chi_squared": float(candidate_score),
        "base_parameter_count": base_parameters,
        "candidate_parameter_count": candidate_parameters,
        "new_parameter_count": new_parameters,
        "fitted_pixels": data_count,
        "evidence_uses_raw_image_chi_squared": raw,
    }


def annotate_subhalos(
    evaluation: dict[str, Any],
    proposal: dict[str, Any],
    observation: Observation,
    base_evaluation: dict[str, Any],
    *,
    base_lens_count: int,
    candidates: Sequence[dict[str, Any]],
    maximum_mass_msun: float,
    raw_evidence: bool,
    minimum_separation: float = 0.0,
) -> dict[str, Any]:
    result = dict(evaluation)
    result.update(evidence_metrics(base_evaluation, evaluation, raw=raw_evidence))
    components = [dict(value) for value in proposal["kwargs_lens"][base_lens_count:]]
    masses = []
    for component in components:
        if component.get("logM") is not None:
            mass = 10.0 ** float(component["logM"])
        else:
            mass = nfw_mass_msun(
                component.get("Rs", 0.0),
                component.get("alpha_Rs", 0.0),
                observation.z_lens,
                observation.z_source,
            )
        masses.append(mass)
    mass_violations = [
        index
        for index, mass in enumerate(masses)
        if not np.isfinite(mass) or mass > maximum_mass_msun
    ]
    center_violations = []
    bounds_checked = []
    for index, (component, candidate) in enumerate(zip(components, candidates)):
        bounds = local_candidate_bounds(candidate, 0.1)
        bounds_checked.append(bounds)
        x = float(component.get("center_x", float("nan")))
        y = float(component.get("center_y", float("nan")))
        if (
            not np.isfinite(x)
            or not np.isfinite(y)
            or not bounds["center_x"][0] <= x <= bounds["center_x"][1]
            or not bounds["center_y"][0] <= y <= bounds["center_y"][1]
        ):
            center_violations.append(index)
    collisions = []
    if minimum_separation > 0:
        for left, first in enumerate(components):
            for right, second in enumerate(components[left + 1 :], start=left + 1):
                separation = math.hypot(
                    float(first["center_x"]) - float(second["center_x"]),
                    float(first["center_y"]) - float(second["center_y"]),
                )
                if separation < minimum_separation:
                    collisions.append((left, right, separation))
    result.update(
        {
            "subhalo_parameters": components,
            "masses_msun": masses,
            "subhalo_mass_cap_msun": maximum_mass_msun,
            "subhalo_mass_limit_ok": not mass_violations,
            "subhalo_mass_violation_indices": mass_violations,
            "subhalo_center_bounds": bounds_checked,
            "subhalo_center_bounds_ok": not center_violations,
            "subhalo_center_violation_indices": center_violations,
            "subhalo_minimum_separation_arcsec": minimum_separation,
            "subhalo_separation_ok": not collisions,
            "subhalo_collision_pairs": collisions,
        }
    )
    if mass_violations or center_violations or collisions:
        result["is_physical"] = False
    return result


@dataclass(frozen=True)
class EvidenceScoringPolicy(ScoringPolicy):
    chi_squared_tiebreak: bool = True

    def quality(
        self,
        evaluation: dict[str, Any],
        proposal: dict[str, Any],
        *,
        diversity: float | None = None,
    ) -> float:
        if (
            evaluation.get("subhalo_mass_limit_ok") is False
            or evaluation.get("subhalo_center_bounds_ok") is False
            or evaluation.get("subhalo_separation_ok") is False
        ):
            return -1.0e12
        quality = _number(evaluation.get("delta_bic"), -1.0e12)
        if self.chi_squared_tiebreak:
            quality -= 0.5 * chi_squared_priority_penalty(
                _number(evaluation.get("reduced_image_chi_squared"), 1.0e6)
            )
        if diversity is not None:
            quality += 5.0 * _number(diversity, 0.0)
        return float(quality)


def evidence_scoring_policy(
    space: ParameterSpace,
    kinematic_weight: float,
    *,
    chi_squared_tiebreak: bool,
) -> EvidenceScoringPolicy:
    quality = QualityConfig(
        image_weight=0.0,
        residual_weight=0.0,
        kinematic_weight=kinematic_weight,
        boundary_weight=0.0,
    )
    return EvidenceScoringPolicy(
        parameter_space=space,
        quality_config=quality,
        residual_weight=0.0,
        diversity_weight=0.5,
        chi_squared_tiebreak=chi_squared_tiebreak,
    )
