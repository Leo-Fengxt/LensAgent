"""Candidate identification and local refinement for fixed-count RSI."""

from __future__ import annotations

import copy
import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from lenstronomy.Cosmo.lens_cosmo import LensCosmo
from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.Data.psf import PSF
from lenstronomy.ImSim.image_linear_solve import ImageLinearFit
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
from scipy import ndimage
from scipy.optimize import minimize

from lensagent.config import CandidateIdentificationConfig
from lensagent.data.observation import Observation
from lensagent.modeling.parameters import ParameterSpace
from lensagent.modeling.profiles import MassConcentrationNFW
from lensagent.rsi.common import COSMOLOGY, effective_einstein_radius
from lensagent.rsi.multisubhalo.physics import curl_metrics, nfw_matched_filter


@dataclass(frozen=True)
class CandidateIdentification:
    detected: bool
    arc_rms: float
    arc_pixels: int
    initial_count: int
    ranked: tuple[dict[str, Any], ...]
    candidate_pool: tuple[dict[str, Any], ...]


def _pixel_to_angle(
    observation: Observation, column: float, row: float
) -> tuple[float, float]:
    transform = observation.transform_pix2angle
    return (
        float(
            observation.ra_at_xy_0 + transform[0, 0] * column + transform[0, 1] * row
        ),
        float(
            observation.dec_at_xy_0 + transform[1, 0] * column + transform[1, 1] * row
        ),
    )


def _connected_candidates(
    pull: np.ndarray,
    observation: Observation,
    theta_e: float,
    *,
    threshold: float,
    minimum_separation: float = 0.1,
) -> list[dict[str, Any]]:
    labels, count = ndimage.label(np.abs(pull) > threshold)
    candidates = []
    for label in range(1, count + 1):
        rows, columns = np.where(labels == label)
        peak = int(np.argmax(np.abs(pull[rows, columns])))
        row, column = int(rows[peak]), int(columns[peak])
        ra, dec = _pixel_to_angle(observation, column, row)
        if math.hypot(ra, dec) > 2.0 * theta_e:
            continue
        candidates.append(
            {
                "ra": ra,
                "dec": dec,
                "absolute_pull": float(abs(pull[row, column])),
                "pull": float(pull[row, column]),
                "pixel_x": float(column),
                "pixel_y": float(row),
            }
        )
    candidates.sort(key=lambda candidate: -candidate["absolute_pull"])
    merged = []
    for candidate in candidates:
        if all(
            math.hypot(
                candidate["ra"] - prior["ra"],
                candidate["dec"] - prior["dec"],
            )
            >= minimum_separation
            for prior in merged
        ):
            merged.append(candidate)
    return merged


def _arc_rms(pull: np.ndarray, theta_e: float, pixel_scale: float) -> tuple[float, int]:
    center = pull.shape[0] / 2.0 - 0.5
    rows, columns = np.mgrid[: pull.shape[0], : pull.shape[1]]
    radius = np.hypot(columns - center, rows - center) * pixel_scale
    selected = (radius > 0.6 * theta_e) & (radius < 1.5 * theta_e)
    values = pull[selected]
    if values.size == 0:
        return 0.0, 0
    return float(np.sqrt(np.mean(values**2))), int(values.size)


def _group_candidates(
    candidates: Sequence[dict[str, Any]], theta_e: float
) -> list[dict[str, Any]]:
    selected = [
        candidate
        for candidate in candidates
        if 0.7 <= math.hypot(candidate["ra"], candidate["dec"]) / theta_e <= 1.3
        and candidate["absolute_pull"] >= 3.0
    ]
    radius = 0.2 * theta_e
    consumed = set()
    groups = []
    for index, candidate in enumerate(selected):
        if index in consumed:
            continue
        members = [candidate]
        consumed.add(index)
        for other_index, other in enumerate(selected):
            if other_index in consumed:
                continue
            if (
                math.hypot(
                    candidate["ra"] - other["ra"],
                    candidate["dec"] - other["dec"],
                )
                < radius
            ):
                members.append(other)
                consumed.add(other_index)
        groups.append(
            {
                "ra": float(np.mean([member["ra"] for member in members])),
                "dec": float(np.mean([member["dec"] for member in members])),
                "member_count": len(members),
                "peak_pull": float(max(member["absolute_pull"] for member in members)),
                "members": [copy.deepcopy(member) for member in members],
            }
        )
    return groups


def _rank_normalized(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if len(array) < 2 or len(set(array.tolist())) == 1:
        return np.full(len(array), 0.5)
    return np.argsort(np.argsort(array)) / (len(array) - 1)


def _rank_groups(
    groups: list[dict[str, Any]],
    pull: np.ndarray,
    model_image: np.ndarray,
    noise_map: np.ndarray,
    observation: Observation,
    parameter_space: ParameterSpace,
    solved_proposal: dict[str, Any],
) -> list[dict[str, Any]]:
    if not groups:
        return []
    lens_model = LensModel(parameter_space.model["lens_model_list"])
    lens_cosmo = LensCosmo(observation.z_lens, observation.z_source, cosmo=COSMOLOGY)
    template_mass = 1.0e9
    concentration = (
        5.71
        * (template_mass / 2.0e12) ** -0.084
        * (1.0 + observation.z_lens) ** -0.47
        * 4.0
    )
    scale_radius, deflection = lens_cosmo.nfw_physical2angle(
        M=template_mass, c=concentration
    )
    for group in groups:
        point = (group["ra"], group["dec"])
        snr, amplitude = nfw_matched_filter(
            point,
            pull,
            model_image,
            noise_map,
            lens_model,
            solved_proposal["kwargs_lens"],
            observation,
            scale_radius,
            deflection,
        )
        curl = curl_metrics(
            point,
            pull,
            model_image,
            noise_map,
            lens_model,
            solved_proposal["kwargs_lens"],
            observation,
        )
        group.update(
            {
                "matched_filter_snr": float(abs(snr)),
                "matched_filter_amplitude": float(amplitude),
                "curl_fraction": curl.curl_fraction,
                "minimum_convergence_ratio": curl.minimum_convergence_ratio,
                "curl_peak_count": curl.peak_count,
                "curl_fitted_pixels": curl.fitted_pixels,
                "mass_guess_msun": float(
                    np.clip(abs(amplitude) * template_mass, 1.0e7, 1.0e11)
                ),
            }
        )

    score = _rank_normalized([group["minimum_convergence_ratio"] for group in groups])
    score += 0.7 * _rank_normalized([-group["curl_fraction"] for group in groups])
    score += 0.7 * _rank_normalized([group["matched_filter_snr"] for group in groups])
    score += 0.3 * _rank_normalized([group["peak_pull"] for group in groups])
    score /= 2.7
    for index, group in enumerate(groups):
        group["score"] = float(score[index])
    groups.sort(key=lambda group: -group["score"])
    for index, group in enumerate(groups, start=1):
        group["source_rank"] = index
    return groups


def _renderer(
    observation: Observation,
    parameter_space: ParameterSpace,
):
    return ImageLinearFit(
        ImageData(**observation.kwargs_data),
        PSF(**observation.kwargs_psf),
        LensModel(parameter_space.model["lens_model_list"]),
        LightModel(parameter_space.model["source_light_model_list"]),
        LightModel(parameter_space.model["lens_light_model_list"]),
        likelihood_mask=observation.likelihood_mask,
        kwargs_numerics=observation.numerics or None,
    )


def _refine_one(
    seed: dict[str, Any],
    base_proposal: dict[str, Any],
    observation: Observation,
    parameter_space: ParameterSpace,
    noise_map: np.ndarray,
    *,
    radius: float,
    iterations: int,
) -> dict[str, Any]:
    MassConcentrationNFW.configure(observation.z_lens, observation.z_source)
    model = copy.deepcopy(parameter_space.model)
    model["lens_model_list"] = list(model["lens_model_list"]) + ["BLANK_PLANE"]
    from lenstronomy.LensModel.Profiles import blank_plane

    blank_plane.BlankPlane = MassConcentrationNFW
    refined_space = replace(parameter_space, model=model)
    image_model = _renderer(observation.with_model(model), refined_space)

    def objective(values):
        center_x, center_y, log_mass = map(float, values)
        if abs(center_x - seed["ra"]) > radius or abs(center_y - seed["dec"]) > radius:
            return 1.0e30
        proposal = {
            "kwargs_lens": list(base_proposal["kwargs_lens"])
            + [
                {
                    "logM": float(np.clip(log_mass, 7.0, 11.0)),
                    "center_x": center_x,
                    "center_y": center_y,
                }
            ],
            "kwargs_lens_light": base_proposal["kwargs_lens_light"],
            "kwargs_source": base_proposal["kwargs_source"],
        }
        image, _, _, _ = image_model.image_linear_solve(**proposal)
        residual = (observation.image_data - image) / noise_map
        if observation.likelihood_mask is not None:
            residual = residual * observation.likelihood_mask
        return float(np.sum(residual**2))

    result = minimize(
        objective,
        [seed["ra"], seed["dec"], 9.0],
        method="Nelder-Mead",
        options={"maxiter": iterations, "xatol": 0.02, "fatol": 3.0},
    )
    center_x, center_y, log_mass = map(float, result.x)
    return {
        **copy.deepcopy(seed),
        "ra": center_x,
        "dec": center_y,
        "logM": log_mass,
        "local_chi_squared": float(result.fun),
        "refined": bool(result.fun < 1.0e30),
        "moved_arcsec": float(
            math.hypot(center_x - seed["ra"], center_y - seed["dec"])
        ),
    }


def _candidate_pool(
    ranked: Sequence[dict[str, Any]],
    refined: Sequence[dict[str, Any]],
    deduplication_radius: float,
) -> list[dict[str, Any]]:
    refined_by_rank = {
        int(candidate["source_rank"]): candidate for candidate in refined
    }
    variants = []
    for raw in ranked:
        rank = int(raw["source_rank"])
        sharpened = copy.deepcopy(refined_by_rank.get(rank, raw))
        sharpened.update(
            {
                "blob_id": rank,
                "coordinate_variant": "refined",
                "is_refined_coordinate": rank in refined_by_rank,
            }
        )
        variants.append(sharpened)
    for raw in ranked:
        rank = int(raw["source_rank"])
        variants.append(
            {
                **copy.deepcopy(raw),
                "blob_id": rank,
                "coordinate_variant": "raw",
                "is_refined_coordinate": False,
                "refined": False,
            }
        )

    selected = []
    for candidate in variants:
        same_identity = [
            prior for prior in selected if prior["blob_id"] == candidate["blob_id"]
        ]
        if all(
            math.hypot(
                candidate["ra"] - prior["ra"],
                candidate["dec"] - prior["dec"],
            )
            > deduplication_radius
            for prior in same_identity
        ):
            candidate["candidate_id"] = len(selected)
            selected.append(candidate)
    return selected


def identify_candidates(
    pull: np.ndarray,
    base_evaluation: dict[str, Any],
    observation: Observation,
    parameter_space: ParameterSpace,
    config: CandidateIdentificationConfig,
) -> CandidateIdentification:
    """Build the refined-plus-raw candidate pool."""
    solved = base_evaluation["proposal"]
    theta_e, _ = effective_einstein_radius(solved, parameter_space)
    arc_rms, arc_pixels = _arc_rms(pull, theta_e, observation.pixel_scale)
    initial = _connected_candidates(
        pull,
        observation,
        theta_e,
        threshold=config.peak_threshold,
    )
    if arc_rms < config.arc_rms_threshold:
        return CandidateIdentification(
            detected=False,
            arc_rms=arc_rms,
            arc_pixels=arc_pixels,
            initial_count=len(initial),
            ranked=(),
            candidate_pool=(),
        )

    model_image = np.asarray(base_evaluation["model_image"], dtype=float)
    noise_map = np.sqrt(
        observation.background_rms**2
        + np.maximum(model_image, 0.0) / observation.exposure_time
    )
    groups = _group_candidates(initial, theta_e)
    ranked = _rank_groups(
        groups,
        pull,
        model_image,
        noise_map,
        observation,
        parameter_space,
        solved,
    )[: config.candidate_limit]
    refined = [
        _refine_one(
            seed,
            solved,
            observation,
            parameter_space,
            np.asarray(observation.background_rms, dtype=float),
            radius=config.refinement_radius_arcsec,
            iterations=config.refinement_iterations,
        )
        for seed in ranked[: config.refinement_limit]
    ]
    pool = _candidate_pool(ranked, refined, config.deduplication_radius_arcsec)
    return CandidateIdentification(
        detected=True,
        arc_rms=arc_rms,
        arc_pixels=arc_pixels,
        initial_count=len(initial),
        ranked=tuple(ranked),
        candidate_pool=tuple(pool),
    )
