"""Image, kinematic, and physical evaluation of lens proposals."""

from __future__ import annotations

import logging
import os
from typing import Any

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np
from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.Data.psf import PSF
from lenstronomy.ImSim.image_linear_solve import ImageLinearFit
from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.Sampling.parameters import Param

from lensagent.data.observation import Observation
from lensagent.modeling.kinematics import KinematicsAPI
from lensagent.modeling.parameters import ParameterSpace, pack_multi_gaussian_components

log = logging.getLogger(__name__)

INVALID_KINEMATIC_CHI_SQUARED = 1.0e6
PHYSICALITY_GRID_PIXELS = 120
PHYSICALITY_GRID_SCALE_ARCSEC = 0.4


def total_model_chi_squared(
    observed: np.ndarray,
    model_image: np.ndarray,
    background_rms: np.ndarray,
    exposure_time: np.ndarray,
    likelihood_mask: np.ndarray | None = None,
    noise_map: np.ndarray | None = None,
) -> tuple[float, np.ndarray, int]:
    """Calculate raw image chi-squared with model Poisson variance."""
    observed = np.asarray(observed, dtype=float)
    model_image = np.asarray(model_image, dtype=float)
    background_rms = np.asarray(background_rms, dtype=float)
    exposure_time = np.asarray(exposure_time, dtype=float)
    if np.any(exposure_time <= 0) or np.any(background_rms <= 0):
        raise ValueError("background RMS and exposure time must be positive")

    residual = observed - model_image
    variance = (np.asarray(noise_map, dtype=float)**2 if noise_map is not None else
                background_rms**2 + np.maximum(model_image, 0.0) / exposure_time)
    normalized = residual / np.sqrt(variance)
    if likelihood_mask is None:
        fitted_pixels = int(residual.size)
        chi_squared = float(np.sum(normalized**2))
    else:
        mask = np.asarray(likelihood_mask, dtype=float)
        if mask.shape != residual.shape:
            raise ValueError("likelihood mask must match the image shape")
        fitted_pixels = int(np.sum(mask))
        chi_squared = float(np.sum(normalized**2 * mask))
        normalized = normalized * mask
    if fitted_pixels <= 0:
        raise ValueError("likelihood mask contains no fitted pixels")
    return chi_squared, normalized, fitted_pixels


def image_parameter_count(
    parameter_space: ParameterSpace,
    image_model: ImageLinearFit,
    kwargs_lens: list[dict[str, Any]],
    kwargs_source: list[dict[str, Any]],
    kwargs_lens_light: list[dict[str, Any]],
    *,
    solve_linear: bool,
    expanded_mass: bool = False,
) -> tuple[int, int]:
    constraints = parameter_space.lenstronomy_constraints
    counting_fixed = parameter_space.fixed_for_parameter_count
    count_model = parameter_space.model
    if expanded_mass and parameter_space.pso_proxy_lens_models:
        from lensagent.workflow.optimizer import optimizer_space
        from dataclasses import replace

        counted = optimizer_space(replace(
            parameter_space, fixed_lens=counting_fixed["kwargs_lens"],
        ))
        count_model = counted.model
        counting_fixed = {**counting_fixed, "kwargs_lens": counted.fixed_lens}
    parameters = Param(
        count_model,
        kwargs_fixed_lens=list(counting_fixed["kwargs_lens"]),
        kwargs_fixed_source=list(counting_fixed["kwargs_source"]),
        kwargs_fixed_lens_light=list(counting_fixed["kwargs_lens_light"]),
        joint_source_with_source=constraints["joint_source_with_source"],
        joint_lens_with_lens=constraints["joint_lens_with_lens"],
        linear_solver=solve_linear,
    )
    nonlinear, _ = parameters.num_param()
    linear = (
        image_model.num_param_linear(kwargs_lens, kwargs_source, kwargs_lens_light, [])
        if solve_linear
        else 0
    )
    return int(nonlinear), int(linear)


def evaluate_proposal(
    proposal: dict[str, Any],
    observation: Observation,
    parameter_space: ParameterSpace,
    *,
    include_kinematics: bool = True,
    solve_linear: bool = True,
    kwargs_anisotropy: dict[str, Any] | None = None,
    kwargs_aperture: dict[str, Any] | None = None,
    kwargs_seeing: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Fit linear amplitudes and evaluate one complete lens proposal."""
    if parameter_space.model != observation.model:
        raise ValueError("parameter space does not match the observation model")

    proposal = pack_multi_gaussian_components(proposal, parameter_space.model)
    kwargs_source = parameter_space.materialize_source_ties(proposal["kwargs_source"])
    kwargs_lens_light = [dict(component) for component in proposal["kwargs_lens_light"]]

    data = ImageData(**observation.kwargs_data)
    psf = PSF(**observation.kwargs_psf)
    lens_model = LensModel(parameter_space.model["lens_model_list"])
    source_model = LightModel(parameter_space.model["source_light_model_list"])
    lens_light_model = LightModel(parameter_space.model["lens_light_model_list"])
    image_model = ImageLinearFit(
        data,
        psf,
        lens_model,
        source_model,
        lens_light_model,
        likelihood_mask=observation.likelihood_mask,
        kwargs_numerics=observation.numerics or None,
    )

    active_image_model = image_model
    if solve_linear:
        model_image, _, _, linear_parameters = image_model.image_linear_solve(
            kwargs_lens=proposal["kwargs_lens"],
            kwargs_source=kwargs_source,
            kwargs_lens_light=kwargs_lens_light,
        )
        if linear_parameters is not None:
            _, kwargs_source, kwargs_lens_light, _ = image_model.update_linear_kwargs(
                linear_parameters,
                proposal["kwargs_lens"],
                kwargs_source,
                kwargs_lens_light,
                [],
            )

        if (
            linear_parameters is not None
            and len(linear_parameters) > 0
            and np.all(linear_parameters == 0)
        ):
            base_light_names = [
                name
                for name in parameter_space.model["lens_light_model_list"]
                if "SHAPELETS" not in name
            ]
            base_light = LightModel(base_light_names)
            active_image_model = ImageLinearFit(
                data,
                psf,
                lens_model,
                source_model,
                base_light,
                likelihood_mask=observation.likelihood_mask,
                kwargs_numerics=observation.numerics or None,
            )
            kwargs_lens_light_base = kwargs_lens_light[: len(base_light_names)]
            model_image, _, _, linear_parameters = (
                active_image_model.image_linear_solve(
                    kwargs_lens=proposal["kwargs_lens"],
                    kwargs_source=kwargs_source,
                    kwargs_lens_light=kwargs_lens_light_base,
                )
            )
            if linear_parameters is not None:
                _, kwargs_source, kwargs_lens_light_base, _ = (
                    active_image_model.update_linear_kwargs(
                        linear_parameters,
                        proposal["kwargs_lens"],
                        kwargs_source,
                        kwargs_lens_light_base,
                        [],
                    )
                )
            kwargs_lens_light = (
                list(kwargs_lens_light_base)
                + kwargs_lens_light[len(base_light_names) :]
            )
    else:
        model_image = image_model.image(
            kwargs_lens=proposal["kwargs_lens"],
            kwargs_source=kwargs_source,
            kwargs_lens_light=kwargs_lens_light,
        )

    solved_proposal = {
        **proposal,
        "kwargs_source": kwargs_source,
        "kwargs_lens_light": kwargs_lens_light,
    }
    lens_light_image = image_model.lens_surface_brightness(
        kwargs_lens_light, unconvolved=False
    )
    chi_squared, residual_map, fitted_pixels = total_model_chi_squared(
        data.data,
        model_image,
        data.background_rms,
        data.exposure_map,
        observation.likelihood_mask,
        observation.noise_map,
    )
    nonlinear, linear = image_parameter_count(
        parameter_space,
        active_image_model,
        proposal["kwargs_lens"],
        kwargs_source,
        kwargs_lens_light,
        solve_linear=solve_linear,
        expanded_mass=observation.hst,
    )
    parameter_count = nonlinear + linear
    degrees_of_freedom = fitted_pixels - parameter_count
    if degrees_of_freedom <= 0:
        raise ValueError(
            f"non-positive image degrees of freedom: N={fitted_pixels}, k={parameter_count}"
        )

    result: dict[str, Any] = {
        "proposal": solved_proposal,
        "model_image": model_image,
        "lens_light_image": lens_light_image,
        "residual_map": residual_map,
        "image_chi_squared": chi_squared,
        "reduced_image_chi_squared": float(chi_squared / degrees_of_freedom),
        "fitted_pixels": fitted_pixels,
        "parameter_count": parameter_count,
        "nonlinear_parameter_count": nonlinear,
        "linear_parameter_count": linear,
        "degrees_of_freedom": degrees_of_freedom,
        "bic": bayesian_information_criterion(
            chi_squared, parameter_count, fitted_pixels
        ),
    }

    if include_kinematics and observation.sigma_obs > 0:
        if not _has_deflector_light(kwargs_lens_light, lens_light_image):
            kinematics = _invalid_kinematics(observation, "zero_deflector_light")
        else:
            kinematics = evaluate_kinematics(
                solved_proposal,
                observation,
                kwargs_anisotropy=kwargs_anisotropy,
                kwargs_aperture=kwargs_aperture,
                kwargs_seeing=kwargs_seeing,
            )
        result.update(kinematics)
        result["total_log_likelihood"] = -0.5 * (
            chi_squared + result["kinematic_chi_squared"]
        )
    else:
        result["total_log_likelihood"] = -0.5 * chi_squared

    try:
        result.update(
            evaluate_physicality(proposal["kwargs_lens"], parameter_space.model)
        )
    except Exception as exc:
        log.debug("physicality evaluation failed: %s", exc)
        result.update({"is_physical": None, "poisson_rmse": None})
    if observation.hst:
        from lensagent.modeling.constraints import MacroFloor

        result["macro_floor_ok"] = MacroFloor.from_model(parameter_space.model).accepts(proposal["kwargs_lens"])
        if not result["macro_floor_ok"]:
            result["is_physical"] = False
    return result


def bayesian_information_criterion(
    chi_squared: float, parameter_count: int, fitted_pixels: int
) -> float:
    if fitted_pixels <= 0 or parameter_count < 0:
        raise ValueError(
            "BIC requires positive pixels and a non-negative parameter count"
        )
    return float(chi_squared + parameter_count * np.log(fitted_pixels))


def delta_bic(base_bic: float, subhalo_bic: float) -> float:
    """Return positive values when the subhalo model is preferred."""
    return float(base_bic - subhalo_bic)


def evaluate_physicality(
    kwargs_lens: list[dict[str, Any]], model: dict[str, list[str]]
) -> dict[str, Any]:
    """Check Poisson consistency and negative convergence on a fixed grid."""
    from lenstronomy.Util import util

    lens_model = LensModel(lens_model_list=model["lens_model_list"])
    x_grid, y_grid = util.make_grid(
        PHYSICALITY_GRID_PIXELS, PHYSICALITY_GRID_SCALE_ARCSEC
    )
    potential = util.array2image(lens_model.potential(x_grid, y_grid, kwargs_lens))
    convergence_vector = lens_model.kappa(x_grid, y_grid, kwargs_lens)
    convergence = util.array2image(convergence_vector)

    dy, dx = np.gradient(potential, PHYSICALITY_GRID_SCALE_ARCSEC)
    _, dxx = np.gradient(dx, PHYSICALITY_GRID_SCALE_ARCSEC)
    dyy, _ = np.gradient(dy, PHYSICALITY_GRID_SCALE_ARCSEC)
    poisson_rmse = float(np.sqrt(np.mean((convergence - 0.5 * (dxx + dyy)) ** 2)))
    minimum_convergence = float(np.min(convergence))
    negative_mass_fraction = float(np.mean(convergence_vector < -0.05))
    is_physical = (
        poisson_rmse <= 0.05
        and minimum_convergence >= -0.2
        and negative_mass_fraction <= 0.05
    )
    return {
        "is_physical": is_physical,
        "physicality_score": -poisson_rmse,
        "poisson_rmse": poisson_rmse,
        "minimum_convergence": minimum_convergence,
        "negative_mass_fraction": negative_mass_fraction,
    }


def _amplitude_sum(value: Any) -> float:
    if value is None:
        return 0.0
    array = np.asarray(value, dtype=float)
    if array.size == 0 or not np.all(np.isfinite(array)):
        return 0.0
    return float(np.sum(np.abs(array)))


def _has_deflector_light(
    kwargs_lens_light: list[dict[str, Any]], lens_light_image: np.ndarray
) -> bool:
    amplitude = sum(
        _amplitude_sum(component.get("amp")) for component in kwargs_lens_light
    )
    if amplitude > 1.0e-12:
        return True
    image = np.asarray(lens_light_image, dtype=float)
    return bool(
        image.size and np.all(np.isfinite(image)) and np.sum(np.abs(image)) > 1.0e-12
    )


def _invalid_kinematics(observation: Observation, reason: str) -> dict[str, Any]:
    return {
        "sigma_predicted": None,
        "sigma_observed": observation.sigma_obs,
        "sigma_observed_err": observation.sigma_obs_err,
        "kinematic_chi_squared": INVALID_KINEMATIC_CHI_SQUARED,
        "kinematic_log_likelihood": -0.5 * INVALID_KINEMATIC_CHI_SQUARED,
        "kinematic_failure_reason": reason,
    }


def evaluate_kinematics(
    proposal: dict[str, Any],
    observation: Observation,
    *,
    kwargs_anisotropy: dict[str, Any] | None = None,
    kwargs_aperture: dict[str, Any] | None = None,
    kwargs_seeing: dict[str, Any] | None = None,
) -> dict[str, Any]:
    kwargs_anisotropy = kwargs_anisotropy or {}
    kwargs_aperture = kwargs_aperture or {
        "aperture_type": "slit",
        "length": 2.66,
        "width": 2.66,
        "angle": 0.0,
    }
    kwargs_seeing = kwargs_seeing or {
        "psf_type": "MOFFAT",
        "fwhm": 1.0,
        "moffat_beta": 2.5,
    }

    lens_center = proposal["kwargs_lens"][0]
    light_center = proposal["kwargs_lens_light"][0]
    kwargs_mge_light = {
        "grid_spacing": 1,
        "grid_num": 100,
        "n_comp": 20,
        "center_x": light_center.get("center_x", 0.0),
        "center_y": light_center.get("center_y", 0.0),
    }
    kwargs_mge_mass = {
        "grid_spacing": 1,
        "grid_num": 100,
        "n_comp": 20,
        "center_x": lens_center.get("center_x", 0.0),
        "center_y": lens_center.get("center_y", 0.0),
    }
    if observation.hst:
        kwargs_mge_mass["num_azimuthal_points"] = 1280
    lens_light_models = observation.model.get("lens_light_model_list", [])
    native_mge_light = bool(lens_light_models) and set(lens_light_models) <= {
        "MULTI_GAUSSIAN",
        "MULTI_GAUSSIAN_ELLIPSE",
    }
    kinematics = KinematicsAPI(
        observation.z_lens,
        observation.z_source,
        observation.model,
        kwargs_aperture,
        kwargs_seeing,
        "isotropic",
        MGE_light=not native_mge_light,
        MGE_mass=True,
        kwargs_mge_light=kwargs_mge_light,
        kwargs_mge_mass=kwargs_mge_mass,
    )
    effective_radius = proposal["kwargs_lens_light"][0].get("R_sersic")
    try:
        predicted = kinematics.velocity_dispersion(
            proposal["kwargs_lens"],
            proposal["kwargs_lens_light"],
            kwargs_anisotropy,
            r_eff=effective_radius,
        )
        sigma_predicted = float(np.ravel(predicted)[0])
        if not np.isfinite(sigma_predicted):
            raise ValueError("non-finite velocity dispersion")
    except Exception as exc:
        log.warning("kinematic solver failed: %s: %s", type(exc).__name__, exc)
        return _invalid_kinematics(observation, f"solver_error:{type(exc).__name__}")

    chi_squared = (
        sigma_predicted - observation.sigma_obs
    ) ** 2 / observation.sigma_obs_err**2
    return {
        "sigma_predicted": sigma_predicted,
        "sigma_observed": observation.sigma_obs,
        "sigma_observed_err": observation.sigma_obs_err,
        "kinematic_chi_squared": float(chi_squared),
        "kinematic_log_likelihood": -0.5 * float(chi_squared),
    }
