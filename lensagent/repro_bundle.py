import json
import os
import pickle
import platform
import sys
from typing import Any, Dict, Optional

import numpy as np


_ARRAY_LABELS = {
    "image_data": {
        "kind": "observation",
        "description": "Processed observed image cutout used in fitting.",
        "units": "same image units as lenstronomy fit data; for current SDSS corrected frames this is nanomaggies/pixel after the cutout pipeline background subtraction",
    },
    "background_rms": {
        "kind": "noise",
        "description": "Background RMS term used in the likelihood.",
        "units": "same units as image_data",
    },
    "exposure_time": {
        "kind": "noise",
        "description": "Effective exposure term used for the Poisson variance.",
        "units": "lenstronomy exposure-map units satisfying counts = image_data * exposure_time; for SDSS calibrated frames this is not literal wall-clock seconds",
    },
    "transform_pix2angle": {
        "kind": "geometry",
        "description": "2x2 pixel-to-angle transform matrix.",
        "units": "arcsec per pixel",
    },
    "kernel_point_source": {
        "kind": "psf",
        "description": "Pixelized PSF kernel used for convolution.",
        "units": "normalized kernel",
    },
    "likelihood_mask": {
        "kind": "mask",
        "description": "Likelihood mask: 1 means included in fit, 0 means ignored.",
        "units": "dimensionless",
    },
    "model_image": {
        "kind": "model",
        "description": "Best-fit model image rendered on the data grid.",
        "units": "same units as image_data",
    },
    "lens_light_image": {
        "kind": "model",
        "description": "Best-fit lens-light component image.",
        "units": "same units as image_data",
    },
    "residual_map": {
        "kind": "diagnostic",
        "description": "Normalized residual map returned by evaluation.",
        "units": "dimensionless; approximately sigma units",
    },
    "pull_map": {
        "kind": "diagnostic",
        "description": "RSI pull/significance map used for candidate detection.",
        "units": "sigma",
    },
}


def _json_default(obj: Any):
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def _array_shape(value: Any):
    arr = np.asarray(value)
    return list(arr.shape)


def _add_array(arrays: Dict[str, np.ndarray], key: str, value: Any):
    if value is None:
        return
    arrays[key] = np.asarray(value)


def _array_map(arrays: Dict[str, np.ndarray]) -> dict:
    mapping = {
        "how_to_load": {
            "npz": "Use numpy.load('repro_arrays.npz') and access arrays by the keys listed below.",
            "pkl": "Use observation_bundle.pkl for the exact serialized ObservationBundle.",
        },
        "arrays": {},
    }
    for key, value in arrays.items():
        meta = dict(_ARRAY_LABELS.get(key, {}))
        if not meta:
            meta = {
                "kind": "extra",
                "description": "Additional array saved for reproducibility.",
                "units": "see repro_manifest.json extra_metadata",
            }
        meta["dtype"] = str(np.asarray(value).dtype)
        meta["shape"] = _array_shape(value)
        mapping["arrays"][key] = meta
    return mapping


def _eval_summary(eval_results: Optional[dict]) -> dict:
    if not eval_results:
        return {}
    keys = [
        "image_chi2",
        "image_chi2_reduced",
        "n_pixels",
        "sigma_predicted",
        "sigma_observed",
        "sigma_observed_err",
        "kin_chi2",
        "quality",
        "delta_bic",
        "is_physical",
        "rmse_poisson",
        "is_significant",
        "ranking",
    ]
    return {k: eval_results.get(k) for k in keys if k in eval_results}


def _versions() -> dict:
    versions = {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": platform.platform(),
    }
    try:
        import lenstronomy
        versions["lenstronomy"] = getattr(lenstronomy, "__version__", "unknown")
    except Exception:
        versions["lenstronomy"] = None
    return versions


def _scoring_context() -> dict:
    try:
        from . import scoring as S
        return {
            "chi2_penalty_mode": S.CHI2_PENALTY,
            "subtracted_chi2": S.SUBTRACTED_CHI2,
            "no_linear_solve": S.NO_LINEAR_SOLVE,
            "kin_soft": S.KIN_SOFT,
            "physicality_mode": S.PHYSICALITY_MODE,
            "alpha": S.ALPHA,
            "beta": S.BETA,
            "gamma": S.GAMMA,
            "delta": S.DELTA,
            "alpha_prl": S.ALPHA_PRL,
            "beta_prl": S.BETA_PRL,
            "gamma_prl": S.GAMMA_PRL,
            "delta_prl": S.DELTA_PRL,
            "chi2_fine_band": S.CHI2_FINE_BAND,
            "chi2_fine_boost": S.CHI2_FINE_BOOST,
            "chi2_ultra_fine_band": S.CHI2_ULTRA_FINE_BAND,
            "chi2_ultra_fine_boost": S.CHI2_ULTRA_FINE_BOOST,
        }
    except Exception:
        return {}


def save_repro_bundle(
    output_dir: str,
    obs,
    *,
    stage: str,
    proposal: Optional[dict] = None,
    model: Optional[dict] = None,
    eval_results: Optional[dict] = None,
    extra_arrays: Optional[dict] = None,
    extra_metadata: Optional[dict] = None,
) -> None:
    """Write a self-contained bundle for future figure regeneration.

    The bundle is intentionally additive: it does not alter existing outputs.
    It stores:
      - the exact observation bundle pickle
      - compressed arrays needed to redraw figures without re-running fits
      - a JSON manifest with proposal/model/score metadata
    """
    os.makedirs(output_dir, exist_ok=True)

    arrays: Dict[str, np.ndarray] = {}
    kwargs_data = obs.kwargs_data_joint["multi_band_list"][0][0]
    kwargs_psf = obs.kwargs_data_joint["multi_band_list"][0][1]

    _add_array(arrays, "image_data", kwargs_data.get("image_data"))
    _add_array(arrays, "background_rms", kwargs_data.get("background_rms"))
    _add_array(arrays, "exposure_time", kwargs_data.get("exposure_time"))
    _add_array(arrays, "transform_pix2angle", kwargs_data.get("transform_pix2angle"))
    _add_array(arrays, "kernel_point_source", kwargs_psf.get("kernel_point_source"))
    _add_array(arrays, "likelihood_mask", getattr(obs, "likelihood_mask", None))

    if eval_results:
        _add_array(arrays, "model_image", eval_results.get("model_image"))
        _add_array(arrays, "lens_light_image", eval_results.get("lens_light_image"))
        _add_array(arrays, "residual_map", eval_results.get("residual_map"))

    if extra_arrays:
        for key, value in extra_arrays.items():
            _add_array(arrays, key, value)

    arrays_path = os.path.join(output_dir, "repro_arrays.npz")
    np.savez_compressed(arrays_path, **arrays)

    array_map_path = os.path.join(output_dir, "repro_arrays_map.json")
    with open(array_map_path, "w") as f:
        json.dump(_array_map(arrays), f, indent=2, default=_json_default)

    obs_path = os.path.join(output_dir, "observation_bundle.pkl")
    with open(obs_path, "wb") as f:
        pickle.dump(obs, f, protocol=pickle.HIGHEST_PROTOCOL)

    manifest = {
        "stage": stage,
        "sdss_name": getattr(obs, "sdss_name", ""),
        "ra_deg": getattr(obs, "ra_deg", None),
        "dec_deg": getattr(obs, "dec_deg", None),
        "pixel_scale": getattr(obs, "pixel_scale", None),
        "z_lens": getattr(obs, "z_lens", None),
        "z_source": getattr(obs, "z_source", None),
        "sigma_obs": getattr(obs, "sigma_obs", None),
        "sigma_obs_err": getattr(obs, "sigma_obs_err", None),
        "model": model if model is not None else getattr(obs, "kwargs_model", None),
        "proposal": proposal,
        "eval_summary": _eval_summary(eval_results),
        "kwargs_data_meta": {
            "ra_at_xy_0": kwargs_data.get("ra_at_xy_0"),
            "dec_at_xy_0": kwargs_data.get("dec_at_xy_0"),
        },
        "kwargs_psf_meta": {
            "psf_type": kwargs_psf.get("psf_type"),
            "pixel_size": kwargs_psf.get("pixel_size"),
        },
        "array_shapes": {k: _array_shape(v) for k, v in arrays.items()},
        "scoring": _scoring_context(),
        "versions": _versions(),
        "files": {
            "arrays": os.path.basename(arrays_path),
            "array_map": os.path.basename(array_map_path),
            "observation_bundle": os.path.basename(obs_path),
        },
        "extra_metadata": extra_metadata or {},
    }

    manifest_path = os.path.join(output_dir, "repro_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=_json_default)
