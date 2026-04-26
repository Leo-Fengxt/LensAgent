"""Subhalo detection and fitting for the RSI phase.

Implements the exp's Subhalo_detection.ipynb pipeline:
1. Compute pull map from PRL (AFMS+PRL) residuals
2. Detect candidate locations via blob_log
3. Evaluate NFW subhalo models appended to PRL params
"""

import copy
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

DEFAULT_MAX_SUBHALO_MASS_MSUN = 1.0e10
INVALID_SUBHALO_MASS_CHI2 = 1.0e6


def compute_pull_map(
    proposal: Dict[str, Any],
    obs,
    timeout_s: int = 30,
    legacy: bool = False,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[Dict]]:
    """Evaluate an AFMS/PRL proposal and return its pull map.

    The pull map is (data - model) / noise, giving per-pixel significance.

    When *legacy=True*, uses the original heuristic noise estimate
    (sqrt of reduced chi2) for backward compatibility with earlier runs.
    When *legacy=False* (default), uses lenstronomy's properly normalized
    residual map which divides by the per-pixel noise model.

    Returns (pull_map, model_image, eval_results) or (None, None, None)
    on failure.
    """
    from .safe_eval import safe_evaluate
    from . import scoring as _S

    result, err = safe_evaluate(
        proposal, obs, include_kinematics=True,
        subtracted_chi2=_S.SUBTRACTED_CHI2,
        no_linear_solve=_S.NO_LINEAR_SOLVE,
        timeout_s=timeout_s)
    if result is None:
        log.warning("AFMS/PRL evaluation failed: %s", err)
        return None, None, None

    model_image = result.get("model_image")
    if model_image is None:
        return None, None, None
    if not isinstance(model_image, np.ndarray):
        model_image = np.array(model_image)

    if legacy:
        observed = obs.image_data
        residual = observed - model_image
        chi2_reduced = result.get("image_chi2_reduced", 1.0)
        noise_estimate = np.sqrt(max(chi2_reduced, 1e-10))
        pull_map = residual / noise_estimate
    else:
        residual_map = result.get("residual_map")
        if residual_map is None:
            observed = obs.image_data
            residual = observed - model_image
            chi2_reduced = result.get("image_chi2_reduced", 1.0)
            noise_estimate = np.sqrt(max(chi2_reduced, 1e-10))
            pull_map = residual / noise_estimate
        else:
            if not isinstance(residual_map, np.ndarray):
                residual_map = np.array(residual_map)
            pull_map = -residual_map

    return pull_map, model_image, result


def detect_candidates(
    pull_map: np.ndarray,
    obs,
    threshold: float = 5.0,  # exp default
) -> List[Dict[str, Any]]:
    """Detect subhalo candidates via Laplacian-of-Gaussian blob detection.

    Returns list of dicts with keys: ra, dec, pull, pix_coord, sigma_blob.
    """
    from skimage.feature import blob_log

    blobs = blob_log(
        np.abs(pull_map),
        min_sigma=0.8,
        max_sigma=3,
        threshold=threshold / 10,
    )

    data = obs.kwargs_data_joint['multi_band_list'][0][0]
    transform = data['transform_pix2angle']
    ra0 = data['ra_at_xy_0']
    dec0 = data['dec_at_xy_0']

    candidates = []
    for blob in blobs:
        y_pix, x_pix, sigma_blob = blob
        ra = ra0 + transform[0, 0] * x_pix + transform[0, 1] * y_pix
        dec = dec0 + transform[1, 0] * x_pix + transform[1, 1] * y_pix
        yi, xi = int(y_pix), int(x_pix)
        if 0 <= yi < pull_map.shape[0] and 0 <= xi < pull_map.shape[1]:
            pull_val = float(pull_map[yi, xi])
        else:
            pull_val = 0.0

        candidates.append({
            "ra": float(ra),
            "dec": float(dec),
            "pull": pull_val,
            "abs_pull": abs(pull_val),
            "pix_coord": (float(x_pix), float(y_pix)),
            "sigma_blob": float(sigma_blob),
        })

    candidates.sort(key=lambda c: c["abs_pull"], reverse=True)
    log.info("Detected %d blob candidates via blob_log on abs(pull_map) "
             "(min_sigma=0.8, max_sigma=3, threshold=%.1f/10)",
             len(candidates), threshold)
    for i, c in enumerate(candidates):
        log.info("  Candidate %d: ra=%.4f dec=%.4f  pull=%.1f sigma",
                 i, c["ra"], c["dec"], c["pull"])
    return candidates


def build_subhalo_model(
    base_proposal: Dict[str, Any],
    obs,
    candidates: List[Dict[str, Any]],
    n_subhalos: int = 1,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Build kwargs_model and kwargs_params for subhalo fitting.

    Matches exp's Subhalo_detection notebook: copies existing
    kwargs_params (base stays FREE) and appends NFW params.

    Returns (kwargs_model, kwargs_params).
    """
    km = copy.deepcopy(obs.kwargs_model)
    for _ in range(n_subhalos):
        km['lens_model_list'].append('NFW')

    from . import scoring as _S
    base = copy.deepcopy(_S.EXP_KWARGS_PARAMS_BASE)

    init_lens, sigma_lens, fixed_lens, lower_lens, upper_lens = base["lens_model"]

    for i in range(n_subhalos):
        cand = candidates[i] if i < len(candidates) else candidates[0]
        ra_c, dec_c = cand["ra"], cand["dec"]

        init_lens.append({
            "Rs": 0.05, "alpha_Rs": 0.01,
            "center_x": ra_c, "center_y": dec_c,
        })
        sigma_lens.append({
            "Rs": 0.02, "alpha_Rs": 0.01,
            "center_x": 0.02, "center_y": 0.02,
        })
        fixed_lens.append({})
        lower_lens.append({
            "Rs": 0.001, "alpha_Rs": 0.0001,
            "center_x": ra_c - 0.1, "center_y": dec_c - 0.1,
        })
        upper_lens.append({
            "Rs": 0.5, "alpha_Rs": 0.5,
            "center_x": ra_c + 0.1, "center_y": dec_c + 0.1,
        })

    kwargs_params = {
        "lens_model": [init_lens, sigma_lens, fixed_lens,
                       lower_lens, upper_lens],
        "source_model": base["source_model"],
        "lens_light_model": base["lens_light_model"],
    }

    return km, kwargs_params


def _effective_image_chi2_reduced_for_bic(
    chi2_reduced: float,
    *,
    raw_bic: bool = False,
) -> float:
    """Return the reduced image chi2 used to build the RSI image logL."""
    chi2_r = float(chi2_reduced)
    if raw_bic or chi2_r >= 1.0:
        return chi2_r
    return 2.0 - chi2_r


def image_log_likelihood_for_bic(
    chi2_reduced: float,
    n_pixels: int,
    *,
    raw_bic: bool = False,
) -> float:
    """Return the image-only logL used by the RSI delta-BIC calculation.

    Default behavior mirrors reduced chi2 values below 1.0 around the optimum
    at 1.0 before converting to image log-likelihood. This preserves the
    exp's ``2 * (logL_new - logL_old) - Delta(k) * log(n)`` structure while
    making "closer to chi2=1" the preferred direction on both sides of 1.
    """
    n = max(int(n_pixels or 0), 1)
    chi2_eff_r = _effective_image_chi2_reduced_for_bic(
        chi2_reduced,
        raw_bic=raw_bic,
    )
    return float(-0.5 * chi2_eff_r * n)


def compute_delta_bic_from_reduced_chi2(
    base_chi2_reduced: float,
    new_chi2_reduced: float,
    n_pixels: int,
    n_original_params: int,
    n_new_params: int,
    *,
    raw_bic: bool = False,
) -> Dict[str, float]:
    """Compute RSI delta-BIC in the same explicit structure as the notebook."""
    import math as _math

    n = max(int(n_pixels or 0), 1)
    logL_original = image_log_likelihood_for_bic(
        base_chi2_reduced,
        n,
        raw_bic=raw_bic,
    )
    logL_new_model = image_log_likelihood_for_bic(
        new_chi2_reduced,
        n,
        raw_bic=raw_bic,
    )
    bic_original = float(-2.0 * logL_original + n_original_params * _math.log(n))
    bic_new = float(-2.0 * logL_new_model + (n_original_params + n_new_params) * _math.log(n))
    delta_bic = float(bic_original - bic_new)
    delta_fit = float(2.0 * (logL_new_model - logL_original))
    param_penalty = float(n_new_params * _math.log(n))
    return {
        "logL_original": float(logL_original),
        "logL_new_model": float(logL_new_model),
        "bic_original": float(bic_original),
        "bic_new": float(bic_new),
        "num_data_points": float(n),
        "num_params_original": int(n_original_params),
        "num_params_new_model": int(n_original_params + n_new_params),
        "num_new_params": int(n_new_params),
        "delta_fit": float(delta_fit),
        "param_penalty": float(param_penalty),
        "delta_bic": float(delta_bic),
        "n_pixels": float(n),
    }


def evaluate_subhalo(
    subhalo_params: List[Dict[str, Any]],
    base_proposal: Dict[str, Any],
    obs,
    base_logL: float,
    n_base_params: int,
    timeout_s: int = 30,
    raw_bic: bool = False,
) -> Optional[Dict[str, Any]]:
    """Evaluate a subhalo proposal (NFW appended to PRL model).

    Returns dict with chi2, delta_bic, mass_msun, subhalo_params, or None
    on failure.

    Default uses the notebook's explicit BIC comparison:
    ``bic_original - bic_new``, where both BIC terms are built from image-only
    log-likelihoods. Reduced chi2 values below 1.0 are mirrored around 1.0
    before constructing those image log-likelihoods, so moves toward chi2=1 are
    rewarded on either side of 1.

    When *raw_bic=True*, the image log-likelihood uses the raw reduced chi2
    directly instead of the mirrored target-1 version.
    """
    from .safe_eval import safe_evaluate

    proposal = copy.deepcopy(base_proposal)
    proposal['kwargs_lens'] = list(proposal['kwargs_lens']) + list(subhalo_params)

    from . import scoring as _S
    result, err = safe_evaluate(
        proposal, obs, include_kinematics=True,
        subtracted_chi2=_S.SUBTRACTED_CHI2,
        no_linear_solve=_S.NO_LINEAR_SOLVE,
        timeout_s=timeout_s)
    if result is None:
        return None

    chi2 = result.get("image_chi2_reduced", 1e6)
    image_chi2 = result.get("image_chi2", 0)
    n_pixels = result.get("n_pixels", 1)

    n_sub_params = 4 * len(subhalo_params)
    base_chi2_r = abs(base_logL) * 2 / max(n_pixels, 1) if n_pixels > 0 else 1e6
    bic_metrics = compute_delta_bic_from_reduced_chi2(
        base_chi2_r,
        chi2,
        n_pixels,
        n_base_params,
        n_sub_params,
        raw_bic=raw_bic,
    )

    masses = []
    for sp in subhalo_params:
        mass = _compute_nfw_mass(
            sp.get("Rs", 0), sp.get("alpha_Rs", 0),
            obs.z_lens, obs.z_source)
        masses.append(mass)

    out = {
        "chi2_reduced": chi2,
        "image_chi2": image_chi2,
        "image_chi2_reduced": chi2,
        "delta_fit": float(bic_metrics["delta_fit"]),
        "delta_bic": float(bic_metrics["delta_bic"]),
        "param_penalty": float(bic_metrics["param_penalty"]),
        "bic_n_pixels": int(bic_metrics["n_pixels"]),
        "num_data_points": int(bic_metrics["num_data_points"]),
        "num_params_original": int(bic_metrics["num_params_original"]),
        "num_params_new_model": int(bic_metrics["num_params_new_model"]),
        "num_new_params": int(bic_metrics["num_new_params"]),
        "logL_original_for_bic": float(bic_metrics["logL_original"]),
        "logL_new_model_for_bic": float(bic_metrics["logL_new_model"]),
        "bic_original": float(bic_metrics["bic_original"]),
        "bic_new": float(bic_metrics["bic_new"]),
        "base_image_log_likelihood_for_bic": float(bic_metrics["logL_original"]),
        "new_image_log_likelihood_for_bic": float(bic_metrics["logL_new_model"]),
        "delta_bic_mode": "raw_image_logl" if raw_bic else "target1_image_logl",
        "n_sub_params": n_sub_params,
        "subhalo_params": subhalo_params,
        "masses_msun": masses,
        "model_image": result.get("model_image"),
        "residual_map": result.get("residual_map"),
        "lens_light_image": result.get("lens_light_image"),
        "sigma_predicted": result.get("sigma_predicted"),
        "sigma_observed": result.get("sigma_observed", obs.sigma_obs),
        "sigma_observed_err": result.get("sigma_observed_err", obs.sigma_obs_err),
        "kin_chi2": result.get("kin_chi2"),
        "is_physical": result.get("is_physical"),
        "rmse_poisson": result.get("rmse_poisson"),
        "min_kappa": result.get("min_kappa"),
        "negative_mass_frac": result.get("negative_mass_frac"),
        "is_significant": bic_metrics["delta_bic"] > 6,
    }
    return out


def _compute_nfw_mass(Rs: float, alpha_Rs: float,
                      z_lens: float, z_source: float) -> float:
    """Convert NFW parameters to M200 in solar masses."""
    try:
        from lenstronomy.Cosmo.lens_cosmo import LensCosmo
        lc = LensCosmo(z_lens=z_lens, z_source=z_source)
        rho0, Rs_phys, c, r200, M200 = lc.nfw_angle2physical(Rs, alpha_Rs)
        return float(M200)
    except Exception as e:
        log.warning("NFW mass computation failed for Rs=%.6f alpha_Rs=%.6f: %s",
                    Rs, alpha_Rs, e)
        return float("nan")


def apply_subhalo_mass_cap(
    eval_results: Dict[str, Any],
    proposal: Dict[str, Any],
    obs,
    *,
    n_base_lens: int,
    max_subhalo_mass_msun: float = DEFAULT_MAX_SUBHALO_MASS_MSUN,
    invalid_chi2: float = INVALID_SUBHALO_MASS_CHI2,
) -> Dict[str, Any]:
    """Annotate and hard-penalize RSI proposals above the subhalo mass cap.

    RSI appends NFW subhalos after the base AFMS/PRL lens components.  This
    helper computes the derived M200 for those appended NFW components only.
    Any proposal with a non-finite mass or M200 above ``max_subhalo_mass_msun``
    is marked invalid and given a very poor chi2 so it is not admitted/ranked.
    """
    result = dict(eval_results)
    lens_kwargs = list(proposal.get("kwargs_lens", []))
    subhalo_params = lens_kwargs[n_base_lens:] if n_base_lens <= len(lens_kwargs) else []
    masses = [
        _compute_nfw_mass(
            sp.get("Rs", 0.0),
            sp.get("alpha_Rs", 0.0),
            obs.z_lens,
            obs.z_source,
        )
        for sp in subhalo_params
    ]

    result["subhalo_params"] = [dict(sp) for sp in subhalo_params]
    result["masses_msun"] = masses
    result["subhalo_mass_cap_msun"] = float(max_subhalo_mass_msun)
    result["subhalo_mass_limit_ok"] = True
    result["subhalo_mass_violation_indices"] = []
    finite_masses = [mass for mass in masses if np.isfinite(mass)]
    if not masses:
        result["subhalo_mass_max_msun"] = 0.0
    else:
        result["subhalo_mass_max_msun"] = max(finite_masses) if finite_masses else float("inf")

    violating = [
        idx for idx, mass in enumerate(masses)
        if not np.isfinite(mass) or mass > max_subhalo_mass_msun
    ]
    if not violating:
        return result

    finite_violating_masses = [
        masses[idx] for idx in violating if np.isfinite(masses[idx])
    ]
    max_mass = max(finite_violating_masses) if finite_violating_masses else float("inf")
    result["subhalo_mass_limit_ok"] = False
    result["subhalo_mass_violation_indices"] = violating
    result["subhalo_mass_max_msun"] = max_mass
    result["subhalo_mass_excess_factor"] = (
        float(max_mass / max_subhalo_mass_msun)
        if np.isfinite(max_mass) and max_subhalo_mass_msun > 0
        else float("inf")
    )
    result["subhalo_mass_violation_reason"] = (
        f"Derived NFW subhalo M200 exceeds cap "
        f"({max_mass:.3e} > {max_subhalo_mass_msun:.3e} Msun)."
    )
    result["is_significant"] = False
    result["is_physical"] = False

    chi2_r = float(result.get("image_chi2_reduced", 0.0) or 0.0)
    penalized_chi2_r = max(float(invalid_chi2), chi2_r)
    result["image_chi2_reduced"] = penalized_chi2_r

    n_pixels = result.get("n_pixels")
    try:
        n_pixels = int(n_pixels) if n_pixels is not None else None
    except (TypeError, ValueError):
        n_pixels = None
    if n_pixels and n_pixels > 0:
        result["image_chi2"] = penalized_chi2_r * n_pixels
    else:
        result["image_chi2"] = max(
            float(result.get("image_chi2", 0.0) or 0.0),
            penalized_chi2_r,
        )

    kin_chi2 = result.get("kin_chi2")
    try:
        kin_term = float(kin_chi2) if kin_chi2 is not None else 0.0
    except (TypeError, ValueError):
        kin_term = 0.0
    result["total_log_likelihood"] = -0.5 * (result["image_chi2"] + kin_term)
    return result


def count_proposal_params(proposal: Dict[str, Any]) -> int:
    """Count proposal parameters from the emitted proposal dictionaries."""
    n_free = 0
    for key in ("kwargs_lens", "kwargs_lens_light", "kwargs_source"):
        for comp in proposal.get(key, []):
            n_free += len(comp)
    return n_free


def compute_base_logL(
    proposal: Dict[str, Any],
    obs,
    timeout_s: int = 60,
) -> Tuple[float, int]:
    """Compute the base model's logL and parameter count for BIC comparison."""
    from .safe_eval import safe_evaluate
    from . import scoring as _S

    result, err = safe_evaluate(
        proposal, obs, include_kinematics=True,
        subtracted_chi2=_S.SUBTRACTED_CHI2,
        no_linear_solve=_S.NO_LINEAR_SOLVE,
        timeout_s=timeout_s)
    if result is None:
        return 0.0, 0

    image_chi2 = result.get("image_chi2", 0)
    logL = -0.5 * image_chi2
    return logL, count_proposal_params(proposal)


NFW_PRIOR_BOUNDS = {
    "Rs": (0.001, 0.5),
    "alpha_Rs": (0.0001, 0.5),
    "center_x": (-2.0, 2.0),
    "center_y": (-2.0, 2.0),
}

NFW_PRIOR_CENTERS = {
    "Rs": 0.05,
    "alpha_Rs": 0.01,
    "center_x": 0.0,
    "center_y": 0.0,
}

SIS_PRIOR_BOUNDS = NFW_PRIOR_BOUNDS
SIS_PRIOR_CENTERS = NFW_PRIOR_CENTERS


def register_subhalo_combo(
    base_proposal: Dict[str, Any],
    obs,
    candidates: List[Dict[str, Any]],
    n_subhalos: int = 1,
    combo_id: int = 99,
    freeze_base_for_pso: bool = True,
    freeze_base_model: bool = False,
) -> Dict[str, Any]:
    """Register the subhalo-extended model as a scoring combo.

    By default the base AFMS/PRL params remain FREE (matching the exp's
    approach of copying kwargs_params and appending NFW). Only the NFW params
    are new.

    When *freeze_base_for_pso=True* (default, backward compatible), PSO
    seeds fix the base lens params and only explore NFW.
    When *freeze_base_for_pso=False*, PSO seeds use AFMS/PRL values as
    init with full bounds, matching the exp's joint optimization.

    When *freeze_base_model=True*, all original non-subhalo lens, lens-light,
    and source parameters are fixed to the AFMS/PRL best proposal so the
    optimization explores only the appended NFW subhalo parameters.

    Returns kwargs_model for the extended model.
    """
    from . import scoring as _S

    base_proposal = _S.pack_mge_proposal(
        copy.deepcopy(base_proposal),
        lens_model_list=obs.kwargs_model.get("lens_model_list", []),
        lens_light_model_list=obs.kwargs_model.get("lens_light_model_list", []),
    )

    km = copy.deepcopy(obs.kwargs_model)
    for _ in range(n_subhalos):
        km['lens_model_list'].append('NFW')

    base_combo = _S.MODEL_COMBOS.get(_S.ACTIVE_COMBO, {})

    base_bounds_lens = base_combo.get("bounds_lens", [])
    base_fixed_lens = base_combo.get("fixed_lens", [])

    bounds_lens = [copy.deepcopy(b) for b in base_bounds_lens]
    centers_lens = [copy.deepcopy(base_proposal['kwargs_lens'][ci])
                    if ci < len(base_proposal.get('kwargs_lens', []))
                    else copy.deepcopy(base_combo.get("centers_lens", [{}])[ci]
                                       if ci < len(base_combo.get("centers_lens", []))
                                       else {})
                    for ci in range(len(base_bounds_lens))]
    if freeze_base_model:
        fixed_lens = []
        for ci in range(len(base_bounds_lens)):
            frozen = copy.deepcopy(base_proposal['kwargs_lens'][ci]
                                   if ci < len(base_proposal.get('kwargs_lens', []))
                                   else {})
            if ci < len(base_fixed_lens):
                frozen.update(copy.deepcopy(base_fixed_lens[ci]))
            fixed_lens.append(frozen)
    else:
        fixed_lens = [copy.deepcopy(f) for f in base_fixed_lens]

    for si in range(n_subhalos):
        cand = candidates[si] if si < len(candidates) else candidates[0]
        ra_c, dec_c = cand["ra"], cand["dec"]
        bounds_lens.append({
            "Rs": (0.001, 0.5),
            "alpha_Rs": (0.0001, 0.5),
            "center_x": (ra_c - 0.1, ra_c + 0.1),
            "center_y": (dec_c - 0.1, dec_c + 0.1),
        })
        centers_lens.append({
            "Rs": 0.05, "alpha_Rs": 0.01,
            "center_x": ra_c,
            "center_y": dec_c,
        })
        fixed_lens.append({})

    bounds_ll = base_combo.get("bounds_ll", [])
    base_ll = base_proposal.get('kwargs_lens_light', [])
    centers_ll = [copy.deepcopy(base_ll[ci]) if ci < len(base_ll)
                  else copy.deepcopy(base_combo.get("centers_ll", [{}])[ci]
                                     if ci < len(base_combo.get("centers_ll", []))
                                     else {})
                  for ci in range(len(bounds_ll))]
    base_fixed_ll = base_combo.get("fixed_ll", [])
    if freeze_base_model:
        fixed_ll = []
        for ci in range(len(bounds_ll)):
            frozen = copy.deepcopy(base_ll[ci]) if ci < len(base_ll) else {}
            if ci < len(base_fixed_ll):
                frozen.update(copy.deepcopy(base_fixed_ll[ci]))
            fixed_ll.append(frozen)
    else:
        fixed_ll = [copy.deepcopy(f) for f in base_fixed_ll]

    bounds_src = base_combo.get("bounds_src", [])
    base_src = base_proposal.get('kwargs_source', [])
    centers_src = [copy.deepcopy(base_src[ci]) if ci < len(base_src)
                   else copy.deepcopy(base_combo.get("centers_src", [{}])[ci]
                                      if ci < len(base_combo.get("centers_src", []))
                                      else {})
                   for ci in range(len(bounds_src))]
    base_fixed_src = base_combo.get("fixed_src", [])
    if freeze_base_model:
        fixed_src = []
        for ci in range(len(bounds_src)):
            frozen = copy.deepcopy(base_src[ci]) if ci < len(base_src) else {}
            if ci < len(base_fixed_src):
                frozen.update(copy.deepcopy(base_fixed_src[ci]))
            fixed_src.append(frozen)
    else:
        fixed_src = [copy.deepcopy(f) for f in base_fixed_src]

    seed_fixed_lens = []
    seed_bounds_lens = []
    seed_centers_lens = []
    for ci in range(len(base_bounds_lens)):
        if freeze_base_model:
            seed_fixed_lens.append(copy.deepcopy(fixed_lens[ci]))
            seed_bounds_lens.append({})
        elif freeze_base_for_pso:
            seed_fixed_lens.append(copy.deepcopy(base_proposal['kwargs_lens'][ci]
                                                  if ci < len(base_proposal['kwargs_lens']) else {}))
            seed_bounds_lens.append({})
        else:
            seed_fixed_lens.append(copy.deepcopy(
                base_combo.get("fixed_lens", [{}])[ci]
                if ci < len(base_combo.get("fixed_lens", [])) else {}))
            seed_bounds_lens.append(copy.deepcopy(base_bounds_lens[ci]))
        seed_centers_lens.append(copy.deepcopy(base_proposal['kwargs_lens'][ci]
                                                if ci < len(base_proposal['kwargs_lens']) else {}))
    for si in range(n_subhalos):
        cand = candidates[si] if si < len(candidates) else candidates[0]
        ra_c, dec_c = cand["ra"], cand["dec"]
        seed_fixed_lens.append({})
        seed_bounds_lens.append({
            "Rs": (0.001, 0.5),
            "alpha_Rs": (0.0001, 0.5),
            "center_x": (ra_c - 0.1, ra_c + 0.1),
            "center_y": (dec_c - 0.1, dec_c + 0.1),
        })
        seed_centers_lens.append({
            "Rs": 0.05, "alpha_Rs": 0.01,
            "center_x": ra_c,
            "center_y": dec_c,
        })

    combo = {
        "label": f"Subhalo ({n_subhalos} NFW)",
        "freeze_base_model": freeze_base_model,
        "kwargs_model": km,
        "bounds_lens": bounds_lens,
        "centers_lens": centers_lens,
        "fixed_lens": fixed_lens,
        "bounds_ll": bounds_ll,
        "centers_ll": centers_ll,
        "fixed_ll": fixed_ll,
        "bounds_src": bounds_src,
        "centers_src": centers_src,
        "fixed_src": fixed_src,
        "seed_fixed_lens": seed_fixed_lens,
        "seed_bounds_lens": seed_bounds_lens,
        "seed_centers_lens": seed_centers_lens,
    }

    _S.MODEL_COMBOS[combo_id] = combo
    _S.set_model_combo(combo_id)
    n_free = sum(len(comp) for bl in _S.PRIOR_BOUNDS.values() for comp in bl)
    log.info("Registered subhalo combo %d: %s  (%d free params)",
             combo_id, combo["label"], n_free)

    return km
