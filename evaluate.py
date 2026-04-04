import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import logging
logging.getLogger("jax._src.xla_bridge").setLevel(logging.WARNING)

import numpy as np
from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.Data.psf import PSF

from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.ImSim.image_linear_solve import ImageLinearFit


from profiles import setup_custom_profiles

setup_custom_profiles()

_ZERO_LIGHT_TOL = 1e-12
_INVALID_KIN_CHI2 = 1e6


def _amp_abs_sum(value):
    if value is None:
        return 0.0
    arr = np.asarray(value, dtype=float)
    if arr.size == 0 or not np.all(np.isfinite(arr)):
        return 0.0
    return float(np.sum(np.abs(arr)))


def _has_nonzero_deflector_light(kwargs_lens_light, lens_light_image=None, tol=_ZERO_LIGHT_TOL):
    total_amp = sum(_amp_abs_sum(comp.get("amp")) for comp in (kwargs_lens_light or []))
    if np.isfinite(total_amp) and total_amp > tol:
        return True
    if lens_light_image is None:
        return False
    image = np.asarray(lens_light_image, dtype=float)
    if image.size == 0 or not np.all(np.isfinite(image)):
        return False
    return float(np.sum(np.abs(image))) > tol


def _invalid_kinematics_result(obs, reason):
    kin_chi2 = float(_INVALID_KIN_CHI2)
    return {
        'sigma_predicted': None,
        'sigma_observed': obs.sigma_obs,
        'sigma_observed_err': obs.sigma_obs_err,
        'kin_chi2': kin_chi2,
        'kin_log_likelihood': -0.5 * kin_chi2,
        'kin_failure_reason': reason,
    }


def evaluate_proposal(proposal, obs,
                      include_kinematics=True,
                      subtracted_chi2=False,
                      no_linear_solve=False,
                      kwargs_anisotropy=None,
                      kwargs_aperture=None,
                      kwargs_seeing=None):
    """Score a proposed mass distribution against observations.

    :param proposal: dict with keys 'kwargs_lens', 'kwargs_lens_light',
        'kwargs_source'.
    :param obs: ObservationBundle instance.
    :param include_kinematics: compute kinematic chi2 when possible.
    :param subtracted_chi2: if True, use lens-subtracted noise model
        (source-only Poisson term) when computing chi2.
    :param no_linear_solve: if True, use proposal amplitudes directly
        instead of running the linear amplitude solver.
    :return: dict with model_image, residual_map, image_chi2,
        image_chi2_reduced, etc.
    """
    try:
        from funsearch.scoring import pack_mge_proposal
        proposal = pack_mge_proposal(
            proposal,
            lens_model_list=obs.kwargs_model.get('lens_model_list', []),
            lens_light_model_list=obs.kwargs_model.get('lens_light_model_list', []),
        )
    except ImportError:
        pass

    data_class = ImageData(**obs.kwargs_data)
    psf_class = PSF(**obs.kwargs_psf)

    lens_model = LensModel(obs.kwargs_model['lens_model_list'])
    source_model = LightModel(obs.kwargs_model['source_light_model_list'])
    lens_light_model = LightModel(obs.kwargs_model['lens_light_model_list'])

    lmask = getattr(obs, 'likelihood_mask', None)
    image_model = ImageLinearFit(
        data_class, psf_class, lens_model, source_model, lens_light_model,
        likelihood_mask=lmask)

    kwargs_source_final = [dict(comp) for comp in proposal['kwargs_source']]
    kwargs_ll_final = proposal['kwargs_lens_light']

    try:
        from funsearch.scoring import MODEL_COMBOS, ACTIVE_COMBO, FIXED_PARAMS, PRIOR_CENTERS, materialize_source_ties
        combo = MODEL_COMBOS.get(ACTIVE_COMBO, {})
        ties = combo.get("shapelet_src_ties", {})
        fixed_src = FIXED_PARAMS.get('kwargs_source', [])
        centers_src = PRIOR_CENTERS.get('kwargs_source', [])
    except ImportError:
        ties = {}
        fixed_src = []
        centers_src = []

    if ties:
        kwargs_source_final = materialize_source_ties(
            kwargs_source_final,
            combo=combo,
            fixed_src=fixed_src,
            centers_src=centers_src,
        )

    if no_linear_solve:
        model_image = image_model.image(
            kwargs_lens=proposal['kwargs_lens'],
            kwargs_source=kwargs_source_final,
            kwargs_lens_light=kwargs_ll_final,
        )
    else:
        model_image, _, _, param = image_model.image_linear_solve(
            kwargs_lens=proposal['kwargs_lens'],
            kwargs_source=kwargs_source_final,
            kwargs_lens_light=kwargs_ll_final,
        )
        if param is not None:
            _, kwargs_source_final, kwargs_ll_final, _ = image_model.update_linear_kwargs(
                param,
                proposal['kwargs_lens'],
                kwargs_source_final,
                kwargs_ll_final,
                [],
            )

        if param is not None and len(param) > 0 and np.all(param == 0):
            ll_models = obs.kwargs_model.get('lens_light_model_list', [])
            base_ll = [m for m in ll_models if 'SHAPELETS' not in m]
            n_base = len(base_ll)
            base_ll_model = LightModel(base_ll)
            base_image_model = ImageLinearFit(
                data_class, psf_class, lens_model, source_model, base_ll_model,
                likelihood_mask=lmask)
            kwargs_ll_base = kwargs_ll_final[:n_base]
            model_image, _, _, param = base_image_model.image_linear_solve(
                kwargs_lens=proposal['kwargs_lens'],
                kwargs_source=kwargs_source_final,
                kwargs_lens_light=kwargs_ll_base,
            )
            if param is not None:
                _, kwargs_source_final, kwargs_ll_base, _ = base_image_model.update_linear_kwargs(
                    param,
                    proposal['kwargs_lens'],
                    kwargs_source_final,
                    kwargs_ll_base,
                    [],
                )
            kwargs_ll_final = list(kwargs_ll_base) + kwargs_ll_final[n_base:]

    solved_proposal = dict(proposal)
    solved_proposal['kwargs_source'] = kwargs_source_final
    solved_proposal['kwargs_lens_light'] = kwargs_ll_final

    lens_light_image = image_model.lens_surface_brightness(
        kwargs_ll_final, unconvolved=False)

    if subtracted_chi2:
        render_sub = model_image - lens_light_image
        observed = data_class.data
        resid = observed - model_image
        bg_rms = data_class.background_rms
        exp_map = data_class.exposure_map
        d_pos = np.maximum(render_sub, 0.0)
        variance_sub = d_pos / exp_map + bg_rms ** 2
        if lmask is not None:
            masked_chi = (resid ** 2 / variance_sub) * lmask
            n_pixels = int(np.sum(lmask))
        else:
            masked_chi = resid ** 2 / variance_sub
            n_pixels = resid.size
        image_chi2_reduced = float(np.sum(masked_chi) / n_pixels)
        norm_residuals = resid / np.sqrt(variance_sub)
        if lmask is not None:
            norm_residuals = norm_residuals * lmask
    else:
        image_chi2_reduced = float(image_model.reduced_chi2(model_image))
        norm_residuals = image_model.reduced_residuals(model_image)
        n_pixels = int(image_model.num_data_evaluate)

    image_chi2 = float(image_chi2_reduced * n_pixels)

    result = {
        'model_image': model_image,
        'lens_light_image': lens_light_image,
        'residual_map': norm_residuals,
        'image_chi2': image_chi2,
        'image_chi2_reduced': image_chi2_reduced,
        'n_pixels': n_pixels,
    }

    if include_kinematics and obs.sigma_obs > 0:
        if not _has_nonzero_deflector_light(kwargs_ll_final, lens_light_image):
            logging.getLogger(__name__).warning(
                "Skipping kinematics for zero-deflector-light solution")
            kin_result = _invalid_kinematics_result(obs, "zero_deflector_light")
        else:
            kin_result = _evaluate_kinematics(
                solved_proposal, obs, kwargs_anisotropy, kwargs_aperture, kwargs_seeing)
        result.update(kin_result)
        result['total_log_likelihood'] = -0.5 * (
            image_chi2 + result['kin_chi2'])
    else:
        result['total_log_likelihood'] = -0.5 * image_chi2

    try:
        phys = compute_physicality(proposal['kwargs_lens'], obs)
        result.update(phys)
    except Exception:
        result['is_physical'] = None
        result['rmse_poisson'] = None

    return result


def compute_physicality(kwargs_lens, obs):
    """Evaluate physical consistency of a lens mass model.

    Checks that the gravitational potential psi and convergence kappa
    satisfy the Poisson equation: 0.5 * laplacian(psi) = kappa.
    Also checks for negative mass (kappa < 0) regions.

    Returns dict with: is_physical, physicality_score, rmse_poisson,
    min_kappa, negative_mass_frac.
    """
    from lenstronomy.LensModel.lens_model import LensModel as _LM
    from lenstronomy.Util import util

    lens_model = _LM(lens_model_list=obs.kwargs_model['lens_model_list'])

    data = obs.kwargs_data_joint['multi_band_list'][0][0]
    num_pix = len(data['image_data'])
    res = np.sqrt(np.abs(np.linalg.det(data['transform_pix2angle'])))
    x_grid, y_grid = util.make_grid(num_pix, res)

    psi_vec = lens_model.potential(x_grid, y_grid, kwargs_lens)
    kappa_vec = lens_model.kappa(x_grid, y_grid, kwargs_lens)

    psi_map = util.array2image(psi_vec)
    kappa_map = util.array2image(kappa_vec)

    dy, dx = np.gradient(psi_map, res)
    _, dxx = np.gradient(dx, res)
    dyy, _ = np.gradient(dy, res)
    half_laplacian = 0.5 * (dxx + dyy)

    poisson_diff = kappa_map - half_laplacian
    rmse_poisson = float(np.sqrt(np.mean(poisson_diff ** 2)))
    min_kappa = float(np.min(kappa_map))
    negative_mass_frac = float(np.sum(kappa_map < -0.05) / len(kappa_vec))

    is_physical = True
    if rmse_poisson > 0.05:
        is_physical = False
    if min_kappa < -0.2 or negative_mass_frac > 0.05:
        is_physical = False

    return {
        'is_physical': is_physical,
        'physicality_score': -rmse_poisson,
        'rmse_poisson': rmse_poisson,
        'min_kappa': min_kappa,
        'negative_mass_frac': negative_mass_frac,
    }


def _evaluate_kinematics(proposal, obs,
                         kwargs_anisotropy=None,
                         kwargs_aperture=None,
                         kwargs_seeing=None):
    """Predict velocity dispersion and compute kinematic chi2.

    Separated from evaluate_proposal so callers can skip the expensive
    kinematic computation when only imaging scores are needed.
    """
    from kinematic_api import KinematicsAPI

    if kwargs_anisotropy is None:
        kwargs_anisotropy = {}
    if kwargs_aperture is None:
        kwargs_aperture = {
            'aperture_type': 'slit',
            'length': 2.66,
            'width': 2.66,
            'angle': 0.0,
        }
    if kwargs_seeing is None:
        kwargs_seeing = {
            'psf_type': 'MOFFAT',
            'fwhm': 1.0,
            'moffat_beta': 2.5,
        }

    center_x = proposal['kwargs_lens'][0].get('center_x', 0)
    center_y = proposal['kwargs_lens'][0].get('center_y', 0)
    center_x_light = proposal['kwargs_lens_light'][0].get('center_x', 0)
    center_y_light = proposal['kwargs_lens_light'][0].get('center_y', 0)

    kwargs_mge_light = {
        'grid_spacing': 1, 'grid_num': 100, 'n_comp': 20,
        'center_x': center_x_light, 'center_y': center_y_light,
    }
    kwargs_mge_mass = {
        'grid_spacing': 1, 'grid_num': 100, 'n_comp': 20,
        'center_x': center_x, 'center_y': center_y,
    }
    ll_models = obs.kwargs_model.get('lens_light_model_list', [])
    native_mge_light = bool(ll_models) and set(ll_models) <= {
        'MULTI_GAUSSIAN', 'MULTI_GAUSSIAN_ELLIPSE'
    }

    api = KinematicsAPI(
        obs.z_lens, obs.z_source, obs.kwargs_model,
        kwargs_aperture, kwargs_seeing, 'isotropic',
        MGE_light=not native_mge_light, MGE_mass=True,
        kwargs_mge_light=kwargs_mge_light,
        kwargs_mge_mass=kwargs_mge_mass,
    )

    r_eff = proposal['kwargs_lens_light'][0].get('R_sersic', None)
    try:
        sigma_pred_array = api.velocity_dispersion(
            proposal['kwargs_lens'],
            proposal['kwargs_lens_light'],
            kwargs_anisotropy,
            r_eff=r_eff,
        )
        sigma_pred = float(np.ravel(sigma_pred_array)[0])
        if not np.isfinite(sigma_pred):
            raise ValueError("non-finite velocity dispersion")
    except Exception as exc:
        logging.getLogger(__name__).warning(
            "Skipping kinematics after solver failure: %s: %s",
            type(exc).__name__,
            exc,
        )
        return _invalid_kinematics_result(
            obs, f"kinematics_error:{type(exc).__name__}"
        )
    kin_chi2 = (sigma_pred - obs.sigma_obs) ** 2 / (obs.sigma_obs_err ** 2)

    return {
        'sigma_predicted': sigma_pred,
        'sigma_observed': obs.sigma_obs,
        'sigma_observed_err': obs.sigma_obs_err,
        'kin_chi2': float(kin_chi2),
        'kin_log_likelihood': -0.5 * float(kin_chi2),
    }
