"""Quality, diversity, selection, and admission metrics.

The scoring module is the sole place where tunable constants (alpha, beta,
gamma, lambda) live.  Everything else in the funsearch package delegates
here for all scoring decisions.
"""

import copy
import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Expert-aligned parameter configurations.
# Shared EPL/SIE/light priors below are taken from `try_all_models (4).ipynb`.
# ---------------------------------------------------------------------------

# --- EPL + SHEAR + MULTIPOLE (expert's Cell 3, new model) ---
_EPL_BOUNDS: Dict[str, Tuple[float, float]] = {
    "theta_E": (0.01, 5.0), "gamma": (1.5, 2.8),
    "e1": (-0.5, 0.5), "e2": (-0.5, 0.5),
    "center_x": (-10.0, 10.0), "center_y": (-10.0, 10.0),
}
_EPL_CENTER: Dict[str, float] = {
    "theta_E": 1.0, "gamma": 2.0, "e1": 0.0, "e2": 0.0,
    "center_x": 0.0, "center_y": 0.0,
}
_EPL_SIGMA: Dict[str, float] = {
    "theta_E": 0.1, "gamma": 0.1, "e1": 0.1, "e2": 0.1,
    "center_x": 0.05, "center_y": 0.05,
}

_SHEAR_BOUNDS: Dict[str, Tuple[float, float]] = {
    "gamma1": (-0.3, 0.3), "gamma2": (-0.3, 0.3),
}
_SHEAR_CENTER: Dict[str, float] = {"gamma1": 0.0, "gamma2": 0.0}
_SHEAR_SIGMA: Dict[str, float] = {"gamma1": 0.05, "gamma2": 0.05}
_SHEAR_FIXED: Dict[str, Any] = {"ra_0": 0, "dec_0": 0}

_MULTIPOLE_BOUNDS: Dict[str, Tuple[float, float]] = {
    "a_m": (-0.1, 0.1), "phi_m": (-math.pi, math.pi),
    "center_x": (-10.0, 10.0), "center_y": (-10.0, 10.0),
}
_MULTIPOLE_CENTER: Dict[str, float] = {
    "a_m": 0.01, "phi_m": 0.0, "center_x": 0.0, "center_y": 0.0,
}
_MULTIPOLE_SIGMA: Dict[str, float] = {
    "a_m": 0.01, "phi_m": 0.1, "center_x": 0.05, "center_y": 0.05,
}
_MULTIPOLE_FIXED: Dict[str, Any] = {"m": 4, "r_E": 1}

_LENS_BOUNDS: List[Dict[str, Tuple[float, float]]] = [
    dict(_EPL_BOUNDS), dict(_SHEAR_BOUNDS), dict(_MULTIPOLE_BOUNDS),
]
_LENS_CENTERS: List[Dict[str, float]] = [
    dict(_EPL_CENTER), dict(_SHEAR_CENTER), dict(_MULTIPOLE_CENTER),
]
_LENS_SIGMAS: List[Dict[str, float]] = [
    dict(_EPL_SIGMA), dict(_SHEAR_SIGMA), dict(_MULTIPOLE_SIGMA),
]
_LENS_FIXED: List[Dict[str, Any]] = [
    {}, dict(_SHEAR_FIXED), dict(_MULTIPOLE_FIXED),
]

# --- Source light (expert's exact values) ---
_SRC_BOUNDS: Dict[str, Tuple[float, float]] = {
    "n_sersic": (0.5, 8.0), "R_sersic": (0.01, 10.0),
    "e1": (-0.5, 0.5), "e2": (-0.5, 0.5),
    "center_x": (-10.0, 10.0), "center_y": (-10.0, 10.0),
}
_SRC_CENTER: Dict[str, float] = {
    "n_sersic": 2.0, "R_sersic": 0.5, "e1": 0.0, "e2": 0.0,
    "center_x": 0.0, "center_y": 0.0,
}
_SRC_SIGMA: Dict[str, float] = {
    "R_sersic": 0.1, "n_sersic": 0.5, "e1": 0.1, "e2": 0.1,
    "center_x": 0.05, "center_y": 0.05,
}
_NFW_SIGMA: Dict[str, float] = {
    "Rs": 2.0, "alpha_Rs": 0.2, "center_x": 0.05, "center_y": 0.05,
}
_PEMD_SIGMA: Dict[str, float] = dict(_EPL_SIGMA)
_SIE_SIGMA: Dict[str, float] = {
    "theta_E": 0.1, "e1": 0.1, "e2": 0.1,
    "center_x": 0.05, "center_y": 0.05,
}
_SIS_SIGMA: Dict[str, float] = {
    "theta_E": 0.1, "center_x": 0.05, "center_y": 0.05,
}

# --- Lens light (same shared `light_registry` prior as try_all_models (4)) ---
# amp is a linear parameter solved by image_linear_solve(); excluded from
# proposal space by default (Mode A).  _WITH_AMP variants are used when
# --no-linear-solve is active (Mode B).
_LL_BOUNDS_WITH_AMP: Dict[str, Tuple[float, float]] = {
    "amp": (0.001, 100000.0),
    "n_sersic": (0.5, 8.0), "R_sersic": (0.01, 10.0),
    "e1": (-0.5, 0.5), "e2": (-0.5, 0.5),
    "center_x": (-10.0, 10.0), "center_y": (-10.0, 10.0),
}
_LL_CENTER_WITH_AMP: Dict[str, float] = {
    "amp": 100.0, "n_sersic": 2.0, "R_sersic": 0.5, "e1": 0.0, "e2": 0.0,
    "center_x": 0.0, "center_y": 0.0,
}
_LL_SIGMA_WITH_AMP: Dict[str, float] = {
    "amp": 10.0, "R_sersic": 0.1, "n_sersic": 0.5, "e1": 0.1, "e2": 0.1,
    "center_x": 0.05, "center_y": 0.05,
}

_LL_BOUNDS: Dict[str, Tuple[float, float]] = {
    "n_sersic": (0.5, 8.0), "R_sersic": (0.01, 10.0),
    "e1": (-0.5, 0.5), "e2": (-0.5, 0.5),
    "center_x": (-10.0, 10.0), "center_y": (-10.0, 10.0),
}
_LL_CENTER: Dict[str, float] = {
    "n_sersic": 2.0, "R_sersic": 0.5, "e1": 0.0, "e2": 0.0,
    "center_x": 0.0, "center_y": 0.0,
}
_LL_SIGMA: Dict[str, float] = {
    "R_sersic": 0.1, "n_sersic": 0.5, "e1": 0.1, "e2": 0.1,
    "center_x": 0.05, "center_y": 0.05,
}

# --- Lens light (older version without amp, from Copy_of_Auto_fit) ---
_LL_BOUNDS_NO_AMP: Dict[str, Tuple[float, float]] = {
    "n_sersic": (0.5, 8.0), "R_sersic": (0.01, 10.0),
    "e1": (-0.5, 0.5), "e2": (-0.5, 0.5),
    "center_x": (-10.0, 10.0), "center_y": (-10.0, 10.0),
}
_LL_CENTER_NO_AMP: Dict[str, float] = {
    "n_sersic": 2.0, "R_sersic": 0.5, "e1": 0.0, "e2": 0.0,
    "center_x": 0.0, "center_y": 0.0,
}

# --- Legacy shapelets (non-bandit path only) ---
_POLAR_BOUNDS: Dict[str, Tuple[float, float]] = {
    "beta": (0.01, 5.0), "center_x": (-1.0, 1.0), "center_y": (-1.0, 1.0),
}
_POLAR_CENTER: Dict[str, float] = {"beta": 0.5, "center_x": 0.0, "center_y": 0.0}
_POLAR_CENTER_LL: Dict[str, float] = {"beta": 1.0, "center_x": 0.0, "center_y": 0.0}

# --- Shapelet source component (added to ALL combos per expert notebook) ---
_SHAPELET_SRC_BOUNDS: Dict[str, Tuple[float, float]] = {
    "beta": (0.02, 0.5),
    "center_x": (-10.0, 10.0),
    "center_y": (-10.0, 10.0),
}
_SHAPELET_SRC_CENTER: Dict[str, float] = {
    "beta": 0.1,
    "center_x": 0.0,
    "center_y": 0.0,
}
_SHAPELET_SRC_SIGMA: Dict[str, float] = {
    "beta": 0.05,
    "center_x": 0.05,
    "center_y": 0.05,
}
_SHAPELET_SRC_FIXED: Dict[str, Any] = {"n_max": 6}

# --- Shared light config (amp excluded – solved by linear solver) ---
_SIMPLE_LL_NO_AMP: Dict[str, Any] = {
    "bounds_ll": [dict(_LL_BOUNDS)],
    "sigmas_ll": [dict(_LL_SIGMA)],
    "centers_ll": [dict(_LL_CENTER)],
    "fixed_ll": [{}],
    "bounds_src": [dict(_SRC_BOUNDS), dict(_SHAPELET_SRC_BOUNDS)],
    "sigmas_src": [dict(_SRC_SIGMA), dict(_SHAPELET_SRC_SIGMA)],
    "centers_src": [dict(_SRC_CENTER), dict(_SHAPELET_SRC_CENTER)],
    "fixed_src": [{}, dict(_SHAPELET_SRC_FIXED)],
}

# --- Shared light config (with amp – for --no-linear-solve mode) ---
_SIMPLE_LL_WITH_AMP: Dict[str, Any] = {
    "bounds_ll": [dict(_LL_BOUNDS_WITH_AMP)],
    "sigmas_ll": [dict(_LL_SIGMA_WITH_AMP)],
    "centers_ll": [dict(_LL_CENTER_WITH_AMP)],
    "fixed_ll": [{}],
    "bounds_src": [dict(_SRC_BOUNDS), dict(_SHAPELET_SRC_BOUNDS)],
    "sigmas_src": [dict(_SRC_SIGMA), dict(_SHAPELET_SRC_SIGMA)],
    "centers_src": [dict(_SRC_CENTER), dict(_SHAPELET_SRC_CENTER)],
    "fixed_src": [{}, dict(_SHAPELET_SRC_FIXED)],
}

# Legacy alias
_SERSIC_BOUNDS = _LL_BOUNDS_WITH_AMP
_SERSIC_CENTER_BULGE = _LL_CENTER_WITH_AMP
_SERSIC_CENTER_ENVELOPE = {
    "n_sersic": 2.0, "R_sersic": 0.5, "e1": 0.0, "e2": 0.0,
    "center_x": 0.0, "center_y": 0.0,
}
_SERSIC_SIGMA: Dict[str, float] = {
    "R_sersic": 0.1, "n_sersic": 0.5, "center_x": 0.05, "center_y": 0.05,
}
_SERSIC_CENTER_ENVELOPE_WITH_AMP = {
    "amp": 100.0, "n_sersic": 2.0, "R_sersic": 0.5, "e1": 0.0, "e2": 0.0,
    "center_x": 0.0, "center_y": 0.0,
}
_SERSIC_SIGMA_WITH_AMP: Dict[str, float] = {
    "amp": 10.0, "R_sersic": 0.1, "n_sersic": 0.5,
    "e1": 0.1, "e2": 0.1, "center_x": 0.05, "center_y": 0.05,
}
_SERSIC_CENTER_SRC = _SRC_CENTER

# ---------------------------------------------------------------------------
# Model combo definitions
# ---------------------------------------------------------------------------

MODEL_COMBOS: Dict[int, Dict[str, Any]] = {
    1: {
        "label": "Sersic ; Sersic",
        "kwargs_model": {
            "lens_model_list": ["EPL", "SHEAR", "MULTIPOLE"],
            "lens_light_model_list": ["SERSIC_ELLIPSE"],
            "source_light_model_list": ["SERSIC_ELLIPSE"],
        },
        "bounds_ll": [dict(_LL_BOUNDS)],
        "bounds_src": [dict(_SRC_BOUNDS)],
        "centers_ll": [dict(_LL_CENTER)],
        "centers_src": [dict(_SRC_CENTER)],
        "fixed_ll": [{}],
        "fixed_src": [{}],
    },
    2: {
        "label": "Sersic+Sersic ; Sersic+PolarShapelet",
        "kwargs_model": {
            "lens_model_list": ["EPL", "SHEAR", "MULTIPOLE"],
            "lens_light_model_list": ["SERSIC_ELLIPSE", "SERSIC_ELLIPSE"],
            "source_light_model_list": ["SERSIC_ELLIPSE", "SHAPELETS"],
        },
        "bounds_ll": [dict(_LL_BOUNDS), dict(_LL_BOUNDS)],
        "bounds_src": [dict(_SRC_BOUNDS), dict(_POLAR_BOUNDS)],
        "centers_ll": [dict(_LL_CENTER), dict(_SERSIC_CENTER_ENVELOPE)],
        "centers_src": [dict(_SRC_CENTER), dict(_POLAR_CENTER)],
        "fixed_ll": [{}, {}],
        "fixed_src": [{}, {"n_max": 6}],
    },
    3: {
        "label": "Sersic+Sersic+Shapelet ; Sersic+Shapelet",
        "kwargs_model": {
            "lens_model_list": ["EPL", "SHEAR", "MULTIPOLE"],
            "lens_light_model_list": ["SERSIC_ELLIPSE", "SERSIC_ELLIPSE", "SHAPELETS"],
            "source_light_model_list": ["SERSIC_ELLIPSE", "SHAPELETS"],
        },
        "bounds_ll": [dict(_LL_BOUNDS), dict(_LL_BOUNDS), dict(_POLAR_BOUNDS)],
        "bounds_src": [dict(_SRC_BOUNDS), dict(_POLAR_BOUNDS)],
        "centers_ll": [dict(_LL_CENTER), dict(_SERSIC_CENTER_ENVELOPE),
                        dict(_POLAR_CENTER_LL)],
        "centers_src": [dict(_SRC_CENTER), dict(_POLAR_CENTER)],
        "fixed_ll": [{}, {}, {"n_max": 6}],
        "fixed_src": [{}, {"n_max": 6}],
    },
    4: {
        "label": "SLIT_STARLETS (requires slitronomy)",
        "kwargs_model": {
            "lens_model_list": ["EPL", "SHEAR", "MULTIPOLE"],
            "lens_light_model_list": ["SERSIC_ELLIPSE"],
            "source_light_model_list": ["SLIT_STARLETS"],
        },
        "bounds_ll": [dict(_LL_BOUNDS)],
        "bounds_src": [{"n_scales": (3, 6), "beta": (0.01, 5.0),
                        "center_x": (-1.0, 1.0), "center_y": (-1.0, 1.0)}],
        "centers_ll": [dict(_LL_CENTER)],
        "centers_src": [{"n_scales": 4, "beta": 1.0, "center_x": 0.0, "center_y": 0.0}],
        "fixed_ll": [{}],
        "fixed_src": [{"n_pixels": 14400, "scale": 1}],
    },
}

_GAUSS_KAPPA_BOUNDS: Dict[str, Tuple[float, float]] = {
    "amp": (0.0, 1.0),
    "sigma": (0.1, 5.0),
    "center_x": (-2.0, 2.0),
    "center_y": (-2.0, 2.0),
}
_GAUSS_KAPPA_CENTER: Dict[str, float] = {
    "amp": 0.1, "sigma": 1.0,
    "center_x": 0.0, "center_y": 0.0,
}
_GAUSSIAN_SUB_BOUNDS: Dict[str, Tuple[float, float]] = {
    "amp": (0.0, 1.0),
    "sigma": (0.1, 5.0),
    "e1": (-0.5, 0.5),
    "e2": (-0.5, 0.5),
    "center_x": (-2.0, 2.0),
    "center_y": (-2.0, 2.0),
}
_GAUSSIAN_SUB_CENTER: Dict[str, float] = {
    "amp": 0.1, "sigma": 1.0,
    "e1": 0.0, "e2": 0.0,
    "center_x": 0.0, "center_y": 0.0,
}
_GAUSSIAN_SUB_SIGMA: Dict[str, float] = {
    "amp": 0.1,
    "sigma": 0.5,
    "e1": 0.1,
    "e2": 0.1,
    "center_x": 0.05,
    "center_y": 0.05,
}
DEFAULT_MGE_COMPONENTS = 10


def build_combo5(n_gaussians: int = 3) -> Dict[str, Any]:
    """Build combo 5: EPL+SHEAR+MULTIPOLE + N Gaussian ellipse kappa blobs."""
    lens_list = ["EPL", "SHEAR", "MULTIPOLE"] + ["GAUSSIAN"] * n_gaussians
    lens_bounds = list(_LENS_BOUNDS) + [dict(_GAUSS_KAPPA_BOUNDS) for _ in range(n_gaussians)]
    lens_centers = list(_LENS_CENTERS) + [dict(_GAUSS_KAPPA_CENTER) for _ in range(n_gaussians)]
    lens_fixed = list(_LENS_FIXED) + [{}] * n_gaussians
    combo = {
        "label": f"EPL+{n_gaussians}xGaussKappa ; Sersic;Sersic",
        "kwargs_model": {
            "lens_model_list": lens_list,
            "lens_light_model_list": ["SERSIC_ELLIPSE"],
            "source_light_model_list": ["SERSIC_ELLIPSE"],
        },
        "bounds_lens": lens_bounds,
        "centers_lens": lens_centers,
        "fixed_lens": lens_fixed,
        **_SIMPLE_LL,
    }
    MODEL_COMBOS[5] = combo
    return combo


def build_combo8(n_gauss: int = DEFAULT_MGE_COMPONENTS) -> Dict[str, Any]:
    """Build combo 8: expert-style MGE mass/light with N Gaussian components."""

    def _build_indexed_mge_priors() -> Tuple[Dict[str, Any], Dict[str, Any]]:
        sigma_grid = np.logspace(np.log10(0.05), np.log10(20.0), n_gauss)
        bounds: Dict[str, Any] = {}
        centers: Dict[str, Any] = {}
        for i in range(n_gauss):
            s = float(sigma_grid[i])
            s_lo = max(0.01, s * 0.3)
            s_hi = s * 3.0
            bounds[f"amp_{i}"] = (0.001, 50.0)
            bounds[f"sigma_{i}"] = (s_lo, s_hi)
            centers[f"amp_{i}"] = 1.0
            centers[f"sigma_{i}"] = s
        bounds["center_x"] = (-10.0, 10.0)
        bounds["center_y"] = (-10.0, 10.0)
        centers["center_x"] = 0.0
        centers["center_y"] = 0.0
        return bounds, centers

    bounds_mge_mass, centers_mge_mass = _build_indexed_mge_priors()
    sigma_grid = [float(s) for s in np.logspace(np.log10(0.05), np.log10(20.0), n_gauss)]
    proxy_joint = [[0, i, ["center_x", "center_y"]] for i in range(1, n_gauss)]

    combo = {
        "label": "MGE Mass Model",
        "kwargs_model": {
            "lens_model_list": ["MULTI_GAUSSIAN", "SHEAR"],
            "lens_light_model_list": ["MULTI_GAUSSIAN"],
            "source_light_model_list": ["SERSIC_ELLIPSE"],
        },
        # lenstronomy's LensParam cannot natively flatten MULTI_GAUSSIAN mass
        # arrays for PSO, so scout/seeding optimize an exactly equivalent sum of
        # circular GAUSSIAN components with shared centers and merge back after.
        "pso_proxy_lens_list": ["GAUSSIAN"] * n_gauss + ["SHEAR"],
        "pso_proxy_joint_lens_with_lens": proxy_joint,
        "n_mge": n_gauss,
        "bounds_lens": [bounds_mge_mass, dict(_LENS_BOUNDS[1])],
        "centers_lens": [centers_mge_mass, dict(_LENS_CENTERS[1])],
        "fixed_lens": [{}, dict(_LENS_FIXED[1])],
        "bounds_ll": [{"center_x": (-10.0, 10.0), "center_y": (-10.0, 10.0)}],
        "centers_ll": [{"center_x": 0.0, "center_y": 0.0}],
        "fixed_ll": [{"sigma": sigma_grid}],
        "bounds_src": [dict(_SRC_BOUNDS)],
        "sigmas_src": [dict(_SRC_SIGMA)],
        "centers_src": [dict(_SRC_CENTER)],
        "fixed_src": [{}],
    }
    MODEL_COMBOS[8] = combo
    return combo




# ---------------------------------------------------------------------------
# Expert mass model families (combos 6-13, from try_all_models.ipynb)
# All use single Sersic light for both lens and source.
# ---------------------------------------------------------------------------

_EPL_BOUNDS = _LENS_BOUNDS[0]
_SHEAR_BOUNDS = _LENS_BOUNDS[1]
_SHEAR_CENTER = _LENS_CENTERS[1]
_SHEAR_FIXED = _LENS_FIXED[1]
_MULTIPOLE_BOUNDS = _LENS_BOUNDS[2]
_MULTIPOLE_CENTER = _LENS_CENTERS[2]
_MULTIPOLE_FIXED = _LENS_FIXED[2]
_EPL_CENTER = _LENS_CENTERS[0]

_SIMPLE_LL = _SIMPLE_LL_NO_AMP

# Expert-aligned base kwargs_params matching the shared priors above.
EXPERT_KWARGS_PARAMS_BASE: Dict[str, Any] = {
    "lens_model": [
        [{"theta_E": 1.0, "gamma": 2.0, "e1": 0.0, "e2": 0.0, "center_x": 0.0, "center_y": 0.0},
         {"gamma1": 0.0, "gamma2": 0.0},
         {"m": 4, "a_m": 0.01, "phi_m": 0.0, "center_x": 0.0, "center_y": 0.0}],
        [{"theta_E": 0.1, "gamma": 0.1, "e1": 0.1, "e2": 0.1, "center_x": 0.05, "center_y": 0.05},
         {"gamma1": 0.05, "gamma2": 0.05},
         {"a_m": 0.01, "phi_m": 0.1, "center_x": 0.05, "center_y": 0.05}],
        [{}, {"ra_0": 0, "dec_0": 0}, {"m": 4, "r_E": 1}],
        [{"theta_E": 0.01, "gamma": 1.5, "e1": -0.5, "e2": -0.5, "center_x": -10.0, "center_y": -10.0},
         {"gamma1": -0.3, "gamma2": -0.3},
         {"a_m": -0.1, "phi_m": -math.pi, "center_x": -10.0, "center_y": -10.0}],
        [{"theta_E": 5.0, "gamma": 2.8, "e1": 0.5, "e2": 0.5, "center_x": 10.0, "center_y": 10.0},
         {"gamma1": 0.3, "gamma2": 0.3},
         {"a_m": 0.1, "phi_m": math.pi, "center_x": 10.0, "center_y": 10.0}],
    ],
    "source_model": [
        [{"n_sersic": 2.0, "R_sersic": 0.5, "e1": 0.0, "e2": 0.0, "center_x": 0.0, "center_y": 0.0}],
        [{"n_sersic": 0.5, "R_sersic": 0.1, "e1": 0.1, "e2": 0.1, "center_x": 0.05, "center_y": 0.05}],
        [{}],
        [{"n_sersic": 0.5, "R_sersic": 0.01, "e1": -0.5, "e2": -0.5, "center_x": -10.0, "center_y": -10.0}],
        [{"n_sersic": 8.0, "R_sersic": 10.0, "e1": 0.5, "e2": 0.5, "center_x": 10.0, "center_y": 10.0}],
    ],
    "lens_light_model": [
        [{"amp": 100.0, "R_sersic": 0.5, "n_sersic": 2.0, "e1": 0.0, "e2": 0.0, "center_x": 0.0, "center_y": 0.0}],
        [{"amp": 100.0, "R_sersic": 0.1, "n_sersic": 0.5, "e1": 0.1, "e2": 0.1, "center_x": 0.05, "center_y": 0.05}],
        [{}],
        [{"amp": 0.001, "R_sersic": 0.01, "n_sersic": 0.5, "e1": -0.5, "e2": -0.5, "center_x": -10.0, "center_y": -10.0}],
        [{"amp": 100000.0, "R_sersic": 10.0, "n_sersic": 8.0, "e1": 0.5, "e2": 0.5, "center_x": 10.0, "center_y": 10.0}],
    ],
}

MODEL_COMBOS.update({
    6: {
        "label": "Standard EPL",
        "kwargs_model": {
            "lens_model_list": ["EPL", "SHEAR"],
            "lens_light_model_list": ["SERSIC_ELLIPSE"],
            "source_light_model_list": ["SERSIC_ELLIPSE"],
        },
        "bounds_lens": [dict(_EPL_BOUNDS), dict(_SHEAR_BOUNDS)],
        "sigmas_lens": [dict(_EPL_SIGMA), dict(_SHEAR_SIGMA)],
        "centers_lens": [dict(_EPL_CENTER), dict(_SHEAR_CENTER)],
        "fixed_lens": [{}, dict(_SHEAR_FIXED)],
        **_SIMPLE_LL,
    },
    7: {
        "label": "EPL + Multipoles",
        "kwargs_model": {
            "lens_model_list": ["EPL", "SHEAR", "MULTIPOLE"],
            "lens_light_model_list": ["SERSIC_ELLIPSE"],
            "source_light_model_list": ["SERSIC_ELLIPSE"],
        },
        "bounds_lens": list(_LENS_BOUNDS),
        "sigmas_lens": list(_LENS_SIGMAS),
        "centers_lens": list(_LENS_CENTERS),
        "fixed_lens": list(_LENS_FIXED),
        **_SIMPLE_LL,
    },
    9: {
        "label": "EPL + Convergence",
        "kwargs_model": {
            "lens_model_list": ["EPL", "SHEAR", "CONVERGENCE"],
            "lens_light_model_list": ["SERSIC_ELLIPSE"],
            "source_light_model_list": ["SERSIC_ELLIPSE"],
        },
        "bounds_lens": [
            dict(_EPL_BOUNDS), dict(_SHEAR_BOUNDS),
            {"kappa": (-0.2, 0.2)},
        ],
        "sigmas_lens": [dict(_EPL_SIGMA), dict(_SHEAR_SIGMA), {}],
        "centers_lens": [
            dict(_EPL_CENTER), dict(_SHEAR_CENTER),
            {"kappa": 0.0},
        ],
        "fixed_lens": [{}, dict(_SHEAR_FIXED), {"ra_0": 0.0, "dec_0": 0.0}],
        **_SIMPLE_LL,
    },
    10: {
        "label": "Stars (Hernquist) + DM (NFW)",
        "kwargs_model": {
            "lens_model_list": ["HERNQUIST", "NFW", "SHEAR"],
            "lens_light_model_list": ["HERNQUIST"],
            "source_light_model_list": ["SERSIC_ELLIPSE"],
        },
        "bounds_lens": [
            {"sigma0": (0.01, 5.0), "Rs": (0.1, 10.0),
             "center_x": (-10.0, 10.0), "center_y": (-10.0, 10.0)},
            {"Rs": (1.0, 100.0), "alpha_Rs": (0.01, 10.0),
             "center_x": (-10.0, 10.0), "center_y": (-10.0, 10.0)},
            dict(_SHEAR_BOUNDS),
        ],
        "sigmas_lens": [{}, dict(_NFW_SIGMA), dict(_SHEAR_SIGMA)],
        "centers_lens": [
            {"sigma0": 1.0, "Rs": 1.0, "center_x": 0.0, "center_y": 0.0},
            {"Rs": 10.0, "alpha_Rs": 1.0, "center_x": 0.0, "center_y": 0.0},
            dict(_SHEAR_CENTER),
        ],
        "fixed_lens": [{}, {}, dict(_SHEAR_FIXED)],
        "bounds_ll": [{"Rs": (0.01, 10.0),
                       "center_x": (-10.0, 10.0), "center_y": (-10.0, 10.0)}],
        "sigmas_ll": [{}],
        "centers_ll": [{"Rs": 1.0, "center_x": 0.0, "center_y": 0.0}],
        "fixed_ll": [{}],
        "bounds_src": [dict(_SRC_BOUNDS)],
        "sigmas_src": [dict(_SRC_SIGMA)],
        "centers_src": [dict(_SRC_CENTER)],
        "fixed_src": [{}],
    },
    11: {
        "label": "Group (EPL + SIS Satellites)",
        "kwargs_model": {
            "lens_model_list": ["EPL", "SHEAR", "SIS"],
            "lens_light_model_list": ["SERSIC_ELLIPSE", "SERSIC"],
            "source_light_model_list": ["SERSIC_ELLIPSE"],
        },
        "bounds_lens": [
            dict(_EPL_BOUNDS), dict(_SHEAR_BOUNDS),
            {"theta_E": (0.01, 2.0),
             "center_x": (-2.0, 2.0), "center_y": (-2.0, 2.0)},
        ],
        "sigmas_lens": [dict(_EPL_SIGMA), dict(_SHEAR_SIGMA), dict(_SIS_SIGMA)],
        "centers_lens": [
            dict(_EPL_CENTER), dict(_SHEAR_CENTER),
            {"theta_E": 0.3, "center_x": 1.0, "center_y": 0.0},
        ],
        "fixed_lens": [{}, dict(_SHEAR_FIXED), {}],
        "bounds_ll": [
            dict(_LL_BOUNDS),
            {"n_sersic": (0.5, 8.0), "R_sersic": (0.01, 10.0),
             "center_x": (-10.0, 10.0), "center_y": (-10.0, 10.0)},
        ],
        "sigmas_ll": [dict(_LL_SIGMA), dict(_SERSIC_SIGMA)],
        "centers_ll": [
            dict(_LL_CENTER),
            {"n_sersic": 2.0, "R_sersic": 0.5,
             "center_x": 0.0, "center_y": 0.0},
        ],
        "fixed_ll": [{}, {}],
        "bounds_src": [dict(_SRC_BOUNDS)],
        "sigmas_src": [dict(_SRC_SIGMA)],
        "centers_src": [dict(_SRC_CENTER)],
        "fixed_src": [{}],
    },
    12: {
        "label": "Substructure (EPL + Gaussian Clumps)",
        "kwargs_model": {
            # lenstronomy does not ship GAUSSIAN_KAPPA; use the native
            # GAUSSIAN_ELLIPSE_KAPPA substitute directly.
            "lens_model_list": ["EPL", "SHEAR", "GAUSSIAN_ELLIPSE_KAPPA"],
            "lens_light_model_list": ["SERSIC_ELLIPSE"],
            "source_light_model_list": ["SERSIC_ELLIPSE"],
        },
        "bounds_lens": [
            dict(_EPL_BOUNDS), dict(_SHEAR_BOUNDS),
            dict(_GAUSSIAN_SUB_BOUNDS),
        ],
        "sigmas_lens": [dict(_EPL_SIGMA), dict(_SHEAR_SIGMA), dict(_GAUSSIAN_SUB_SIGMA)],
        "centers_lens": [
            dict(_EPL_CENTER), dict(_SHEAR_CENTER),
            dict(_GAUSSIAN_SUB_CENTER),
        ],
        "fixed_lens": [{}, dict(_SHEAR_FIXED), {}],
        **_SIMPLE_LL,
    },
    13: {
        "label": "Merger/Dual Center",
        "kwargs_model": {
            "lens_model_list": ["EPL", "EPL", "SHEAR"],
            "lens_light_model_list": ["SERSIC_ELLIPSE", "SERSIC_ELLIPSE"],
            "source_light_model_list": ["SERSIC_ELLIPSE"],
        },
        "bounds_lens": [
            dict(_EPL_BOUNDS),
            dict(_EPL_BOUNDS),
            dict(_SHEAR_BOUNDS),
        ],
        "sigmas_lens": [dict(_EPL_SIGMA), dict(_EPL_SIGMA), dict(_SHEAR_SIGMA)],
        "centers_lens": [
            dict(_EPL_CENTER),
            dict(_EPL_CENTER),
            dict(_SHEAR_CENTER),
        ],
        "fixed_lens": [{}, {}, dict(_SHEAR_FIXED)],
        "bounds_ll": [dict(_LL_BOUNDS), dict(_LL_BOUNDS)],
        "sigmas_ll": [dict(_LL_SIGMA), dict(_LL_SIGMA)],
        "centers_ll": [dict(_LL_CENTER), dict(_SERSIC_CENTER_ENVELOPE)],
        "fixed_ll": [{}, {}],
        "bounds_src": [dict(_SRC_BOUNDS)],
        "sigmas_src": [dict(_SRC_SIGMA)],
        "centers_src": [dict(_SRC_CENTER)],
        "fixed_src": [{}],
    },
})

# ---------------------------------------------------------------------------
# Expert v2 model families (from try_all_models (1).ipynb)
# PEMD/IEMD mass models + double Sersic lens light + GAUSSIAN_KAPPA
# ---------------------------------------------------------------------------

_PEMD_BOUNDS: Dict[str, Tuple[float, float]] = {
    "theta_E": (0.01, 5.0), "gamma": (1.5, 2.8),
    "e1": (-0.5, 0.5), "e2": (-0.5, 0.5),
    "center_x": (-10.0, 10.0), "center_y": (-10.0, 10.0),
}
_PEMD_CENTER: Dict[str, float] = {
    "theta_E": 1.0, "gamma": 2.0, "e1": 0.0, "e2": 0.0,
    "center_x": 0.0, "center_y": 0.0,
}
_SIE_BOUNDS: Dict[str, Tuple[float, float]] = {
    "theta_E": (0.01, 5.0),
    "e1": (-0.5, 0.5), "e2": (-0.5, 0.5),
    "center_x": (-10.0, 10.0), "center_y": (-10.0, 10.0),
}
_SIE_CENTER: Dict[str, float] = {
    "theta_E": 1.0, "e1": 0.0, "e2": 0.0,
    "center_x": 0.0, "center_y": 0.0,
}
_DOUBLE_LL: Dict[str, Any] = {
    "bounds_ll": [dict(_LL_BOUNDS), dict(_LL_BOUNDS)],
    "sigmas_ll": [dict(_LL_SIGMA), dict(_LL_SIGMA)],
    "centers_ll": [dict(_LL_CENTER), dict(_SERSIC_CENTER_ENVELOPE)],
    "fixed_ll": [{}, {}],
    "bounds_src": [dict(_SRC_BOUNDS), dict(_SHAPELET_SRC_BOUNDS)],
    "sigmas_src": [dict(_SRC_SIGMA), dict(_SHAPELET_SRC_SIGMA)],
    "centers_src": [dict(_SRC_CENTER), dict(_SHAPELET_SRC_CENTER)],
    "fixed_src": [{}, dict(_SHAPELET_SRC_FIXED)],
}
_DOUBLE_LL_WITH_AMP: Dict[str, Any] = {
    "bounds_ll": [dict(_LL_BOUNDS_WITH_AMP), dict(_LL_BOUNDS_WITH_AMP)],
    "sigmas_ll": [dict(_LL_SIGMA_WITH_AMP), dict(_LL_SIGMA_WITH_AMP)],
    "centers_ll": [dict(_LL_CENTER_WITH_AMP), dict(_SERSIC_CENTER_ENVELOPE_WITH_AMP)],
    "fixed_ll": [{}, {}],
    "bounds_src": [dict(_SRC_BOUNDS), dict(_SHAPELET_SRC_BOUNDS)],
    "sigmas_src": [dict(_SRC_SIGMA), dict(_SHAPELET_SRC_SIGMA)],
    "centers_src": [dict(_SRC_CENTER), dict(_SHAPELET_SRC_CENTER)],
    "fixed_src": [{}, dict(_SHAPELET_SRC_FIXED)],
}

_HERNQUIST_LL_BOUNDS: Dict[str, Tuple[float, float]] = {
    "Rs": (0.01, 10.0),
    "center_x": (-10.0, 10.0), "center_y": (-10.0, 10.0),
}
_HERNQUIST_LL_CENTER: Dict[str, float] = {
    "Rs": 1.0, "center_x": 0.0, "center_y": 0.0,
}

MODEL_COMBOS.update({
    14: {
        "label": "PEMD + Shear",
        "kwargs_model": {
            "lens_model_list": ["PEMD", "SHEAR"],
            "lens_light_model_list": ["SERSIC_ELLIPSE"],
            "source_light_model_list": ["SERSIC_ELLIPSE"],
        },
        "bounds_lens": [dict(_PEMD_BOUNDS), dict(_SHEAR_BOUNDS)],
        "sigmas_lens": [dict(_PEMD_SIGMA), dict(_SHEAR_SIGMA)],
        "centers_lens": [dict(_PEMD_CENTER), dict(_SHEAR_CENTER)],
        "fixed_lens": [{}, dict(_SHEAR_FIXED)],
        **_SIMPLE_LL,
    },
    15: {
        "label": "SIE + Shear",
        "kwargs_model": {
            "lens_model_list": ["SIE", "SHEAR"],
            "lens_light_model_list": ["SERSIC_ELLIPSE"],
            "source_light_model_list": ["SERSIC_ELLIPSE"],
        },
        "bounds_lens": [dict(_SIE_BOUNDS), dict(_SHEAR_BOUNDS)],
        "sigmas_lens": [dict(_SIE_SIGMA), dict(_SHEAR_SIGMA)],
        "centers_lens": [dict(_SIE_CENTER), dict(_SHEAR_CENTER)],
        "fixed_lens": [{}, dict(_SHEAR_FIXED)],
        **_SIMPLE_LL,
    },
    16: {
        "label": "Stars (Hernquist) + DM (NFW) v2",
        "kwargs_model": {
            "lens_model_list": ["HERNQUIST", "NFW", "SHEAR"],
            "lens_light_model_list": ["HERNQUIST"],
            "source_light_model_list": ["SERSIC_ELLIPSE"],
        },
        "bounds_lens": [
            {"sigma0": (0.01, 5.0), "Rs": (0.1, 10.0),
             "center_x": (-10.0, 10.0), "center_y": (-10.0, 10.0)},
            {"Rs": (1.0, 100.0), "alpha_Rs": (0.01, 10.0),
             "center_x": (-10.0, 10.0), "center_y": (-10.0, 10.0)},
            dict(_SHEAR_BOUNDS),
        ],
        "sigmas_lens": [{}, dict(_NFW_SIGMA), dict(_SHEAR_SIGMA)],
        "centers_lens": [
            {"sigma0": 1.0, "Rs": 1.0, "center_x": 0.0, "center_y": 0.0},
            {"Rs": 10.0, "alpha_Rs": 1.0, "center_x": 0.0, "center_y": 0.0},
            dict(_SHEAR_CENTER),
        ],
        "fixed_lens": [{}, {}, dict(_SHEAR_FIXED)],
        "bounds_ll": [dict(_HERNQUIST_LL_BOUNDS)],
        "sigmas_ll": [{}],
        "centers_ll": [dict(_HERNQUIST_LL_CENTER)],
        "fixed_ll": [{}],
        "bounds_src": [dict(_SRC_BOUNDS)],
        "sigmas_src": [dict(_SRC_SIGMA)],
        "centers_src": [dict(_SRC_CENTER)],
        "fixed_src": [{}],
    },
    17: {
        "label": "Group (EPL + SIS) v2",
        "kwargs_model": {
            "lens_model_list": ["EPL", "SHEAR", "SIS"],
            "lens_light_model_list": ["SERSIC_ELLIPSE", "SERSIC"],
            "source_light_model_list": ["SERSIC_ELLIPSE"],
        },
        "bounds_lens": [
            dict(_EPL_BOUNDS), dict(_SHEAR_BOUNDS),
            {"theta_E": (0.01, 2.0),
             "center_x": (-2.0, 2.0), "center_y": (-2.0, 2.0)},
        ],
        "sigmas_lens": [dict(_EPL_SIGMA), dict(_SHEAR_SIGMA), dict(_SIS_SIGMA)],
        "centers_lens": [
            dict(_EPL_CENTER), dict(_SHEAR_CENTER),
            {"theta_E": 0.3, "center_x": 1.0, "center_y": 0.0},
        ],
        "fixed_lens": [{}, dict(_SHEAR_FIXED), {}],
        "bounds_ll": [dict(_LL_BOUNDS),
                      {"n_sersic": (0.5, 8.0), "R_sersic": (0.01, 10.0),
                       "center_x": (-10.0, 10.0), "center_y": (-10.0, 10.0)}],
        "sigmas_ll": [dict(_LL_SIGMA), dict(_SERSIC_SIGMA)],
        "centers_ll": [dict(_LL_CENTER),
                       {"n_sersic": 2.0, "R_sersic": 0.5,
                        "center_x": 0.0, "center_y": 0.0}],
        "fixed_ll": [{}, {}],
        "bounds_src": [dict(_SRC_BOUNDS)],
        "sigmas_src": [dict(_SRC_SIGMA)],
        "centers_src": [dict(_SRC_CENTER)],
        "fixed_src": [{}],
    },
    18: {
        "label": "Substructure (GAUSSIAN_KAPPA) v2",
        "kwargs_model": {
            "lens_model_list": ["EPL", "SHEAR", "GAUSSIAN_ELLIPSE_KAPPA"],
            "lens_light_model_list": ["SERSIC_ELLIPSE"],
            "source_light_model_list": ["SERSIC_ELLIPSE"],
        },
        "bounds_lens": [
            dict(_EPL_BOUNDS), dict(_SHEAR_BOUNDS),
            dict(_GAUSSIAN_SUB_BOUNDS),
        ],
        "sigmas_lens": [dict(_EPL_SIGMA), dict(_SHEAR_SIGMA), dict(_GAUSSIAN_SUB_SIGMA)],
        "centers_lens": [
            dict(_EPL_CENTER), dict(_SHEAR_CENTER),
            dict(_GAUSSIAN_SUB_CENTER),
        ],
        "fixed_lens": [{}, dict(_SHEAR_FIXED), {}],
        **_SIMPLE_LL,
    },
    19: {
        "label": "Merger/Dual Center v2",
        "kwargs_model": {
            "lens_model_list": ["EPL", "EPL", "SHEAR"],
            "lens_light_model_list": ["SERSIC_ELLIPSE", "SERSIC_ELLIPSE"],
            "source_light_model_list": ["SERSIC_ELLIPSE"],
        },
        "bounds_lens": [
            dict(_EPL_BOUNDS),
            dict(_EPL_BOUNDS),
            dict(_SHEAR_BOUNDS),
        ],
        "sigmas_lens": [dict(_EPL_SIGMA), dict(_EPL_SIGMA), dict(_SHEAR_SIGMA)],
        "centers_lens": [
            dict(_EPL_CENTER),
            dict(_EPL_CENTER),
            dict(_SHEAR_CENTER),
        ],
        "fixed_lens": [{}, {}, dict(_SHEAR_FIXED)],
        **_DOUBLE_LL,
    },
})

# ---------------------------------------------------------------------------
# Active priors (set by set_model_combo, default = combo 2)
# ---------------------------------------------------------------------------


build_combo8(DEFAULT_MGE_COMPONENTS)


def _inject_shapelet_src() -> None:
    """Ensure every combo has SHAPELETS in source_light_model_list.

    The expert notebook uses [SERSIC_ELLIPSE, SHAPELETS] for source light.
    This adds one SHAPELETS component tied to the first source SERSIC_ELLIPSE.
    Stores ``combo["shapelet_src_ties"]`` mapping the shapelet index to its
    parent source-light index.
    """
    for cid, combo in MODEL_COMBOS.items():
        src_list = combo["kwargs_model"].get("source_light_model_list", [])
        if "SHAPELETS" in src_list:
            n_non = sum(1 for m in src_list if "SHAPELETS" not in m)
            combo["shapelet_src_ties"] = {
                i: j for j, (i, m) in enumerate(
                    (idx, m) for idx, m in enumerate(src_list) if "SHAPELETS" in m
                )
            }
            continue

        non_shapelet = [m for m in src_list if "SHAPELETS" not in m]
        n_parent = len(non_shapelet)
        if n_parent == 0:
            continue

        combo["kwargs_model"]["source_light_model_list"] = list(non_shapelet) + ["SHAPELETS"]

        old_bounds = combo.get("bounds_src", [])
        old_centers = combo.get("centers_src", [])
        old_fixed = combo.get("fixed_src", [])

        new_bounds = list(old_bounds[:n_parent])
        new_centers = list(old_centers[:n_parent])
        new_fixed = list(old_fixed[:n_parent])
        while len(new_bounds) < n_parent:
            new_bounds.append({})
        while len(new_centers) < n_parent:
            new_centers.append({})
        while len(new_fixed) < n_parent:
            new_fixed.append({})

        new_bounds.append(dict(_SHAPELET_SRC_BOUNDS))
        new_centers.append(dict(_SHAPELET_SRC_CENTER))
        new_fixed.append(dict(_SHAPELET_SRC_FIXED))

        combo["bounds_src"] = new_bounds
        combo["centers_src"] = new_centers
        combo["fixed_src"] = new_fixed

        combo["shapelet_src_ties"] = {n_parent: 0}


_inject_shapelet_src()


def materialize_source_ties(
    kwargs_source: List[Dict[str, Any]],
    *,
    combo: Optional[Dict[str, Any]] = None,
    fixed_src: Optional[List[Dict[str, Any]]] = None,
    centers_src: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Return a concrete source-kwargs list with native tie metadata applied.

    FittingSequence handles source-shapelet ties natively through
    ``joint_source_with_source`` constraints. Low-level image evaluation does
    not consume a constraints object, so we materialize the tied coordinates
    here from the active combo metadata rather than mutating proposals in-place
    inside the evaluator.
    """
    src = [dict(comp) for comp in kwargs_source]
    combo = combo or MODEL_COMBOS.get(ACTIVE_COMBO, {})
    fixed_src = fixed_src or FIXED_PARAMS.get("kwargs_source", combo.get("fixed_src", []))
    centers_src = centers_src or PRIOR_CENTERS.get("kwargs_source", combo.get("centers_src", []))

    for idx, fixed in enumerate(fixed_src):
        while len(src) <= idx:
            src.append(dict(centers_src[idx]) if idx < len(centers_src) else {})
        for key, value in fixed.items():
            src[idx].setdefault(key, value)

    for shapelet_idx, parent_idx in combo.get("shapelet_src_ties", {}).items():
        while len(src) <= shapelet_idx:
            src.append(dict(centers_src[shapelet_idx]) if shapelet_idx < len(centers_src) else {})
        if parent_idx >= len(src):
            continue
        parent = src[parent_idx]
        child = src[shapelet_idx]
        parent_center_x = parent.get(
            "center_x",
            centers_src[parent_idx].get("center_x", 0.0) if parent_idx < len(centers_src) else 0.0,
        )
        parent_center_y = parent.get(
            "center_y",
            centers_src[parent_idx].get("center_y", 0.0) if parent_idx < len(centers_src) else 0.0,
        )
        child["center_x"] = parent_center_x
        child["center_y"] = parent_center_y
        if shapelet_idx < len(fixed_src):
            for key, value in fixed_src[shapelet_idx].items():
                child.setdefault(key, value)
    return src


def inject_fixed_params(
    proposal: Dict[str, Any],
    *,
    fixed_params: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    prior_centers: Optional[Dict[str, List[Dict[str, Any]]]] = None,
) -> Dict[str, Any]:
    """Return a proposal with fixed params reapplied component-by-component.

    This is the server-side counterpart to prompt-level guidance: if a caller
    marks components as fixed, any conflicting values in ``proposal`` are
    overwritten here before evaluation so frozen params stay frozen.
    """
    fixed_src = fixed_params if fixed_params is not None else FIXED_PARAMS
    centers_src = prior_centers if prior_centers is not None else PRIOR_CENTERS
    out = copy.deepcopy(proposal)

    for key in ("kwargs_lens", "kwargs_lens_light", "kwargs_source"):
        fixed_list = fixed_src.get(key, [])
        params_list = list(out.get(key, []))
        if len(params_list) > len(fixed_list):
            raise ValueError(
                f"{key}: you provided {len(params_list)} components but "
                f"the model expects exactly {len(fixed_list)}. "
                f"Remove the extra components.")
        centers_list = centers_src.get(key, [{}] * len(fixed_list))
        while len(params_list) < len(fixed_list):
            idx = len(params_list)
            params_list.append(
                dict(centers_list[idx]) if idx < len(centers_list) else {}
            )
        for idx, fixed in enumerate(fixed_list):
            if fixed:
                params_list[idx].update(fixed)
        out[key] = params_list

    combo = MODEL_COMBOS.get(ACTIVE_COMBO, {})
    out["kwargs_source"] = materialize_source_ties(
        out.get("kwargs_source", []),
        combo=combo,
        fixed_src=fixed_src.get("kwargs_source"),
        centers_src=centers_src.get("kwargs_source"),
    )
    return out


def pack_mge_proposal(
    proposal: Dict[str, Any],
    lens_model_list: list = None,
    lens_light_model_list: list = None,
) -> Dict[str, Any]:
    """Pack indexed MULTI_GAUSSIAN kwargs into lenstronomy's array form."""

    def _has_indexed_mge(components: List[Dict[str, Any]]) -> bool:
        for comp in components:
            if any(k.startswith("amp_") or k.startswith("sigma_") for k in comp):
                return True
        return False

    def _pack_components(
        components: List[Dict[str, Any]],
        valid_keys: set[str],
    ) -> List[Dict[str, Any]]:
        packed = []
        for comp in components:
            has_indexed = any(k.startswith("amp_") or k.startswith("sigma_") for k in comp)
            if not has_indexed:
                packed.append(comp)
                continue
            amps = []
            sigmas = []
            other = {}
            for k, v in comp.items():
                if k.startswith("amp_"):
                    idx = int(k.split("_", 1)[1])
                    while len(amps) <= idx:
                        amps.append(0.0)
                    amps[idx] = v
                elif k.startswith("sigma_"):
                    idx = int(k.split("_", 1)[1])
                    while len(sigmas) <= idx:
                        sigmas.append(0.1)
                    sigmas[idx] = v
                elif k in valid_keys:
                    other[k] = v
            other["amp"] = amps
            other["sigma"] = sigmas
            packed.append(other)
        return packed

    has_lens_mge = bool(lens_model_list and "MULTI_GAUSSIAN" in lens_model_list)
    if not has_lens_mge:
        combo = MODEL_COMBOS.get(ACTIVE_COMBO, {})
        has_lens_mge = combo.get("n_mge", 0) > 0
    if not has_lens_mge:
        has_lens_mge = _has_indexed_mge(proposal.get("kwargs_lens", []))

    has_light_mge = bool(lens_light_model_list and "MULTI_GAUSSIAN" in lens_light_model_list)
    if not has_light_mge:
        has_light_mge = _has_indexed_mge(proposal.get("kwargs_lens_light", []))

    if not has_lens_mge and not has_light_mge:
        return proposal

    packed = dict(proposal)
    if has_lens_mge:
        packed["kwargs_lens"] = _pack_components(
            proposal.get("kwargs_lens", []),
            {"center_x", "center_y", "scale_factor"},
        )
    if has_light_mge:
        packed["kwargs_lens_light"] = _pack_components(
            proposal.get("kwargs_lens_light", []),
            {"center_x", "center_y"},
        )
    return packed


PRIOR_BOUNDS: Dict[str, List[Dict[str, Tuple[float, float]]]] = {}
PRIOR_CENTERS: Dict[str, List[Dict[str, float]]] = {}
PRIOR_SIGMAS: Dict[str, List[Dict[str, float]]] = {}
FIXED_PARAMS: Dict[str, List[Dict[str, Any]]] = {}
ACTIVE_COMBO: int = 0
BLIND_MODE: bool = False
KIN_SOFT: bool = False
PHYSICALITY_MODE: str = "off"
CHI2_PENALTY: str = "linear"  # "linear" = |chi2 - 1| (default), "log" = |log(chi2)|
SUBTRACTED_CHI2: bool = False  # compute chi2 on lens-light-subtracted residuals
NO_LINEAR_SOLVE: bool = False  # if True, LLM predicts amp; no linear solver


def _inject_amp_for_no_linear_solve(
    bounds_ll: list, centers_ll: list, sigmas_ll: list,
    bounds_src: list, centers_src: list, sigmas_src: list,
) -> None:
    """When NO_LINEAR_SOLVE is active, ensure amp is in light bounds/centers."""

    def _has_indexed_amp(comp: Dict[str, Any]) -> bool:
        return any(k.startswith("amp_") for k in comp)

    for bd, sd in zip(bounds_ll, sigmas_ll):
        if "amp" not in bd and not _has_indexed_amp(bd):
            bd["amp"] = (0.001, 100000.0)
            sd["amp"] = 10.0
    for cd in centers_ll:
        if "amp" not in cd and not _has_indexed_amp(cd):
            cd["amp"] = 100.0
    for bd, sd in zip(bounds_src, sigmas_src):
        if "amp" not in bd and "n_scales" not in bd and not _has_indexed_amp(bd):
            bd["amp"] = (0.001, 100000.0)
            sd["amp"] = 10.0
    for cd in centers_src:
        if "amp" not in cd and "n_scales" not in cd and not _has_indexed_amp(cd):
            cd["amp"] = 100.0


def set_model_combo(combo_id: int) -> Dict[str, Any]:
    """Activate a model combo, updating module-level priors in-place."""
    global ACTIVE_COMBO
    if combo_id not in MODEL_COMBOS:
        raise ValueError(f"Unknown combo {combo_id}. Choose from {list(MODEL_COMBOS)}")
    c = MODEL_COMBOS[combo_id]

    bounds_ll = [dict(d) for d in c["bounds_ll"]]
    centers_ll = [dict(d) for d in c["centers_ll"]]
    sigmas_ll = [dict(d) for d in c.get("sigmas_ll", [{} for _ in bounds_ll])]
    bounds_src = [dict(d) for d in c["bounds_src"]]
    centers_src = [dict(d) for d in c["centers_src"]]
    sigmas_src = [dict(d) for d in c.get("sigmas_src", [{} for _ in bounds_src])]
    bounds_lens = [dict(d) for d in c.get("bounds_lens", list(_LENS_BOUNDS))]
    centers_lens = [dict(d) for d in c.get("centers_lens", list(_LENS_CENTERS))]
    sigmas_lens = [dict(d) for d in c.get("sigmas_lens", [{} for _ in bounds_lens])]

    if NO_LINEAR_SOLVE:
        _inject_amp_for_no_linear_solve(
            bounds_ll, centers_ll, sigmas_ll, bounds_src, centers_src, sigmas_src)

    PRIOR_BOUNDS.clear()
    PRIOR_BOUNDS.update({
        "kwargs_lens": bounds_lens,
        "kwargs_lens_light": bounds_ll,
        "kwargs_source": bounds_src,
    })
    PRIOR_CENTERS.clear()
    PRIOR_CENTERS.update({
        "kwargs_lens": centers_lens,
        "kwargs_lens_light": centers_ll,
        "kwargs_source": centers_src,
    })
    PRIOR_SIGMAS.clear()
    PRIOR_SIGMAS.update({
        "kwargs_lens": sigmas_lens,
        "kwargs_lens_light": sigmas_ll,
        "kwargs_source": sigmas_src,
    })
    FIXED_PARAMS.clear()
    FIXED_PARAMS.update({
        "kwargs_lens": c.get("fixed_lens", list(_LENS_FIXED)),
        "kwargs_lens_light": c["fixed_ll"],
        "kwargs_source": c["fixed_src"],
    })
    ACTIVE_COMBO = combo_id
    n_free = sum(len(comp) for bl in PRIOR_BOUNDS.values() for comp in bl)
    log.info("Model combo %d: %s  (%d free params, no_linear_solve=%s)",
             combo_id, c["label"], n_free, NO_LINEAR_SOLVE)
    return c["kwargs_model"]


set_model_combo(2)

# ---------------------------------------------------------------------------
# Tunable constants
# ---------------------------------------------------------------------------

ALPHA = 5.0      # weight on |log(chi2)| — chi2=1 is optimal
BETA = 0.5       # weight on kinematic chi2
GAMMA = 0.5      # weight on boundary penalty
DELTA = 2.0      # weight on residual_randomness (0-1 scale, lower=better)
EPSILON = 10.0   # weight on physicality penalty (only when rmse > 0.05 threshold)

ALPHA_P15 = 8.0  # pass1.5: stronger chi2 emphasis for fine-tuning
BETA_P15 = 0.5   # pass1.5: kinematics unchanged
GAMMA_P15 = 0.3  # pass1.5: lighter boundary (already near-optimal)
DELTA_P15 = 1.0  # pass1.5: halved residual-randomness weight
DIVERSITY_K = 5  # k for k-nearest-neighbor diversity
DEDUP_EPSILON = 0.01  # min L2 distance in normalized param space for dedup

# Once chi2 is already within 1%, keep all behavior unchanged outside that band
# but sharply reward the closest-to-1 solutions inside it.
CHI2_FINE_BAND = 0.01
CHI2_FINE_BOOST = 10.0

# Inside an ultra-tiny inner band, further amplify microscopic chi2
# differences so values like 1.00002 can decisively outrank 1.0002, but
# without changing behavior outside this narrow window.
CHI2_ULTRA_FINE_BAND = 2.0e-4
CHI2_ULTRA_FINE_BOOST = 50.0

# ---------------------------------------------------------------------------
# Quality
# ---------------------------------------------------------------------------


def _boundary_penalty(proposal: Dict[str, Any]) -> float:
    """Sum of exponential penalties for parameters near prior boundaries."""

    def _bound_value(params: Dict[str, Any], pname: str):
        val = params.get(pname)
        if val is not None:
            return val
        if pname.startswith("amp_"):
            amps = params.get("amp")
            if isinstance(amps, (list, tuple, np.ndarray)):
                idx = int(pname.split("_", 1)[1])
                if idx < len(amps):
                    return amps[idx]
        if pname.startswith("sigma_"):
            sigmas = params.get("sigma")
            if isinstance(sigmas, (list, tuple, np.ndarray)):
                idx = int(pname.split("_", 1)[1])
                if idx < len(sigmas):
                    return sigmas[idx]
        return None

    penalty = 0.0
    for key in ("kwargs_lens", "kwargs_lens_light", "kwargs_source"):
        bounds_list = PRIOR_BOUNDS.get(key, [])
        params_list = proposal.get(key, [])
        for comp_idx, bounds in enumerate(bounds_list):
            if comp_idx >= len(params_list):
                continue
            params = params_list[comp_idx]
            for pname, (lo, hi) in bounds.items():
                val = _bound_value(params, pname)
                if val is None:
                    continue
                span = hi - lo
                if span <= 0:
                    continue
                dist_lo = (val - lo) / span
                dist_hi = (hi - val) / span
                margin = 0.05
                if dist_lo < margin:
                    exp_arg = -dist_lo / margin
                    penalty += math.exp(min(exp_arg, 20.0))
                if dist_hi < margin:
                    exp_arg = -dist_hi / margin
                    penalty += math.exp(min(exp_arg, 20.0))
    return penalty


def residual_randomness(eval_results: Dict[str, Any]) -> Optional[float]:
    """Measure how structured (non-random) the residual map is.

    Uses the normalized radial autocorrelation of the residual.
    For pure noise, autocorrelation drops to ~0 at lag > 0.
    For structured residuals (rings, arcs, gradients), autocorrelation
    stays high at larger lags.

    Returns the mean absolute autocorrelation at lags 3-15 pixels,
    normalized so pure noise gives ~0.02 and strong structure gives
    ~0.3-0.5.  Lower = more random = better.
    """
    residual = eval_results.get("residual_map")
    if residual is None:
        return None
    if hasattr(residual, '__len__') and not isinstance(residual, np.ndarray):
        residual = np.array(residual)
    if residual.ndim != 2 or residual.size == 0:
        return None

    r = residual - residual.mean()
    var = np.mean(r ** 2)
    if var < 1e-12:
        return 0.0

    fft = np.fft.fft2(r)
    power = np.abs(fft) ** 2
    autocorr = np.real(np.fft.ifft2(power)) / (var * r.size)

    ny, nx = r.shape
    cy, cx = ny // 2, nx // 2
    autocorr_centered = np.fft.fftshift(autocorr)

    y, x = np.mgrid[:ny, :nx]
    dist = np.sqrt((y - cy)**2 + (x - cx)**2)

    values = []
    for lag in range(3, 16):
        mask = (dist >= lag - 0.5) & (dist < lag + 0.5)
        if mask.any():
            values.append(float(np.abs(autocorr_centered[mask]).mean()))

    return float(np.mean(values)) if values else 0.0


KIN_CHI2_MISSING_PENALTY = 50.0


def _chi2_penalty_base(chi2: float, penalty_mode: Optional[str] = None) -> float:
    """Base chi2 penalty before any near-1 emphasis."""
    mode = penalty_mode or CHI2_PENALTY
    chi2_val = float(chi2)
    if mode == "log":
        return abs(math.log(max(chi2_val, 1e-6)))
    return abs(chi2_val - 1.0)


def _band_edge_chi2(chi2_val: float, band: float) -> float:
    """Return the signed band-edge chi2 value on the same side of 1."""
    if chi2_val > 1.0:
        return 1.0 + band
    if chi2_val < 1.0:
        return 1.0 - band
    return 1.0 + band


def _chi2_penalty_fine_band(chi2: float, penalty_mode: Optional[str] = None) -> float:
    """Chi2 penalty with the outer near-1 emphasis applied."""
    chi2_val = float(chi2)
    base_penalty = _chi2_penalty_base(chi2_val, penalty_mode)
    raw_delta = abs(chi2_val - 1.0)
    if raw_delta >= CHI2_FINE_BAND:
        return base_penalty

    edge_penalty = _chi2_penalty_base(
        _band_edge_chi2(chi2_val, CHI2_FINE_BAND), penalty_mode)
    return base_penalty - CHI2_FINE_BOOST * (edge_penalty - base_penalty)


def chi2_priority_penalty(chi2: float, penalty_mode: Optional[str] = None) -> float:
    """Chi2 penalty with nested near-1 emphasis bands.

    Outside ``|chi2-1| < CHI2_FINE_BAND`` this returns the unchanged base
    penalty. Inside that band it smoothly boosts quality in proportion to how
    much closer the fit is to 1.0 than the band edge, making chi2 dominate
    tie-breaks among already-good fits without affecting broader behavior.

    Inside ``|chi2-1| < CHI2_ULTRA_FINE_BAND`` it applies a second, much
    narrower boost so microscopic differences very close to 1.0 are still
    meaningful on the scale of the kinematic term.
    """
    chi2_val = float(chi2)
    fine_penalty = _chi2_penalty_fine_band(chi2_val, penalty_mode)
    raw_delta = abs(chi2_val - 1.0)
    if raw_delta >= CHI2_ULTRA_FINE_BAND:
        return fine_penalty

    edge_penalty = _chi2_penalty_fine_band(
        _band_edge_chi2(chi2_val, CHI2_ULTRA_FINE_BAND), penalty_mode)
    return fine_penalty - CHI2_ULTRA_FINE_BOOST * (edge_penalty - fine_penalty)


def compute_quality(
    eval_results: Dict[str, Any],
    proposal: Dict[str, Any],
    alpha: float = ALPHA,
    beta: float = BETA,
    gamma: float = GAMMA,
    delta: float = DELTA,
) -> float:
    """Scalar quality score (higher is better).

    ``-alpha * chi2_penalty - delta * residual_randomness - beta * chi2_kin - gamma * boundary``
    """
    chi2_img = _coerce_float(eval_results.get("image_chi2_reduced"), 1e6)
    chi2_penalty = chi2_priority_penalty(chi2_img)
    bp = _boundary_penalty(proposal)
    if BLIND_MODE:
        return -alpha * chi2_penalty - gamma * bp
    raw_kin = eval_results.get("kin_chi2")
    chi2_kin = float(raw_kin) if raw_kin is not None else KIN_CHI2_MISSING_PENALTY
    if KIN_SOFT:
        sigma_pred = eval_results.get("sigma_predicted")
        sigma_obs = eval_results.get("sigma_observed")
        sigma_err = eval_results.get("sigma_observed_err")
        if sigma_pred is not None and sigma_obs is not None and sigma_err and sigma_err > 0:
            deviation = abs(sigma_pred - sigma_obs)
            if deviation <= sigma_err:
                chi2_kin = 0.0
            else:
                chi2_kin = ((deviation - sigma_err) / sigma_err) ** 2
    rand = _coerce_float(eval_results.get("residual_randomness"), 0.5)
    q = -alpha * chi2_penalty - delta * rand - beta * chi2_kin - gamma * bp

    if PHYSICALITY_MODE == "active":
        rmse_p = _coerce_float(eval_results.get("rmse_poisson"), 0.0)
        phys_penalty = max(0.0, rmse_p - 0.05)
        q -= EPSILON * phys_penalty

    return q


def compute_quality_pass15(
    eval_results: Dict[str, Any],
    proposal: Dict[str, Any],
    alpha: float = ALPHA_P15,
    beta: float = BETA_P15,
    gamma: float = GAMMA_P15,
    delta: float = DELTA_P15,
) -> float:
    """Pass-1.5 quality score: tighter chi2 emphasis for fine-tuning.

    Same structure as compute_quality but with rebalanced weights
    that prioritize chi2 closeness to 1.0 over secondary terms,
    since the model is already well-fitted at this stage.
    """
    return compute_quality(eval_results, proposal,
                           alpha=alpha, beta=beta,
                           gamma=gamma, delta=delta)


QUALITY_FN = compute_quality


def set_quality_fn(fn):
    """Switch the active quality function used by all scoring call sites."""
    global QUALITY_FN
    QUALITY_FN = fn


def _coerce_float(value: Any, default: float) -> float:
    """Convert optional numeric-like values without crashing on None."""
    try:
        if value is None:
            return float(default)
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def compute_subhalo_quality(eval_results: Dict[str, Any]) -> float:
    """Quality score for subhalo fitting (higher is better).

    Uses the reported delta-BIC and adds a modest chi2-to-1 tie-breaker so
    subhalo fits with similar evidence still prefer noise-limited image fits.
    """
    delta_bic = _coerce_float(eval_results.get("delta_bic"), 0.0)
    chi2 = _coerce_float(eval_results.get("chi2_reduced"), 1e6)
    chi2_penalty = chi2_priority_penalty(chi2)
    return delta_bic - 0.5 * chi2_penalty


# ---------------------------------------------------------------------------
# Behavior vector  (for diversity in observable space)
# ---------------------------------------------------------------------------


def compute_behavior_vector(
    eval_results: Dict[str, Any],
    proposal: Dict[str, Any],
) -> np.ndarray:
    """5-D behavior descriptor in observable space."""
    chi2_img = _coerce_float(eval_results.get("image_chi2_reduced"), 1e6)
    sigma_pred = _coerce_float(eval_results.get("sigma_predicted"), 0.0)
    theta_E = _coerce_float(proposal.get("kwargs_lens", [{}])[0].get("theta_E"), 0.0)
    gamma = _coerce_float(proposal.get("kwargs_lens", [{}])[0].get("gamma"), 0.0)
    model_image = eval_results.get("model_image")
    total_flux = float(np.sum(model_image)) if model_image is not None else 0.0
    return np.array([chi2_img, sigma_pred, theta_E, gamma, total_flux], dtype=np.float64)


# ---------------------------------------------------------------------------
# Diversity  (mean L2 to k-nearest neighbors in normalized behavior space)
# ---------------------------------------------------------------------------


def _normalize_behavior_matrix(vecs: np.ndarray) -> np.ndarray:
    """Column-wise min-max normalization to [0, 1]."""
    if len(vecs) == 0:
        return vecs
    mins = vecs.min(axis=0)
    maxs = vecs.max(axis=0)
    spans = maxs - mins
    spans[spans == 0] = 1.0
    return (vecs - mins) / spans


def compute_diversity(
    behavior_vec: np.ndarray,
    all_behavior_vecs: np.ndarray,
    k: int = DIVERSITY_K,
) -> float:
    """Mean L2 distance to k-nearest neighbors in normalized behavior space."""
    if len(all_behavior_vecs) == 0:
        return 1.0
    combined = np.vstack([all_behavior_vecs, behavior_vec.reshape(1, -1)])
    normed = _normalize_behavior_matrix(combined)
    target = normed[-1]
    others = normed[:-1]
    dists = np.linalg.norm(others - target, axis=1)
    k_actual = min(k, len(dists))
    nearest = np.sort(dists)[:k_actual]
    return float(np.mean(nearest)) if k_actual > 0 else 1.0


# ---------------------------------------------------------------------------
# Proposal flattening  (for dedup checks)
# ---------------------------------------------------------------------------


def _flatten_proposal(proposal: Dict[str, Any]) -> np.ndarray:
    """Flatten free parameters to a 1-D vector in prior-normalized [0,1] space."""

    def _bound_value(params: Dict[str, Any], pname: str, default: float):
        if pname in params:
            return params[pname]
        if pname.startswith("amp_"):
            amps = params.get("amp")
            if isinstance(amps, (list, tuple, np.ndarray)):
                idx = int(pname.split("_", 1)[1])
                if idx < len(amps):
                    return amps[idx]
        if pname.startswith("sigma_"):
            sigmas = params.get("sigma")
            if isinstance(sigmas, (list, tuple, np.ndarray)):
                idx = int(pname.split("_", 1)[1])
                if idx < len(sigmas):
                    return sigmas[idx]
        return default

    vals = []
    for key in ("kwargs_lens", "kwargs_lens_light", "kwargs_source"):
        bounds_list = PRIOR_BOUNDS.get(key, [])
        params_list = proposal.get(key, [])
        for comp_idx, bounds in enumerate(bounds_list):
            if comp_idx >= len(params_list):
                continue
            params = params_list[comp_idx]
            for pname, (lo, hi) in bounds.items():
                val = _bound_value(params, pname, (lo + hi) / 2)
                span = hi - lo if hi != lo else 1.0
                vals.append((val - lo) / span)
    return np.array(vals, dtype=np.float64)


def is_duplicate(
    proposal: Dict[str, Any],
    existing_proposals: List[Dict[str, Any]],
    epsilon: float = DEDUP_EPSILON,
) -> bool:
    """True if the proposal is too close to any existing entry in param space."""
    if not existing_proposals:
        return False
    target = _flatten_proposal(proposal)
    for existing in existing_proposals:
        other = _flatten_proposal(existing)
        if target.shape == other.shape and np.linalg.norm(target - other) < epsilon:
            return True
    return False


# ---------------------------------------------------------------------------
# Admission gate  (Pareto non-dominance + dedup)
# ---------------------------------------------------------------------------


def should_admit(
    quality: float,
    diversity: float,
    all_qualities: np.ndarray,
    all_diversities: np.ndarray,
) -> bool:
    """Admit if quality or diversity is high enough.

    Admission paths (checked in order, first match wins):
      1. Quality above the 40th percentile of the database -> admit.
      2. Diversity in the top 20% -> admit (even if quality is mediocre).
      3. Not dominated by any existing entry on both axes -> admit.
      4. Otherwise -> reject.
    """
    if len(all_qualities) == 0:
        return True

    if quality > float(np.percentile(all_qualities, 40)):
        return True

    if diversity > float(np.percentile(all_diversities, 80)):
        return True

    for q, d in zip(all_qualities, all_diversities):
        if q > quality and d > diversity:
            return False

    return True


# ---------------------------------------------------------------------------
# Tiered sampling
# ---------------------------------------------------------------------------


SAMPLING_TEMPERATURE = 20.0


def tiered_sample(
    entries: list,
    n: int = 2,
    rng: Optional[np.random.Generator] = None,
) -> list:
    """Stochastically sample *n* entries with high temperature.

    Uses a flat temperature so that even mediocre entries have a real
    chance of being sampled.  This prevents the best basin from
    dominating all context windows.

    At temperature=50, a quality gap of 10 gives only a ~1.2x
    weight advantage (exp(10/50) = 1.22), so the sampling is
    nearly uniform with a mild quality preference.
    """
    if rng is None:
        rng = np.random.default_rng()
    if len(entries) <= n:
        return list(entries)

    qs = np.array([e.quality for e in entries])
    qs_shifted = qs - qs.max()
    q_weights = np.exp(qs_shifted / SAMPLING_TEMPERATURE)

    weights = np.maximum(q_weights, 1e-12)
    probs = weights / weights.sum()

    idxs = rng.choice(len(entries), size=min(n, len(entries)),
                       replace=False, p=probs)
    return [entries[i] for i in idxs]


# ---------------------------------------------------------------------------
# Random proposal generation  (for seeding the database)
# ---------------------------------------------------------------------------


def random_proposal(rng: Optional[np.random.Generator] = None) -> Dict[str, Any]:
    """Generate a random proposal uniformly sampled across the prior bounds."""
    if rng is None:
        rng = np.random.default_rng()
    proposal: Dict[str, Any] = {}
    for key in ("kwargs_lens", "kwargs_lens_light", "kwargs_source"):
        bounds_list = PRIOR_BOUNDS[key]
        fixed_list = FIXED_PARAMS.get(key, [{}] * len(bounds_list))
        components = []
        for comp_idx, bounds in enumerate(bounds_list):
            comp: Dict[str, Any] = {}
            fixed = fixed_list[comp_idx] if comp_idx < len(fixed_list) else {}
            for pname, (lo, hi) in bounds.items():
                if pname in fixed:
                    continue
                val = float(rng.uniform(lo, hi))
                comp[pname] = val
            if fixed:
                comp.update(fixed)
            components.append(comp)
        proposal[key] = components
    return proposal
