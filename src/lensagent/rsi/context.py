"""Parameter maps for the lens and source stages of single-subhalo RSI."""

import copy
from dataclasses import replace
import math
import warnings

import numpy as np
from lenstronomy.Workflow.fitting_sequence import FittingSequence

from lensagent.modeling.constraints import ImageFootprint, MacroFloor
from lensagent.modeling.evaluate import evaluate_proposal
from lensagent.modeling.parameters import pack_multi_gaussian_components
from lensagent.rsi.common import independent_nfw_space, nfw_mass_msun, _freeze_components
from lensagent.workflow.optimizer import optimizer_space, optimizer_proposal, native_proposal
from lensagent.workflow.pso import fitting_parameters
from lensagent.workflow.swarm import evaluation_deadline, handoff_optimizer_params, validate_optimizer_start


def dispersion_ok(evaluation, observation):
    sigma = evaluation.get("sigma_predicted")
    return bool(sigma is not None and np.isfinite(sigma) and observation.sigma_obs_err > 0
                and abs(sigma - observation.sigma_obs) <= observation.sigma_obs_err)


def same_components(actual, expected, *, light=False):
    if len(actual) != len(expected):
        return False
    return all(name in found and np.allclose(found[name], value, rtol=1e-12, atol=1e-12)
               for found, saved in zip(actual, expected, strict=True)
               for name, value in saved.items() if not (light and name == "amp"))


def stage_space(base, macro, candidate, stage, parent=None):
    if stage not in {"lens", "source"}:
        raise ValueError("RSI stage must be lens or source")
    if stage == "source" and parent is None:
        raise ValueError("the source stage requires a completed lens-stage fit")
    start = copy.deepcopy(parent["proposal"] if parent else macro)
    base_start = copy.deepcopy(start)
    seeds = []
    if candidate is not None:
        seed = dict(candidate)
        if parent:
            subhalo = base_start["kwargs_lens"].pop()
            seed.update(ra=subhalo["center_x"], dec=subhalo["center_y"],
                        Rs=subhalo["Rs"], alpha_Rs=subhalo["alpha_Rs"])
        seeds.append(seed)
    space = independent_nfw_space(base, base_start, seeds, freeze_smooth_model=False,
                                  center_half_width=0.1)
    space = replace(
        space,
        centers_lens_light=tuple(copy.deepcopy(base_start["kwargs_lens_light"])),
        centers_source=tuple(copy.deepcopy(base_start["kwargs_source"])),
        fixed_lens_light=_freeze_components(base_start["kwargs_lens_light"], base.fixed_lens_light),
        bounds_lens_light=tuple({} for _ in base.fixed_lens_light),
        pso_proxy_lens_models=(base.pso_proxy_lens_models + tuple("NFW" for _ in seeds)
                               if base.pso_proxy_lens_models else ()),
    )
    if stage == "lens":
        space = replace(space, fixed_source=_freeze_components(base_start["kwargs_source"], base.fixed_source),
                        bounds_source=tuple({} for _ in base.fixed_source))
    else:
        count = len(base.fixed_lens)
        space = replace(space, fixed_lens=tuple(copy.deepcopy(base_start["kwargs_lens"])) + tuple({} for _ in seeds),
                        bounds_lens=tuple({} for _ in range(count)) + space.bounds_lens[count:])
    return space


class RSIContext:
    def __init__(self, observation, base_space, macro, candidate, stage, maximum_mass, parent=None):
        macro = pack_multi_gaussian_components(macro, base_space.model)
        self.stage, self.candidate, self.parent = stage, candidate, parent
        self.maximum_mass = float(maximum_mass)
        if not math.isfinite(self.maximum_mass) or self.maximum_mass <= 0:
            raise ValueError("maximum subhalo mass must be finite and positive")
        self.base_count = len(base_space.model["lens_model_list"])
        self.space = stage_space(base_space, macro, candidate, stage, parent)
        self.observation = observation.with_model(self.space.model)
        self.floor = MacroFloor.from_model(base_space.model)
        self.footprint = ImageFootprint.from_observation(observation)
        optimized = optimizer_space(self.space)
        parameters = fitting_parameters(optimized, optimized.model)
        inherited = optimizer_proposal(parent["proposal"], self.space) if parent else None
        self.parameters, self.handoff_bounds = handoff_optimizer_params(
            parameters, len(optimized.fixed_lens) - int(candidate is not None), inherited)
        likelihood = {"check_bounds": True, "custom_logL_addition": self.prior}
        if observation.likelihood_mask is not None:
            likelihood["image_likelihood_mask_list"] = [observation.likelihood_mask]
        self.fit = FittingSequence(self.observation.kwargs_data_joint, optimized.model,
                                   optimized.lenstronomy_constraints, likelihood, self.parameters)
        initial = self.proposal(validate_optimizer_start(self.fit))
        expected = parent["proposal"] if parent else macro
        for group in ("kwargs_lens", "kwargs_source", "kwargs_lens_light"):
            actual = initial[group]
            if group == "kwargs_lens" and parent is None:
                actual = actual[:self.base_count]
            if not same_components(actual, expected[group], light=group != "kwargs_lens"):
                raise ValueError(f"RSI handoff changed inherited {group}")
        self.best = None
        self.validated = {}
        self.selection_scores = {}

    def prior(self, kwargs_lens, **kwargs):
        lenses = native_proposal({"kwargs_lens": kwargs_lens, "kwargs_source": [], "kwargs_lens_light": []},
                                 self.space)["kwargs_lens"]
        if not self.floor.accepts(lenses[:self.base_count]):
            return -1e30
        if self.candidate:
            sub = lenses[-1]
            if not self.footprint.contains(sub["center_x"], sub["center_y"]):
                return -1e30
            try:
                mass = nfw_mass_msun(sub["Rs"], sub["alpha_Rs"],
                                     self.observation.z_lens, self.observation.z_source)
                if not math.isfinite(mass) or not 0 < mass <= self.maximum_mass:
                    return -1e30
            except (ValueError, RuntimeError, FloatingPointError):
                return -1e30
        return 0.0

    def proposal(self, vector):
        return native_proposal(self.fit.param_class.args2kwargs(vector), self.space)

    def normalize(self, proposal):
        optimized = optimizer_proposal(proposal, self.space)
        vector = np.asarray(self.fit.param_class.kwargs2args(**{
            name: optimized[name] for name in ("kwargs_lens", "kwargs_source", "kwargs_lens_light")}), dtype=float)
        lower, upper = self.fit.param_class.param_limits()
        if not np.isfinite(vector).all() or np.any(vector < np.asarray(lower) - 1e-12) or np.any(vector > np.asarray(upper) + 1e-12):
            raise ValueError("proposal is outside the RSI optimizer bounds")
        return self.proposal(vector)

    def annotate(self, proposal, evaluation):
        result = dict(evaluation)
        result["macro_floor_ok"] = self.floor.accepts(proposal["kwargs_lens"][:self.base_count])
        if self.candidate:
            sub = proposal["kwargs_lens"][-1]
            mass = nfw_mass_msun(sub["Rs"], sub["alpha_Rs"], self.observation.z_lens, self.observation.z_source)
            result.update(masses_msun=[mass], subhalo_mass_limit_ok=bool(np.isfinite(mass) and 0 < mass <= self.maximum_mass),
                          subhalo_center_bounds_ok=self.footprint.contains(sub["center_x"], sub["center_y"]))
        if not all(result.get(flag, True) for flag in ("macro_floor_ok", "subhalo_mass_limit_ok", "subhalo_center_bounds_ok")):
            result["is_physical"] = False
        return result

    def eligible(self, evaluation):
        return (math.isfinite(float(evaluation.get("image_chi_squared", math.inf)))
                and evaluation.get("is_physical") is True and dispersion_ok(evaluation, self.observation)
                and all(evaluation.get(flag, True) for flag in
                        ("macro_floor_ok", "subhalo_mass_limit_ok", "subhalo_center_bounds_ok")))

    def measure(self, proposal):
        state = np.random.get_state()
        try:
            np.random.seed(17429)
            with evaluation_deadline(10), warnings.catch_warnings():
                warnings.filterwarnings("error", category=RuntimeWarning, module=r"lenstronomy\.GalKin\.light_profile")
                evaluation = evaluate_proposal(proposal, self.observation, self.space)
            return self.annotate(proposal, evaluation)
        finally:
            np.random.set_state(state)

    def validate(self, vector, value):
        if not math.isfinite(value) or value <= -1e29:
            return False
        key = np.asarray(vector, dtype=float).tobytes()
        if key in self.validated:
            return self.validated[key]
        proposal = self.proposal(vector)
        try:
            evaluation = self.measure(proposal)
        except TimeoutError:
            self.validated[key] = False
            return False
        if not np.isclose(-2 * value, evaluation["image_chi_squared"], rtol=1e-7, atol=1e-5):
            raise ValueError("optimizer and evaluator use different image likelihoods")
        valid = self.eligible(evaluation)
        self.validated[key] = valid
        self.selection_scores[key] = -0.5 * evaluation["image_chi_squared"]
        if valid and (self.best is None or evaluation["image_chi_squared"] < self.best["evaluation"]["image_chi_squared"]):
            self.best = {"proposal": proposal, "evaluation": evaluation}
        return valid

    def selection_score(self, vector, value):
        return self.selection_scores[np.asarray(vector, dtype=float).tobytes()]

    def verify_parent(self):
        if self.parent is None:
            return
        initial = validate_optimizer_start(self.fit)
        if not self.validate(initial, float(self.fit.likelihood_class.logL(initial))):
            raise ValueError("lens-stage fit is not feasible at the source-stage start")
        saved, current = self.parent["evaluation"], self.best["evaluation"]
        for key in ("image_chi_squared", "reduced_image_chi_squared", "sigma_predicted"):
            if not np.isclose(saved[key], current[key], rtol=1e-9, atol=1e-6):
                raise ValueError(f"stage handoff changed {key}")
        for key in ("parameter_count", "fitted_pixels", "degrees_of_freedom"):
            if saved[key] != current[key]:
                raise ValueError(f"stage handoff changed {key}")
