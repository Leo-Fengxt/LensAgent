"""Feasible PSO checkpoints and optimizer handoff validation."""

import copy
import math
import signal
import time
from contextlib import contextmanager

import numpy as np
from lenstronomy.Sampling.Samplers.pso import ParticleSwarmOptimizer

class _EvaluationExpired(BaseException):
    pass


@contextmanager
def evaluation_deadline(seconds):
    """Interrupt numerical loops even when library code catches Exception."""
    def expired(signum, frame):
        raise _EvaluationExpired()

    if seconds <= 0:
        raise ValueError("evaluation timeout must be positive")
    if signal.getitimer(signal.ITIMER_REAL)[0]:
        raise RuntimeError("cannot replace an active evaluation timer")
    previous = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    except _EvaluationExpired as error:
        raise TimeoutError(f"evaluation exceeded {seconds:g} seconds") from error
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)



def handoff_optimizer_params(params, macro_lens_count, proposal=None):
    """Include inherited macro values without changing the scoring priors."""
    result = copy.deepcopy(params)
    adjustments = []
    for block, key in (("lens_model", "kwargs_lens"),
                       ("source_model", "kwargs_source"),
                       ("lens_light_model", "kwargs_lens_light")):
        initial, _, fixed, lower, upper = result[block]
        if proposal is not None and len(proposal[key]) != len(initial):
            raise ValueError(f"{key}: handoff component count does not match the optimizer")
        for index, values in enumerate(initial):
            inherited = block == "source_model" or (
                block == "lens_model" and index < macro_lens_count)
            for name in values:
                if name in fixed[index]:
                    continue
                if proposal is not None and name in proposal[key][index]:
                    values[name] = copy.deepcopy(proposal[key][index][name])
                value = float(values[name])
                if not math.isfinite(value):
                    raise ValueError(f"non-finite handoff value: {key}[{index}].{name}")
                if not inherited or name not in lower[index] or name not in upper[index]:
                    continue
                lo, hi = lower[index][name], upper[index][name]
                if lo <= value <= hi:
                    continue
                new_lo, new_hi = min(lo, value), max(hi, value)
                lower[index][name], upper[index][name] = new_lo, new_hi
                adjustments.append({"group": key, "component": index, "parameter": name,
                                    "start": value, "prior_bounds": [lo, hi],
                                    "optimizer_bounds": [new_lo, new_hi]})
    return result, adjustments


def validate_optimizer_start(fit):
    """Check the actual parameter map before submitting any PSO replicas."""
    pc = fit.param_class
    initial = np.asarray(pc.kwargs2args(**fit.best_fit()), dtype=float)
    lower, upper = (np.asarray(values, dtype=float) for values in pc.param_limits())
    invalid = ~np.isfinite(initial) | (initial < lower) | (initial > upper)
    if np.any(invalid):
        names = pc.num_param()[1]
        details = "; ".join(f"{names[i]}={initial[i]} outside [{lower[i]}, {upper[i]}]"
                            for i in np.flatnonzero(invalid))
        raise ValueError(f"saved RSI start is outside its registered bounds: {details}")
    return initial


class CheckpointSwarm(ParticleSwarmOptimizer):
    """Image PSO with a bounded, measured-dispersion archive at checkpoints."""

    def __init__(self, objective, validate, lower, upper, particles, initial,
                 initial_low, initial_high, initial_fitness, selection_score=None):
        self.validate = validate
        self.selection_score = selection_score or (lambda position, score: score)
        self.checked = 0
        self.accepted = 0
        self.image_calls = 0
        self.archive = {}
        self.validated = set()
        super().__init__(objective, initial_low, initial_high, particles)
        self.low, self.high = list(lower), list(upper)
        self.set_global_best(initial, np.zeros(len(initial)), initial_fitness)
        self.swarm[0].position = list(initial)
        for particle in self.swarm:
            particle.fitness = -np.inf
            particle.update_personal_best()
        key = np.asarray(initial, dtype=float).tobytes()
        self.validated.add(key)
        self.checked = 1
        if math.isfinite(initial_fitness):
            self.archive[key] = (np.asarray(initial).copy(), initial_fitness,
                                 float(self.selection_score(initial, initial_fitness)))
            self.accepted = 1

    def _converged(self, *args, **kwargs):
        # Resetting guides at a checkpoint is not numerical convergence.
        return False

    def _get_fitness(self, swarm):
        for particle in swarm:
            position = np.asarray(particle.position)
            if not np.isfinite(position).all() or np.any(position < self.low) or np.any(position > self.high):
                particle.fitness = -np.inf
                continue
            value = float(self.func(particle.position))
            self.image_calls += 1
            if not math.isfinite(value) or value <= -1e29:
                particle.fitness = -np.inf
                continue
            particle.fitness = value

    @property
    def feasible_best(self):
        return max(self.archive.values(), key=lambda row: row[2]) if self.archive else None

    def checkpoint(self, limit):
        rows = {}
        for particle in [self.global_best, *self.swarm,
                         *(p.personal_best for p in self.swarm)]:
            key = np.asarray(particle.position, dtype=float).tobytes()
            if key not in self.validated and math.isfinite(particle.fitness) and particle.fitness > -1e29:
                rows[key] = (np.asarray(particle.position).copy(), float(particle.fitness))
        pending = sorted(rows.values(), key=lambda row: row[1], reverse=True)
        chosen, pending = pending[:limit // 2], pending[limit // 2:]
        scale = np.maximum(np.asarray(self.high) - self.low, 1e-12)
        # Half by likelihood, half by separation, so nearly identical image
        # minima do not consume the whole kinematic-check budget.
        while pending and len(chosen) < limit:
            distances = [min(np.linalg.norm((row[0] - other[0]) / scale)
                             for other in chosen) if chosen else 0 for row in pending]
            chosen.append(pending.pop(int(np.argmax(distances))))
        for position, score in chosen:
            key = position.astype(float).tobytes()
            self.validated.add(key)
            self.checked += 1
            if self.validate(position, score):
                self.archive[key] = (position.copy(), score, float(self.selection_score(position, score)))
                self.accepted += 1
        best = self.feasible_best
        if best is None:
            return
        self.set_global_best(best[0], np.zeros(self.param_count), best[1])
        guides = list(self.archive.values())
        for particle in self.swarm:
            position, velocity = particle.position, particle.velocity
            guide, score, _ = min(guides, key=lambda row: np.linalg.norm((row[0] - position) / scale))
            particle.position, particle.velocity, particle.fitness = list(guide), [0.] * self.param_count, score
            particle.update_personal_best()
            particle.position, particle.velocity, particle.fitness = position, velocity, -np.inf


def run_feasible_pso(fit, validate, *, particles, iterations, seed, progress=None,
                     checkpoint_interval=25, checkpoint_candidates=8, selection_score=None):
    """Use fast image PSO and select the best fully evaluated feasible fit."""
    param = fit.param_class
    if particles < 2 or iterations < 1 or checkpoint_interval < 1 or checkpoint_candidates < 1:
        raise ValueError("invalid PSO or checkpoint budget")
    initial = validate_optimizer_start(fit)
    lower, upper = (np.asarray(x, dtype=float) for x in param.param_limits())
    if not len(initial):
        value = float(fit.likelihood_class.logL(initial))
        valid = validate(initial, value)
        score = (selection_score(initial, value) if selection_score else value) if valid else -np.inf
        return initial, {"positions": [initial], "logL": [score],
                         "image_calls": 1, "kinematic_checks": 1,
                         "accepted_improvements": int(valid), "iterations": 0}
    np.random.seed(seed)
    value = float(fit.likelihood_class.logL(initial))
    feasible = math.isfinite(value) and value > -1e29 and validate(initial, value)
    swarm = CheckpointSwarm(fit.likelihood_class.logL, validate, lower, upper,
                          particles, initial, np.maximum(lower, initial - .1 * (upper - lower)),
                          np.minimum(upper, initial + .1 * (upper - lower)),
                          value if feasible else -np.inf, selection_score=selection_score)
    positions, scores = [initial.copy()], [swarm.feasible_best[2] if feasible else -np.inf]
    started = time.monotonic()
    for index, _ in enumerate(swarm.sample(iterations, verbose=False), start=1):
        checkpoint = index % checkpoint_interval == 0 or index == iterations
        if checkpoint:
            swarm.checkpoint(checkpoint_candidates)
        feasible_best = swarm.feasible_best
        positions.append(feasible_best[0].copy() if feasible_best else initial.copy())
        scores.append(float(feasible_best[2]) if feasible_best else -np.inf)
        if progress and (index == 1 or index % 10 == 0 or checkpoint):
            progress({"iteration": index, "iterations": iterations,
                      "best_image_logL": float(swarm.global_best.fitness),
                      "best_feasible_logL": scores[-1], "checkpoint": checkpoint,
                      "image_calls": swarm.image_calls, "kinematic_checks": swarm.checked,
                      "accepted_improvements": swarm.accepted,
                      "elapsed_s": time.monotonic() - started})
    feasible_best = swarm.feasible_best
    best = feasible_best[0].copy() if feasible_best else initial.copy()
    positions.append(best)
    scores.append(float(feasible_best[2]) if feasible_best else -np.inf)
    return best, {"positions": positions, "logL": scores, "image_calls": swarm.image_calls,
                  "kinematic_checks": swarm.checked, "accepted_improvements": swarm.accepted,
                  "checkpoint_interval": checkpoint_interval,
                  "checkpoint_candidates": checkpoint_candidates,
                  "iterations": len(positions) - 2}
