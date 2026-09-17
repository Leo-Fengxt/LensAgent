"""Reference selection and local refinement for AFMS and PRL."""

import copy
import math

import numpy as np



def _number(value, default=math.inf):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


class RefinementPolicy:
    objective = "target_chi_squared"
    reference_roles = ("Best eligible fit",
                       "Best image fit; handoff requirements not yet satisfied",
                       "Best image alternative")

    def __init__(self, observation, scoring, *, quality_factor=2.0, epsilon=0.01, tolerance=1e-8):
        self.scoring = scoring
        self.quality_factor = quality_factor
        self.epsilon = epsilon
        self.tolerance = tolerance
        self.sigma_obs = _number(observation.sigma_obs)
        self.sigma_err = _number(observation.sigma_obs_err, 0.0)
        self.bounds = scoring.parameter_space.bounds
        self.weights = {"kwargs_lens": 2.0, "kwargs_lens_light": 0.7, "kwargs_source": 0.7}
        self.linear_solve = True

    def normalize(self, proposal):
        result = self.scoring.inject_fixed(proposal)
        result["kwargs_source"] = self.scoring.parameter_space.materialize_source_ties(result["kwargs_source"])
        return result

    def effective(self, proposal):
        normalized = self.normalize(proposal)
        result = {}
        for group in ("kwargs_lens", "kwargs_lens_light", "kwargs_source"):
            result[group] = []
            for component in normalized[group]:
                values = dict(component)
                # Only brightness amplitudes are replaced by the linear solver.
                if self.linear_solve and group != "kwargs_lens":
                    values = {k: v for k, v in values.items()
                              if k != "amp" and not k.startswith("amp_")}
                for stem in ("amp", "sigma"):
                    array = values.pop(stem, None)
                    if isinstance(array, (list, tuple, np.ndarray)):
                        for i, value in enumerate(array):
                            values[f"{stem}_{i}"] = value
                    elif array is not None:
                        values[stem] = array
                result[group].append(values)
        return result

    def key(self, proposal):
        def freeze(value):
            if isinstance(value, dict):
                return tuple((k, freeze(v)) for k, v in sorted(value.items()))
            if isinstance(value, (list, tuple, np.ndarray)):
                return tuple(freeze(v) for v in value)
            if isinstance(value, (float, int, np.number)):
                if not math.isfinite(float(value)):
                    raise ValueError("nonfinite model parameter")
                return float(value)
            return value
        return freeze(self.effective(proposal))

    def vector(self, proposal):
        return self.scoring.flatten(self.effective(proposal))

    def distance(self, entry):
        chi2 = _number(entry.evaluation.get("reduced_image_chi_squared"))
        if chi2 <= 0 or not math.isfinite(chi2):
            return math.inf
        return abs(math.log(chi2))

    def usable(self, entry):
        result = entry.evaluation
        if (not math.isfinite(self.distance(entry))
                or result.get("is_physical") is False
                or result.get("kinematic_failure_reason") == "zero_deflector_light"
                or result.get("error") or result.get("evaluation_error")):
            return False
        try:
            self.key(entry.proposal)
        except (ValueError, TypeError, IndexError):
            return False
        return True

    def eligible(self, entry):
        return (self.usable(entry) and entry.evaluation.get("is_physical") is True
                and self.sigma_err > 0 and math.isfinite(self.sigma_obs)
                and abs(_number(entry.evaluation.get("sigma_predicted"))
                        - self.sigma_obs) <= self.sigma_err)

    def sigma_distance(self, entry):
        return abs(_number(entry.evaluation.get("sigma_predicted")) - self.sigma_obs)

    def champions(self, entries):
        pool = sorted((e for e in entries if self.usable(e)),
                      key=lambda e: (self.distance(e), e.timestamp, e.id))
        accepted = [e for e in pool if self.eligible(e)]
        return (accepted[0] if accepted else None, pool[0] if pool else None)

    def protected_ids(self, entries):
        return {e.id for e in self.champions(entries) if e is not None}

    def reference_distance(self, entry):
        return self.distance(entry)

    def _separation(self, target, others):
        total = np.zeros(len(others))
        weight_sum = 0.0
        start = 0
        for group in ("kwargs_lens", "kwargs_lens_light", "kwargs_source"):
            size = sum(len(b) for b in self.bounds.get(group, []))
            if size:
                weight = self.weights.get(group, 1.0)
                delta = others[:, start:start + size] - target[start:start + size]
                total += weight * np.sqrt(np.mean(delta ** 2, axis=1))
                weight_sum += weight
            start += size
        return total / weight_sum if weight_sum else total

    def sample(self, entries, n, rng, island):
        eligible, image = self.champions(entries)
        if image is None or n <= 0:
            return []
        pool = [e for e in entries if self.usable(e)
                and self.reference_distance(e) <= self.reference_distance(image) + math.log(self.quality_factor)]
        chosen, seen = [], set()

        def add(entry, role):
            key = self.key(entry.proposal)
            if len(chosen) < n and key not in seen:
                reference = copy.copy(entry)
                reference.reference_role = role
                chosen.append(reference)
                seen.add(key)

        add(eligible or image, self.reference_roles[0] if eligible else self.reference_roles[1])
        add(image, self.reference_roles[2])
        local = [e for e in pool if e.island == island and self.key(e.proposal) not in seen]
        if local:
            add(min(local, key=self.distance), "Island reference")
        vectors = {e.id: self.vector(e.proposal) for e in pool + chosen}
        while len(chosen) < n:
            remaining = [e for e in pool if self.key(e.proposal) not in seen]
            if not remaining:
                break
            matrix = np.array([vectors[e.id] for e in chosen])
            remaining.sort(key=lambda e: (
                -float(self._separation(vectors[e.id], matrix).min()),
                self.distance(e), e.id))
            add(remaining[int(rng.integers(min(3, len(remaining))))], "Alternative fit")
        return chosen

    def decide(self, candidate, entries, exploration_reason):
        """Return a decision and at most one dominated neighbour to replace."""
        info = {"target_distance": self.distance(candidate), "replaced_id": None}
        if not self.usable(candidate):
            return dict(info, outcome="rejected", admission_reason="invalid_fit")
        key = self.key(candidate.proposal)
        pool = []
        for entry in entries:
            try:
                if self.key(entry.proposal) == key:
                    return dict(info, outcome="duplicate", admission_reason="exact_duplicate",
                                nearest_id=entry.id, nearest_distance=0.0)
                pool.append(entry)
            except (ValueError, TypeError, IndexError):
                continue
        vector = self.vector(candidate.proposal)
        neighbours = sorted(
            ((float(np.linalg.norm(vector - self.vector(e.proposal))), e) for e in pool),
            key=lambda pair: (pair[0], pair[1].id))
        if neighbours:
            distance, nearest = neighbours[0]
            info.update(nearest_id=nearest.id, nearest_distance=distance,
                        nearest_target_distance=self.distance(nearest))
        near = [e for d, e in neighbours if d < self.epsilon]
        eligible, image = self.champions(entries)
        new_eligible = self.eligible(candidate) and (eligible is None or
            self.distance(candidate) < self.distance(eligible) - self.tolerance)
        new_image = image is None or self.distance(candidate) < self.distance(image) - self.tolerance
        improves = [e for e in near if self.distance(candidate) < self.distance(e) - self.tolerance
                    and (self.eligible(candidate) or not self.eligible(e))]
        if improves:
            replaced = improves[0]
            return dict(info, outcome="admitted", admission_reason="nearby_improvement",
                        replaced_id=replaced.id, replaced_target_distance=self.distance(replaced))
        if new_eligible or new_image:
            return dict(info, outcome="admitted", admission_reason=(
                "new_eligible_best" if new_eligible else "new_image_best"))
        # A sigma improvement or an image/sigma tradeoff can still enter via
        # the existing exploration gate, but cannot evict a protected fit.
        tradeoff = any(
            (self.eligible(candidate) and not self.eligible(e)) or
            self.sigma_distance(candidate) < self.sigma_distance(e) - self.tolerance or
            self.distance(candidate) < self.distance(e) - self.tolerance for e in near)
        if near and not tradeoff:
            return dict(info, outcome="duplicate", admission_reason="nearby_no_improvement")
        return dict(info, outcome="rejected" if exploration_reason == "dominated"
                    else "admitted", admission_reason=exploration_reason)


def context_record(entries):
    return {"context_ids": [e.id for e in entries],
            "context_roles": [getattr(e, "reference_role", None) for e in entries],
            "best_fit_id_at_launch": entries[0].id if entries else None}
