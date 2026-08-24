"""Quality, diversity, admission, and sampling for LensAgent proposals."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from lensagent.config import QualityConfig
from lensagent.modeling.parameters import ParameterSpace


class ScoredRecord(Protocol):
    quality: float
    diversity: float
    behavior_vector: Sequence[float]


def _number(value: Any, default: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result if math.isfinite(result) else float(default)


def _parameter_value(
    parameters: dict[str, Any], name: str, default: float | None = None
):
    if name in parameters:
        return parameters[name]
    for prefix, packed_name in (("amp_", "amp"), ("sigma_", "sigma")):
        if name.startswith(prefix):
            values = parameters.get(packed_name)
            if isinstance(values, (list, tuple, np.ndarray)):
                index = int(name.removeprefix(prefix))
                if index < len(values):
                    return values[index]
    return default


def residual_randomness(evaluation: dict[str, Any]) -> float | None:
    """Mean radial residual autocorrelation at lags of 3 to 15 pixels."""
    residual = evaluation.get("residual_map")
    if residual is None:
        return None
    residual = np.asarray(residual, dtype=float)
    if residual.ndim != 2 or residual.size == 0:
        return None

    centered = residual - residual.mean()
    variance = np.mean(centered**2)
    if variance < 1.0e-12:
        return 0.0
    autocorrelation = np.real(np.fft.ifft2(np.abs(np.fft.fft2(centered)) ** 2)) / (
        variance * centered.size
    )
    autocorrelation = np.fft.fftshift(autocorrelation)
    rows, columns = centered.shape
    y, x = np.mgrid[:rows, :columns]
    radius = np.sqrt((y - rows // 2) ** 2 + (x - columns // 2) ** 2)
    values = []
    for lag in range(3, 16):
        annulus = (radius >= lag - 0.5) & (radius < lag + 0.5)
        if annulus.any():
            values.append(float(np.abs(autocorrelation[annulus]).mean()))
    return float(np.mean(values)) if values else 0.0


def chi_squared_priority_penalty(chi_squared: float) -> float:
    """Log-distance from one with additional resolution close to one."""
    value = max(float(chi_squared), 1.0e-6)

    def base(point: float) -> float:
        return abs(math.log(max(point, 1.0e-6)))

    def edge(point: float, band: float) -> float:
        return 1.0 + band if point >= 1.0 else 1.0 - band

    penalty = base(value)
    if abs(value - 1.0) < 0.01:
        edge_penalty = base(edge(value, 0.01))
        penalty -= 10.0 * (edge_penalty - penalty)
    if abs(value - 1.0) < 2.0e-4:
        edge_value = edge(value, 2.0e-4)
        edge_penalty = base(edge_value)
        if abs(edge_value - 1.0) < 0.01:
            outer_edge = base(edge(edge_value, 0.01))
            edge_penalty -= 10.0 * (outer_edge - edge_penalty)
        penalty -= 50.0 * (edge_penalty - penalty)
    return penalty


@dataclass(frozen=True)
class ScoringPolicy:
    parameter_space: ParameterSpace
    quality_config: QualityConfig
    residual_weight: float
    diversity_weight: float = 0.5

    def boundary_penalty(self, proposal: dict[str, Any]) -> float:
        penalty = 0.0
        for group, bounds_list in self.parameter_space.bounds.items():
            parameters_list = proposal.get(group, [])
            for index, bounds in enumerate(bounds_list):
                if index >= len(parameters_list):
                    continue
                parameters = parameters_list[index]
                for name, interval in bounds.items():
                    low, high = map(float, interval)
                    value = _parameter_value(parameters, name)
                    if value is None or high <= low:
                        continue
                    lower_distance = (float(value) - low) / (high - low)
                    upper_distance = (high - float(value)) / (high - low)
                    if lower_distance < 0.05:
                        penalty += math.exp(min(-lower_distance / 0.05, 20.0))
                    if upper_distance < 0.05:
                        penalty += math.exp(min(-upper_distance / 0.05, 20.0))
        return penalty

    def quality(
        self,
        evaluation: dict[str, Any],
        proposal: dict[str, Any],
        *,
        diversity: float | None = None,
    ) -> float:
        image_chi_squared = _number(evaluation.get("reduced_image_chi_squared"), 1.0e6)
        kinematic_chi_squared = _number(
            evaluation.get("kinematic_chi_squared"),
            self.quality_config.missing_kinematic_penalty,
        )
        randomness = _number(evaluation.get("residual_randomness"), 0.5)
        result = (
            -self.quality_config.image_weight
            * chi_squared_priority_penalty(image_chi_squared)
            - self.residual_weight * randomness
            - self.quality_config.kinematic_weight * kinematic_chi_squared
            - self.quality_config.boundary_weight * self.boundary_penalty(proposal)
        )
        if diversity is not None:
            result += (
                self.diversity_weight
                * self.quality_config.diversity_scale
                * _number(diversity, 0.0)
            )
        return float(result)

    def flatten(self, proposal: dict[str, Any]) -> np.ndarray:
        values: list[float] = []
        for group, bounds_list in self.parameter_space.bounds.items():
            parameters_list = proposal.get(group, [])
            for index, bounds in enumerate(bounds_list):
                parameters = (
                    parameters_list[index] if index < len(parameters_list) else {}
                )
                for name, interval in bounds.items():
                    low, high = map(float, interval)
                    value = _parameter_value(parameters, name, (low + high) / 2.0)
                    try:
                        normalized = (float(value) - low) / (high - low or 1.0)
                    except (TypeError, ValueError):
                        normalized = 0.5
                    if not math.isfinite(normalized):
                        normalized = 0.5
                    values.append(float(np.clip(normalized, 0.0, 1.0)))
        return np.asarray(values, dtype=np.float64)

    def diversity(
        self, behavior_vector: Sequence[float], neighbors: Iterable[Sequence[float]]
    ) -> float:
        target = np.asarray(behavior_vector, dtype=np.float64).reshape(-1)
        others = _neighbor_matrix(neighbors, target.size)
        if not np.all(np.isfinite(target)):
            return 0.0
        if len(others) == 0:
            return 1.0

        difference = np.abs(others - target.reshape(1, -1))
        threshold = self.quality_config.diversity_threshold
        hard_fraction = (difference > threshold).mean(axis=1)
        effective = max(
            1,
            min(self.quality_config.diversity_effective_dimensions, target.size),
        )
        largest = np.sort(difference, axis=1)[:, -effective:]
        soft_rms = np.sqrt(
            np.mean(np.clip(largest / max(threshold, 1.0e-9), 0.0, 1.0) ** 2, axis=1)
        )
        fraction = np.clip(0.5 * hard_fraction + 0.5 * soft_rms, 0.0, 1.0)
        distance = self._group_weighted_distance(target, others)
        count = min(self.quality_config.diversity_neighbors, len(distance))
        nearest = np.argsort(distance)[:count]
        return float(np.clip(np.mean(fraction[nearest]), 0.0, 1.0))

    def _group_weighted_distance(
        self, target: np.ndarray, others: np.ndarray
    ) -> np.ndarray:
        weights = {
            "kwargs_lens": 2.0,
            "kwargs_lens_light": 0.7,
            "kwargs_source": 0.7,
        }
        total = np.zeros(len(others), dtype=np.float64)
        total_weight = 0.0
        start = 0
        for group, bounds_list in self.parameter_space.bounds.items():
            width = sum(len(bounds) for bounds in bounds_list)
            stop = start + width
            if width:
                difference = others[:, start:stop] - target[start:stop]
                weight = weights[group]
                total += weight * np.sqrt(np.mean(difference**2, axis=1))
                total_weight += weight
            start = stop
        return total / total_weight

    def is_duplicate(
        self,
        proposal: dict[str, Any],
        existing: Iterable[dict[str, Any]],
        epsilon: float = 0.01,
    ) -> bool:
        target = self.flatten(proposal)
        return any(
            (other := self.flatten(item)).shape == target.shape
            and np.linalg.norm(target - other) < epsilon
            for item in existing
        )

    def random_proposal(self, rng: np.random.Generator) -> dict[str, Any]:
        proposal: dict[str, Any] = {}
        fixed_groups = self.parameter_space.fixed
        for group, bounds_list in self.parameter_space.bounds.items():
            components = []
            for index, bounds in enumerate(bounds_list):
                fixed = fixed_groups[group][index]
                component = {
                    name: float(rng.uniform(float(interval[0]), float(interval[1])))
                    for name, interval in bounds.items()
                    if name not in fixed
                }
                component.update(fixed)
                components.append(component)
            proposal[group] = components
        return proposal

    def inject_fixed(self, proposal: dict[str, Any]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for group, fixed_list in self.parameter_space.fixed.items():
            supplied = [dict(component) for component in proposal.get(group, [])]
            if len(supplied) > len(fixed_list):
                raise ValueError(f"{group} has too many components")
            centers = self.parameter_space.centers[group]
            while len(supplied) < len(fixed_list):
                supplied.append(dict(centers[len(supplied)]))
            for index, fixed in enumerate(fixed_list):
                supplied[index].update(fixed)
            result[group] = supplied
        return result


def _neighbor_matrix(neighbors: Iterable[Sequence[float]], width: int) -> np.ndarray:
    rows = [np.asarray(row, dtype=np.float64).reshape(-1) for row in neighbors]
    rows = [row for row in rows if row.size == width and np.all(np.isfinite(row))]
    return np.vstack(rows) if rows else np.empty((0, width), dtype=np.float64)


def admission_reason(
    quality: float,
    diversity: float,
    qualities: Sequence[float],
    diversities: Sequence[float],
) -> str:
    quality_array = np.asarray(qualities, dtype=float)
    diversity_array = np.asarray(diversities, dtype=float)
    finite_quality = quality_array[np.isfinite(quality_array)]
    finite_diversity = diversity_array[np.isfinite(diversity_array)]
    if not len(finite_quality):
        return "empty_database"
    if quality > float(np.percentile(finite_quality, 40)):
        return "quality_percentile"
    if not len(finite_diversity) or diversity > float(
        np.percentile(finite_diversity, 80)
    ):
        return "diversity_percentile"
    for existing_quality, existing_diversity in zip(quality_array, diversity_array):
        if (
            np.isfinite(existing_quality)
            and np.isfinite(existing_diversity)
            and existing_quality > quality
            and existing_diversity > diversity
        ):
            return "dominated"
    return "pareto_frontier"


def tiered_sample(
    entries: Sequence[ScoredRecord],
    count: int,
    rng: np.random.Generator,
    *,
    diversity_weight: float = 0.5,
) -> list[ScoredRecord]:
    pool = [entry for entry in entries if math.isfinite(entry.quality)] or list(entries)
    if len(pool) <= count:
        return list(pool)
    ranked = sorted(pool, key=lambda entry: entry.quality, reverse=True)
    selected = [ranked[0]]
    remaining = ranked[1:]
    novelty_probability = min(0.85, max(0.05, 0.10 + 1.40 * diversity_weight))
    while len(selected) < count and remaining:
        if rng.random() < novelty_probability:
            choices = sorted(
                remaining,
                key=lambda entry: (entry.diversity, entry.quality),
                reverse=True,
            )[: min(5, len(remaining))]
            choice = choices[int(rng.integers(0, len(choices)))]
        else:
            ranks = np.arange(len(remaining), dtype=float)
            probabilities = 1.0 / np.power(ranks + 1.0, 1.0)
            probabilities /= probabilities.sum()
            choice = remaining[int(rng.choice(len(remaining), p=probabilities))]
        selected.append(choice)
        remaining = [entry for entry in remaining if entry is not choice]
    return selected
