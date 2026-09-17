"""Relative macro-lens strength and image-footprint constraints."""

from dataclasses import dataclass
import copy
import math

import numpy as np

THETA_PROFILES = {"EPL", "SIS", "SIE", "PEMD", "SPEP", "SPP"}


@dataclass(frozen=True)
class MacroFloor:
    indices: tuple[int, ...]
    fraction: float = 0.1

    def __post_init__(self):
        if not 0 <= self.fraction <= 1:
            raise ValueError("macro floor must be a fraction in [0, 1]")

    @classmethod
    def from_model(cls, model, fraction=0.1):
        return cls(tuple(i for i, name in enumerate(model["lens_model_list"])
                         if name in THETA_PROFILES), fraction)

    def accepts(self, lenses):
        if self.fraction == 0 or len(self.indices) < 2:
            return True
        values = [float(lenses[i].get("theta_E", math.nan)) for i in self.indices]
        return (all(math.isfinite(v) and v > 0 for v in values)
                and min(values) / max(values) >= self.fraction - 1e-12)

    def project_start(self, parameters):
        result = copy.deepcopy(parameters)
        if self.fraction == 0 or len(self.indices) < 2:
            return result
        initial, _, fixed, lower, upper = result["lens_model"]
        floor = self.fraction * max(float(initial[i]["theta_E"]) for i in self.indices) * (1 + 1e-8)
        for i in self.indices:
            if "theta_E" in fixed[i]:
                if fixed[i]["theta_E"] < floor / (1 + 1e-8) - 1e-12:
                    raise ValueError("fixed macro component violates the floor")
                continue
            if floor > upper[i]["theta_E"]:
                raise ValueError("macro floor cannot be represented within the bounds")
            initial[i]["theta_E"] = max(initial[i]["theta_E"], floor, lower[i]["theta_E"])
        if not self.accepts(initial):
            raise ValueError("macro starting point violates the floor")
        return result


@dataclass(frozen=True)
class ImageFootprint:
    shape: tuple[int, int]
    transform: tuple[tuple[float, float], tuple[float, float]]
    origin: tuple[float, float]

    @classmethod
    def from_observation(cls, observation):
        return cls(observation.image_data.shape,
                   tuple(tuple(row) for row in observation.transform_pix2angle),
                   (observation.ra_at_xy_0, observation.dec_at_xy_0))

    @property
    def bounds(self):
        height, width = self.shape
        corners = np.array([[-0.5, -0.5], [width - 0.5, -0.5],
                            [-0.5, height - 0.5], [width - 0.5, height - 0.5]])
        angles = corners @ np.asarray(self.transform).T + self.origin
        return {name: (float(angles[:, i].min()), float(angles[:, i].max()))
                for i, name in enumerate(("center_x", "center_y"))}

    def contains(self, x, y):
        pixels = np.linalg.solve(self.transform, np.array([x, y]) - self.origin)
        return bool(np.isfinite(pixels).all() and -0.5 - 1e-9 <= pixels[0] <= self.shape[1] - 0.5 + 1e-9
                    and -0.5 - 1e-9 <= pixels[1] <= self.shape[0] - 0.5 + 1e-9)
