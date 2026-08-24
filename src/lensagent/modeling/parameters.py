"""Parameter spaces for lenstronomy model families."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

from lensagent.modeling.families import ModelFamily

ParameterComponents = tuple[dict[str, Any], ...]


def _components(value: Any, length: int = 0) -> ParameterComponents:
    records = tuple(copy.deepcopy(item) for item in (value or []))
    if len(records) < length:
        records += tuple({} for _ in range(length - len(records)))
    return records


@dataclass(frozen=True)
class ParameterSpace:
    family: str
    model: dict[str, list[str]]
    bounds_lens: ParameterComponents
    bounds_lens_light: ParameterComponents
    bounds_source: ParameterComponents
    centers_lens: ParameterComponents
    centers_lens_light: ParameterComponents
    centers_source: ParameterComponents
    sigmas_lens: ParameterComponents
    sigmas_lens_light: ParameterComponents
    sigmas_source: ParameterComponents
    fixed_lens: ParameterComponents
    fixed_lens_light: ParameterComponents
    fixed_source: ParameterComponents
    shapelet_source_ties: tuple[tuple[int, int], ...] = ()
    joint_lens_components: tuple[tuple[int, int, tuple[str, ...]], ...] = ()
    lens_centered_geometry: dict[str, Any] | None = None
    pso_proxy_lens_models: tuple[str, ...] = ()
    mge_components: int = 0
    counting_fixed_lens: ParameterComponents | None = None
    counting_fixed_lens_light: ParameterComponents | None = None
    counting_fixed_source: ParameterComponents | None = None

    @classmethod
    def from_family(cls, family: ModelFamily) -> ParameterSpace:
        item = family.working_copy()
        bounds_lens = _components(item["bounds_lens"])
        bounds_lens_light = _components(item["bounds_ll"])
        bounds_source = _components(item["bounds_src"])
        ties = tuple(
            sorted(
                (int(child), int(parent))
                for child, parent in item.get("shapelet_src_ties", {}).items()
            )
        )
        joint_lens = tuple(
            (int(parent), int(child), tuple(names))
            for parent, child, names in item.get("pso_proxy_joint_lens_with_lens", [])
        )
        return cls(
            family=family.slug,
            model=copy.deepcopy(item["kwargs_model"]),
            bounds_lens=bounds_lens,
            bounds_lens_light=bounds_lens_light,
            bounds_source=bounds_source,
            centers_lens=_components(item["centers_lens"], len(bounds_lens)),
            centers_lens_light=_components(item["centers_ll"], len(bounds_lens_light)),
            centers_source=_components(item["centers_src"], len(bounds_source)),
            sigmas_lens=_components(item.get("sigmas_lens"), len(bounds_lens)),
            sigmas_lens_light=_components(
                item.get("sigmas_ll"), len(bounds_lens_light)
            ),
            sigmas_source=_components(item.get("sigmas_src"), len(bounds_source)),
            fixed_lens=_components(item["fixed_lens"], len(bounds_lens)),
            fixed_lens_light=_components(item["fixed_ll"], len(bounds_lens_light)),
            fixed_source=_components(item["fixed_src"], len(bounds_source)),
            shapelet_source_ties=ties,
            joint_lens_components=joint_lens,
            lens_centered_geometry=copy.deepcopy(item.get("lens_centered_geometry")),
            pso_proxy_lens_models=tuple(item.get("pso_proxy_lens_list", ())),
            mge_components=int(item.get("n_mge", 0)),
        )

    @property
    def bounds(self) -> dict[str, list[dict[str, Any]]]:
        return {
            "kwargs_lens": copy.deepcopy(list(self.bounds_lens)),
            "kwargs_lens_light": copy.deepcopy(list(self.bounds_lens_light)),
            "kwargs_source": copy.deepcopy(list(self.bounds_source)),
        }

    @property
    def centers(self) -> dict[str, list[dict[str, Any]]]:
        return {
            "kwargs_lens": copy.deepcopy(list(self.centers_lens)),
            "kwargs_lens_light": copy.deepcopy(list(self.centers_lens_light)),
            "kwargs_source": copy.deepcopy(list(self.centers_source)),
        }

    @property
    def sigmas(self) -> dict[str, list[dict[str, Any]]]:
        return {
            "kwargs_lens": copy.deepcopy(list(self.sigmas_lens)),
            "kwargs_lens_light": copy.deepcopy(list(self.sigmas_lens_light)),
            "kwargs_source": copy.deepcopy(list(self.sigmas_source)),
        }

    @property
    def fixed(self) -> dict[str, list[dict[str, Any]]]:
        return {
            "kwargs_lens": copy.deepcopy(list(self.fixed_lens)),
            "kwargs_lens_light": copy.deepcopy(list(self.fixed_lens_light)),
            "kwargs_source": copy.deepcopy(list(self.fixed_source)),
        }

    @property
    def lenstronomy_constraints(self) -> dict[str, list[list[Any]]]:
        return {
            "joint_source_with_source": [
                [parent, child, ["center_x", "center_y"]]
                for child, parent in self.shapelet_source_ties
            ],
            "joint_lens_with_lens": [
                [parent, child, list(names)]
                for parent, child, names in self.joint_lens_components
            ],
        }

    @property
    def fixed_for_parameter_count(self) -> dict[str, ParameterComponents]:
        return {
            "kwargs_lens": self.counting_fixed_lens or self.fixed_lens,
            "kwargs_lens_light": (
                self.counting_fixed_lens_light or self.fixed_lens_light
            ),
            "kwargs_source": self.counting_fixed_source or self.fixed_source,
        }

    def materialize_source_ties(
        self, components: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        source = [dict(component) for component in components]
        while len(source) < len(self.fixed_source):
            index = len(source)
            source.append(dict(self.centers_source[index]))

        for index, fixed in enumerate(self.fixed_source):
            for name, value in fixed.items():
                source[index].setdefault(name, value)

        for child_index, parent_index in self.shapelet_source_ties:
            parent = source[parent_index]
            child = source[child_index]
            child["center_x"] = parent.get(
                "center_x", self.centers_source[parent_index].get("center_x", 0.0)
            )
            child["center_y"] = parent.get(
                "center_y", self.centers_source[parent_index].get("center_y", 0.0)
            )
        return source


def pack_multi_gaussian_components(
    proposal: dict[str, Any], model: dict[str, list[str]]
) -> dict[str, Any]:
    """Convert indexed Gaussian parameters to lenstronomy array parameters."""

    def pack(
        components: list[dict[str, Any]], valid_keys: set[str]
    ) -> list[dict[str, Any]]:
        packed: list[dict[str, Any]] = []
        for component in components:
            indexed = any(name.startswith(("amp_", "sigma_")) for name in component)
            if not indexed:
                packed.append(dict(component))
                continue
            amplitudes: list[Any] = []
            sigmas: list[Any] = []
            result = {
                name: value for name, value in component.items() if name in valid_keys
            }
            for name, value in component.items():
                if name.startswith("amp_"):
                    index = int(name.removeprefix("amp_"))
                    amplitudes.extend(0.0 for _ in range(index + 1 - len(amplitudes)))
                    amplitudes[index] = value
                elif name.startswith("sigma_"):
                    index = int(name.removeprefix("sigma_"))
                    sigmas.extend(0.1 for _ in range(index + 1 - len(sigmas)))
                    sigmas[index] = value
            result["amp"] = amplitudes
            result["sigma"] = sigmas
            packed.append(result)
        return packed

    packed = copy.deepcopy(proposal)
    if "MULTI_GAUSSIAN" in model.get("lens_model_list", []):
        packed["kwargs_lens"] = pack(
            proposal.get("kwargs_lens", []), {"center_x", "center_y", "scale_factor"}
        )
    if "MULTI_GAUSSIAN" in model.get("lens_light_model_list", []):
        packed["kwargs_lens_light"] = pack(
            proposal.get("kwargs_lens_light", []), {"center_x", "center_y"}
        )
    return packed
