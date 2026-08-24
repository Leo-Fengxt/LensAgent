"""Lens-model families available to AFMS."""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
from types import MappingProxyType
from typing import Any

from lensagent.config import MOCK_MODEL_FAMILIES, REAL_MODEL_FAMILIES, DatasetKind


@dataclass(frozen=True)
class ModelFamily:
    slug: str
    label: str
    definition: Mapping[str, Any]

    @classmethod
    def from_record(cls, slug: str, record: Mapping[str, Any]) -> ModelFamily:
        stored_slug = record.get("slug")
        if stored_slug != slug:
            raise ValueError(
                f"model-family key {slug!r} does not match {stored_slug!r}"
            )
        label = record.get("label")
        if not isinstance(label, str) or not label:
            raise ValueError(f"model family {slug!r} has no label")

        definition = {
            key: copy.deepcopy(value)
            for key, value in record.items()
            if key not in {"slug", "label"}
        }
        _validate_definition(slug, definition)
        return cls(slug=slug, label=label, definition=MappingProxyType(definition))

    def working_copy(self) -> dict[str, Any]:
        """Return a mutable family definition for one fit."""
        return copy.deepcopy(dict(self.definition))


def _validate_definition(slug: str, definition: Mapping[str, Any]) -> None:
    required = {
        "kwargs_model",
        "bounds_lens",
        "centers_lens",
        "fixed_lens",
        "bounds_ll",
        "centers_ll",
        "fixed_ll",
        "bounds_src",
        "sigmas_src",
        "centers_src",
        "fixed_src",
    }
    missing = required.difference(definition)
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"model family {slug!r} is missing: {names}")

    model = definition["kwargs_model"]
    for key in ("lens_model_list", "lens_light_model_list", "source_light_model_list"):
        if key not in model:
            raise ValueError(f"model family {slug!r} is missing kwargs_model.{key}")


@lru_cache(maxsize=1)
def _records() -> dict[str, Any]:
    path = files("lensagent.resources").joinpath("manifests/model_families.json")
    with path.open("r", encoding="utf-8") as handle:
        records = json.load(handle)
    if set(records) != {"sdss", "mock"}:
        raise ValueError("model-family registry must contain sdss and mock groups")
    return records


def family_registry(dataset: DatasetKind | str) -> Mapping[str, ModelFamily]:
    """Load the model-family set for a dataset."""
    kind = DatasetKind(dataset)
    group = "sdss" if kind is DatasetKind.SDSS else "mock"
    expected = REAL_MODEL_FAMILIES if group == "sdss" else MOCK_MODEL_FAMILIES
    records = _records()[group]
    if tuple(records) != expected:
        raise ValueError(f"unexpected {group} model-family order")

    registry = {slug: ModelFamily.from_record(slug, records[slug]) for slug in expected}
    return MappingProxyType(registry)


def model_family(dataset: DatasetKind | str, slug: str) -> ModelFamily:
    try:
        return family_registry(dataset)[slug]
    except KeyError as exc:
        raise KeyError(
            f"model family {slug!r} is not available for {DatasetKind(dataset).value}"
        ) from exc
