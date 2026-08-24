"""Input validation for resumable fixed-count searches."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from lensagent.data.observation import Observation
from lensagent.modeling.parameters import ParameterSpace
from lensagent.output.artifacts import NumpyEncoder, write_json


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        cls=NumpyEncoder,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def ensure_archive_context(
    path: str | Path,
    *,
    stage: str,
    observation: Observation,
    parameter_space: ParameterSpace,
    configuration: Any,
    inputs: dict[str, Any],
) -> None:
    destination = Path(path)
    config_value = (
        asdict(configuration) if is_dataclass(configuration) else configuration
    )
    context = {
        "stage": stage,
        "observation_sha256": observation.fingerprint(),
        "parameter_space": asdict(parameter_space),
        "configuration": config_value,
        "inputs": inputs,
    }
    fingerprint = hashlib.sha256(_canonical_json(context).encode()).hexdigest()
    record = {"schema": 1, "fingerprint": fingerprint, "context": context}
    if destination.exists():
        saved = json.loads(destination.read_text(encoding="utf-8"))
        if saved.get("schema") != 1 or saved.get("fingerprint") != fingerprint:
            raise ValueError(
                f"existing search artifacts do not match the current inputs: {destination}"
            )
        return
    write_json(destination, record, sort_keys=True)
