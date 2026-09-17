"""Timeout isolation for lenstronomy proposal evaluation."""

from __future__ import annotations

import multiprocessing as mp
import signal
import sys
import warnings
from typing import Any

import numpy as np

from lensagent.data.observation import Observation
from lensagent.modeling.parameters import ParameterSpace

FULL_TIMEOUT_SECONDS = 30
IMAGING_TIMEOUT_SECONDS = 10


def _start_method() -> str:
    if any(name in sys.modules for name in ("jax", "jaxlib", "jaxtronomy")):
        methods = mp.get_all_start_methods()
        return "forkserver" if "forkserver" in methods else "spawn"
    return "fork"


def _worker(
    proposal: dict[str, Any],
    observation: Observation,
    parameter_space: ParameterSpace,
    include_kinematics: bool,
    connection,
) -> None:
    try:
        warnings.filterwarnings("ignore")
        if observation.hst:
            np.random.seed(17429)
            warnings.filterwarnings("error", category=RuntimeWarning,
                                    module=r"lenstronomy\.GalKin\.light_profile")
        from lensagent.modeling.evaluate import evaluate_proposal

        if "BLANK_PLANE" in parameter_space.model.get("lens_model_list", []):
            from lensagent.rsi.common import configure_tied_mass_space

            configure_tied_mass_space(parameter_space, observation)

        result = evaluate_proposal(
            proposal,
            observation,
            parameter_space,
            include_kinematics=include_kinematics,
        )
        connection.send(
            (
                "ok",
                {
                    name: value.tolist() if isinstance(value, np.ndarray) else value
                    for name, value in result.items()
                },
            )
        )
    except Exception as exc:
        connection.send(("error", f"{type(exc).__name__}: {exc}"))
    finally:
        connection.close()


def _unexpected_exit(process: mp.Process) -> str:
    code = process.exitcode
    if code is None:
        return "evaluation worker exited unexpectedly"
    if code < 0:
        try:
            name = signal.Signals(-code).name
        except ValueError:
            name = f"signal {-code}"
        return f"evaluation worker exited unexpectedly ({name})"
    return f"evaluation worker exited unexpectedly (exit code {code})"


def _run(
    proposal: dict[str, Any],
    observation: Observation,
    parameter_space: ParameterSpace,
    include_kinematics: bool,
    timeout_seconds: int,
) -> tuple[dict[str, Any] | None, str | None]:
    context = mp.get_context(_start_method())
    parent, child = context.Pipe(duplex=False)
    process = context.Process(
        target=_worker,
        args=(proposal, observation, parameter_space, include_kinematics, child),
    )
    process.start()
    child.close()
    if parent.poll(timeout_seconds):
        try:
            status, payload = parent.recv()
        except (EOFError, OSError):
            process.join(timeout=2)
            if process.is_alive():
                process.kill()
                process.join(timeout=2)
            return None, _unexpected_exit(process)
        process.join(timeout=2)
        if status == "error":
            return None, payload
        for name in ("model_image", "residual_map", "lens_light_image"):
            if isinstance(payload.get(name), list):
                payload[name] = np.asarray(payload[name])
        return payload, None
    process.join(timeout=0)
    if process.exitcode is not None:
        return None, _unexpected_exit(process)
    process.kill()
    process.join(timeout=2)
    return None, f"evaluation timed out after {timeout_seconds}s"


def safe_evaluate(
    proposal: dict[str, Any],
    observation: Observation,
    parameter_space: ParameterSpace,
    *,
    include_kinematics: bool = True,
    timeout_seconds: int = FULL_TIMEOUT_SECONDS,
) -> tuple[dict[str, Any] | None, str | None]:
    result, error = _run(
        proposal,
        observation,
        parameter_space,
        include_kinematics,
        timeout_seconds,
    )
    if error and include_kinematics and "timed out" in error:
        return _run(
            proposal,
            observation,
            parameter_space,
            False,
            IMAGING_TIMEOUT_SECONDS,
        )
    return result, error
