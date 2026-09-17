"""Fixed-support mass polish for fixed-count RSI."""

from __future__ import annotations

import copy
import json
import math
import os
import random
import time
from collections.abc import Sequence
from concurrent.futures import as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from lensagent.config import MassPolishConfig, PSOConfig
from lensagent.data.observation import Observation
from lensagent.modeling.parameters import ParameterSpace
from lensagent.output.artifacts import NumpyEncoder, write_json
from lensagent.rsi.common import configure_tied_mass_space, tied_mass_nfw_space
from lensagent.rsi.multisubhalo.artifacts import ensure_archive_context
from lensagent.rsi.multisubhalo.support import ExactCountSelection
from lensagent.workflow.pso import run_family_pso
from lensagent.workflow.processes import process_pool


@dataclass(frozen=True)
class MassPolishFit:
    stage: str
    replica: int
    bic: float
    log_likelihood: float
    eligible: bool
    proposal: dict[str, Any] | None
    fitted_subhalos: tuple[dict[str, Any], ...]
    fitted_base_lens: tuple[dict[str, Any], ...]
    minimum_separation_arcsec: float | None
    random_seed: int
    elapsed_seconds: float
    error: str | None = None


@dataclass(frozen=True)
class MassPolishResult:
    known_count: int
    medium_replicas: int
    high_replicas: int
    failed_replicas: int
    candidates: tuple[dict[str, Any], ...]
    selected: MassPolishFit


def _minimum_separation(subhalos: Sequence[dict[str, Any]]) -> float | None:
    minimum = None
    for index, first in enumerate(subhalos):
        for second in subhalos[index + 1 :]:
            distance = math.hypot(
                float(first["center_x"]) - float(second["center_x"]),
                float(first["center_y"]) - float(second["center_y"]),
            )
            minimum = distance if minimum is None else min(minimum, distance)
    return minimum


def _polish_candidates(
    selection: ExactCountSelection,
    position_half_width: float,
) -> list[dict[str, Any]]:
    candidates = []
    for subhalo in selection.selected.fitted_subhalos:
        mass = float(subhalo["mass_msun"])
        if not np.isfinite(mass) or mass <= 0:
            raise ValueError(
                f"candidate {subhalo['candidate_id']} has no finite NFW mass"
            )
        center_x = float(subhalo["center_x"])
        center_y = float(subhalo["center_y"])
        candidates.append(
            {
                "candidate_id": int(subhalo["candidate_id"]),
                "blob_id": int(subhalo["blob_id"]),
                "source_rank": int(subhalo["source_rank"]),
                "coordinate_variant": subhalo["coordinate_variant"],
                "ra": center_x,
                "dec": center_y,
                "logM": float(np.clip(math.log10(mass), 7.0, 11.0)),
                "handoff_mass_msun": mass,
                "center_bounds": {
                    "center_x": (
                        center_x - position_half_width,
                        center_x + position_half_width,
                    ),
                    "center_y": (
                        center_y - position_half_width,
                        center_y + position_half_width,
                    ),
                },
            }
        )
    return candidates


def _fit_replica(payload: dict[str, Any]) -> MassPolishFit:
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = "1"
    started = time.monotonic()
    stage = str(payload["stage"])
    replica = int(payload["replica"])
    seed = int(payload["random_seed"])
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    try:
        config: MassPolishConfig = payload["config"]
        candidates = copy.deepcopy(payload["candidates"])
        base_space: ParameterSpace = payload["parameter_space"]
        base_proposal = payload["base_proposal"]
        space = tied_mass_nfw_space(
            base_space,
            base_proposal,
            candidates,
            center_half_width=config.position_half_width_arcsec,
            macro_thaw={
                "theta_E": config.theta_e_half_width,
                "gamma": config.slope_half_width,
                "e1": config.ellipticity_half_width,
                "e2": config.ellipticity_half_width,
            },
            warm_base_lens=payload.get("warm_base_lens"),
        )
        observation: Observation = payload["observation"].with_model(space.model)
        configure_tied_mass_space(space, observation)
        result = run_family_pso(
            observation,
            space,
            PSOConfig(
                seeds=0,
                runs=1,
                particles=int(payload["particles"]),
                iterations=int(payload["iterations"]),
                sigma_scale=float(payload["sigma_scale"]),
            ),
        )
        fit = result.fits[0]
        base_lens_count = len(base_proposal["kwargs_lens"])
        fitted_subhalos = []
        for candidate, component in zip(
            candidates, fit.proposal["kwargs_lens"][base_lens_count:]
        ):
            fitted_subhalos.append(
                {
                    "candidate_id": int(candidate["candidate_id"]),
                    "blob_id": int(candidate["blob_id"]),
                    "source_rank": int(candidate["source_rank"]),
                    "coordinate_variant": candidate["coordinate_variant"],
                    "seed_ra": float(candidate["ra"]),
                    "seed_dec": float(candidate["dec"]),
                    "center_bounds": copy.deepcopy(candidate["center_bounds"]),
                    "center_x": float(component["center_x"]),
                    "center_y": float(component["center_y"]),
                    "logM": float(component["logM"]),
                    "mass_msun": 10.0 ** float(component["logM"]),
                }
            )
        separation = _minimum_separation(fitted_subhalos)
        eligible = separation is None or separation >= config.minimum_separation_arcsec
        return MassPolishFit(
            stage=stage,
            replica=replica,
            bic=float(fit.bic),
            log_likelihood=float(fit.log_likelihood),
            eligible=eligible,
            proposal=fit.proposal,
            fitted_subhalos=tuple(fitted_subhalos),
            fitted_base_lens=tuple(
                copy.deepcopy(fit.proposal["kwargs_lens"][:base_lens_count])
            ),
            minimum_separation_arcsec=separation,
            random_seed=seed,
            elapsed_seconds=time.monotonic() - started,
        )
    except Exception as exc:
        return MassPolishFit(
            stage=stage,
            replica=replica,
            bic=float("inf"),
            log_likelihood=float("-inf"),
            eligible=False,
            proposal=None,
            fitted_subhalos=(),
            fitted_base_lens=(),
            minimum_separation_arcsec=None,
            random_seed=seed,
            elapsed_seconds=time.monotonic() - started,
            error=f"{type(exc).__name__}: {exc}",
        )


def _append(path: Path, fit: MassPolishFit) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(asdict(fit), cls=NumpyEncoder) + "\n")


def _load_archive(path: Path) -> list[MassPolishFit]:
    if not path.exists():
        return []
    latest = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            item["fitted_subhalos"] = tuple(item["fitted_subhalos"])
            item["fitted_base_lens"] = tuple(item["fitted_base_lens"])
            fit = MassPolishFit(**item)
            latest[(fit.stage, fit.replica)] = fit
    return list(latest.values())


def _run_stage(
    stage: str,
    replicas: int,
    candidates: Sequence[dict[str, Any]],
    base_proposal: dict[str, Any],
    observation: Observation,
    parameter_space: ParameterSpace,
    config: MassPolishConfig,
    archive: Path,
    *,
    particles: int,
    iterations: int,
    sigma_scale: float,
    warm_fit: MassPolishFit | None = None,
    existing: Sequence[MassPolishFit] = (),
) -> list[MassPolishFit]:
    stage_candidates = copy.deepcopy(list(candidates))
    warm_base = None
    if warm_fit is not None:
        fitted_by_id = {
            int(subhalo["candidate_id"]): subhalo
            for subhalo in warm_fit.fitted_subhalos
        }
        for candidate in stage_candidates:
            fitted = fitted_by_id[int(candidate["candidate_id"])]
            candidate["ra"] = fitted["center_x"]
            candidate["dec"] = fitted["center_y"]
            candidate["logM"] = fitted["logM"]
        warm_base = warm_fit.fitted_base_lens
    completed = {
        fit.replica: fit for fit in existing if fit.stage == stage and fit.error is None
    }
    payloads = []
    for replica in range(replicas):
        if replica in completed:
            continue
        payloads.append(
            {
                "stage": stage,
                "replica": replica,
                "random_seed": config.random_seed
                + (1_000_000 if stage == "high" else 0)
                + replica,
                "config": config,
                "candidates": stage_candidates,
                "base_proposal": base_proposal,
                "observation": observation,
                "parameter_space": parameter_space,
                "warm_base_lens": warm_base,
                "particles": particles,
                "iterations": iterations,
                "sigma_scale": sigma_scale,
            }
        )
    results = list(completed.values())
    with process_pool(config.workers) as executor:
        futures = [executor.submit(_fit_replica, payload) for payload in payloads]
        for future in as_completed(futures):
            fit = future.result()
            results.append(fit)
            _append(archive, fit)
    return results


def polish_masses(
    selection: ExactCountSelection,
    base_proposal: dict[str, Any],
    observation: Observation,
    parameter_space: ParameterSpace,
    config: MassPolishConfig,
    output_directory: str | Path,
) -> MassPolishResult:
    """Polish the selected support with tied mass and concentration."""
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    candidates = _polish_candidates(selection, config.position_half_width_arcsec)
    ensure_archive_context(
        output / "archive_context.json",
        stage="tied_mass_polish",
        observation=observation,
        parameter_space=parameter_space,
        configuration=config,
        inputs={
            "selection": asdict(selection),
            "candidates": candidates,
            "base_proposal": base_proposal,
        },
    )
    archive = output / "mass_polish.jsonl"
    existing = _load_archive(archive)
    medium = _run_stage(
        "medium",
        config.medium_replicas,
        candidates,
        base_proposal,
        observation,
        parameter_space,
        config,
        archive,
        particles=config.medium_particles,
        iterations=config.medium_iterations,
        sigma_scale=config.medium_sigma_scale,
        existing=existing,
    )
    eligible_medium = [fit for fit in medium if fit.eligible]
    if not eligible_medium:
        raise RuntimeError("mass polish produced no eligible medium-budget fit")
    best_medium = min(eligible_medium, key=lambda fit: fit.bic)
    high = _run_stage(
        "high",
        config.high_replicas,
        candidates,
        base_proposal,
        observation,
        parameter_space,
        config,
        archive,
        particles=config.high_particles,
        iterations=config.high_iterations,
        sigma_scale=config.high_sigma_scale,
        warm_fit=best_medium,
        existing=existing,
    )
    eligible_high = [fit for fit in high if fit.eligible]
    if not eligible_high:
        raise RuntimeError("mass polish produced no eligible high-budget fit")
    selected = min(eligible_high, key=lambda fit: fit.bic)
    result = MassPolishResult(
        known_count=selection.known_count,
        medium_replicas=len(medium),
        high_replicas=len(high),
        failed_replicas=sum(fit.error is not None for fit in medium + high),
        candidates=tuple(candidates),
        selected=selected,
    )
    write_json(output / "mass_polish_selection.json", asdict(result))
    return result
