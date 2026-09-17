"""Known-count joint support search for fixed-count RSI."""

from __future__ import annotations

import copy
import itertools
import json
import math
import os
import random
import time
from collections import Counter, defaultdict, deque
from collections.abc import Iterable, Sequence
from concurrent.futures import FIRST_COMPLETED, wait
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from lensagent.config import ExactCountSearchConfig, PSOConfig
from lensagent.data.observation import Observation
from lensagent.modeling.parameters import ParameterSpace
from lensagent.output.artifacts import NumpyEncoder, write_json
from lensagent.rsi.common import independent_nfw_space, nfw_mass_msun
from lensagent.rsi.multisubhalo.artifacts import ensure_archive_context
from lensagent.workflow.pso import run_family_pso
from lensagent.workflow.processes import process_pool


@dataclass(frozen=True)
class SupportFit:
    stage: str
    support: tuple[int, ...]
    support_blob_ids: tuple[int, ...]
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
class ExactCountSelection:
    known_count: int
    candidate_count: int
    candidate_identity_count: int
    support_universe_size: int
    medium_evaluated: int
    high_evaluated: int
    failed_evaluations: int
    selected: SupportFit


def _groups(candidates: Sequence[dict[str, Any]]) -> dict[int, list[int]]:
    grouped: dict[int, list[int]] = defaultdict(list)
    for candidate in candidates:
        grouped[int(candidate["blob_id"])].append(int(candidate["candidate_id"]))
    for indices in grouped.values():
        indices.sort()
    return dict(grouped)


def _ordered_blobs(candidates: Sequence[dict[str, Any]]) -> list[int]:
    grouped = _groups(candidates)
    return sorted(
        grouped,
        key=lambda blob: min(
            (
                int(candidates[index].get("source_rank", index + 1)),
                index,
            )
            for index in grouped[blob]
        ),
    )


def support_universe_size(blob_ids: Iterable[int], known_count: int) -> int:
    multiplicities = Counter(int(blob_id) for blob_id in blob_ids)
    if known_count < 0 or known_count > len(multiplicities):
        return 0
    coefficients = [0] * (known_count + 1)
    coefficients[0] = 1
    for multiplicity in multiplicities.values():
        for size in range(known_count, 0, -1):
            coefficients[size] += coefficients[size - 1] * multiplicity
    return coefficients[known_count]


def _all_supports(
    candidate_count: int,
    known_count: int,
    blob_ids: Sequence[int],
) -> Iterable[tuple[int, ...]]:
    for support in itertools.combinations(range(candidate_count), known_count):
        if len({blob_ids[index] for index in support}) == known_count:
            yield support


def _weighted_support(
    candidates: Sequence[dict[str, Any]],
    known_count: int,
    rng: random.Random,
) -> tuple[int, ...]:
    grouped = _groups(candidates)
    remaining = _ordered_blobs(candidates)
    selected_blobs = []
    while len(selected_blobs) < known_count:
        weights = [
            1.0
            / math.sqrt(
                max(
                    1,
                    min(
                        int(candidates[index].get("source_rank", index + 1))
                        for index in grouped[blob]
                    ),
                )
            )
            for blob in remaining
        ]
        blob = rng.choices(remaining, weights=weights, k=1)[0]
        selected_blobs.append(blob)
        remaining.remove(blob)
    selected = []
    for blob in selected_blobs:
        choices = grouped[blob]
        weights = [1.0 / math.sqrt(index + 1.0) for index in choices]
        selected.append(rng.choices(choices, weights=weights, k=1)[0])
    return tuple(sorted(selected))


def _initial_supports(
    candidates: Sequence[dict[str, Any]],
    known_count: int,
    limit: int,
    rng: random.Random,
) -> list[tuple[int, ...]]:
    grouped = _groups(candidates)
    blobs = _ordered_blobs(candidates)
    primary = {blob: grouped[blob][0] for blob in blobs}
    result = []
    seen = set()

    def add(values: Iterable[int]) -> None:
        support = tuple(sorted(int(value) for value in values))
        if len(support) != known_count or support in seen:
            return
        if len({candidates[index]["blob_id"] for index in support}) != known_count:
            return
        seen.add(support)
        result.append(support)

    for offset in range(len(blobs)):
        window = [blobs[(offset + index) % len(blobs)] for index in range(known_count)]
        add(primary[blob] for blob in window)
    for combination in itertools.combinations(blobs[: known_count + 5], known_count):
        add(primary[blob] for blob in combination)
        if len(result) >= limit:
            return result
    initial = tuple(primary[blob] for blob in blobs[:known_count])
    for selected_index, candidate_id in enumerate(initial):
        blob = int(candidates[candidate_id]["blob_id"])
        for variant in grouped[blob][1:]:
            replacement = list(initial)
            replacement[selected_index] = variant
            add(replacement)
    attempts = 0
    while len(result) < limit and attempts < limit * 100:
        attempts += 1
        add(_weighted_support(candidates, known_count, rng))
    return result


def _weakest_candidates(fit: SupportFit) -> list[int]:
    scored = []
    for candidate_id, subhalo in zip(fit.support, fit.fitted_subhalos):
        alpha = float(subhalo.get("alpha_Rs", 0.0) or 0.0)
        scale = float(subhalo.get("Rs", 0.0) or 0.0)
        boundary = 0.0
        bounds = subhalo.get("center_bounds")
        if bounds:
            x_low, x_high = bounds["center_x"]
            y_low, y_high = bounds["center_y"]
            x_scale = max(float(x_high) - float(x_low), 1.0e-6) / 2.0
            y_scale = max(float(y_high) - float(y_low), 1.0e-6) / 2.0
            boundary = max(
                abs(float(subhalo["center_x"]) - float(subhalo["seed_ra"])) / x_scale,
                abs(float(subhalo["center_y"]) - float(subhalo["seed_dec"])) / y_scale,
            )
        duplicate = 0.0
        for other in fit.fitted_subhalos:
            if other is subhalo:
                continue
            distance = math.hypot(
                float(subhalo["center_x"]) - float(other["center_x"]),
                float(subhalo["center_y"]) - float(other["center_y"]),
            )
            if distance < 0.2:
                duplicate = max(duplicate, 1.0 - distance / 0.2)
        score = 1.0 / max(alpha, 1.0e-5) + 0.2 / max(scale, 1.0e-4)
        score += 5.0 if boundary > 0.92 else 0.0
        score += 4.0 * duplicate + candidate_id * 1.0e-4
        scored.append((score, candidate_id))
    return [candidate_id for _, candidate_id in sorted(scored, reverse=True)]


def _children(
    fit: SupportFit,
    candidates: Sequence[dict[str, Any]],
    known_count: int,
    limit: int,
    rng: random.Random,
) -> list[tuple[int, ...]]:
    grouped = _groups(candidates)
    support = tuple(fit.support)
    selected_by_blob = {int(candidates[index]["blob_id"]): index for index in support}
    absent_blobs = [
        blob for blob in _ordered_blobs(candidates) if blob not in selected_by_blob
    ]
    result = []
    seen = set()

    def add(values: Iterable[int]) -> None:
        child = tuple(sorted(int(value) for value in values))
        if child == support or child in seen or len(child) != known_count:
            return
        if len({candidates[index]["blob_id"] for index in child}) != known_count:
            return
        seen.add(child)
        result.append(child)

    for candidate_id in support:
        blob = int(candidates[candidate_id]["blob_id"])
        for variant in grouped[blob]:
            if variant != candidate_id:
                add((set(support) - {candidate_id}) | {variant})
                if len(result) >= limit:
                    return result
    for drop_id in _weakest_candidates(fit):
        for add_blob in absent_blobs:
            for add_id in grouped[add_blob]:
                add((set(support) - {drop_id}) | {add_id})
                if len(result) >= limit:
                    return result
    attempts = 0
    while len(result) < limit and absent_blobs and attempts < limit * 50:
        attempts += 1
        drop_id = rng.choice(support)
        add_blob = rng.choice(absent_blobs)
        add((set(support) - {drop_id}) | {rng.choice(grouped[add_blob])})
    return result


def _jaccard(left: Sequence[int], right: Sequence[int]) -> float:
    left_set, right_set = set(left), set(right)
    return len(left_set & right_set) / max(1, len(left_set | right_set))


def _select_diverse(
    fits: Sequence[SupportFit], limit: int, threshold: float
) -> list[SupportFit]:
    selected = []
    for fit in sorted(fits, key=lambda item: item.bic):
        if all(_jaccard(fit.support, prior.support) <= threshold for prior in selected):
            selected.append(fit)
            if len(selected) >= limit:
                break
    return selected


class _SupportGenerator:
    def __init__(
        self,
        candidates: Sequence[dict[str, Any]],
        known_count: int,
        config: ExactCountSearchConfig,
        rng: random.Random,
    ) -> None:
        self.candidates = candidates
        self.known_count = known_count
        self.config = config
        self.rng = rng
        self.blob_ids = [int(candidate["blob_id"]) for candidate in candidates]
        self.universe_size = support_universe_size(self.blob_ids, known_count)
        self.exhaustive = self.universe_size <= config.exhaustive_limit
        self.completed: list[SupportFit] = []
        self.proposed: set[tuple[int, ...]] = set()
        self.returned: set[tuple[int, ...]] = set()
        self.queue: deque[tuple[int, ...]] = deque()
        if self.exhaustive:
            supports = list(_all_supports(len(candidates), known_count, self.blob_ids))
            rng.shuffle(supports)
            self.queue.extend(supports)
        else:
            limit = max(
                config.workers * config.queue_multiplier * 2,
                config.beam_width,
            )
            self.queue.extend(_initial_supports(candidates, known_count, limit, rng))

    def _add(self, support: tuple[int, ...]) -> None:
        support = tuple(sorted(set(support)))
        if len(support) != self.known_count or support in self.proposed:
            return
        if len({self.blob_ids[index] for index in support}) != self.known_count:
            return
        self.proposed.add(support)
        self.queue.append(support)

    def _refill(self) -> None:
        if self.exhaustive:
            return
        eligible = [fit for fit in self.completed if fit.eligible]
        for parent in _select_diverse(
            eligible, self.config.beam_width, self.config.jaccard_threshold
        ):
            for child in _children(
                parent,
                self.candidates,
                self.known_count,
                self.config.children_per_parent,
                self.rng,
            ):
                self._add(child)
        target = self.config.workers * self.config.queue_multiplier
        attempts = 0
        while len(self.queue) < target and attempts < target * 200:
            attempts += 1
            self._add(_weighted_support(self.candidates, self.known_count, self.rng))

    def next(self) -> tuple[int, ...] | None:
        while self.queue:
            support = self.queue.popleft()
            if support not in self.returned:
                self.proposed.add(support)
                self.returned.add(support)
                return support
        self._refill()
        while self.queue:
            support = self.queue.popleft()
            if support not in self.returned:
                self.proposed.add(support)
                self.returned.add(support)
                return support
        return None


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


def _fit_support(payload: dict[str, Any]) -> SupportFit:
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = "1"
    started = time.monotonic()
    support = tuple(payload["support"])
    candidates = copy.deepcopy(payload["candidates"])
    stage = payload["stage"]
    seed = int(payload["random_seed"])
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    try:
        selected = [candidates[index] for index in support]
        config: ExactCountSearchConfig = payload["config"]
        for candidate in selected:
            candidate.setdefault(
                "center_bounds",
                {
                    "center_x": (
                        candidate["ra"] - config.center_half_width_arcsec,
                        candidate["ra"] + config.center_half_width_arcsec,
                    ),
                    "center_y": (
                        candidate["dec"] - config.center_half_width_arcsec,
                        candidate["dec"] + config.center_half_width_arcsec,
                    ),
                },
            )
        base_space: ParameterSpace = payload["parameter_space"]
        base_proposal = payload["base_proposal"]
        space = independent_nfw_space(
            base_space,
            base_proposal,
            selected,
            freeze_smooth_model=True,
            center_half_width=config.center_half_width_arcsec,
            macro_thaw={
                "theta_E": config.theta_e_half_width,
                "gamma": config.slope_half_width,
                "e1": config.ellipticity_half_width,
                "e2": config.ellipticity_half_width,
            },
            warm_base_lens=payload.get("warm_base_lens"),
        )
        observation: Observation = payload["observation"].with_model(space.model)
        result = run_family_pso(
            observation,
            space,
            PSOConfig(
                seeds=0,
                runs=int(payload["runs"]),
                particles=int(payload["particles"]),
                iterations=int(payload["iterations"]),
                sigma_scale=float(payload["sigma_scale"]),
            ),
        )
        base_lens_count = len(base_proposal["kwargs_lens"])
        eligible_fits = []
        all_fits = []
        for fit in result.fits:
            proposal = fit.proposal
            subhalos = []
            for candidate, component in zip(
                selected, proposal["kwargs_lens"][base_lens_count:]
            ):
                mass = nfw_mass_msun(
                    component["Rs"],
                    component["alpha_Rs"],
                    observation.z_lens,
                    observation.z_source,
                )
                subhalos.append(
                    {
                        "candidate_id": int(candidate["candidate_id"]),
                        "blob_id": int(candidate["blob_id"]),
                        "coordinate_variant": candidate["coordinate_variant"],
                        "source_rank": int(candidate["source_rank"]),
                        "seed_ra": float(candidate["ra"]),
                        "seed_dec": float(candidate["dec"]),
                        "center_bounds": copy.deepcopy(candidate["center_bounds"]),
                        "center_x": float(component["center_x"]),
                        "center_y": float(component["center_y"]),
                        "Rs": float(component["Rs"]),
                        "alpha_Rs": float(component["alpha_Rs"]),
                        "mass_msun": mass,
                    }
                )
            separation = _minimum_separation(subhalos)
            item = (fit, tuple(subhalos), separation)
            all_fits.append(item)
            if separation is None or separation >= config.minimum_separation_arcsec:
                eligible_fits.append(item)
        fit, subhalos, separation = min(
            eligible_fits or all_fits, key=lambda item: item[0].bic
        )
        return SupportFit(
            stage=stage,
            support=support,
            support_blob_ids=tuple(
                int(candidates[index]["blob_id"]) for index in support
            ),
            bic=float(fit.bic),
            log_likelihood=float(fit.log_likelihood),
            eligible=bool(eligible_fits),
            proposal=fit.proposal,
            fitted_subhalos=subhalos,
            fitted_base_lens=tuple(
                copy.deepcopy(fit.proposal["kwargs_lens"][:base_lens_count])
            ),
            minimum_separation_arcsec=separation,
            random_seed=seed,
            elapsed_seconds=time.monotonic() - started,
        )
    except Exception as exc:
        return SupportFit(
            stage=stage,
            support=support,
            support_blob_ids=(),
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


def _append_archive(path: Path, fit: SupportFit) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(asdict(fit), cls=NumpyEncoder) + "\n")


def _load_archive(path: Path) -> list[SupportFit]:
    if not path.exists():
        return []
    latest = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            item["support"] = tuple(item["support"])
            item["support_blob_ids"] = tuple(item["support_blob_ids"])
            item["fitted_subhalos"] = tuple(item["fitted_subhalos"])
            item["fitted_base_lens"] = tuple(item["fitted_base_lens"])
            fit = SupportFit(**item)
            latest[(fit.stage, fit.support)] = fit
    return list(latest.values())


def _evaluate_supports(
    supports: Iterable[tuple[int, ...]],
    candidates: Sequence[dict[str, Any]],
    base_proposal: dict[str, Any],
    observation: Observation,
    parameter_space: ParameterSpace,
    config: ExactCountSearchConfig,
    *,
    stage: str,
    particles: int,
    iterations: int,
    runs: int,
    sigma_scale: float,
    maximum: int,
    deadline: float,
    archive_path: Path,
    warm_fits: dict[tuple[int, ...], SupportFit] | None = None,
    on_complete=None,
) -> list[SupportFit]:
    iterator = iter(supports)
    completed = []
    submitted = 0
    exhausted = False
    pending = {}
    with process_pool(config.workers) as executor:
        while pending or (
            not exhausted and submitted < maximum and time.monotonic() < deadline
        ):
            while (
                not exhausted
                and len(pending) < config.workers * config.queue_multiplier
                and submitted < maximum
                and time.monotonic() < deadline
            ):
                try:
                    support = next(iterator)
                except StopIteration:
                    exhausted = True
                    break
                warm = warm_fits.get(support) if warm_fits else None
                seed = (
                    config.random_seed
                    + (1_000_000 if stage == "high" else 0)
                    + sum(
                        (index + 1) * (value + 1) * 101
                        for index, value in enumerate(support)
                    )
                )
                payload = {
                    "support": support,
                    "candidates": candidates,
                    "stage": stage,
                    "random_seed": seed,
                    "config": config,
                    "base_proposal": base_proposal,
                    "observation": observation,
                    "parameter_space": parameter_space,
                    "particles": particles,
                    "iterations": iterations,
                    "runs": runs,
                    "sigma_scale": sigma_scale,
                    "warm_base_lens": warm.fitted_base_lens if warm else None,
                }
                if warm:
                    warmed = copy.deepcopy(list(candidates))
                    for subhalo in warm.fitted_subhalos:
                        candidate = warmed[int(subhalo["candidate_id"])]
                        candidate["ra"] = subhalo["center_x"]
                        candidate["dec"] = subhalo["center_y"]
                        candidate["Rs"] = subhalo["Rs"]
                        candidate["alpha_Rs"] = subhalo["alpha_Rs"]
                    payload["candidates"] = warmed
                pending[executor.submit(_fit_support, payload)] = support
                submitted += 1
            if not pending:
                break
            finished, _ = wait(pending, timeout=30, return_when=FIRST_COMPLETED)
            for future in finished:
                pending.pop(future)
                fit = future.result()
                completed.append(fit)
                if on_complete is not None:
                    on_complete(fit)
                _append_archive(archive_path, fit)
    return completed


def search_exact_count(
    candidates: Sequence[dict[str, Any]],
    known_count: int,
    base_proposal: dict[str, Any],
    observation: Observation,
    parameter_space: ParameterSpace,
    config: ExactCountSearchConfig,
    output_directory: str | Path,
) -> ExactCountSelection:
    """Search candidate combinations and select the minimum-BIC support."""
    if known_count < 1:
        raise ValueError("known_count must be positive")
    search_candidates = copy.deepcopy(list(candidates))
    for candidate in search_candidates:
        candidate.setdefault(
            "center_bounds",
            {
                "center_x": (
                    candidate["ra"] - config.center_half_width_arcsec,
                    candidate["ra"] + config.center_half_width_arcsec,
                ),
                "center_y": (
                    candidate["dec"] - config.center_half_width_arcsec,
                    candidate["dec"] + config.center_half_width_arcsec,
                ),
            },
        )
    identities = {int(candidate["blob_id"]) for candidate in search_candidates}
    if known_count > len(identities):
        raise ValueError(
            f"known count {known_count} exceeds {len(identities)} candidate identities"
        )
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    ensure_archive_context(
        output / "archive_context.json",
        stage="exact_count_support_search",
        observation=observation,
        parameter_space=parameter_space,
        configuration=config,
        inputs={
            "known_count": known_count,
            "candidates": search_candidates,
            "base_proposal": base_proposal,
        },
    )
    archive_path = output / "support_search.jsonl"
    existing = _load_archive(archive_path)
    rng = random.Random(config.random_seed)
    generator = _SupportGenerator(search_candidates, known_count, config, rng)
    existing_medium = [
        fit for fit in existing if fit.stage == "medium" and fit.error is None
    ]
    existing_high = [
        fit for fit in existing if fit.stage == "high" and fit.error is None
    ]
    generator.completed.extend(existing_medium)
    generator.returned.update(fit.support for fit in existing_medium)
    generator.proposed.update(fit.support for fit in existing_medium)
    medium_target = (
        generator.universe_size
        if generator.exhaustive
        else min(generator.universe_size, config.maximum_medium_evaluations)
    )
    started = time.monotonic()
    deadline = started + config.time_budget_seconds
    search_deadline = started + config.time_budget_seconds * config.search_fraction

    def stream():
        while time.monotonic() < search_deadline:
            support = generator.next()
            if support is None:
                break
            yield support

    new_medium = _evaluate_supports(
        stream(),
        search_candidates,
        base_proposal,
        observation,
        parameter_space,
        config,
        stage="medium",
        particles=config.medium_particles,
        iterations=config.medium_iterations,
        runs=config.medium_runs,
        sigma_scale=config.medium_sigma_scale,
        maximum=max(0, medium_target - len(existing_medium)),
        deadline=search_deadline,
        archive_path=archive_path,
        on_complete=generator.completed.append,
    )
    medium = existing_medium + new_medium
    eligible_medium = [fit for fit in medium if fit.eligible]
    promotions = _select_diverse(
        eligible_medium,
        min(config.high_budget_promotions, len(eligible_medium)),
        config.jaccard_threshold,
    )
    selected_keys = {fit.support for fit in promotions}
    for fit in sorted(eligible_medium, key=lambda item: item.bic):
        if len(promotions) >= config.high_budget_promotions:
            break
        if fit.support not in selected_keys:
            promotions.append(fit)
            selected_keys.add(fit.support)
    warm = {fit.support: fit for fit in promotions}
    existing_high_supports = {fit.support for fit in existing_high}
    pending_promotions = [
        fit for fit in promotions if fit.support not in existing_high_supports
    ]
    new_high = _evaluate_supports(
        [fit.support for fit in pending_promotions],
        search_candidates,
        base_proposal,
        observation,
        parameter_space,
        config,
        stage="high",
        particles=config.high_particles,
        iterations=config.high_iterations,
        runs=config.high_runs,
        sigma_scale=config.high_sigma_scale,
        maximum=len(pending_promotions),
        deadline=deadline,
        archive_path=archive_path,
        warm_fits=warm,
    )
    high = existing_high + new_high
    selection_pool = [fit for fit in high if fit.eligible] or eligible_medium
    if not selection_pool:
        raise RuntimeError("known-count support search produced no eligible fit")
    selected = min(selection_pool, key=lambda fit: fit.bic)
    result = ExactCountSelection(
        known_count=known_count,
        candidate_count=len(search_candidates),
        candidate_identity_count=len(identities),
        support_universe_size=generator.universe_size,
        medium_evaluated=len(medium),
        high_evaluated=len(high),
        failed_evaluations=sum(fit.error is not None for fit in medium + high),
        selected=selected,
    )
    write_json(output / "support_selection.json", asdict(result))
    return result
