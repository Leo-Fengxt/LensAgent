"""Persistent proposal records for LensAgent searches."""

from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from lensagent.modeling.scoring import ScoringPolicy, tiered_sample
from lensagent.output.artifacts import write_json

IMAGE_FIELDS = {"model_image", "residual_map", "lens_light_image"}


def serializable_evaluation(evaluation: dict[str, Any]) -> dict[str, Any]:
    result = {}
    for name, value in evaluation.items():
        if name in IMAGE_FIELDS or isinstance(value, np.ndarray):
            continue
        result[name] = value
    return result


@dataclass
class ProposalRecord:
    id: str
    proposal: dict[str, Any]
    evaluation: dict[str, Any]
    quality: float
    diversity: float
    behavior_vector: list[float]
    island: int = 0
    timestamp: float = field(default_factory=time.time)


class ProposalDatabase:
    def __init__(self, path: str | Path, scoring: ScoringPolicy):
        self.path = Path(path)
        self.scoring = scoring
        self.refinement = None
        self._records: list[ProposalRecord] = []
        self._lock = threading.Lock()
        if self.path.exists() and self.path.stat().st_size > 2:
            self.load()

    @property
    def records(self) -> tuple[ProposalRecord, ...]:
        return tuple(self._records)

    @property
    def best(self) -> ProposalRecord | None:
        return max(self._records, key=lambda record: record.quality, default=None)

    @property
    def size(self) -> int:
        return len(self._records)

    def create(
        self, proposal: dict[str, Any], evaluation: dict[str, Any], *, island: int = 0
    ) -> ProposalRecord:
        evaluation = dict(evaluation)
        evaluation.setdefault(
            "residual_randomness", residual_randomness_or_none(evaluation)
        )
        behavior = self.scoring.flatten(proposal)
        diversity = self.scoring.diversity(
            behavior, (record.behavior_vector for record in self._records)
        )
        quality = self.scoring.quality(evaluation, proposal, diversity=diversity)
        return ProposalRecord(
            id=uuid.uuid4().hex[:12],
            proposal=proposal,
            evaluation=serializable_evaluation(evaluation),
            quality=quality,
            diversity=diversity,
            behavior_vector=behavior.tolist(),
            island=island,
        )

    def add(self, record: ProposalRecord) -> None:
        with self._lock:
            self._records.append(record)
            self._refresh_unlocked()
            self.save()

    def _refresh_unlocked(self) -> None:
        for index, record in enumerate(self._records):
            neighbors = (
                other.behavior_vector
                for other_index, other in enumerate(self._records)
                if other_index != index
            )
            record.diversity = self.scoring.diversity(record.behavior_vector, neighbors)
            record.quality = self.scoring.quality(
                record.evaluation, record.proposal, diversity=record.diversity
            )

    def sample(
        self, count: int, rng: np.random.Generator, *, island: int | None = None
    ) -> list[ProposalRecord]:
        if self.refinement is not None:
            return self.refinement.sample(self._records, count, rng, island)
        pool = (
            [record for record in self._records if record.island == island]
            if island is not None
            else list(self._records)
        )
        if len(pool) < count:
            pool = list(self._records)
        return list(
            tiered_sample(
                pool,
                count,
                rng,
                diversity_weight=self.scoring.diversity_weight,
            )
        )

    def records_in_island(self, island: int) -> list[ProposalRecord]:
        return [record for record in self._records if record.island == island]

    def trim_island(self, island: int, maximum: int = 50) -> int:
        with self._lock:
            pool = self.records_in_island(island)
            if len(pool) <= maximum:
                return 0
            keep: set[str] = (self.refinement.protected_ids(self._records)
                              if self.refinement is not None else set())
            for record in sorted(pool, key=lambda item: item.quality, reverse=True)[
                : round(maximum * 0.6)
            ]:
                keep.add(record.id)
            for record in sorted(pool, key=lambda item: item.diversity, reverse=True):
                if len(keep) >= round(maximum * 0.8):
                    break
                keep.add(record.id)
            pareto = [
                record
                for record in pool
                if not any(
                    other.quality > record.quality
                    and other.diversity > record.diversity
                    for other in pool
                    if other is not record
                )
            ]
            for record in sorted(
                pareto, key=lambda item: (item.quality, item.diversity), reverse=True
            ):
                if len(keep) >= maximum:
                    break
                keep.add(record.id)
            for record in sorted(pool, key=lambda item: item.quality, reverse=True):
                if len(keep) >= maximum:
                    break
                keep.add(record.id)
            before = len(self._records)
            self._records = [
                record
                for record in self._records
                if record.island != island or record.id in keep
            ]
            self._refresh_unlocked()
            self.save()
            return before - len(self._records)

    def save(self) -> None:
        records = [
            {
                "id": record.id,
                "proposal": record.proposal,
                "evaluation": record.evaluation,
                "quality": record.quality,
                "diversity": record.diversity,
                "behavior_vector": record.behavior_vector,
                "island": record.island,
                "timestamp": record.timestamp,
            }
            for record in self._records
        ]
        write_json(self.path, records)

    def remove(self, record_id: str) -> None:
        with self._lock:
            self._records = [record for record in self._records if record.id != record_id]
            self._refresh_unlocked()
            self.save()

    def load(self) -> None:
        data = json.loads(self.path.read_text(encoding="utf-8"))
        self._records = [ProposalRecord(**item) for item in data]

    def qualities(self) -> np.ndarray:
        return np.asarray([record.quality for record in self._records], dtype=float)

    def diversities(self) -> np.ndarray:
        return np.asarray([record.diversity for record in self._records], dtype=float)

    def proposals(self) -> list[dict[str, Any]]:
        return [record.proposal for record in self._records]

    def set_scoring(self, scoring: ScoringPolicy) -> None:
        with self._lock:
            self.scoring = scoring
            self._refresh_unlocked()
            self.save()


def residual_randomness_or_none(evaluation: dict[str, Any]) -> float | None:
    from lensagent.modeling.scoring import residual_randomness

    return residual_randomness(evaluation)
