"""Catalog records for SDSS and mock systems."""

from __future__ import annotations

import csv
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

from lensagent.config import DatasetKind


@dataclass(frozen=True)
class CatalogEntry:
    system_id: str
    dataset: DatasetKind
    z_lens: float
    z_source: float
    sigma_obs: float
    sigma_obs_err: float
    ra_deg: float = 0.0
    dec_deg: float = 0.0
    observation: str = ""
    subhalo_count: int | None = None

    @classmethod
    def from_row(cls, row: dict[str, str]) -> CatalogEntry:
        count = row.get("subhalo_count", "").strip()
        return cls(
            system_id=row["system_id"].strip(),
            dataset=DatasetKind(row["dataset"].strip()),
            z_lens=float(row["z_lens"]),
            z_source=float(row["z_source"]),
            sigma_obs=float(row["sigma_obs"]),
            sigma_obs_err=float(row["sigma_obs_err"]),
            ra_deg=float(row.get("ra_deg") or 0.0),
            dec_deg=float(row.get("dec_deg") or 0.0),
            observation=row.get("observation", "").strip(),
            subhalo_count=int(count) if count else None,
        )


class Catalog:
    def __init__(self, entries: Iterable[CatalogEntry]):
        items = tuple(entries)
        by_id = {entry.system_id: entry for entry in items}
        if len(by_id) != len(items):
            raise ValueError("catalog system_id values must be unique")
        self._entries = items
        self._by_id = by_id

    def __iter__(self) -> Iterator[CatalogEntry]:
        return iter(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def __getitem__(self, system_id: str) -> CatalogEntry:
        try:
            return self._by_id[system_id]
        except KeyError as exc:
            raise KeyError(f"system {system_id!r} is not in the catalog") from exc

    @classmethod
    def load(cls, path: str | Path) -> Catalog:
        with Path(path).open("r", encoding="utf-8", newline="") as handle:
            return cls(CatalogEntry.from_row(row) for row in csv.DictReader(handle))

    def select(self, dataset: DatasetKind | str) -> tuple[CatalogEntry, ...]:
        kind = DatasetKind(dataset)
        return tuple(entry for entry in self if entry.dataset is kind)
