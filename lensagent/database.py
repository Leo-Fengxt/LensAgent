"""JSON-backed proposal database with scoring and persistence."""

import json
import logging
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from . import scoring as S

log = logging.getLogger(__name__)


@dataclass
class ProposalEntry:
    id: str
    proposal: Dict[str, Any]
    eval_results: Dict[str, Any]
    quality: float
    diversity: float
    behavior_vector: List[float]
    island: int = 0
    timestamp: float = field(default_factory=time.time)

    def summary(self, show_diversity: bool = True) -> str:
        chi2_img = self.eval_results.get("image_chi2_reduced", None)
        chi2_kin = self.eval_results.get("kin_chi2", None)
        sigma = self.eval_results.get("sigma_predicted", None)
        parts = [f"[{self.id[:8]}]  quality={self.quality:+.3f}"]
        if show_diversity:
            parts.append(f"diversity={self.diversity:.4f}")
        parts.append(f"chi2_img={chi2_img}  chi2_kin={chi2_kin}  sigma={sigma}")
        return "  ".join(parts)


_IMAGE_KEYS = {"model_image", "residual_map", "lens_light_image"}


def _strip_images(eval_results: Dict[str, Any]) -> Dict[str, Any]:
    """Remove image arrays/lists before JSON serialization."""
    out = {}
    for k, v in eval_results.items():
        if isinstance(v, np.ndarray):
            continue
        if k in _IMAGE_KEYS:
            continue
        out[k] = v
    return out


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        if hasattr(obj, 'tolist'):
            return obj.tolist()
        return super().default(obj)


class ProposalDatabase:
    """Persistent collection of scored proposals.

    Data is stored as a JSON file on disk.  The database is loaded into
    memory on construction and flushed after every mutation.
    Thread-safe: all mutations acquire an internal lock.
    """

    def __init__(self, db_path: str = "lensagent_db.json"):
        import threading
        self.db_path = Path(db_path)
        self._entries: List[ProposalEntry] = []
        self._lock = threading.Lock()
        if self.db_path.exists() and self.db_path.stat().st_size > 2:
            self.load()
            log.info("Loaded database with %d entries from %s",
                     len(self._entries), self.db_path)
        else:
            log.info("New database (will write to %s)", self.db_path)

    @property
    def size(self) -> int:
        return len(self._entries)

    @property
    def all_entries(self) -> List[ProposalEntry]:
        return list(self._entries)

    @property
    def best(self) -> Optional[ProposalEntry]:
        if not self._entries:
            return None
        return max(self._entries, key=lambda e: e.quality)

    def all_proposals(self) -> List[Dict[str, Any]]:
        return [e.proposal for e in self._entries]

    def all_behavior_vecs(self) -> np.ndarray:
        if not self._entries:
            return np.empty((0, 5))
        return np.array([e.behavior_vector for e in self._entries])

    def all_qualities(self) -> np.ndarray:
        return np.array([e.quality for e in self._entries])

    def all_diversities(self) -> np.ndarray:
        return np.array([e.diversity for e in self._entries])

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------

    def add(self, entry: ProposalEntry) -> None:
        with self._lock:
            self._entries.append(entry)
            self.save()
        log.debug("db.add  %s  size=%d",
                  entry.summary(show_diversity=False), self.size)

    def update_all_diversity(self) -> None:
        """Recompute diversity for every entry (call after adding)."""
        with self._lock:
            if len(self._entries) < 2:
                for e in self._entries:
                    e.diversity = 1.0
                return
            all_vecs = self.all_behavior_vecs()
            for i, entry in enumerate(self._entries):
                others = np.delete(all_vecs, i, axis=0)
                entry.diversity = S.compute_diversity(
                    np.array(entry.behavior_vector), others)
            self.save()
        divs = self.all_diversities()
        log.debug("Diversity updated: min=%.4f  mean=%.4f  max=%.4f",
                  float(divs.min()), float(divs.mean()), float(divs.max()))

    def trim_island(self, island: int, max_size: int = 50) -> int:
        """Evict lowest-quality entries from an island. Returns count evicted."""
        with self._lock:
            pool = [e for e in self._entries if e.island == island]
            if len(pool) <= max_size:
                return 0
            pool.sort(key=lambda e: e.quality, reverse=True)
            keep_ids = {e.id for e in pool[:max_size]}
            before = len(self._entries)
            self._entries = [e for e in self._entries
                         if e.island != island or e.id in keep_ids]
        evicted = before - len(self._entries)
        if evicted:
            self.save()
        return evicted

    def sample(self, n: int = 5, rng: Optional[np.random.Generator] = None,
               island: Optional[int] = None) -> List[ProposalEntry]:
        if island is not None:
            pool = [e for e in self._entries if e.island == island]
            if len(pool) < n:
                pool = list(self._entries)
        else:
            pool = list(self._entries)
        return S.tiered_sample(pool, n=n, rng=rng)

    def entries_in_island(self, island: int) -> List[ProposalEntry]:
        return [e for e in self._entries if e.island == island]

    def island_sizes(self, n_islands: int) -> Dict[int, int]:
        counts: Dict[int, int] = {i: 0 for i in range(n_islands)}
        for e in self._entries:
            counts[e.island] = counts.get(e.island, 0) + 1
        return counts

    def best_per_island(self, n_islands: int) -> Dict[int, Optional[ProposalEntry]]:
        result: Dict[int, Optional[ProposalEntry]] = {}
        for i in range(n_islands):
            pool = self.entries_in_island(i)
            result[i] = max(pool, key=lambda e: e.quality) if pool else None
        return result

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self) -> None:
        records = []
        for e in self._entries:
            rec = {
                "id": e.id,
                "proposal": e.proposal,
                "eval_results": _strip_images(e.eval_results),
                "quality": e.quality,
                "diversity": e.diversity,
                "behavior_vector": list(e.behavior_vector),
                "island": e.island,
                "timestamp": e.timestamp,
            }
            records.append(rec)
        abs_path = self.db_path.resolve()
        abs_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = abs_path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(records, f, cls=_NumpyEncoder, indent=1)
        try:
            tmp.rename(abs_path)
        except OSError:
            import shutil
            shutil.move(str(tmp), str(abs_path))

    def load(self) -> None:
        with open(self.db_path) as f:
            records = json.load(f)
        self._entries = []
        for rec in records:
            self._entries.append(ProposalEntry(
                id=rec["id"],
                proposal=rec["proposal"],
                eval_results=rec["eval_results"],
                quality=rec["quality"],
                diversity=rec["diversity"],
                behavior_vector=rec["behavior_vector"],
                island=rec.get("island", 0),
                timestamp=rec.get("timestamp", 0.0),
            ))

    # ------------------------------------------------------------------
    # Factory helper
    # ------------------------------------------------------------------

    @staticmethod
    def make_entry(
        proposal: Dict[str, Any],
        eval_results: Dict[str, Any],
    ) -> ProposalEntry:
        """Create a fully-scored ProposalEntry from raw proposal + eval."""
        quality = S.QUALITY_FN(eval_results, proposal)
        bvec = S.compute_behavior_vector(eval_results, proposal)
        return ProposalEntry(
            id=uuid.uuid4().hex[:12],
            proposal=proposal,
            eval_results=_strip_images(eval_results),
            quality=quality,
            diversity=0.0,
            behavior_vector=bvec.tolist(),
        )

    def stats_summary(self) -> str:
        if not self._entries:
            return "Database: empty"
        qs = self.all_qualities()
        return (
            f"Database: {self.size} entries  "
            f"quality: best={qs.max():.3f}  median={float(np.median(qs)):.3f}  "
            f"worst={qs.min():.3f}"
        )
