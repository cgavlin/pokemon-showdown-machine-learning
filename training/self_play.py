"""
Self-play opponent pool:
  - save successful versions of the agent
  - add previous versions to an opponent pool
  - sample opponents from that pool during training
  - periodically evaluate the current agent against older versions
  - keep particularly challenging opponents available
"""

from __future__ import annotations

import json
import random
import shutil
import uuid
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PoolEntry:
    checkpoint_path: str
    win_rate_vs_current: float = 0.5  # updated as it's evaluated
    times_sampled: int = 0
    is_pinned_hard: bool = False  # "particularly challenging" opponents kept around


class SelfPlayPool:
    """
    A lightweight on-disk registry of past checkpoints usable as
    opponents. Sampling favors a mix of recency and difficulty so the
    agent doesn't just learn to beat one static past version.

    A single pool_dir can safely be shared across multiple separate
    training runs/processes (e.g. resuming training later, or several
    trainers contributing to the same pool): add_checkpoint() always
    gives each copy a unique filename, so two runs that happen to
    produce identically-named source checkpoints (e.g. both call theirs
    "checkpoint_step40000.pt") never collide or silently overwrite one
    another inside the pool.
    """

    def __init__(self, pool_dir: str | Path, max_pool_size: int = 20):
        self.pool_dir = Path(pool_dir)
        self.pool_dir.mkdir(parents=True, exist_ok=True)
        self.max_pool_size = max_pool_size
        self._manifest_path = self.pool_dir / "manifest.json"
        self.entries: list[PoolEntry] = self._load_manifest()

    def _load_manifest(self) -> list[PoolEntry]:
        if not self._manifest_path.exists():
            return []
        data = json.loads(self._manifest_path.read_text())
        return [PoolEntry(**e) for e in data]

    def _save_manifest(self) -> None:
        self._manifest_path.write_text(
            json.dumps([e.__dict__ for e in self.entries], indent=2)
        )

    def add_checkpoint(self, checkpoint_path: str | Path, hard: bool = False) -> None:
        checkpoint_path = Path(checkpoint_path)
        # Always give the pool's own copy a unique filename (source name
        # + a short random suffix). Two different training runs can
        # easily produce checkpoints with the identical name (e.g. both
        # named "checkpoint_step40000.pt" because each run's own step
        # counter restarts at 0); without this, the second run's
        # add_checkpoint would silently overwrite the first run's file
        # on disk while a stale PoolEntry still pointed at that same
        # path from the manifest.
        dest = self.pool_dir / f"{checkpoint_path.stem}_{uuid.uuid4().hex[:8]}{checkpoint_path.suffix}"
        shutil.copy2(checkpoint_path, dest)

        self.entries.append(PoolEntry(checkpoint_path=str(dest), is_pinned_hard=hard))

        # Evict weakest, non-pinned, least-challenging entries once over capacity.
        if len(self.entries) > self.max_pool_size:
            evictable = [e for e in self.entries if not e.is_pinned_hard]
            evictable.sort(key=lambda e: e.win_rate_vs_current, reverse=True)
            while len(self.entries) > self.max_pool_size and evictable:
                victim = evictable.pop()
                self.entries.remove(victim)
                Path(victim.checkpoint_path).unlink(missing_ok=True)

        self._save_manifest()

    def sample(self, rng: random.Random | None = None) -> PoolEntry | None:
        if not self.entries:
            return None
        rng = rng or random.Random()

        # Weight sampling toward opponents that have historically been
        # challenging (lower recorded win_rate_vs_current means they beat
        # the "current" agent more often at the time they were measured),
        # so the pool doesn't collapse to only-easy past selves.
        weights = [max(0.05, 1.0 - e.win_rate_vs_current) for e in self.entries]
        entry = rng.choices(self.entries, weights=weights, k=1)[0]
        entry.times_sampled += 1
        self._save_manifest()
        return entry

    def update_win_rate(self, checkpoint_path: str, win_rate_vs_current: float) -> None:
        for e in self.entries:
            if e.checkpoint_path == checkpoint_path:
                e.win_rate_vs_current = win_rate_vs_current
                # Auto-pin opponents that are proving unusually tough.
                if win_rate_vs_current < 0.35:
                    e.is_pinned_hard = True
                break
        self._save_manifest()
