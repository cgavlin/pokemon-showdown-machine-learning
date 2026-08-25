#!/usr/bin/env python3
"""
Inspect a SelfPlayPool: lists every checkpoint currently in the pool,
its win rate against the learner, how many times it's been sampled as
an opponent, and whether it's pinned as a "hard" opponent.

Without this, the only way to see what's in a pool was to read
manifest.json by hand.

Usage:
    python scripts/inspect_pool.py --pool-dir runs/stage4_self_play_pool
    python scripts/inspect_pool.py --pool-dir runs/stage4_self_play_pool --sort win_rate
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.self_play import SelfPlayPool

_SORT_KEYS = {
    "win_rate": lambda e: e.win_rate_vs_current,
    "times_sampled": lambda e: -e.times_sampled,
    "name": lambda e: Path(e.checkpoint_path).name,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool-dir", type=str, required=True)
    parser.add_argument(
        "--sort",
        type=str,
        default="win_rate",
        choices=list(_SORT_KEYS),
        help="win_rate: hardest opponents (lowest learner win rate) first. "
        "times_sampled: most-sampled first. name: alphabetical.",
    )
    args = parser.parse_args()

    pool = SelfPlayPool(pool_dir=args.pool_dir)

    if not pool.entries:
        print(f"Pool at {pool.pool_dir} is empty.")
        return

    entries = sorted(pool.entries, key=_SORT_KEYS[args.sort])

    name_width = max(len(Path(e.checkpoint_path).name) for e in entries)
    header = f"{'checkpoint':<{name_width}}  {'win_rate':>8}  {'sampled':>7}  pinned"
    print(header)
    print("-" * len(header))
    for entry in entries:
        name = Path(entry.checkpoint_path).name
        pinned = "yes" if entry.is_pinned_hard else ""
        print(f"{name:<{name_width}}  {entry.win_rate_vs_current:>8.2f}  {entry.times_sampled:>7}  {pinned}")

    print()
    print(f"{len(entries)} checkpoint(s) in pool (max_pool_size={pool.max_pool_size}) at {pool.pool_dir}")
    pinned_count = sum(1 for e in entries if e.is_pinned_hard)
    if pinned_count:
        print(f"{pinned_count} pinned as hard opponents (win_rate_vs_current < 0.35 when last evaluated).")


if __name__ == "__main__":
    main()
