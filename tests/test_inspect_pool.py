"""
Tests for scripts/inspect_pool.py.

Runs the script as a real subprocess (matching how a user would
actually invoke it) against a small SelfPlayPool built directly via
training/self_play.py, rather than importing script internals -- the
script itself has no reusable-module surface, so exercising the actual
CLI entry point is the most faithful test.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from training.self_play import SelfPlayPool

SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "inspect_pool.py"


def _run(pool_dir, *extra_args) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT_PATH), "--pool-dir", str(pool_dir), *extra_args],
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_empty_pool_reports_empty(tmp_path):
    result = _run(tmp_path / "empty_pool")
    assert result.returncode == 0
    assert "is empty" in result.stdout


def test_lists_entries_sorted_by_win_rate_hardest_first(tmp_path):
    pool_dir = tmp_path / "pool"
    pool = SelfPlayPool(pool_dir=pool_dir, max_pool_size=10)
    win_rates = [0.9, 0.2, 0.5]
    for i, wr in enumerate(win_rates):
        cp = tmp_path / f"checkpoint_step{i}.pt"
        cp.write_bytes(b"fake")
        pool.add_checkpoint(cp)
        pool.update_win_rate(pool.entries[-1].checkpoint_path, win_rate_vs_current=wr)

    result = _run(pool_dir, "--sort", "win_rate")
    assert result.returncode == 0

    # First data row (after header + separator) must be the hardest
    # opponent (lowest win_rate_vs_current = 0.2), matching the
    # "hardest first" contract documented in the script's --help.
    # Take exactly the 3 known entry rows rather than filtering blank
    # lines (which would blend the summary section into "last row").
    lines = result.stdout.splitlines()
    data_lines = lines[2:5]  # skip header + separator, take 3 entry rows
    assert "0.20" in data_lines[0]
    assert "0.90" in data_lines[-1]
    assert "3 checkpoint(s) in pool" in result.stdout


def test_pinned_entries_are_flagged(tmp_path):
    pool_dir = tmp_path / "pool"
    pool = SelfPlayPool(pool_dir=pool_dir, max_pool_size=10)
    cp = tmp_path / "checkpoint_step0.pt"
    cp.write_bytes(b"fake")
    pool.add_checkpoint(cp)
    # Below the 0.35 auto-pin threshold in SelfPlayPool.update_win_rate.
    pool.update_win_rate(pool.entries[-1].checkpoint_path, win_rate_vs_current=0.1)

    result = _run(pool_dir)
    assert result.returncode == 0
    assert "pinned as hard opponents" in result.stdout


def test_sort_by_times_sampled(tmp_path):
    pool_dir = tmp_path / "pool"
    pool = SelfPlayPool(pool_dir=pool_dir, max_pool_size=10)
    for i in range(2):
        cp = tmp_path / f"checkpoint_step{i}.pt"
        cp.write_bytes(b"fake")
        pool.add_checkpoint(cp)

    # Sample the first entry (index 0) several times more than the second.
    for _ in range(5):
        pool.entries[0].times_sampled += 1
    pool._save_manifest()

    result = _run(pool_dir, "--sort", "times_sampled")
    assert result.returncode == 0
    lines = result.stdout.splitlines()
    data_lines = lines[2:4]  # skip header + separator, take 2 entry rows
    assert Path(pool.entries[0].checkpoint_path).name in data_lines[0]
