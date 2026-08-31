#!/usr/bin/env python3
"""
Live training monitor: tails a run's metrics.jsonl (written by
Trainer, SelfPlayTrainer, or PooledSelfPlayTrainer  and plots a rolling-average
episode reward curve plus epsilon decay, refreshing every few seconds
while training runs in another terminal.

This is read-only and offline: it never touches the Showdown server or
the training process, it just re-reads the same metrics.jsonl file
scripts/train.py (or self_play_train.py / pooled_self_play_train.py)
is appending to.

Handles both metrics formats in this project:
  - Trainer / PooledSelfPlayTrainer: "episode_reward" per episode_end line.
  - SelfPlayTrainer (mirror self-play): "player1_reward"/"player2_reward"
    per episode_end line instead -- averaged into one curve here.

Usage:
    python scripts/watch_training.py --run-dir runs/stage1_basic_mechanics_abcd1234
    python scripts/watch_training.py --run-dir runs/<run_id> --window 50 --refresh 3

Requires matplotlib (not needed for training itself -- only for this
monitor): pip install matplotlib
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
except ImportError:
    print(
        "This script needs matplotlib, which isn't in requirements.txt "
        "(training itself doesn't need it). Install it with:\n"
        "    pip install matplotlib",
        file=sys.stderr,
    )
    sys.exit(1)


def _rolling_mean(values: list[float], window: int) -> list[float]:
    out = []
    total = 0.0
    for i, v in enumerate(values):
        total += v
        if i >= window:
            total -= values[i - window]
        out.append(total / min(i + 1, window))
    return out


def _read_episode_rewards(metrics_path: Path) -> tuple[list[int], list[float], list[float]]:
    """Returns (steps, episode_rewards, epsilons) parsed from every
    'episode_end' line currently in the file. Missing/partial trailing
    lines (the writer may be mid-flush) are skipped, not fatal."""
    steps: list[int] = []
    rewards: list[float] = []
    epsilons: list[float] = []

    if not metrics_path.exists():
        return steps, rewards, epsilons

    with open(metrics_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue  # a line being written right now -- try again next refresh

            if record.get("event") != "episode_end":
                continue

            if "episode_reward" in record:
                reward = record["episode_reward"]
            elif "player1_reward" in record and "player2_reward" in record:
                # Mirror self-play: one shared network plays both sides
                # every episode, so average the two perspectives into a
                # single learning-progress curve.
                reward = (record["player1_reward"] + record["player2_reward"]) / 2.0
            else:
                continue

            steps.append(record.get("step", len(steps)))
            rewards.append(reward)
            epsilons.append(record.get("epsilon", float("nan")))

    return steps, rewards, epsilons


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir", type=str, required=True,
        help="A run directory containing metrics.jsonl (e.g. runs/stage1_basic_mechanics_abcd1234).",
    )
    parser.add_argument(
        "--window", type=int, default=20,
        help="Number of episodes to average over for the rolling reward curve (default: 20).",
    )
    parser.add_argument(
        "--refresh", type=float, default=5.0,
        help="Seconds between re-reading metrics.jsonl (default: 5).",
    )
    args = parser.parse_args()

    metrics_path = Path(args.run_dir) / "metrics.jsonl"
    print(f"Watching {metrics_path} (refreshing every {args.refresh:.0f}s)...")

    fig, (ax_reward, ax_epsilon) = plt.subplots(2, 1, sharex=True, figsize=(9, 6))
    fig.suptitle(f"Training progress: {Path(args.run_dir).name}")

    def _update(_frame):
        steps, rewards, epsilons = _read_episode_rewards(metrics_path)

        ax_reward.clear()
        ax_epsilon.clear()

        if not steps:
            ax_reward.set_title("Waiting for episode data...")
            return

        rolling = _rolling_mean(rewards, args.window)

        ax_reward.plot(steps, rewards, color="lightsteelblue", linewidth=0.8, label="episode reward")
        ax_reward.plot(steps, rolling, color="navy", linewidth=2.0, label=f"rolling avg ({args.window} ep)")
        ax_reward.set_ylabel("episode reward")
        ax_reward.legend(loc="upper left")
        ax_reward.set_title(f"{len(steps)} episodes so far -- latest rolling avg: {rolling[-1]:.2f}")

        ax_epsilon.plot(steps, epsilons, color="darkorange", linewidth=1.5)
        ax_epsilon.set_ylabel("epsilon")
        ax_epsilon.set_xlabel("training step")

        fig.tight_layout(rect=[0, 0, 1, 0.96])

    anim = FuncAnimation(fig, _update, interval=args.refresh * 1000, cache_frame_data=False)
    plt.show()


if __name__ == "__main__":
    main()