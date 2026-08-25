"""
Automatic curriculum advancement.

Wires curriculum.py's per-stage min_eval_win_rate_to_advance gates
into an actual runnable process, per CLAUDE.md's "Curriculum Learning"
section: train a stage to completion, evaluate the result against a
held-out opponent (per CLAUDE.md's held-out evaluation requirement --
always a DIFFERENT opponent type than the one trained against, so
passing the gate reflects generalization rather than memorizing one
scripted bot's quirks), and only proceed to the next stage if that
gate is met. Before this existed, comparing a trained checkpoint
against its stage's gate and deciding whether to continue was a fully
manual process: copy the yaml, change curriculum_stage, rerun
scripts/train.py, separately rerun scripts/evaluate.py, eyeball the
number against curriculum.py's source.

Stops automatically before stage5_human_opponents: live battles remain
behind the explicit, independent safety gate in showdown/integration.py
and are never triggered by this or any other automated process, no
matter how well earlier stages did.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Optional

from agents.inference import TrainedAgent
from environment.battle_env import make_env
from evaluation.benchmarks import evaluate_agent
from training.curriculum import CURRICULUM, OPPONENT_FACTORIES, CurriculumStage
from training.trainer_factory import build_trainer

logger = logging.getLogger(__name__)

LIVE_BATTLE_STAGE_NAME = "stage5_human_opponents"

# Stages where CLAUDE.md's curriculum explicitly calls for self-play.
# CurriculumRunner forces self_play.enabled to match this set for every
# stage it trains, REGARDLESS of what the base config's self_play block
# says -- the base config's self_play sub-fields (mode, pool_dir,
# episodes_per_opponent, bootstrap_opponent) still apply once a stage
# turns self-play on; only whether it's on is decided by the stage.
SELF_PLAY_STAGE_NAMES = {"stage4_competitive_play"}

# A stage's held-out eval opponent must differ from the opponent type
# it trained against, so passing the gate reflects generalization
# instead of memorizing one scripted bot's quirks (CLAUDE.md's
# held-out evaluation requirement). Fixed mapping to a different,
# reasonably-strong opponent type per training opponent.
_HELD_OUT_OPPONENT_FOR = {
    "random": "heuristic",
    "max_base_power": "heuristic",
    "heuristic": "max_base_power",
}


def held_out_opponent_kind(stage: CurriculumStage) -> str:
    return _HELD_OUT_OPPONENT_FOR.get(stage.opponent_factory, "heuristic")


class CurriculumRunner:
    def __init__(
        self,
        base_config: dict,
        run_dir: str | Path,
        start_stage: Optional[str] = None,
        eval_battles: int = 200,
        eval_device: str = "cpu",
        build_trainer_fn: Callable = build_trainer,
        evaluate_fn: Optional[Callable[[Path, CurriculumStage], dict]] = None,
    ):
        self.base_config = base_config
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.start_stage = start_stage
        self.eval_battles = eval_battles
        self.eval_device = eval_device
        # Both injectable so tests can exercise the actual advance/stop
        # gate logic (stage skipping, the self-play-only-at-stage4
        # override, stopping before live battles) without needing a
        # live server, a real checkpoint, or torch at all.
        self._build_trainer_fn = build_trainer_fn
        self._evaluate_fn = evaluate_fn or self._default_evaluate_checkpoint
        self.progress_log_path = self.run_dir / "curriculum_progress.jsonl"

    def _log_progress(self, **kwargs) -> None:
        kwargs["timestamp"] = time.time()
        with open(self.progress_log_path, "a") as f:
            f.write(json.dumps(kwargs) + "\n")

    def _default_evaluate_checkpoint(self, checkpoint_path: Path, stage: CurriculumStage) -> dict:
        eval_opponent_kind = held_out_opponent_kind(stage)
        eval_opponent = OPPONENT_FACTORIES[eval_opponent_kind](stage.battle_format, None)
        eval_env = make_env(opponent=eval_opponent, battle_format=stage.battle_format, local=True)
        try:
            agent = TrainedAgent(checkpoint_path, n_actions=eval_env.action_space.n, device=self.eval_device)
            metrics = evaluate_agent(eval_env, agent, n_battles=self.eval_battles)
        finally:
            eval_env.close()
        summary = metrics.summary()
        summary["eval_opponent"] = eval_opponent_kind
        return summary

    def _stage_config(self, stage: CurriculumStage) -> dict:
        stage_config = dict(self.base_config)
        stage_config["curriculum_stage"] = stage.name

        self_play_cfg = dict(self.base_config.get("self_play") or {})
        self_play_cfg["enabled"] = stage.name in SELF_PLAY_STAGE_NAMES
        stage_config["self_play"] = self_play_cfg

        return stage_config

    def run(self) -> None:
        started = self.start_stage is None

        for stage in CURRICULUM:
            if not started:
                if stage.name == self.start_stage:
                    started = True
                else:
                    continue

            if stage.name == LIVE_BATTLE_STAGE_NAME:
                logger.info(
                    "Reached %s -- automatic advancement stops here. Live "
                    "battles require the explicit, independently-gated path "
                    "in showdown/integration.py and are never triggered "
                    "automatically, regardless of earlier results.",
                    LIVE_BATTLE_STAGE_NAME,
                )
                self._log_progress(stage=stage.name, event="stopped_before_live_battles")
                return

            logger.info("Curriculum: starting stage %s", stage.name)
            stage_config = self._stage_config(stage)
            trainer = self._build_trainer_fn(stage_config, stage, self.run_dir)
            try:
                checkpoint_path = trainer.train()
            finally:
                trainer.env.close()

            logger.info("Curriculum: evaluating stage %s checkpoint %s", stage.name, checkpoint_path)
            eval_summary = self._evaluate_fn(checkpoint_path, stage)
            passed = eval_summary["win_rate"] >= stage.min_eval_win_rate_to_advance

            self._log_progress(
                stage=stage.name,
                event="stage_evaluated",
                checkpoint=str(checkpoint_path),
                eval_summary=eval_summary,
                gate=stage.min_eval_win_rate_to_advance,
                passed=passed,
            )

            logger.info(
                "Curriculum: stage %s win_rate=%.2f vs gate=%.2f (opponent=%s) -> %s",
                stage.name,
                eval_summary["win_rate"],
                stage.min_eval_win_rate_to_advance,
                eval_summary.get("eval_opponent", "?"),
                "ADVANCING" if passed else "STOPPING (gate not met)",
            )

            if not passed:
                return

        logger.info("Curriculum: all automatable stages completed.")
