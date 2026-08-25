"""
Tests for training/curriculum_runner.py's CurriculumRunner.

Exercises the actual advance/stop decision logic -- the whole point of
wiring curriculum.py's min_eval_win_rate_to_advance gates into
something runnable -- entirely offline, via the injectable
build_trainer_fn/evaluate_fn hooks: no real training, no real
evaluation, no live server, no torch checkpoints. A fake trainer
factory just returns a stub with a .train()/.env.close(), and a fake
evaluate_fn returns pre-scripted win rates per stage so tests can
assert on exactly what CurriculumRunner does with a "pass" vs a "fail".
"""

from __future__ import annotations

import json

from training.curriculum import CURRICULUM
from training.curriculum_runner import (
    LIVE_BATTLE_STAGE_NAME,
    SELF_PLAY_STAGE_NAMES,
    CurriculumRunner,
    held_out_opponent_kind,
)


class _FakeTrainer:
    def __init__(self, stage_config, stage, run_dir):
        self.stage_config = stage_config
        self.stage = stage
        self.run_dir = run_dir
        self.env = self

    def train(self):
        return self.run_dir / f"{self.stage.name}_checkpoint.pt"

    def close(self):
        pass


def _make_runner(tmp_path, win_rates: dict, start_stage=None, base_config=None):
    """win_rates maps stage name -> win rate the fake evaluator returns
    for that stage; stages not present default to 1.0 (always passes)."""
    built_stage_configs = []

    def fake_build_trainer(stage_config, stage, run_dir):
        built_stage_configs.append(stage_config)
        return _FakeTrainer(stage_config, stage, run_dir)

    def fake_evaluate(checkpoint_path, stage):
        return {"win_rate": win_rates.get(stage.name, 1.0), "eval_opponent": held_out_opponent_kind(stage)}

    runner = CurriculumRunner(
        base_config=base_config or {"self_play": {"enabled": False}},
        run_dir=tmp_path,
        start_stage=start_stage,
        build_trainer_fn=fake_build_trainer,
        evaluate_fn=fake_evaluate,
    )
    return runner, built_stage_configs


def test_all_stages_pass_advances_through_and_stops_before_live_battles(tmp_path):
    runner, built = _make_runner(tmp_path, win_rates={})  # everything passes
    runner.run()

    lines = [json.loads(l) for l in runner.progress_log_path.read_text().strip().split("\n")]
    evaluated_stages = [l["stage"] for l in lines if l.get("event") == "stage_evaluated"]

    # Every automatable stage (everything except stage5) gets trained
    # and evaluated, and the run stops right before stage5 rather than
    # attempting it.
    automatable = [s.name for s in CURRICULUM if s.name != LIVE_BATTLE_STAGE_NAME]
    assert evaluated_stages == automatable
    assert lines[-1]["event"] == "stopped_before_live_battles"
    assert lines[-1]["stage"] == LIVE_BATTLE_STAGE_NAME

    assert all(l["passed"] for l in lines if l.get("event") == "stage_evaluated")
    assert len(built) == len(automatable)


def test_failing_a_gate_stops_the_run_immediately(tmp_path):
    # stage2 fails its gate -> stage3/stage4 must never be attempted.
    runner, built = _make_runner(tmp_path, win_rates={"stage2_tactical_decisions": 0.0})
    runner.run()

    lines = [json.loads(l) for l in runner.progress_log_path.read_text().strip().split("\n")]
    evaluated_stages = [l["stage"] for l in lines if l.get("event") == "stage_evaluated"]

    assert evaluated_stages == ["stage1_basic_mechanics", "stage2_tactical_decisions"]
    assert lines[-1]["stage"] == "stage2_tactical_decisions"
    assert lines[-1]["passed"] is False
    assert len(built) == 2  # never got to stage3/stage4


def test_start_stage_skips_earlier_stages(tmp_path):
    runner, built = _make_runner(tmp_path, win_rates={}, start_stage="stage3_strategic_play")
    runner.run()

    lines = [json.loads(l) for l in runner.progress_log_path.read_text().strip().split("\n")]
    evaluated_stages = [l["stage"] for l in lines if l.get("event") == "stage_evaluated"]

    assert evaluated_stages == ["stage3_strategic_play", "stage4_competitive_play"]


def test_self_play_is_forced_only_for_stage4_regardless_of_base_config(tmp_path):
    # Base config says self_play.enabled: true globally -- CurriculumRunner
    # must still only turn it on for stage4, not stages 1-3.
    base_config = {"self_play": {"enabled": True, "mode": "pooled"}}
    runner, built = _make_runner(tmp_path, win_rates={}, base_config=base_config)
    runner.run()

    by_stage = {cfg["curriculum_stage"]: cfg for cfg in built}
    for name in SELF_PLAY_STAGE_NAMES:
        assert by_stage[name]["self_play"]["enabled"] is True
    for name in by_stage:
        if name not in SELF_PLAY_STAGE_NAMES:
            assert by_stage[name]["self_play"]["enabled"] is False

    # mode/other self-play sub-fields from the base config still carry
    # through once a stage does turn self-play on.
    assert by_stage["stage4_competitive_play"]["self_play"]["mode"] == "pooled"


def test_self_play_forced_off_even_when_base_config_omits_the_block(tmp_path):
    runner, built = _make_runner(tmp_path, win_rates={}, base_config={})
    runner.run()

    by_stage = {cfg["curriculum_stage"]: cfg for cfg in built}
    assert by_stage["stage1_basic_mechanics"]["self_play"]["enabled"] is False
    assert by_stage["stage4_competitive_play"]["self_play"]["enabled"] is True


def test_held_out_opponent_always_differs_from_training_opponent():
    for stage in CURRICULUM:
        if stage.name == LIVE_BATTLE_STAGE_NAME:
            continue
        assert held_out_opponent_kind(stage) != stage.opponent_factory
