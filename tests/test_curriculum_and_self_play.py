import tempfile
from pathlib import Path

from training.curriculum import CURRICULUM, get_stage
from training.self_play import SelfPlayPool


def test_curriculum_has_five_stages():
    names = [s.name for s in CURRICULUM]
    assert names == [
        "stage1_basic_mechanics",
        "stage2_tactical_decisions",
        "stage3_strategic_play",
        "stage4_competitive_play",
        "stage5_human_opponents",
    ]


def test_get_stage_roundtrip():
    stage = get_stage("stage1_basic_mechanics")
    assert stage.opponent_factory == "random"


def test_get_stage_unknown_raises():
    try:
        get_stage("not_a_real_stage")
        assert False, "expected KeyError"
    except KeyError:
        pass


def test_self_play_pool_add_and_sample(tmp_path):
    pool = SelfPlayPool(pool_dir=tmp_path / "pool", max_pool_size=5)
    checkpoint = tmp_path / "checkpoint_1.pt"
    checkpoint.write_bytes(b"fake checkpoint contents")

    pool.add_checkpoint(checkpoint)
    assert len(pool.entries) == 1

    sampled = pool.sample()
    assert sampled is not None
    assert sampled.times_sampled == 1


def test_self_play_pool_pins_hard_opponents(tmp_path):
    pool = SelfPlayPool(pool_dir=tmp_path / "pool", max_pool_size=5)
    checkpoint = tmp_path / "checkpoint_1.pt"
    checkpoint.write_bytes(b"fake")
    pool.add_checkpoint(checkpoint)

    entry = pool.entries[0]
    pool.update_win_rate(entry.checkpoint_path, win_rate_vs_current=0.2)
    assert pool.entries[0].is_pinned_hard is True


def test_self_play_pool_evicts_when_over_capacity(tmp_path):
    pool = SelfPlayPool(pool_dir=tmp_path / "pool", max_pool_size=2)
    for i in range(4):
        cp = tmp_path / f"checkpoint_{i}.pt"
        cp.write_bytes(b"fake")
        pool.add_checkpoint(cp)
        # mark each as "easy" (high win rate vs current) so it's evictable
        pool.update_win_rate(pool.entries[-1].checkpoint_path, win_rate_vs_current=0.9)

    assert len(pool.entries) <= 2


def test_self_play_pool_add_checkpoint_never_collides_on_same_source_name(tmp_path):
    """Two different runs producing identically-named source checkpoints
    (e.g. both "checkpoint_step40000.pt" because each run's own step
    counter restarts) must not silently overwrite each other once
    copied into a shared pool."""
    pool = SelfPlayPool(pool_dir=tmp_path / "pool", max_pool_size=10)

    run1_dir = tmp_path / "run1"
    run2_dir = tmp_path / "run2"
    run1_dir.mkdir()
    run2_dir.mkdir()
    cp1 = run1_dir / "checkpoint_step40000.pt"
    cp2 = run2_dir / "checkpoint_step40000.pt"
    cp1.write_bytes(b"run1 weights")
    cp2.write_bytes(b"run2 weights")

    pool.add_checkpoint(cp1)
    pool.add_checkpoint(cp2)

    assert len(pool.entries) == 2
    paths = [Path(e.checkpoint_path) for e in pool.entries]
    assert paths[0] != paths[1]
    assert paths[0].read_bytes() == b"run1 weights"
    assert paths[1].read_bytes() == b"run2 weights"
