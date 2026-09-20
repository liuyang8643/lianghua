from datetime import date
import json
from pathlib import Path

import pytest

from ai.ga import get_mode_configs, get_profile_preload_range
from ai.ga.train import parse_args as ga_args, training_date_range
from ai.bundle import policy_source_sha256
from ai.rl.train import build_parser
from configs.training import DEFAULT_EVALUATION_SPLITS_PATH, read_evaluation_splits


def test_default_ga_and_ppo_share_periods_and_requested_budgets():
    ga = ga_args([])
    ppo = build_parser().parse_args(["--runtime", "runtime.npz"])
    splits, _ = read_evaluation_splits(ga.evaluation_splits)
    assert Path(ga.evaluation_splits) == DEFAULT_EVALUATION_SPLITS_PATH
    assert get_profile_preload_range() == tuple(map(date.fromisoformat, splits["train"]))
    for name, bounds in splits.items():
        assert [getattr(ppo, name + "_start"), getattr(ppo, name + "_end")] == bounds
    assert get_mode_configs()[ga.mode]["generations"] == 10000
    assert ppo.rollouts == 100000


def test_ga_explicit_protocol_and_debug_do_not_acquire_implicit_holdouts():
    assert ga_args(["--evaluation-splits", "custom.json"]).evaluation_splits == "custom.json"
    assert ga_args(["--mode", "debug"]).evaluation_splits is None
    assert ga_args(["--mode", "debug", "--evaluation-splits", "custom.json"]).evaluation_splits == "custom.json"


def test_explicit_training_budget_and_ppo_period_overrides_remain_available():
    ga = ga_args(["--generations", "7", "--population-size", "8"])
    assert (ga.generations, ga.population_size) == (7, 8)
    ppo = build_parser().parse_args(["--runtime", "runtime.npz", "--rollouts", "3",
                                   "--train-start", "2015-01-01", "--train-end", "2016-12-31"])
    assert (ppo.rollouts, ppo.train_start, ppo.train_end) == (3, "2015-01-01", "2016-12-31")


def test_explicit_shared_split_file_controls_ga_train_period(tmp_path):
    split_file = tmp_path / "splits.json"
    split_file.write_text(json.dumps({"train": ["2014-01-01", "2021-12-31"],
        "validation": ["2022-01-01", "2024-12-31"], "test": ["2025-01-01", "2026-08-28"]}), encoding="utf8")
    args = ga_args(["--evaluation-splits", str(split_file)])
    assert training_date_range(args) == (date(2014, 1, 1), date(2021, 12, 31))


def test_shared_period_reader_is_bound_by_existing_source_fingerprint(tmp_path):
    config = tmp_path / "configs"
    config.mkdir()
    reader = config / "training.py"
    reader.write_text("# original\n", encoding="utf8")
    original = policy_source_sha256(tmp_path)
    reader.write_text("# changed\n", encoding="utf8")
    assert policy_source_sha256(tmp_path) != original
