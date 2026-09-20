"""Independent diagnostic-mode isolation checks on a tiny synthetic learner."""
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from ai.ga import build_individual_config
from ai.rl.device import load_cuda_ppo
from ai.rl import train as training
from env.backtest import EpisodeSession, run_day_config_episode
from utils.atomic_file import file_sha256
from rl_test_data import write_runtime






def test_changed_test_scores_cannot_change_learner_or_validation_selection(tmp_path, monkeypatch):
    """Real SB3 updates/persistence; deterministic synthetic evaluation scores.

    Only evaluation traces are replaced to independently control validation and
    diagnostic test ranks. No real market data or subprocess model replay runs.
    """
    runtime = tmp_path / "runtime.npz"
    write_runtime(runtime)
    monkeypatch.setattr(training, "read_financial_snapshot_manifest", lambda path: {
        "manifest_sha256": "a" * 64, "snapshot_sha256": file_sha256(Path(path)),
        "financial_identity": {"sha256": "c" * 64},
        "panel_builder_version": "synthetic-only", "financial_replay_version": "synthetic-only",
        "availability": "synthetic fixture", "pit_evidence_limit": "not a real archive certification",
    })
    fit_calls = []
    real_fit = training._fit_train_normalizer

    def fit(episode, **kwargs):
        fit_calls.append((episode.runtime.manifest.requested_start, episode.runtime.manifest.requested_end))
        return real_fit(episode, **kwargs)

    monkeypatch.setattr(training, "_fit_train_normalizer", fit)
    monkeypatch.setattr(training, "_summary_calmar", lambda _summary: 1e9)

    def local_trace(episode, schema):
        config = schema.from_serialized_day_config(build_individual_config(turnover_rate=0.1))
        return run_day_config_episode(EpisodeSession(episode), lambda _observation: config)

    def static_replay(episode, action_schema, _normalizer, **kwargs):
        assert isinstance(kwargs["provider"], training.FixedConfigProvider)
        if isinstance(episode, training.SharedPreparedEpisodeDescriptor):
            with episode.attach() as attached:
                return local_trace(attached.episode, action_schema)
        return local_trace(episode, action_schema)

    monkeypatch.setattr(training, "_backtest_provider", static_replay)
    replay_calls = []
    variant = 0

    def controlled_replay(snapshot, descriptor, *, task_id, action_schema, **_kwargs):
        snapshot.validate()
        split = "validation" if "validation" in task_id else "test" if "test" in task_id else "train"
        replay_calls.append((variant, split, snapshot.timesteps, snapshot.sha256))
        with descriptor.attach() as attached:
            trace = local_trace(attached.episode, action_schema)
        if split == "validation":
            calmar = {0: 99.0, 4: 0.8, 8: 0.5, 12: 0.6}[snapshot.timesteps]
        elif split == "test":
            calmar = ({0: 0.1, 4: 0.1, 8: 9.0, 12: 0.2} if variant == 0
                      else {0: 20.0, 4: 30.0, 8: 0.1, 12: 19.0})[snapshot.timesteps]
        else:
            calmar = 0.2
        returns = np.zeros_like(trace.portfolio_returns)
        returns[0] = -0.01
        target_nav = (1 + calmar * 0.01) ** (len(returns) / 252)
        returns[-1] = target_nav / np.prod(1 + returns[:-1]) - 1
        nav = float(trace.nav[0]) * np.r_[1., np.cumprod(1. + returns)]
        return replace(trace, portfolio_returns=returns, nav=nav)

    monkeypatch.setattr(training, "_evaluate_frozen_on_descriptor", controlled_replay)
    weights = []
    for variant in range(2):
        output = tmp_path / f"run{variant}"
        args = training.build_parser().parse_args([
            "--runtime", str(runtime), "--output", str(output),
            "--train-start", "2020-06-20", "--train-end", "2020-06-24",
            "--validation-start", "2020-06-25", "--validation-end", "2020-06-28",
            "--test-start", "2020-06-29", "--test-end", "2020-07-02",
            "--evaluation-execution", "blocking",
            "--seed", "2345", "--rollouts", "3", "--n-envs", "1", "--n-steps", "4", "--batch-size", "4",
            "--n-epochs", "1", "--eval-every-rollouts", "1", "--min-episode-transitions", "1",
            "--lookback", "4", "--rollout-backend", "dummy", "--backtest-workers", "1", "--device", "cuda",
        ])
        assert training.train(args) == output.resolve()
        curves = json.loads((output / "evaluation_curves.json").read_text("utf-8"))
        assert [row["timesteps"] for row in curves["train"]] == [0, 4, 8, 12]
        for split in ("validation", "test"):
            assert [row["timesteps"] for row in curves[split]] == [0, 4, 8, 12]
        assert all(not row["eligible_for_selection"] for row in curves["test"])
        assert curves["validation"][0]["eligible_for_selection"] is False
        for index in range(4):
            assert len({curves[split][index]["checkpoint_sha256"] for split in ("train", "validation", "test")}) == 1
        initial = json.loads((output / "initial_model.json").read_text("utf-8"))
        assert initial["sha256"] == curves["train"][0]["checkpoint_sha256"]
        selected = json.loads((output / "model.json").read_text("utf-8"))
        assert selected["timesteps"] == 4
        assert selected["role"] == "validation_calmar_selected_checkpoint"
        assert not any(key.startswith("test") for key in selected["metrics"])
        assert not (output / "manifest.json").exists()
        assert not (output / "pretest_qualification.json").exists()
        identity = json.loads((output / "run_identity.json").read_text("utf-8"))
        assert identity["identity_version"] == training.RUN_IDENTITY_VERSION
        assert identity["contract"]["algorithm"]["evaluation_protocol"] == "periodic_three_split_validation_selection"
        model = load_cuda_ppo(output / "latest_train_model.zip")
        assert model.num_timesteps == 12
        weights.append({key: value.clone() for key, value in model.policy.state_dict().items()})
    assert fit_calls == [("2020-06-20", "2020-06-24")] * 2
    assert len(replay_calls) == 24
    assert weights[0].keys() == weights[1].keys()
    assert all(torch.equal(weights[0][key], weights[1][key]) for key in weights[0])
