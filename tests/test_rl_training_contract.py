from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import ai.rl.train as train_module
from ai.bundle import DEPLOYMENT_GATE_NAMES
from ai.rl.train import (
    MATERIAL_SCORE_IMPROVEMENT,
    PLATEAU_EVALUATIONS,
    TrainEvaluationCallback,
    _TrainCheckpointState,
    _dynamicity,
    _initialize_policy_from_static,
    _record_post_update_evaluation,
    _select_model_after_training,
    _technical_convergence,
    _validate_split_boundaries,
)
from env.action_schema import ActionSchema


_DYNAMIC = {
    "continuous_parameter_dynamic": True,
    "binary_parameter_dynamic": True,
    "discrete_parameter_dynamic": True,
    "enum_parameter_dynamic": True,
}


class _SavingModel:
    def __init__(self) -> None:
        self.saved: list[Path] = []

    def save(self, path) -> None:
        target = Path(path)
        if target.suffix != ".zip":
            target = target.with_suffix(".zip")
        target.write_bytes(b"model")
        self.saved.append(target)


def _callback(tmp_path: Path, *, initial_score: float = 1.0):
    callback = TrainEvaluationCallback(
        episode=SimpleNamespace(),
        action_schema=SimpleNamespace(),
        normalizer=SimpleNamespace(),
        folds=(),
        output_dir=tmp_path,
        eval_freq=10,
        initial_score=initial_score,
        initial_cash=1_000.0,
        seed=7,
    )
    callback.model = _SavingModel()
    return callback


def _record(
    callback: TrainEvaluationCallback,
    *,
    score: float,
    timesteps: int,
    phase: str = "scheduled",
    ppo_updates: int = 1,
):
    return callback._record_evaluation(
        timesteps=timesteps,
        phase=phase,
        ppo_updates=ppo_updates,
        score=score,
        average_exposure=0.75,
        dynamicity=_DYNAMIC,
        optimizer={},
        evaluation={"robust": {"robust_calmar": score}},
    )


def _args(**changes):
    values = {
        "train_start": "2010-01-01",
        "train_end": "2018-12-31",
        "validation_start": "2019-01-01",
        "validation_end": "2022-12-31",
        "test_start": "2023-01-01",
        "test_end": "2026-08-27",
        "skip_holdout": False,
    }
    values.update(changes)
    return Namespace(**values)


def test_split_boundaries_require_ordered_disjoint_train_validation_test():
    _validate_split_boundaries(_args())

    with pytest.raises(ValueError, match="strictly ordered and disjoint"):
        _validate_split_boundaries(_args(validation_start="2018-12-31"))
    with pytest.raises(ValueError, match="strictly ordered and disjoint"):
        _validate_split_boundaries(_args(test_start="2022-12-31"))


def test_skip_holdout_still_validates_training_interval():
    _validate_split_boundaries(_args(skip_holdout=True))
    with pytest.raises(ValueError, match="training split"):
        _validate_split_boundaries(
            _args(skip_holdout=True, train_start="2019-01-01")
        )


def test_ppo_actor_starts_at_static_config_but_binary_means_are_explorable():
    schema = ActionSchema()
    fixed = schema.encode_static_config(
        {
            "weights": dict(zip(schema.factor_names, (0.4, 0.9, 0.1, 0.6))),
            "filter_factors": {name: True for name in schema.filter_names},
            "buy_n": 20,
            "sell_m": 25,
            "cash_reserve_ratio": 0.25,
            "rebalance": True,
            "holding_period": 1,
            "limit_up_protection": True,
        }
    )
    action_net = torch.nn.Linear(3, schema.action_dim)
    model = SimpleNamespace(policy=SimpleNamespace(action_net=action_net))

    initialized = _initialize_policy_from_static(model, schema, fixed)

    assert schema.decode(initialized) == schema.decode(fixed)
    assert np.allclose(action_net.weight.detach().numpy(), 0.0)
    assert np.allclose(action_net.bias.detach().numpy(), initialized)
    for field in schema.layout:
        if field.kind == "binary":
            expected = 0.02 if field.name == "limit_up_protection" else 0.10
            assert abs(float(initialized[field.index])) == pytest.approx(expected)


def test_dynamicity_requires_material_range_and_two_covered_categories():
    categorical = {
        "factor_enabled.a": {
            "unique_count": 2,
            "coverage": {"False": 0.02, "True": 0.98},
        },
        "rebalance_now": {"unique_count": 1, "coverage": {"True": 1.0}},
        "limit_up_protection": {"unique_count": 1, "coverage": {"True": 1.0}},
        "buy_n": {"unique_count": 2, "coverage": {"20": 0.8, "25": 0.2}},
        "sell_m": {"unique_count": 1, "coverage": {"25": 1.0}},
        "rebalance": {"unique_count": 1, "coverage": {"True": 1.0}},
    }
    result = _dynamicity(
        {
            "continuous": {
                "target_exposure": {
                    "minimum": 0.70,
                    "maximum": 0.72,
                    "std": 0.003,
                }
            },
            "categorical": categorical,
        }
    )

    assert result == {
        "continuous_parameter_dynamic": True,
        "binary_parameter_dynamic": True,
        "discrete_parameter_dynamic": True,
        "enum_parameter_dynamic": False,
    }


def test_post_update_evaluation_is_kept_when_scheduled_timestep_matches(tmp_path):
    callback = _callback(tmp_path)
    scheduled = _record(
        callback,
        score=1.01,
        timesteps=256,
        phase="scheduled",
    )
    post_update = _record(
        callback,
        score=1.03,
        timesteps=256,
        phase="post_update",
    )

    assert scheduled["timesteps"] == post_update["timesteps"] == 256
    assert [row["phase"] for row in callback.history] == [
        "scheduled",
        "post_update",
    ]
    assert [row["evaluation_index"] for row in callback.history] == [1, 2]
    assert callback.best_trained_step == 256
    assert callback.best_trained_phase == "post_update"

    calls: list[str] = []

    class Probe:
        def evaluate_current_model(self, *, phase):
            calls.append(phase)
            return {"phase": phase}

    assert _record_post_update_evaluation(Probe()) == {"phase": "post_update"}
    assert calls == ["post_update"]


def test_checkpoint_zero_cannot_become_a_best_trained_checkpoint(tmp_path):
    callback = _callback(tmp_path)
    row = _record(
        callback,
        score=2.0,
        timesteps=128,
        ppo_updates=0,
    )

    assert row["eligible_checkpoint"] is False
    assert row["improved"] is False
    assert callback.best_trained_step is None
    assert not callback.best_path.with_suffix(".zip").exists()


def test_no_eligible_checkpoint_saves_final_diagnostic_and_selects_checkpoint_zero(
    tmp_path,
    monkeypatch,
):
    callback = _callback(tmp_path)
    checkpoint_zero = tmp_path / "checkpoint_0.zip"
    checkpoint_zero.write_bytes(b"checkpoint-zero")
    final_model = _SavingModel()
    loaded: list[Path] = []
    selected_model = object()

    def fake_load(path, *, env, device):
        assert env == "train-env"
        assert device == "cpu"
        loaded.append(Path(path))
        return selected_model

    monkeypatch.setattr(train_module.PPO, "load", staticmethod(fake_load))
    selected, selection = _select_model_after_training(
        final_model,
        callback,
        "train-env",
        tmp_path,
    )

    assert selected is selected_model
    assert loaded == [checkpoint_zero]
    assert selection == {
        "source": "checkpoint_0",
        "selected_step": 0,
        "selected_phase": "initial",
        "best_trained_step": None,
        "best_trained_phase": None,
        "trained_final_model": "trained_final_model.zip",
        "selected_file": "checkpoint_0.zip",
    }
    assert (tmp_path / "trained_final_model.zip").exists()
    assert not (tmp_path / "best_train_model.zip").exists()

    gates = {name: True for name in DEPLOYMENT_GATE_NAMES}
    gates["trained_checkpoint_selected"] = False
    assert _technical_convergence(gates) is False
    gates["trained_checkpoint_selected"] = True
    assert _technical_convergence(gates) is True
    gates[DEPLOYMENT_GATE_NAMES[0]] = False
    assert _technical_convergence(gates) is False


def test_real_eligible_checkpoint_is_selected_without_final_diagnostic(
    tmp_path,
    monkeypatch,
):
    callback = _callback(tmp_path)
    _record(callback, score=1.1, timesteps=512, phase="post_update")
    final_model = _SavingModel()
    loaded: list[Path] = []
    selected_model = object()

    def fake_load(path, *, env, device):
        loaded.append(Path(path))
        return selected_model

    monkeypatch.setattr(train_module.PPO, "load", staticmethod(fake_load))
    selected, selection = _select_model_after_training(
        final_model,
        callback,
        "train-env",
        tmp_path,
    )

    assert selected is selected_model
    assert loaded == [tmp_path / "best_train_model.zip"]
    assert selection["source"] == "best_trained_model"
    assert selection["selected_step"] == 512
    assert selection["selected_phase"] == "post_update"
    assert selection["trained_final_model"] is None
    assert not (tmp_path / "trained_final_model.zip").exists()


def test_plateau_uses_cumulative_material_anchor_not_drifting_best_score():
    state = _TrainCheckpointState(initial_score=1.0)
    material_flags = []
    for evaluation_index, score in enumerate(
        (1.005, 1.010, 1.015, 1.020001),
        start=1,
    ):
        improved, material_improved = state.observe(
            score=score,
            eligible=True,
            timesteps=evaluation_index * 100,
            phase="scheduled",
            evaluation_index=evaluation_index,
        )
        assert improved is True
        material_flags.append(material_improved)

    assert MATERIAL_SCORE_IMPROVEMENT == 0.02
    assert material_flags == [False, False, False, True]
    assert state.best_score == pytest.approx(1.020001)
    assert state.material_best_score == pytest.approx(1.020001)
    assert state.last_material_improvement_eval == 4
    assert state.plateau_reached(4 + PLATEAU_EVALUATIONS - 1) is False
    assert state.plateau_reached(4 + PLATEAU_EVALUATIONS) is True
