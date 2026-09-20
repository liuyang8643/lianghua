"""PPO has one CUDA execution contract and no CPU fallback."""
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch as th

from ai.rl import device as devices
from ai.rl import train as training
from ai.rl.evaluation import _FrozenPPOCallable
from ai.rl.policy import RLPolicy, predict_encoded_action
from ai.rl.typed_policy import TypedActorCriticPolicy
from env.action_schema import ActionSchema


@pytest.mark.parametrize("device", ["cpu", "auto", "mps"])
def test_non_cuda_requests_fail_without_loading_any_checkpoint(monkeypatch, device):
    load = Mock()
    monkeypatch.setattr(devices.PPO, "load", load)
    with pytest.raises(ValueError, match="CUDA"):
        devices.load_cuda_ppo("missing.zip", device=device)
    with pytest.raises(ValueError, match="CUDA"):
        TypedActorCriticPolicy.load("missing.pth", device=device)
    load.assert_not_called()


def test_missing_cuda_fails_before_loading_checkpoint_or_runtime(monkeypatch):
    monkeypatch.setattr(th.cuda, "is_available", lambda: False)
    load = Mock()
    prepare = Mock()
    monkeypatch.setattr(devices.PPO, "load", load)
    monkeypatch.setattr(training, "_prepare_split", prepare)
    with pytest.raises(RuntimeError, match="available CUDA GPU"):
        devices.load_cuda_ppo("missing.zip")
    with pytest.raises(RuntimeError, match="available CUDA GPU"):
        TypedActorCriticPolicy(None, None, lambda _: 1e-3, encoded_schema={})
    args = training.build_parser().parse_args(["--runtime", "must-not-load.npz"])
    with pytest.raises(RuntimeError, match="available CUDA GPU"):
        training.train(args)
    load.assert_not_called()
    prepare.assert_not_called()


@pytest.mark.parametrize("device", ["cpu", "auto"])
def test_training_cli_rejects_removed_devices(device):
    with pytest.raises(SystemExit):
        training.build_parser().parse_args(["--runtime", "missing.npz", "--device", device])


def test_cuda_loader_and_model_guard_have_one_explicit_device(monkeypatch):
    monkeypatch.setattr(th.cuda, "is_available", lambda: True)
    monkeypatch.setattr(th.cuda, "device_count", lambda: 1)
    model = SimpleNamespace(device=th.device("cuda:0"), policy=SimpleNamespace(device=th.device("cuda:0")))
    load = Mock(return_value=model)
    monkeypatch.setattr(devices.PPO, "load", load)
    assert devices.load_cuda_ppo("model.zip") is model
    assert load.call_args.kwargs["device"].type == "cuda"
    with pytest.raises(ValueError, match="does not exist"):
        devices.require_cuda_device("cuda:1")
    model.policy.device = th.device("cpu")
    with pytest.raises(ValueError, match="CUDA"):
        devices.require_cuda_model(model)


def test_cpu_models_are_rejected_by_live_frozen_and_prediction_adapters():
    cpu = SimpleNamespace(device=th.device("cpu"), policy=SimpleNamespace(device=th.device("cpu")))
    schema = ActionSchema()
    with pytest.raises(ValueError, match="CUDA"):
        predict_encoded_action(cpu, np.zeros(1, dtype=np.float32))
    with pytest.raises(ValueError, match="CUDA"):
        _FrozenPPOCallable(cpu, schema)
    with pytest.raises(ValueError, match="CUDA"):
        RLPolicy(cpu, schema, None, None, prefilter_n=300)
