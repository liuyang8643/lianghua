"""Deterministic decisions retain real endpoints; samples remain PPO-safe."""
import numpy as np
import torch
import pytest

from stable_baselines3.common.distributions import DiagGaussianDistribution
from env.action_schema import ActionSchema


def test_endpoint_mode_can_disable_factor_and_check_at_most_ten_holdings():
    schema = ActionSchema()
    modes = torch.full((2, schema.action_dim), 0.5)
    modes[:, 0] = 0.0
    modes[:, 1] = 1.0
    modes[:, -1] = torch.tensor([0.0, 1.0])
    law = DiagGaussianDistribution(schema.action_dim).proba_distribution(2 * modes - 1, torch.zeros(schema.action_dim))
    actions = law.mode()
    assert torch.isfinite(law.log_prob(actions)).all()
    np.testing.assert_array_equal(actions[:, -1].numpy(), [-1.0, 1.0])
    for row, replacements in zip(actions.numpy(), (0, 10)):
        config = schema.decode(row)
        assert config.replacement_limit == replacements
        assert config.factor_weights[schema.factor_names[0]] == 0.0
        assert config.factor_weights[schema.factor_names[1]] == 1.0
    samples = law.sample()
    assert torch.isfinite(law.log_prob(samples)).all()
    assert torch.all((samples.clamp(-1, 1) >= -1) & (samples.clamp(-1, 1) <= 1))


@pytest.mark.parametrize("upper", [0.2, 1.0, 1])
def test_turnover_range_is_explicit_and_hash_sealed(upper):
    schema = ActionSchema(turnover_maximum=upper)
    assert ActionSchema.from_dict(schema.to_dict()) == schema
    config = schema.decode(np.ones(schema.action_dim))
    assert config.turnover_rate == upper
    assert config.replacement_limit == int(50 * upper)
    payload = schema.to_dict()
    del payload["turnover_maximum"]
    with pytest.raises(KeyError):
        ActionSchema.from_dict(payload)
    payload = schema.to_dict()
    payload["turnover_maximum"] = True
    with pytest.raises(ValueError):
        ActionSchema.from_dict(payload)


@pytest.mark.parametrize("upper", [0., -0.1, 1.01, float("nan"), float("inf"), True])
def test_invalid_turnover_upper_bound_is_rejected(upper):
    with pytest.raises(ValueError):
        ActionSchema(turnover_maximum=upper)
