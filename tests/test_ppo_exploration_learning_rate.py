import math

import pytest

from ai.rl.train import build_parser, exploration_scaled_learning_rate


def test_narrow_gaussian_cannot_increase_rate_to_scale_ratio():
    reference = math.exp(-1.6)
    for scale in (reference, .05, .0085, .006, 1.0):
        rate = exploration_scaled_learning_rate(.003, math.log(scale), -1.6)
        assert rate <= .003
        assert rate / scale <= .003 / reference * (1 + 1e-12)
    assert exploration_scaled_learning_rate(.003, -1.6, -1.6) == .003


def test_restored_policy_scale_has_identical_rate_without_local_clock():
    # A canonical checkpoint restores log_std. No new segment counter or running
    # history may change the scale-based multiplier on continuation.
    logs = [-1.6, -2.0, -3.0, -5.0]
    whole = [exploration_scaled_learning_rate(.003, value, -1.6) for value in logs]
    restored = [exploration_scaled_learning_rate(.003, value, -1.6) for value in logs[2:]]
    assert restored == whole[2:]


@pytest.mark.parametrize('rate,minimum,reference', [
    (0, -1.6, -1.6), (.003, float('nan'), -1.6),
    (.003, -1.6, float('inf')), (.003, -10000, -1.6),
])
def test_invalid_scale_fails_instead_of_silently_freezing_training(rate, minimum, reference):
    with pytest.raises(ValueError):
        exploration_scaled_learning_rate(rate, minimum, reference)


def test_scaling_is_an_explicit_new_run_option():
    parser = build_parser()
    assert not parser.parse_args(['--runtime', 'unused.npz']).learning_rate_std_scaling
    assert parser.parse_args(['--runtime', 'unused.npz', '--learning-rate-std-scaling']).learning_rate_std_scaling
