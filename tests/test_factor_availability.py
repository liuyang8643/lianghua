from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from env.action_schema import ActionSchema
from env.contracts import AccountState
from env.encoder import ObservationEncoder, RawMarketStore
from env.observation import ObservationBuilder
from env.scoring import score_factor_ranks
from factor import factor_coverage, precompute_factors
from rl_test_data import build_episode
from test_rl_observation import synthetic_inputs, sample_account, make_builder


def test_partial_unit_weight_scores_are_centered_and_no_signal_has_stable_tail():
    schema = ActionSchema(factor_names=("first", "second"))
    config = schema.decode(np.array([1.0, 1.0, -1.0]))
    ranks = {"first": np.array([0.8, 0.5, 0.8, np.nan]), "second": np.array([0.2, np.nan, 0.8, np.nan])}
    valid = {name: np.isfinite(values) for name, values in ranks.items()}
    scores = score_factor_ranks(ranks, valid, config, 4)
    assert scores[0] == pytest.approx(scores[1])
    assert scores[0] == pytest.approx(1.0)
    assert scores[2] == pytest.approx(1.6)
    assert scores[3] < min(scores[:3])
    assert np.isfinite(scores).all()


def test_completely_observed_scores_preserve_original_sum_exactly():
    schema = ActionSchema()
    config = schema.decode(np.linspace(-0.7, 0.9, schema.action_dim))
    rng = np.random.default_rng(45)
    ranks = {name: rng.random(97) for name in schema.factor_names}
    valid = {name: np.ones(97, dtype=bool) for name in schema.factor_names}
    expected = sum(ranks[name] * config.factor_weights[name] for name in schema.factor_names)
    np.testing.assert_array_equal(score_factor_ranks(ranks, valid, config, 97), expected)


def test_factor_coverage_counts_pit_members_and_missing_intervals(tmp_path):
    episode = build_episode(tmp_path / "runtime.npz")
    factors = episode.factors
    mask = factors.validity.copy()
    left = factors.decision_start
    mask[left:left + 2, 0] = False
    changed = replace(factors, validity=mask)
    audit = factor_coverage(episode.runtime, changed)
    item = audit["factors"][factors.factor_names[0]]
    assert item["missing_cells"] == 2 * episode.runtime.n_stocks
    assert item["fully_missing_intervals"] == [{"start": str(factors.trade_dates[left]), "end": str(factors.trade_dates[left + 1]), "trading_rows": 2}]
    assert item["coverage"] < 1.0
    assert item["minimum_daily_coverage"] == 0.0


def test_actor_uses_raw_state_independently_of_factor_availability():
    runtime, factors = synthetic_inputs(date_count=30)
    before = make_builder(runtime, factors, lookback=4)
    missing = SimpleNamespace(**vars(factors))
    missing.validity = factors.validity.copy()
    missing.validity[25:, 0] = False
    after = make_builder(runtime, missing, lookback=4)
    original = before.build(27,sample_account(runtime))
    changed = after.build(27,sample_account(runtime))
    np.testing.assert_array_equal(original.stock_panel, changed.stock_panel)
    assert not any(name.startswith(("factor_rank.", "filter_pass.")) for name in after.schema.stock_feature_names)
    assert not any("stock_mask" in name or "feature_mask" in name for name in after.schema.stock_feature_names)


def test_nonmember_finite_factor_values_cannot_change_member_ranks(tmp_path):
    episode = build_episode(tmp_path / "runtime.npz")
    data = {name: value.copy() for name, value in episode.runtime.data.items()}
    # A delisted column can retain lagged source values, but no longer belongs
    # in today's cross-sectional denominator.
    data["delisted_mask"][:, -1] = True
    runtime = replace(episode.runtime, data=data)
    baseline = precompute_factors(runtime)
    edited = {name: value.copy() for name, value in data.items()}
    for name in ("open", "high", "low", "close", "preClose", "amount", "volume", "total_share"):
        edited[name][:, -1] *= 1000.0
    changed = precompute_factors(replace(runtime, data=edited))
    np.testing.assert_array_equal(baseline.ranks[:, :, :-1], changed.ranks[:, :, :-1])
    assert not np.any(changed.ranks[:, :, -1])




@pytest.mark.parametrize("field", ["amount", "volume", "total_share", "issue_price"])
def test_actor_raw_fields_ignore_finite_nonmember_values(field):
    runtime, factors = synthetic_inputs(date_count=30)
    runtime.data["delisted_mask"][:, -1] = True
    before = make_builder(runtime, factors, lookback=4)
    initial = before.build(27, sample_account(runtime))
    # Mutate only a nonmember nominal feature, without changing member inputs.
    changed_runtime = SimpleNamespace(**vars(runtime))
    changed_runtime.data = {name: values.copy() for name, values in runtime.data.items()}
    changed_runtime.data[field][..., -1] *= 1e-6
    after = make_builder(changed_runtime, factors, lookback=4)
    changed = after.build(27, sample_account(runtime))
    encoder = ObservationEncoder(before.schema)
    np.testing.assert_array_equal(initial.stock_panel[:, :-1], changed.stock_panel[:, :-1])
    np.testing.assert_array_equal(initial.stock_panel, changed.stock_panel)
