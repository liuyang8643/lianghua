from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from env.contracts import Observation
from env.encoder import (
    ObservationEncoder,
    StaticMarketEncodingCache,
    TrainOnlyNormalizer,
)
from env.observation import ObservationBuilder
from test_rl_observation import sample_account, synthetic_inputs


def copy_observation(observation: Observation, **changes) -> Observation:
    values = {
        "stock_panel": observation.stock_panel.copy(),
        "market_panel": observation.market_panel.copy(),
        "position_panel": observation.position_panel.copy(),
        "portfolio": observation.portfolio.copy(),
        "feature_mask": observation.feature_mask.copy(),
        "stock_mask": observation.stock_mask.copy(),
        "time_mask": observation.time_mask.copy(),
        "schema_version": observation.schema_version,
        "decision_date": observation.decision_date,
    }
    values.update(changes)
    return Observation(**values)


def test_encoder_is_deterministic_finite_and_independent_of_stock_count() -> None:
    runtime, factors = synthetic_inputs()
    builder = ObservationBuilder(runtime, factors, lookback=4)
    encoder = ObservationEncoder(builder.schema)
    observation = builder.build(5, sample_account(runtime))

    first = encoder.encode(observation)
    second = encoder.encode(observation)

    np.testing.assert_array_equal(first, second)
    assert first.shape == (encoder.output_dimension,)
    assert first.dtype == np.float32
    assert np.isfinite(first).all()
    assert encoder.output_schema.dimension == len(encoder.output_schema.feature_names)
    assert len(encoder.output_schema.schema_hash) == 64

    larger_codes = tuple(f"{index:06d}.SZ" for index in range(7))
    larger_runtime, larger_factors = synthetic_inputs(stock_codes=larger_codes)
    larger_builder = ObservationBuilder(larger_runtime, larger_factors, lookback=4)
    larger_encoder = ObservationEncoder(larger_builder.schema)
    assert larger_encoder.output_dimension == encoder.output_dimension
    assert larger_encoder.output_schema.feature_names == encoder.output_schema.feature_names


def test_every_registered_panel_and_account_dimension_is_consumed() -> None:
    runtime, factors = synthetic_inputs()
    builder = ObservationBuilder(runtime, factors, lookback=4)
    encoder = ObservationEncoder(builder.schema)
    observation = builder.build(5, sample_account(runtime))
    baseline = encoder.encode(observation)

    for feature_index in range(observation.stock_panel.shape[2]):
        changed_panel = observation.stock_panel.copy()
        changed_panel[-1, 0, feature_index] += np.float32(0.123 + feature_index / 100.0)
        changed = encoder.encode(copy_observation(observation, stock_panel=changed_panel))
        assert not np.array_equal(changed, baseline), f"stock feature {feature_index} was ignored"

    # The oldest row and last stock must contribute as well.
    changed_panel = observation.stock_panel.copy()
    changed_panel[0, -1, 0] += 3.0
    assert not np.array_equal(
        encoder.encode(copy_observation(observation, stock_panel=changed_panel)), baseline
    )

    for feature_index in range(observation.market_panel.shape[1]):
        changed_market = observation.market_panel.copy()
        changed_market[0, feature_index] += np.float32(0.25)
        changed = encoder.encode(copy_observation(observation, market_panel=changed_market))
        assert not np.array_equal(changed, baseline), f"market feature {feature_index} was ignored"

    for feature_index in range(observation.position_panel.shape[1]):
        changed_positions = observation.position_panel.copy()
        changed_positions[0, feature_index] += np.float32(0.25)
        changed = encoder.encode(copy_observation(observation, position_panel=changed_positions))
        assert not np.array_equal(changed, baseline), f"position feature {feature_index} was ignored"

    for feature_index in range(observation.portfolio.shape[0]):
        changed_portfolio = observation.portfolio.copy()
        changed_portfolio[feature_index] += np.float32(0.25)
        changed = encoder.encode(copy_observation(observation, portfolio=changed_portfolio))
        assert not np.array_equal(changed, baseline), f"portfolio feature {feature_index} was ignored"


def test_factor_rank_stock_return_pairing_changes_joint_market_encoding() -> None:
    runtime, factors = synthetic_inputs()
    builder = ObservationBuilder(runtime, factors, lookback=4)
    encoder = ObservationEncoder(builder.schema)
    observation = builder.build(5, sample_account(runtime))
    factor_index = builder.schema.stock_feature_names.index(
        "factor_rank.AmihudIlliquidity"
    )
    open_index = builder.schema.stock_feature_names.index("open")
    preclose_index = builder.schema.stock_feature_names.index("preClose")

    baseline_panel = observation.stock_panel.copy()
    baseline_panel[-1, :, factor_index] = (0.0, 0.25, 1.0)
    reordered_panel = baseline_panel.copy()
    reordered_panel[-1, :, factor_index] = baseline_panel[-1, ::-1, factor_index]
    stock_returns = (
        baseline_panel[-1, :, open_index]
        / baseline_panel[-1, :, preclose_index]
        - 1.0
    )
    assert not np.isclose(
        baseline_panel[-1, :, factor_index] @ stock_returns,
        reordered_panel[-1, :, factor_index] @ stock_returns,
    )
    np.testing.assert_array_equal(
        np.sort(baseline_panel[-1, :, factor_index]),
        np.sort(reordered_panel[-1, :, factor_index]),
    )
    baseline_observation = copy_observation(
        observation,
        stock_panel=baseline_panel,
    )
    reordered_observation = copy_observation(
        observation,
        stock_panel=reordered_panel,
    )

    baseline_encoding = encoder.encode_market(baseline_observation)
    reordered_encoding = encoder.encode_market(reordered_observation)

    assert not np.array_equal(baseline_encoding, reordered_encoding)
    changed_names = {
        encoder.output_schema.feature_names[index]
        for index in np.flatnonzero(baseline_encoding != reordered_encoding)
    }
    assert any(name.startswith("factor_joint.AmihudIlliquidity") for name in changed_names)


def test_holding_same_position_on_different_factor_stock_changes_joint_account_encoding() -> None:
    runtime, factors = synthetic_inputs()
    builder = ObservationBuilder(runtime, factors, lookback=4)
    encoder = ObservationEncoder(builder.schema)
    cache = StaticMarketEncodingCache.precompute(builder, encoder, (5,))
    first = builder.build_account(5, sample_account(runtime))
    moved_positions = np.zeros_like(first.position_panel)
    moved_positions[2] = first.position_panel[0]
    second = replace(first, position_panel=moved_positions)
    np.testing.assert_array_equal(
        np.sort(first.position_panel, axis=0),
        np.sort(second.position_panel, axis=0),
    )

    first_account = encoder.encode_account(first)
    second_account = encoder.encode_account(second)
    first_full = cache.encode_account_state(first, encoder)
    second_full = cache.encode_account_state(second, encoder)

    assert not np.array_equal(first_account, second_account)
    np.testing.assert_array_equal(
        first_full[: encoder.market_dimension],
        second_full[: encoder.market_dimension],
    )
    assert not np.array_equal(first_full, second_full)
    changed_names = {
        encoder.output_schema.feature_names[encoder.market_dimension + index]
        for index in np.flatnonzero(first_account != second_account)
    }
    assert any(name.startswith("account_factor.") for name in changed_names)


def test_future_and_t_post_open_mutations_do_not_change_encoding() -> None:
    runtime, factors = synthetic_inputs()
    baseline_builder = ObservationBuilder(runtime, factors, lookback=4)
    encoder = ObservationEncoder(baseline_builder.schema)
    baseline = encoder.encode(
        baseline_builder.build(5, sample_account(runtime))
    )

    runtime2, factors2 = synthetic_inputs()
    for field in ("high", "low", "close", "volume", "amount", "total_share"):
        runtime2.data[field][5:] = 123_456.0
    factors2.ranks[6:] = 0.999
    changed_builder = ObservationBuilder(runtime2, factors2, lookback=4)
    changed = encoder.encode(
        changed_builder.build(5, sample_account(runtime2))
    )
    np.testing.assert_array_equal(changed, baseline)


def test_static_market_cache_matches_direct_encoding_exactly() -> None:
    runtime, factors = synthetic_inputs()
    builder = ObservationBuilder(runtime, factors, lookback=4)
    encoder = ObservationEncoder(builder.schema)
    cache = StaticMarketEncodingCache.precompute(
        builder,
        encoder,
        (4, 5, 6),
        chunk_rows=2,
    )
    account = sample_account(runtime, 5)
    account_part = builder.build_account(5, account)

    cached = cache.encode_account_state(account_part, encoder)
    direct = encoder.encode(builder.build(5, account))

    np.testing.assert_array_equal(cached, direct)
    with pytest.raises(ValueError, match="decision date mismatch"):
        cache.market_for(5, decision_date="2099-01-01", encoder=encoder)


def test_encoder_fails_fast_on_schema_or_shape_mismatch() -> None:
    runtime, factors = synthetic_inputs()
    builder = ObservationBuilder(runtime, factors, lookback=4)
    encoder = ObservationEncoder(builder.schema)
    observation = builder.build(5, sample_account(runtime))

    wrong_schema = copy_observation(observation, schema_version="other-schema")
    with pytest.raises(ValueError, match="schema identifier"):
        encoder.encode(wrong_schema)

    wrong_shape = copy_observation(
        observation,
        stock_panel=observation.stock_panel[:, :-1],
        feature_mask=observation.feature_mask[:, :-1],
        stock_mask=observation.stock_mask[:, :-1],
        position_panel=observation.position_panel[:-1],
    )
    with pytest.raises(ValueError, match="stock_panel shape mismatch"):
        encoder.encode(wrong_shape)


def test_train_only_normalizer_round_trip_and_role_guard(tmp_path) -> None:
    runtime, factors = synthetic_inputs()
    builder = ObservationBuilder(runtime, factors, lookback=4)
    encoder = ObservationEncoder(builder.schema)
    account = sample_account(runtime, 5)
    base = encoder.encode(builder.build(5, account))
    rows = np.stack((base, base + 0.25, base - 0.5))

    with pytest.raises(ValueError, match="training split"):
        TrainOnlyNormalizer.fit(
            rows,
            encoder.output_schema,
            dataset_role="validation",
        )

    normalizer = TrainOnlyNormalizer.fit(
        rows,
        encoder.output_schema,
        dataset_role="train",
    )
    transformed = normalizer.transform(rows)
    assert transformed.shape == rows.shape
    assert np.isfinite(transformed).all()

    path = tmp_path / "normalizer.json"
    normalizer.save(path)
    loaded = TrainOnlyNormalizer.load(path, expected_schema=encoder.output_schema)
    np.testing.assert_array_equal(loaded.mean, normalizer.mean)
    np.testing.assert_array_equal(loaded.scale, normalizer.scale)
    np.testing.assert_array_equal(loaded.transform(rows), transformed)

    tampered = normalizer.to_dict()
    tampered["clip"] = 1.0
    with pytest.raises(ValueError, match="state hash"):
        TrainOnlyNormalizer.from_dict(tampered)
