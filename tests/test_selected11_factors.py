"""Selected-eleven continuity and financial soft-score/causality contracts."""

from dataclasses import replace

import numpy as np
import pytest

import factor.compute as compute_module
from env.contracts import DayConfig
from env.scoring import score_factor_ranks
from factor import PRODUCTION_FACTORS, PRODUCTION_FACTOR_NAMES, precompute_factors
from factor.base import FACTOR_SCHEMA_VERSION
from factor.compute import scores_to_ranks
from factor.library.bilibili import calculate_bilibili_scores
from factor.library.financial import (
    HighOperatingProfitRevenueGrowthSpread,
    LowCashOutflowProfitGrowthSpread,
    PBBelowTwoROEAbove10Signal,
)
from factor.library.abnormal_gross_profit import (
    ABNORMAL_GROSS_PROFIT_PANEL_FIELDS, HighAbnormalGrossProfit, abnormal_gross_profit,
)
from factor.registry import get_factor_definition
from offline_data import RuntimeManifest, RuntimeSlice
from test_rl_runtime_slice import _runtime_arrays


SELECTED7 = (
    "TrueMarketCap", "VolumeCV", "AmountBasedSmallCap", "CompletedReversal20",
    "CompletedMomentum252Skip21", "BiliAdjustedIssueDiscount", "BiliHighLifetimeRangeRatio",
)
FINANCIAL_NAMES = (
    "PBBelowTwoROEAbove10Signal", "LowCashOutflowProfitGrowthSpread",
    "HighOperatingProfitRevenueGrowthSpread",
)
FINANCIAL_FIELDS = (
    "financial_profit_ttm", "financial_equity", "financial_cash_outflow_yoy",
    "financial_profit_yoy", "financial_operating_profit_yoy", "financial_revenue_yoy",
)


@pytest.fixture
def runtime():
    # Construct the public snapshot contract directly: financial source loading
    # and identity sealing are owned by the separate offline-data workstream.
    arrays = _runtime_arrays(rows=280, stocks=6)
    dates = arrays.pop("trade_dates")
    codes = tuple(arrays.pop("stock_codes"))
    for name, value in zip(FINANCIAL_FIELDS, (2e8, 1e9, 0.125, 0.5, 0.75, 0.25)):
        arrays[name] = np.full(arrays["open"].shape, value, dtype=np.float64)
    for name, value in zip(ABNORMAL_GROSS_PROFIT_PANEL_FIELDS,
                           (160., 100., 100., 100., 60., 80., 1000.)):
        arrays[name] = np.full(arrays["open"].shape, value, dtype=np.float64)
    dates.flags.writeable = False
    for values in arrays.values():
        values.flags.writeable = False
    manifest = RuntimeManifest(
        schema_version="test-financial-runtime", schema_hash="runtime-schema",
        source_path="synthetic", source_sha256="synthetic-source", source_size=0,
        stock_vocabulary_sha256="synthetic-stocks", requested_start=str(dates[260]),
        requested_end=str(dates[-1]), loaded_start=str(dates[0]), loaded_end=str(dates[-1]),
        requested_preload_rows=100000, actual_preload_rows=260, fields=(),
    )
    return RuntimeSlice(codes, dates, arrays, 260, len(dates), manifest)


def _changed(runtime, **fields):
    data = dict(runtime.data)
    for name, values in fields.items():
        values.flags.writeable = False
        data[name] = values
    return replace(runtime, data=data)


def _config(names, weights):
    return DayConfig(
        factor_weights=dict(zip(names, weights)),
        factor_enabled=dict(zip(names, (weight != 0 for weight in weights))),
        filter_flags={}, buy_n=20, turnover_rate=1.0, limit_up_protection=True,
        rebalance_band_pct=0.01, single_buy_pct=0.05,
    )


def test_selected11_vocabulary_metadata_and_schema(runtime):
    assert PRODUCTION_FACTOR_NAMES == SELECTED7 + FINANCIAL_NAMES + ("HighAbnormalGrossProfit",)
    assert tuple(item.metadata.name for item in PRODUCTION_FACTORS) == PRODUCTION_FACTOR_NAMES
    assert FACTOR_SCHEMA_VERSION == "wbr.production-factors.v9-selected11-momentum-interior-gaps"
    assert tuple(item.metadata.hist_days for item in PRODUCTION_FACTORS) == (
        1, 20, 60, 20, 252, 100000, 100000, 1, 0, 0, 0,
    )
    binary = get_factor_definition(FINANCIAL_NAMES[0]).metadata
    assert binary.score_semantics == "binary"
    assert binary.as_dict()["score_semantics"] == "binary"
    assert binary.lagged_fields == ("total_share",)
    for definition in PRODUCTION_FACTORS:
        if definition.metadata.name != binary.name:
            assert definition.metadata.score_semantics == "continuous"
        assert len(definition.metadata.implementation_hash) == 64
    batch = precompute_factors(runtime)
    assert batch.schema_version == FACTOR_SCHEMA_VERSION
    assert batch.raw.shape == batch.ranks.shape == batch.validity.shape == (280, 11, 6)
    assert batch.raw.dtype == batch.ranks.dtype == np.float32
    assert batch.validity.dtype == np.bool_
    for values in (batch.raw, batch.ranks, batch.validity, batch.filters):
        assert values.flags.c_contiguous and not values.flags.writeable


@pytest.mark.parametrize("name", ("AmihudIlliquidity", "CompletedAmountImbalance20", "CompletedCloseLocation20"))
def test_unselected_production8_entries_are_removed_but_research_classes_remain(name):
    from factor.library import CompletedAmountImbalance20, CompletedCloseLocation20
    from factor_db.factors.AmihudIlliquidity import AmihudIlliquidity

    with pytest.raises(KeyError):
        get_factor_definition(name)
    assert {cls.__name__ for cls in (AmihudIlliquidity, CompletedAmountImbalance20, CompletedCloseLocation20)} >= {name}


def test_continuous_average_ties_are_permutation_equivariant():
    raw = np.array([[4, 4, 1, -2, -2, np.nan, np.inf, -np.inf]], dtype=np.float64)
    denominator = 5
    expected = np.zeros_like(raw, dtype=np.float32)
    expected[0, :5] = 1 - np.array([0.5, 0.5, 2, 3.5, 3.5], dtype=np.float32) / denominator
    ranks = scores_to_ranks(raw)
    np.testing.assert_array_equal(ranks, expected)
    rng = np.random.default_rng(17)
    for _ in range(20):
        permutation = rng.permutation(raw.shape[1])
        np.testing.assert_array_equal(scores_to_ranks(raw[:, permutation]), ranks[:, permutation])


def test_no_ties_keep_existing_positional_normalization():
    raw = np.array([[9, 7, 5, -1, np.nan], [np.nan, 3, np.nan, np.nan, np.nan]], dtype=np.float64)
    expected = np.zeros(raw.shape, dtype=np.float32)
    denominator = 4
    expected[0, :4] = 1 - np.arange(4, dtype=np.float32) / denominator
    expected[1, 1] = 1
    np.testing.assert_array_equal(scores_to_ranks(raw), expected)


def test_all_ties_and_all_missing_are_defined():
    raw = np.array([[2, 2, 2, 2], [np.nan, np.nan, np.inf, -np.inf]])
    np.testing.assert_array_equal(scores_to_ranks(raw), [[0.625] * 4, [0] * 4])


@pytest.mark.parametrize("raw", (
    [[1, 1, 0, 0, np.nan, np.inf]], [[1, 1, 1, 1, 1, 1]], [[0, 0, 0, 0, 0, 0]],
))
def test_binary_groups_preserve_fixed_scores(raw):
    raw = np.asarray(raw, dtype=np.float64)
    ranks = scores_to_ranks(raw, score_semantics="binary")
    np.testing.assert_array_equal(ranks, np.where(np.isfinite(raw), raw, 0))
    np.testing.assert_array_equal(
        scores_to_ranks(raw[:, ::-1], score_semantics="binary"), ranks[:, ::-1],
    )


@pytest.mark.parametrize("value", (-1, 0.5, 2))
def test_binary_semantics_reject_nonbinary_finite_outputs(value):
    with pytest.raises(ValueError, match="binary factor scores"):
        scores_to_ranks(np.array([[value]]), score_semantics="binary")


def test_semantics_are_validated_and_sealed_in_batch_identity(runtime):
    definition = get_factor_definition(FINANCIAL_NAMES[0])
    with pytest.raises(ValueError, match="score_semantics"):
        replace(definition.metadata, score_semantics="unknown")
    with pytest.raises(ValueError, match="score_semantics"):
        scores_to_ranks(np.ones((1, 1)), score_semantics="unknown")
    continuous = replace(definition, metadata=replace(definition.metadata, score_semantics="continuous"))
    binary_batch = precompute_factors(runtime, definitions=(definition,))
    continuous_batch = precompute_factors(runtime, definitions=(continuous,))
    assert binary_batch.schema_hash != continuous_batch.schema_hash
    np.testing.assert_array_equal(binary_batch.raw, continuous_batch.raw)
    assert not np.array_equal(binary_batch.ranks, continuous_batch.ranks)


def test_binary_thresholds_and_finite_negative_profit_are_known_rejections():
    panel = {
        "open": np.array([[1, 1, 2, 1, 1, 1, np.nextafter(2.0, 0.0)]]),
        "total_share": np.full((1, 7), 100.0),
        "financial_equity": np.full((1, 7), 100.0),
        "financial_profit_ttm": np.array([[11, 10, 11, 0, -5, 10 + 1e-10, 11]]),
    }
    actual = PBBelowTwoROEAbove10Signal().calc_batch(panel)
    assert actual.dtype == np.float64
    np.testing.assert_array_equal(actual, [[1, 0, 0, 0, 0, 1, 1]])


@pytest.mark.parametrize("field", ("open", "total_share", "financial_equity", "financial_profit_ttm"))
@pytest.mark.parametrize("value", (np.nan, np.inf, -np.inf))
def test_binary_missing_required_primitive_is_nan(field, value):
    panel = {"open": np.array([[1.0]]), "total_share": np.array([[100.0]]),
             "financial_equity": np.array([[100.0]]), "financial_profit_ttm": np.array([[20.0]])}
    panel[field][0, 0] = value
    assert np.isnan(PBBelowTwoROEAbove10Signal().calc_batch(panel)).all()


@pytest.mark.parametrize("field", ("open", "total_share", "financial_equity"))
@pytest.mark.parametrize("value", (0.0, -1.0))
def test_binary_nonpositive_price_shares_or_equity_is_invalid(field, value):
    panel = {"open": np.array([[1.0]]), "total_share": np.array([[100.0]]),
             "financial_equity": np.array([[100.0]]), "financial_profit_ttm": np.array([[20.0]])}
    panel[field][0, 0] = value
    assert np.isnan(PBBelowTwoROEAbove10Signal().calc_batch(panel)).all()


def test_binary_nonfinite_cap_is_invalid_but_finite_negative_profit_remains_known():
    panel = {"open": np.array([[1e300, 1.0]]), "total_share": np.array([[1e300, 1.0]]),
             "financial_equity": np.array([[1.0, 1e-30]]), "financial_profit_ttm": np.array([[1.0, -1e300]])}
    np.testing.assert_array_equal(PBBelowTwoROEAbove10Signal().calc_batch(panel), [[np.nan, 0]])


@pytest.mark.parametrize("implementation,preferred,other", (
    (LowCashOutflowProfitGrowthSpread, "financial_profit_yoy", "financial_cash_outflow_yoy"),
    (HighOperatingProfitRevenueGrowthSpread, "financial_operating_profit_yoy", "financial_revenue_yoy"),
))
def test_spreads_have_correct_orientation_float64_ties_and_missingness(implementation, preferred, other):
    panel = {preferred: np.array([[0.5, -0.25, 0, np.nan, 0.5, np.inf, 1e-10]], dtype=np.float64),
             other: np.array([[0.25, -0.5, 0, 0, np.nan, 0, 0]], dtype=np.float64)}
    result = implementation().calc_batch(panel)
    assert result.dtype == np.float64
    np.testing.assert_array_equal(result, [[0.25, 0.25, 0, np.nan, np.nan, np.nan, 1e-10]])
    np.testing.assert_array_equal(scores_to_ranks(result), [[0.875, 0.875, 0.25, 0, 0, 0, 0.5]])


def test_precompute_preserves_binary_reject_zero_separately_from_missing(runtime):
    profit = runtime.field("financial_profit_ttm").copy()
    profit[:, 2:] = [-1, 0, 1e8, np.nan]
    batch = precompute_factors(_changed(runtime, financial_profit_ttm=profit))
    index = batch.factor_names.index(FINANCIAL_NAMES[0])
    np.testing.assert_array_equal(batch.raw[260, index], [1, 1, 0, 0, 0, np.nan])
    np.testing.assert_array_equal(batch.ranks[260, index], [1, 1, 0, 0, 0, 0])
    np.testing.assert_array_equal(batch.validity[260, index], [True, True, True, True, True, False])
    config = _config(FINANCIAL_NAMES[:1], (0.7,))
    scores = score_factor_ranks(
        {FINANCIAL_NAMES[0]: batch.ranks[260, index]},
        {FINANCIAL_NAMES[0]: batch.validity[260, index]}, config, runtime.n_stocks,
    )
    np.testing.assert_array_equal(scores[:5], [0.7, 0.7, 0, 0, 0])
    assert scores[-1] < min(scores[:-1])


def test_eleven_factor_scores_use_env_weighted_sum_without_binary_filtering(runtime):
    profit = runtime.field("financial_profit_ttm").copy()
    profit[:, 2:] = [-1, 0, 1e8, 2e8]
    batch = precompute_factors(_changed(runtime, financial_profit_ttm=profit))
    assert batch.validity[260].all()
    weights = tuple((index + 1) / 20 for index in range(11))
    config = _config(batch.factor_names, weights)
    ranks = dict(zip(batch.factor_names, batch.ranks[260]))
    validity = dict(zip(batch.factor_names, batch.validity[260]))
    expected = sum(row.astype(np.float64) * weight for row, weight in zip(batch.ranks[260], weights))
    actual = score_factor_ranks(ranks, validity, config, runtime.n_stocks)
    np.testing.assert_array_equal(actual, expected)
    assert np.isfinite(actual).all()
    assert batch.filters[260].all()


def test_share_lag_is_exactly_one_day_and_t_open_is_used(runtime):
    baseline = precompute_factors(runtime)
    shares = runtime.field("total_share").copy()
    shares[260, 0] *= 100
    changed = precompute_factors(_changed(runtime, total_share=shares))
    for name in ("TrueMarketCap", FINANCIAL_NAMES[0]):
        index = baseline.factor_names.index(name)
        np.testing.assert_array_equal(baseline.raw[:261, index], changed.raw[:261, index])
        assert baseline.raw[261, index, 0] != changed.raw[261, index, 0]
        np.testing.assert_array_equal(baseline.raw[262:, index], changed.raw[262:, index])
    index = baseline.factor_names.index(FINANCIAL_NAMES[0])
    assert np.isnan(baseline.raw[0, index]).all()
    opening = runtime.field("open").copy()
    opening[260, 0] *= 100
    changed = precompute_factors(_changed(runtime, open=opening))
    assert baseline.raw[260, index, 0] == 1
    assert changed.raw[260, index, 0] == 0


def test_financial_asof_rows_are_not_lagged_twice_and_future_rows_are_causal(runtime):
    baseline = precompute_factors(runtime)
    fields = {name: runtime.field(name).copy() for name in FINANCIAL_FIELDS}
    for values in fields.values():
        values[261:] = np.nan
    fields["financial_profit_ttm"][260, 0] = -1
    changed = precompute_factors(_changed(runtime, **fields))
    np.testing.assert_array_equal(baseline.raw[:260], changed.raw[:260])
    index = baseline.factor_names.index(FINANCIAL_NAMES[0])
    assert baseline.raw[260, index, 0] == 1
    assert changed.raw[260, index, 0] == 0
    assert np.isnan(changed.raw[261:, 7:10]).all()


def test_future_nonmember_stock_cannot_change_existing_financial_scores(runtime):
    ages = runtime.field("listing_age").copy()
    ages[:, -1] = -1
    baseline = precompute_factors(_changed(runtime, listing_age=ages))
    fields = {name: runtime.field(name).copy() for name in FINANCIAL_FIELDS}
    for values in fields.values():
        values[:, -1] *= 1e10
    changed = precompute_factors(_changed(runtime, listing_age=ages, **fields))
    np.testing.assert_array_equal(baseline.ranks[:, :, :-1], changed.ranks[:, :, :-1])
    assert not changed.ranks[:, :, -1].any()


def test_missing_financial_field_fails_explicitly(runtime):
    fields = dict(runtime.data)
    del fields["financial_equity"]
    with pytest.raises(ValueError, match="financial_equity"):
        precompute_factors(replace(runtime, data=fields))


def test_selected_lifetime_factors_reuse_one_canonical_calculation(runtime, monkeypatch):
    calls = []

    def recorded(dates, panel, **kwargs):
        calls.append((dates, panel))
        assert kwargs["factor_names"] == ("BiliAdjustedIssueDiscount", "BiliHighLifetimeRangeRatio")
        return calculate_bilibili_scores(dates, panel, **kwargs)

    monkeypatch.setattr(compute_module, "calculate_bilibili_scores", recorded)
    batch = precompute_factors(runtime)
    assert len(calls) == 1
    expected = calculate_bilibili_scores(runtime.trade_dates, runtime.data)
    for name in SELECTED7[-2:]:
        index = batch.factor_names.index(name)
        np.testing.assert_array_equal(batch.raw[:, index], expected[name].astype(np.float32))
        np.testing.assert_array_equal(batch.ranks[:, index], scores_to_ranks(expected[name]))


def test_selected_lifetime_factors_require_full_history_preload(runtime):
    manifest = replace(runtime.manifest, requested_preload_rows=260)
    with pytest.raises(ValueError, match="runtime's first row"):
        precompute_factors(replace(runtime, manifest=manifest))


def test_completed_styles_share_one_official_return_calculation(runtime, monkeypatch):
    calls = []
    original = compute_module._official_log_returns
    def recorded(close, preclose):
        calls.append((close.shape, preclose.shape))
        return original(close, preclose)
    monkeypatch.setattr(compute_module, "_official_log_returns", recorded)
    batch = precompute_factors(runtime)
    assert len(calls) == 1
    for name in ("CompletedReversal20", "CompletedMomentum252Skip21"):
        definition = get_factor_definition(name)
        expected = definition.implementation().calc_batch(runtime.data)
        np.testing.assert_array_equal(batch.raw[:, batch.factor_names.index(name)], expected.astype(np.float32))


@pytest.mark.parametrize("lagged_fields",[("close",),("preClose",),("close","close")])
def test_completed_return_sharing_separates_different_registered_lags(runtime, monkeypatch, lagged_fields):
    reversal = get_factor_definition("CompletedReversal20")
    momentum = get_factor_definition("CompletedMomentum252Skip21")
    shifted = replace(momentum, metadata=replace(momentum.metadata,
        name="CompletedMomentumWithExtraLag", lagged_fields=lagged_fields))
    calls = []
    original = compute_module._official_log_returns
    def recorded(close, preclose):
        calls.append(close.copy())
        return original(close, preclose)
    monkeypatch.setattr(compute_module, "_official_log_returns", recorded)
    batch = precompute_factors(runtime, definitions=(reversal, shifted))
    assert len(calls) == 2
    for index, definition in enumerate((reversal, shifted)):
        panel = dict(runtime.data)
        for field in definition.metadata.lagged_fields:
            panel[field] = np.concatenate((np.full_like(panel[field][:1], np.nan), panel[field][:-1]))
        expected = definition.implementation().calc_batch(panel)
        np.testing.assert_array_equal(batch.raw[:, index], expected.astype(np.float32))
        np.testing.assert_array_equal(batch.validity[:, index], np.isfinite(expected))
        np.testing.assert_array_equal(batch.ranks[:, index], scores_to_ranks(expected))


def test_completed_raw_runtime_view_preserves_calc_batch_float64_semantics(runtime):
    rng = np.random.default_rng(20260917)
    shape = runtime.data["close"].shape
    runtime = _changed(runtime, close=rng.uniform(8.,12.,shape).astype(np.float32),
                       preClose=rng.uniform(8.,12.,shape).astype(np.float32))
    definition = replace(get_factor_definition("CompletedReversal20"), raw_runtime_view=True)
    expected = definition.implementation().calc_batch(runtime.data)
    batch = precompute_factors(runtime, definitions=(definition,))
    np.testing.assert_array_equal(batch.raw[:, 0], expected.astype(np.float32))
    np.testing.assert_array_equal(batch.validity[:, 0], np.isfinite(expected))
    np.testing.assert_array_equal(batch.ranks[:, 0], scores_to_ranks(expected))
    rounded = definition.implementation().calc_from_returns(
        compute_module._official_log_returns(runtime.data["close"], runtime.data["preClose"]))
    assert not np.array_equal(rounded.astype(np.float32), expected.astype(np.float32), equal_nan=True)


def test_new_abnormal_factor_reuses_research_formula_and_already_known_rows(runtime):
    definition = get_factor_definition("HighAbnormalGrossProfit")
    assert definition.metadata.required_fields == ABNORMAL_GROSS_PROFIT_PANEL_FIELDS
    assert definition.metadata.lagged_fields == ()
    assert definition.metadata.hist_days == 0
    expected = abnormal_gross_profit(runtime.data)
    np.testing.assert_array_equal(HighAbnormalGrossProfit().calc_batch(runtime.data), expected)
    np.testing.assert_allclose(expected, .01)
    fields = {name: runtime.field(name).copy() for name in ABNORMAL_GROSS_PROFIT_PANEL_FIELDS}
    fields["abnormal_revenue_quarter"][260, 0] += 10
    for values in fields.values():
        values[261:] = np.nan
    baseline = precompute_factors(runtime)
    changed = precompute_factors(_changed(runtime, **fields))
    np.testing.assert_array_equal(baseline.raw[:260], changed.raw[:260])
    assert changed.raw[260, -1, 0] == np.float32(.02)
    assert np.isnan(changed.raw[261:, -1]).all()
    assert not changed.validity[261:, -1].any()
    assert not changed.ranks[261:, -1].any()


def test_new_factor_preserves_all_original_ten_values_and_zero_weight_scores(runtime):
    old = precompute_factors(runtime, definitions=PRODUCTION_FACTORS[:-1])
    new = precompute_factors(runtime)
    for name in ("raw", "ranks", "validity"):
        np.testing.assert_array_equal(getattr(old, name), getattr(new, name)[:, :10])
    np.testing.assert_array_equal(old.filters, new.filters)
    weights = (.9, .1, .6, 0., 0., 0., 0., 0., 0., 0.)
    old_scores = score_factor_ranks(dict(zip(old.factor_names, old.ranks[260])),
                                    dict(zip(old.factor_names, old.validity[260])),
                                    _config(old.factor_names, weights), runtime.n_stocks)
    fields = {name: np.full_like(runtime.field(name), np.nan) for name in ABNORMAL_GROSS_PROFIT_PANEL_FIELDS}
    missing = precompute_factors(_changed(runtime, **fields))
    new_scores = score_factor_ranks(dict(zip(missing.factor_names, missing.ranks[260])),
                                    dict(zip(missing.factor_names, missing.validity[260])),
                                    _config(missing.factor_names, weights + (0.,)), runtime.n_stocks)
    np.testing.assert_array_equal(old_scores, new_scores)


def test_abnormal_future_nonmember_values_do_not_change_member_ranks(runtime):
    ages = runtime.field("listing_age").copy()
    ages[:, -1] = -1
    baseline = precompute_factors(_changed(runtime, listing_age=ages))
    revenue = runtime.field("abnormal_revenue_quarter").copy()
    revenue[:, -1] = 1e30
    changed = precompute_factors(_changed(runtime, listing_age=ages, abnormal_revenue_quarter=revenue))
    np.testing.assert_array_equal(baseline.ranks[:, :, :-1], changed.ranks[:, :, :-1])
    assert not changed.ranks[:, -1, -1].any()


@pytest.mark.parametrize("field", ABNORMAL_GROSS_PROFIT_PANEL_FIELDS)
def test_abnormal_required_public_primitive_is_not_synthesized(runtime, field):
    data = dict(runtime.data)
    del data[field]
    with pytest.raises(ValueError, match=field):
        precompute_factors(replace(runtime, data=data))


def test_static_configuration_appends_zero_without_replacing_old_vocabulary():
    import json
    from pathlib import Path
    import yaml

    payload = json.loads(Path("configs/config.json").read_text(encoding="utf-8"))["individual_config"]
    assert tuple(payload["weights"]) == PRODUCTION_FACTOR_NAMES
    assert payload["weights"]["HighAbnormalGrossProfit"] == 0.
    strategy = yaml.safe_load(Path("configs/strategy.yaml").read_text(encoding="utf-8"))
    assert tuple(strategy["profiles"]["current4"]["factor_classes"]) == PRODUCTION_FACTOR_NAMES
