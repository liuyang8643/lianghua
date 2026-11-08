from __future__ import annotations

from pathlib import Path
import hashlib
import inspect
import warnings

import numpy as np
import pytest

from factor import (
    FactorDefinition,
    FactorMetadata,
    PRODUCTION_FACTORS,
    PRODUCTION_FACTOR_NAMES,
    PRODUCTION_FILTER_NAMES,
    get_factor_definition,
    precompute_factors,
)
from offline_data import load_runtime_slice
from factor.compute import scores_to_ranks
from factor_db.factors.AmihudIlliquidity import AmihudIlliquidity

from test_rl_runtime_slice import _runtime_arrays


FACTOR_PRELOAD_ROWS = max(
    definition.metadata.hist_days for definition in PRODUCTION_FACTORS
)


@pytest.mark.parametrize("semantics", ["continuous", "binary"])
def test_direct_membership_ranking_matches_explicit_missing_panel(semantics):
    values = np.asarray([[0., 1., 1., np.nan, np.inf], [0., 0., 1., 1., 0.]])
    member = np.asarray([[True, True, False, True, True], [False, True, True, False, True]])
    expected = scores_to_ranks(np.where(member, values, np.nan), score_semantics=semantics)
    actual = scores_to_ranks(values, score_semantics=semantics, membership=member)
    np.testing.assert_array_equal(actual, expected)


def test_direct_membership_ranking_rejects_wrong_axis_or_type():
    values = np.ones((3, 4))
    for membership in (np.ones((3, 5), dtype=bool), np.ones((3, 4))):
        with pytest.raises(ValueError, match="matching boolean"):
            scores_to_ranks(values, membership=membership)
# Retired production factors remain explicit research inputs for regressions.
AMIHUD_DEFINITION = FactorDefinition(
    metadata=FactorMetadata(
        name="AmihudIlliquidity",
        version="legacy-known-close-amount-v1",
        hist_days=20,
        required_fields=("close", "amount"),
        implementation_hash=hashlib.sha256(
            inspect.getsource(inspect.getmodule(AmihudIlliquidity)).encode("utf-8")
        ).hexdigest(),
    ),
    implementation=AmihudIlliquidity,
)
LEGACY_FOUR_DEFINITIONS = (
    AMIHUD_DEFINITION,
    *(get_factor_definition(name) for name in ("TrueMarketCap", "VolumeCV", "AmountBasedSmallCap")),
)


def _load(tmp_path, filename: str, data: dict[str, np.ndarray]):
    path = tmp_path / filename
    np.savez(path, **data)
    return load_runtime_slice(
        path,
        data["trade_dates"][130],
        data["trade_dates"][170],
        preload_rows=FACTOR_PRELOAD_ROWS,
    )


def test_production_vocabulary_metadata_and_cache_layout(tmp_path):
    data = _runtime_arrays()
    runtime = _load(tmp_path, "runtime.npz", data)

    batch = precompute_factors(runtime)

    assert batch.schema_version == "wbr.production-factors.v10-selected12-completed-amihud"
    assert batch.factor_names == PRODUCTION_FACTOR_NAMES
    assert batch.filter_names == PRODUCTION_FILTER_NAMES
    assert tuple(item.metadata.name for item in PRODUCTION_FACTORS) == (
        "TrueMarketCap",
        "VolumeCV",
        "AmountBasedSmallCap",
        "CompletedReversal20",
        "CompletedMomentum252Skip21",
        "BiliAdjustedIssueDiscount",
        "BiliHighLifetimeRangeRatio",
        "PBBelowTwoROEAbove10Signal",
        "LowCashOutflowProfitGrowthSpread",
        "HighOperatingProfitRevenueGrowthSpread",
        "HighAbnormalGrossProfit",
        "CompletedAmihudIlliquidity20",
    )
    assert [item.metadata.required_fields for item in PRODUCTION_FACTORS] == [
        ("open", "total_share"),
        ("volume",),
        ("amount",),
        ("close", "preClose"),
        ("close", "preClose"),
        ("open", "high", "low", "close", "preClose", "issue_price", "issue_date", "listing_age", "delisted_mask"),
        ("open", "high", "low", "close", "preClose", "issue_price", "issue_date", "listing_age", "delisted_mask"),
        ("open", "total_share", "financial_profit_ttm", "financial_equity"),
        ("financial_cash_outflow_yoy", "financial_profit_yoy"),
        ("financial_operating_profit_yoy", "financial_revenue_yoy"),
        ("abnormal_revenue_quarter", "abnormal_cost_quarter", "abnormal_sales_cash_quarter",
         "abnormal_revenue_prior_year_quarter", "abnormal_cost_prior_year_quarter",
         "abnormal_sales_cash_prior_year_quarter", "abnormal_total_assets"),
        ("close", "preClose", "amount"),
    ]
    assert [item.metadata.hist_days for item in PRODUCTION_FACTORS] == [
        1, 20, 60, 20, 252, 100_000, 100_000, 1, 0, 0, 0, 20,
    ]
    assert [item.metadata.lagged_fields for item in PRODUCTION_FACTORS] == [
        ("total_share",), (), (), (), (), (), (), ("total_share",), (), (), (), (),
    ]
    assert [item.metadata.score_semantics for item in PRODUCTION_FACTORS] == [
        *("continuous",) * 7, "binary", "continuous", "continuous", "continuous", "continuous",
    ]
    assert all(len(item.metadata.implementation_hash) == 64 for item in PRODUCTION_FACTORS)
    assert all(item.metadata.version for item in PRODUCTION_FACTORS)

    expected = (runtime.n_dates, 12, runtime.n_stocks)
    assert batch.raw.shape == expected
    assert batch.ranks.shape == expected
    assert batch.validity.shape == expected
    assert batch.filters.shape == (runtime.n_dates, 2, runtime.n_stocks)
    assert batch.raw.dtype == np.float32
    assert batch.ranks.dtype == np.float32
    assert batch.validity.dtype == np.bool_
    assert batch.filters.dtype == np.bool_
    assert batch.raw.flags.c_contiguous and not batch.raw.flags.writeable
    assert batch.ranks.flags.c_contiguous and not batch.ranks.flags.writeable
    assert batch.validity.flags.c_contiguous and not batch.validity.flags.writeable
    assert batch.filters.flags.c_contiguous and not batch.filters.flags.writeable

    day = batch.day(runtime.decision_start)
    assert day.raw.shape == (12, runtime.n_stocks)
    assert day.ranks.shape == (12, runtime.n_stocks)
    assert day.validity.shape == (12, runtime.n_stocks)
    assert day.filters.shape == (2, runtime.n_stocks)
    np.testing.assert_array_equal(day.validity, np.isfinite(day.raw))
    assert day.validity[[0, 1, 2, 3, 5, 6, 7, 8, 9, 10]].all()
    assert not day.validity[4].any()  # The 252-row factor has insufficient history.
    # A known binary rejection is valid and must remain zero after scoring.
    np.testing.assert_array_equal(day.raw[7], [0.0, 1.0, 1.0])
    np.testing.assert_array_equal(day.ranks[7], [0.0, 1.0, 1.0])
    assert np.all((day.ranks >= 0.0) & (day.ranks <= 1.0))


def test_only_pit_members_participate_in_ranking_and_mask_api_is_removed(tmp_path):
    data = _runtime_arrays()
    data["listing_age"][:140, 1] = -1
    data["listing_age"][140:, 1] = np.arange(len(data["trade_dates"]) - 140)
    data["issue_date"][1] = data["trade_dates"][140]
    data["delisted_mask"][150:, 2] = True
    runtime = _load(tmp_path, "runtime.npz", data)

    batch = precompute_factors(runtime)

    np.testing.assert_array_equal(batch.validity, np.isfinite(batch.raw))
    member = (runtime.field("listing_age") >= 0) & ~runtime.field("delisted_mask")
    for factor_index, definition in enumerate(PRODUCTION_FACTORS):
        # Rank the float64 calculation, not the lossy float32 raw cache.
        panel = dict(runtime.data, trade_dates=runtime.trade_dates)
        for field in definition.metadata.required_fields:
            if np.issubdtype(panel[field].dtype, np.number):
                panel[field] = panel[field].astype(np.float64)
        for field in definition.metadata.lagged_fields:
            panel[field] = np.concatenate(
                (np.full((1, runtime.n_stocks), np.nan), panel[field][:-1])
            )
        with np.errstate(all="ignore"):
            calculated = definition.implementation().calc_batch(panel)
        expected = scores_to_ranks(
            np.where(member, calculated, np.nan),
            score_semantics=definition.metadata.score_semantics,
        )
        np.testing.assert_array_equal(
            batch.ranks[:, factor_index, :],
            expected,
        )
        np.testing.assert_array_equal(batch.ranks[:, factor_index, :][~member], 0.0)
    assert batch.validity[130, 7, 1]  # Availability is distinct from membership.
    assert batch.validity[150, 7, 2]

    with pytest.raises(TypeError, match="rank_universe_mask"):
        precompute_factors(  # type: ignore[call-arg]
            runtime,
            rank_universe_mask=np.ones(runtime.n_stocks, dtype=bool),
        )


def test_cross_sectional_rank_ties_share_average_positions_and_ignore_column_order():
    raw = np.array(
        [
            [3.0, 3.0, 1.0, 0.0],
            [2.0, 2.0, 0.0, np.nan],
            [np.nan, 2.0, 2.0, 0.0],
        ],
        dtype=np.float32,
    )

    ranks = scores_to_ranks(raw)

    np.testing.assert_allclose(
        ranks,
        [[0.875, 0.875, 0.5, 0.25], [5 / 6, 5 / 6, 1 / 3, 0.0], [0.0, 5 / 6, 5 / 6, 1 / 3]],
        rtol=0,
        atol=np.finfo(np.float32).eps,
    )
    permutation = np.asarray([2, 0, 3, 1])
    np.testing.assert_array_equal(scores_to_ranks(raw[:, permutation]), ranks[:, permutation])


def test_extreme_finite_amihud_stays_valid_and_is_clipped_without_warning(
    tmp_path,
):
    data = _runtime_arrays()
    data["amount"][:, 0] = 1e-34
    runtime = _load(tmp_path, "extreme-amihud.npz", data)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        batch = precompute_factors(runtime, definitions=(AMIHUD_DEFINITION,))

    runtime_warnings = [
        warning
        for warning in caught
        if issubclass(warning.category, RuntimeWarning)
    ]
    assert runtime_warnings == []
    assert batch.schema_version == "wbr.research-factors.v1-pit-ranks"
    assert batch.factor_names == ("AmihudIlliquidity",)
    date_index = runtime.decision_start
    factor_index = batch.factor_names.index("AmihudIlliquidity")
    assert batch.validity[date_index, factor_index, 0]
    assert batch.ranks[date_index, factor_index, 0] == 1.0
    assert batch.raw[date_index, factor_index, 0] == np.finfo(np.float32).max


def test_financial_float64_thresholds_and_ranks_survive_float32_raw_cache(tmp_path):
    data = _runtime_arrays(stocks=4)
    data["financial_equity"][:] = 1e9
    data["financial_profit_ttm"][:] = [1e8 + 0.01, 1e8, 1e8 + 0.02, np.nan]
    data["financial_cash_outflow_yoy"][:] = 0.0
    data["financial_profit_yoy"][:] = [1.0 + 1e-9, 1.0 + 2e-9, 1.0 + 3e-9, np.nan]
    data["financial_operating_profit_yoy"][:] = [1.0 + 3e-9, 1.0 + 2e-9, 1.0 + 1e-9, np.nan]
    data["financial_revenue_yoy"][:] = 0.0
    runtime = _load(tmp_path, "financial-precision.npz", data)
    for field in (
        "financial_profit_ttm", "financial_equity", "financial_cash_outflow_yoy",
        "financial_profit_yoy", "financial_operating_profit_yoy", "financial_revenue_yoy",
    ):
        assert runtime.field(field).dtype == np.float64
        assert not runtime.field(field).flags.writeable
        np.testing.assert_array_equal(runtime.field(field), data[field][:runtime.n_dates])

    batch = precompute_factors(runtime)
    day = batch.day(runtime.decision_start)
    binary_index = batch.factor_names.index("PBBelowTwoROEAbove10Signal")
    np.testing.assert_array_equal(day.validity[binary_index], [True, True, True, False])
    np.testing.assert_array_equal(day.raw[binary_index], [1.0, 0.0, 1.0, np.nan])
    np.testing.assert_array_equal(day.ranks[binary_index], [1.0, 0.0, 1.0, 0.0])
    # Float32 collapses these continuous raw scores into ties, but the ranks
    # must still distinguish the original float64 financial differences.
    for name, expected in (
        ("LowCashOutflowProfitGrowthSpread", [1 / 3, 2 / 3, 1.0, 0.0]),
        ("HighOperatingProfitRevenueGrowthSpread", [1.0, 2 / 3, 1 / 3, 0.0]),
    ):
        index = batch.factor_names.index(name)
        np.testing.assert_array_equal(day.raw[index], [1.0, 1.0, 1.0, np.nan])
        np.testing.assert_array_equal(day.validity[index], [True, True, True, False])
        np.testing.assert_allclose(day.ranks[index], expected, rtol=0, atol=np.finfo(np.float32).eps)


def test_t_day_hlcva_and_future_values_do_not_change_t_factor_cache(tmp_path):
    base = _runtime_arrays()
    target_source_index = 145
    target_date = base["trade_dates"][target_source_index]
    changed = {name: values.copy() for name, values in base.items()}
    for field in ("high", "low", "close", "volume", "amount"):
        changed[field][target_source_index] *= 1000.0
    for field in (
        "open", "high", "low", "close", "volume", "amount", "total_share",
        "financial_profit_ttm", "financial_equity", "financial_cash_outflow_yoy",
        "financial_profit_yoy", "financial_operating_profit_yoy", "financial_revenue_yoy",
    ):
        changed[field][target_source_index + 1 :] *= 2000.0
    changed["st_mask"][target_source_index + 1 :] = True

    baseline = precompute_factors(_load(tmp_path, "base.npz", base))
    perturbed = precompute_factors(_load(tmp_path, "changed.npz", changed))
    index = baseline.index_of(target_date)

    np.testing.assert_array_equal(baseline.raw[index], perturbed.raw[index])
    np.testing.assert_array_equal(baseline.ranks[index], perturbed.ranks[index])
    np.testing.assert_array_equal(baseline.validity[index], perturbed.validity[index])
    np.testing.assert_array_equal(baseline.filters[index], perturbed.filters[index])


def test_true_market_cap_uses_t_open_and_t_minus_one_total_share(tmp_path):
    base = _runtime_arrays()
    target_source_index = 145
    target_date = base["trade_dates"][target_source_index]
    share_changed = {name: values.copy() for name, values in base.items()}
    share_changed["total_share"][target_source_index, 0] *= 10.0

    baseline = precompute_factors(_load(tmp_path, "base.npz", base))
    changed = precompute_factors(_load(tmp_path, "share.npz", share_changed))
    index = baseline.index_of(target_date)
    factor_index = baseline.factor_names.index("TrueMarketCap")

    assert baseline.raw[index, factor_index, 0] == changed.raw[index, factor_index, 0]
    assert baseline.raw[index + 1, factor_index, 0] != changed.raw[index + 1, factor_index, 0]

    open_changed = {name: values.copy() for name, values in base.items()}
    open_changed["open"][target_source_index, 0] *= 2.0
    changed = precompute_factors(_load(tmp_path, "open.npz", open_changed))
    assert baseline.raw[index, factor_index, 0] != changed.raw[index, factor_index, 0]


def test_explicit_star_st_mask_does_not_reintroduce_removed_production_filter(tmp_path):
    data = _runtime_arrays()
    target_source_index = 145
    data["star_st_mask"] = np.zeros_like(data["st_mask"])
    data["star_st_mask"][target_source_index, 2] = True
    runtime = _load(tmp_path, "runtime.npz", data)
    batch = precompute_factors(runtime)
    index = batch.index_of(data["trade_dates"][target_source_index])
    assert batch.filter_names == ("FilterST", "FilterLowPrice")
    assert batch.day(index).filters.tolist() == [
        [True, True, True],
        [True, True, True],
    ]


def test_real_train_2022_research_four_factor_topn_matches_float64_legacy_ranking():
    from factor.compute import scores_to_ranks
    from factor_db.factors.AmihudIlliquidity import AmihudIlliquidity
    from factor_db.factors.AmountBasedSmallCap import AmountBasedSmallCap
    from factor_db.factors.TrueMarketCap import TrueMarketCap
    from factor_db.factors.VolumeCV import VolumeCV

    runtime_files = sorted(
        (Path(__file__).resolve().parents[1] / "data" / "runtime").glob(
            "runtime_*.npz"
        )
    )
    if not runtime_files:
        pytest.skip("production runtime NPZ is not available")
    path = runtime_files[-1]
    try:
        runtime = load_runtime_slice(
            path,
            "2022-01-04",
            "2022-12-31",
            preload_rows=max(item.metadata.hist_days for item in LEGACY_FOUR_DEFINITIONS),
        )
    except ValueError as exc:
        if "missing registered fields" not in str(exc):
            raise
        pytest.skip(f"production runtime uses an obsolete schema: {exc}")
    batch = precompute_factors(runtime, definitions=LEGACY_FOUR_DEFINITIONS)
    assert batch.factor_names == (
        "AmihudIlliquidity", "TrueMarketCap", "VolumeCV", "AmountBasedSmallCap",
    )
    assert batch.schema_version == "wbr.research-factors.v1-pit-ranks"

    with np.load(path, allow_pickle=False) as npz:
        source_dates = npz["trade_dates"].astype("datetime64[D]")
        row_start = int(np.searchsorted(source_dates, runtime.trade_dates[0]))
        row_stop = int(
            np.searchsorted(
                source_dates,
                runtime.trade_dates[-1],
                side="right",
            )
        )
        panel = {
            name: np.array(npz[name][row_start:row_stop], copy=True)
            for name in ("open", "close", "volume", "amount", "total_share")
        }

    lagged_share = np.empty_like(panel["total_share"])
    lagged_share[0] = np.nan
    lagged_share[1:] = panel["total_share"][:-1]
    legacy_raw = []
    for implementation, share_override in (
        (AmihudIlliquidity, None),
        (TrueMarketCap, lagged_share),
        (VolumeCV, None),
        (AmountBasedSmallCap, None),
    ):
        factor_panel = dict(panel)
        if share_override is not None:
            factor_panel["total_share"] = share_override
        with np.errstate(divide="ignore", invalid="ignore"):
            legacy_raw.append(implementation().calc_batch(factor_panel))

    member = (runtime.field("listing_age") >= 0) & ~runtime.field("delisted_mask")
    legacy_ranks = [scores_to_ranks(np.where(member, raw, np.nan)) for raw in legacy_raw]
    weights = np.array([0.4, 0.9, 0.1, 0.6], dtype=np.float64)
    for date_index in range(runtime.decision_start, runtime.decision_stop):
        legacy_valid = np.logical_and.reduce(
            [np.isfinite(raw[date_index]) for raw in legacy_raw]
        )
        candidate_mask = (
            legacy_valid
            & batch.validity[date_index, :4].all(axis=0)
            & batch.filters[date_index].all(axis=0)
        )
        candidates = np.flatnonzero(candidate_mask)
        batch_score = (
            batch.ranks[date_index, :4][:, candidates].T * weights
        ).sum(axis=1)
        legacy_score = np.stack(
            [ranks[date_index, candidates] for ranks in legacy_ranks],
            axis=1,
        ) @ weights
        batch_order = candidates[np.argsort(-batch_score)]
        legacy_order = candidates[np.argsort(-legacy_score)]
        np.testing.assert_array_equal(
            batch_order[:20],
            legacy_order[:20],
            err_msg=f"buy_n parity failed on {runtime.trade_dates[date_index]}",
        )
        assert set(batch_order[:25]) == set(legacy_order[:25]), (
            f"retention parity failed on {runtime.trade_dates[date_index]}"
        )
