from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pytest

from factor import (
    PRODUCTION_FACTORS,
    PRODUCTION_FACTOR_NAMES,
    PRODUCTION_FILTER_NAMES,
    precompute_factors,
)
from offline_data import load_runtime_slice
from factor.compute import _cross_sectional_ranks

from test_rl_runtime_slice import _runtime_arrays


def _load(tmp_path, filename: str, data: dict[str, np.ndarray]):
    path = tmp_path / filename
    np.savez(path, **data)
    return load_runtime_slice(
        path,
        data["trade_dates"][130],
        data["trade_dates"][170],
    )


def test_production_vocabulary_metadata_and_cache_layout(tmp_path):
    data = _runtime_arrays()
    runtime = _load(tmp_path, "runtime.npz", data)

    batch = precompute_factors(runtime)

    assert batch.schema_version == "wbr.production-factors.v2"
    assert batch.factor_names == PRODUCTION_FACTOR_NAMES
    assert batch.filter_names == PRODUCTION_FILTER_NAMES
    assert tuple(item.metadata.name for item in PRODUCTION_FACTORS) == (
        "AmihudIlliquidity",
        "TrueMarketCap",
        "VolumeCV",
        "AmountBasedSmallCap",
    )
    assert [item.metadata.required_fields for item in PRODUCTION_FACTORS] == [
        ("close", "amount"),
        ("open", "total_share"),
        ("volume",),
        ("amount",),
    ]
    assert [item.metadata.hist_days for item in PRODUCTION_FACTORS] == [20, 1, 20, 60]
    assert all(len(item.metadata.implementation_hash) == 64 for item in PRODUCTION_FACTORS)
    assert all(item.metadata.version for item in PRODUCTION_FACTORS)

    expected = (runtime.n_dates, 4, runtime.n_stocks)
    assert batch.raw.shape == expected
    assert batch.ranks.shape == expected
    assert batch.validity.shape == expected
    assert batch.filters.shape == (runtime.n_dates, 3, runtime.n_stocks)
    assert batch.raw.dtype == np.float32
    assert batch.ranks.dtype == np.float32
    assert batch.validity.dtype == np.bool_
    assert batch.filters.dtype == np.bool_
    assert batch.rank_universe_mask.dtype == np.bool_
    assert batch.rank_universe_mask.tolist() == [True, True, True]
    assert len(batch.rank_universe_sha256) == 64
    assert batch.raw.flags.c_contiguous and not batch.raw.flags.writeable
    assert batch.ranks.flags.c_contiguous and not batch.ranks.flags.writeable
    assert batch.validity.flags.c_contiguous and not batch.validity.flags.writeable
    assert batch.filters.flags.c_contiguous and not batch.filters.flags.writeable

    day = batch.day(runtime.decision_start)
    assert day.raw.shape == (4, runtime.n_stocks)
    assert day.ranks.shape == (4, runtime.n_stocks)
    assert day.validity.shape == (4, runtime.n_stocks)
    assert day.filters.shape == (3, runtime.n_stocks)
    assert day.rank_universe_mask.shape == (runtime.n_stocks,)
    assert np.isfinite(day.raw).all()
    assert np.all((day.ranks >= 0.0) & (day.ranks <= 1.0))


def test_rank_universe_is_applied_before_ranking_and_persisted(tmp_path):
    data = _runtime_arrays()
    runtime = _load(tmp_path, "runtime.npz", data)
    universe = np.array([True, True, False])

    batch = precompute_factors(runtime, rank_universe_mask=universe)

    np.testing.assert_array_equal(batch.rank_universe_mask, universe)
    assert not batch.validity[:, :, 2].any()
    assert not batch.ranks[:, :, 2].any()
    for factor_index in range(len(batch.factor_names)):
        expected = _cross_sectional_ranks(
            batch.raw[:, factor_index, :2]
        )
        np.testing.assert_array_equal(
            batch.ranks[:, factor_index, :2],
            expected,
        )

    with pytest.raises(ValueError, match="fixed stock vocabulary"):
        precompute_factors(runtime, rank_universe_mask=np.ones(2, dtype=bool))


def test_cross_sectional_rank_ties_match_existing_reverse_argsort_semantics():
    from core.scoring import scores_to_ranks

    raw = np.array(
        [
            [3.0, 3.0, 1.0, 0.0],
            [2.0, 2.0, 0.0, np.nan],
            [np.nan, 2.0, 2.0, 0.0],
        ],
        dtype=np.float32,
    )

    ranks = _cross_sectional_ranks(raw)

    np.testing.assert_array_equal(ranks, scores_to_ranks(raw))


def test_extreme_finite_amihud_stays_valid_and_is_clipped_without_warning(
    tmp_path,
):
    data = _runtime_arrays()
    data["amount"][:, 0] = 1e-34
    runtime = _load(tmp_path, "extreme-amihud.npz", data)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        batch = precompute_factors(runtime)

    runtime_warnings = [
        warning
        for warning in caught
        if issubclass(warning.category, RuntimeWarning)
    ]
    assert runtime_warnings == []
    date_index = runtime.decision_start
    factor_index = batch.factor_names.index("AmihudIlliquidity")
    assert batch.validity[date_index, factor_index, 0]
    assert batch.ranks[date_index, factor_index, 0] == 1.0
    assert batch.raw[date_index, factor_index, 0] == np.finfo(np.float32).max


def test_t_day_hlcva_and_future_values_do_not_change_t_factor_cache(tmp_path):
    base = _runtime_arrays()
    target_source_index = 145
    target_date = base["trade_dates"][target_source_index]
    changed = {name: values.copy() for name, values in base.items()}
    for field in ("high", "low", "close", "volume", "amount"):
        changed[field][target_source_index] *= 1000.0
    for field in ("open", "high", "low", "close", "volume", "amount", "total_share"):
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


def test_soft_filters_are_independent_and_star_st_falls_back_to_st(tmp_path):
    data = _runtime_arrays()
    target_source_index = 145
    data["open"][target_source_index, 0] = 1.99
    data["st_mask"][target_source_index, 1] = True
    runtime = _load(tmp_path, "runtime.npz", data)
    batch = precompute_factors(runtime)
    index = batch.index_of(data["trade_dates"][target_source_index])

    filters = batch.day(index).filters
    filter_index = {name: i for i, name in enumerate(batch.filter_names)}
    assert filters[filter_index["FilterLowPrice"]].tolist() == [False, True, True]
    assert filters[filter_index["FilterST"]].tolist() == [True, False, True]
    np.testing.assert_array_equal(
        filters[filter_index["FilterST"]],
        filters[filter_index["FilterStarST"]],
    )


def test_explicit_star_st_mask_is_used_when_present(tmp_path):
    data = _runtime_arrays()
    target_source_index = 145
    data["star_st_mask"] = np.zeros_like(data["st_mask"])
    data["star_st_mask"][target_source_index, 2] = True
    runtime = _load(tmp_path, "runtime.npz", data)
    batch = precompute_factors(runtime)
    index = batch.index_of(data["trade_dates"][target_source_index])
    filters = batch.day(index).filters
    filter_index = {name: i for i, name in enumerate(batch.filter_names)}

    assert filters[filter_index["FilterST"]].tolist() == [True, True, True]
    assert filters[filter_index["FilterStarST"]].tolist() == [True, True, False]


def test_real_2024_config_topn_matches_float64_legacy_ranking():
    from core.scoring import scores_to_ranks
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
    runtime = load_runtime_slice(path, "2024-01-02", "2024-12-31")
    universe = np.array(
        [
            code.startswith(("60", "00", "30", "688"))
            for code in runtime.stock_codes
        ],
        dtype=np.bool_,
    )
    batch = precompute_factors(runtime, rank_universe_mask=universe)
    universe_columns = np.flatnonzero(universe)

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

    legacy_ranks = [
        scores_to_ranks(raw[:, universe_columns]) for raw in legacy_raw
    ]
    weights = np.array([0.4, 0.9, 0.1, 0.6], dtype=np.float64)
    for date_index in range(runtime.decision_start, runtime.decision_stop):
        legacy_valid = np.logical_and.reduce(
            [np.isfinite(raw[date_index]) for raw in legacy_raw]
        )
        candidate_mask = (
            universe
            & legacy_valid
            & batch.validity[date_index].all(axis=0)
            & batch.filters[date_index].all(axis=0)
        )
        candidates = np.flatnonzero(candidate_mask)
        batch_score = (
            batch.ranks[date_index][:, candidates].T * weights
        ).sum(axis=1)
        local_candidates = np.searchsorted(universe_columns, candidates)
        legacy_score = np.stack(
            [ranks[date_index, local_candidates] for ranks in legacy_ranks],
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
            f"sell_m parity failed on {runtime.trade_dates[date_index]}"
        )
