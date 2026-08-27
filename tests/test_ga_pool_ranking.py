from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

import testback.run_ga as run_ga
from core.backtest import _backtest_direct, _compute_factor_scores
from core.scoring import scores_to_ranks
from testback.run_ga import (
    FACTOR_RANKING_VERSION,
    _build_run_metadata,
    _factor_worker,
    _resolve_profile_ranking_pool,
    _validate_config_ranking_pool,
    _validate_resume_metadata,
    _worker_evaluate,
)


class _RawPanelFactor:
    hist_days = 0

    def calc_batch(self, data):
        return np.asarray(data["factor_raw"], dtype=np.float64)


def _run_factor_worker(tmp_path: Path, raw: np.ndarray, rank_cols: list[int]):
    tmp_path.mkdir()
    raw_path = tmp_path / "raw.bin"
    raw.tofile(raw_path)
    base_info = {
        "factor_raw": (str(raw_path), raw.shape, str(raw.dtype)),
    }
    codes = [f"code_{idx}" for idx in range(raw.shape[1])]
    dates = [datetime(2024, 1, 2)]
    name, entry, raw_valid = _factor_worker((
        _RawPanelFactor,
        base_info,
        codes,
        dates,
        str(tmp_path),
        None,
        False,
        np.asarray(rank_cols, dtype=np.intp),
    ))
    path, shape, dtype_str = entry
    values = np.memmap(
        path, dtype=np.dtype(dtype_str), mode="r", shape=shape,
    ).copy()
    return name, values, raw_valid


def test_outside_extreme_and_appended_code_do_not_change_pool_ranks(tmp_path):
    pool_raw = np.asarray([[10.0, 30.0, 20.0]], dtype=np.float64)
    with_outside = np.column_stack([pool_raw, np.asarray([[1.0e30]])])
    with_appended = np.column_stack([
        pool_raw,
        np.asarray([[1.0e30, -1.0e30]]),
    ])

    _, base_values, _ = _run_factor_worker(
        tmp_path / "base", pool_raw, [0, 1, 2],
    )
    _, outside_values, _ = _run_factor_worker(
        tmp_path / "outside", with_outside, [0, 1, 2],
    )
    _, appended_values, _ = _run_factor_worker(
        tmp_path / "appended", with_appended, [0, 1, 2],
    )

    expected = scores_to_ranks(pool_raw.astype(np.float32))
    np.testing.assert_array_equal(base_values[:, :3], expected)
    np.testing.assert_array_equal(outside_values[:, :3], expected)
    np.testing.assert_array_equal(appended_values[:, :3], expected)
    np.testing.assert_array_equal(outside_values[:, 3:], 0.0)
    np.testing.assert_array_equal(appended_values[:, 3:], 0.0)


def test_ga_factor_and_single_backtest_paths_match_for_same_late_period(
    tmp_path, monkeypatch,
):
    codes = np.asarray([
        "600001.SH",
        "000001.SZ",
        "300001.SZ",
        "688001.SH",
    ])
    trade_dates = np.asarray(
        ["2024-01-02", "2024-01-03", "2024-01-04"],
        dtype="datetime64[D]",
    )
    raw = np.asarray([
        [1.0, 2.0, 3.0, 1.0e30],
        [3.0, 2.0, 1.0, -1.0e30],
        [1.0, 3.0, 2.0, 1.0e30],
    ])
    shape = raw.shape
    data = {
        "stock_codes": codes,
        "trade_dates": trade_dates,
        "factor_raw": raw,
        "open": np.full(shape, 10.0),
        "high": np.full(shape, 10.1),
        "low": np.full(shape, 9.9),
        "close": np.asarray([
            [10.0, 10.0, 10.0, 10.0],
            [10.3, 10.2, 10.1, 10.0],
            [10.1, 10.4, 10.2, 10.0],
        ]),
        "preClose": np.full(shape, 10.0),
        "volume": np.full(shape, 1_000_000.0),
        "amount": np.full(shape, 10_000_000.0),
        "total_share": np.full(shape, 100_000_000.0),
        "st_mask": np.zeros(shape, dtype=bool),
        "issue_price": np.full(shape[1], 10.0),
    }
    later_dates = [datetime(2024, 1, 3), datetime(2024, 1, 4)]
    pool_stocks = codes[:3].tolist()
    stock_indices = {str(code): idx for idx, code in enumerate(codes)}
    rank_cols = [stock_indices[code] for code in pool_stocks]

    base_info = {}
    ga_dir = tmp_path / "ga"
    ga_dir.mkdir()
    for key, value in data.items():
        if key in {"stock_codes", "trade_dates"}:
            continue
        path = ga_dir / f"{key}.bin"
        np.asarray(value).tofile(path)
        base_info[key] = (
            str(path), np.asarray(value).shape, str(np.asarray(value).dtype),
        )
    factor_name, entry, _ = _factor_worker((
        _RawPanelFactor,
        base_info,
        codes.tolist(),
        [value.astype("datetime64[D]").item() for value in trade_dates],
        str(ga_dir),
        (1, 3),
        False,
        np.asarray(rank_cols, dtype=np.intp),
    ))
    ga_path, ga_shape, ga_dtype = entry
    ga_scores = np.memmap(
        ga_path, dtype=np.dtype(ga_dtype), mode="r", shape=ga_shape,
    ).copy()

    single_result = _compute_factor_scores(
        later_dates,
        pool_stocks,
        weights={factor_name: 1.0},
        factor_classes=[_RawPanelFactor],
        data=data,
    )
    assert single_result is not None
    (
        _,
        single_scores,
        _,
        valid_dates,
        date_indices,
        valid_stocks,
        single_stock_indices,
    ) = single_result

    np.testing.assert_array_equal(
        ga_scores[1:3],
        single_scores[factor_name][1:3],
    )
    monkeypatch.setattr("core.backtest.get_delist_stock_info", lambda: {})
    common = {
        "data": data,
        "valid_dates": valid_dates,
        "date_indices": date_indices,
        "valid_stocks": valid_stocks,
        "weights": {factor_name: 1.0},
        "buy_n": 1,
        "sell_m": 1,
        "lightweight": True,
        "market_order_freeze": False,
        "list_dates_map": {},
    }
    ga_result = _backtest_direct(
        all_scores={factor_name: ga_scores},
        stock_indices=stock_indices,
        **common,
    )
    single_backtest_result = _backtest_direct(
        all_scores=single_scores,
        stock_indices=single_stock_indices,
        **common,
    )

    assert ga_result["daily_topn"] == single_backtest_result["daily_topn"]
    np.testing.assert_array_equal(
        ga_result["daily_returns"],
        single_backtest_result["daily_returns"],
    )


def test_multi_pool_profile_fails_closed_before_pre_ranking(monkeypatch):
    monkeypatch.setattr(
        run_ga,
        "get_profile",
        lambda _name: {
            "fixed_parameters": {},
            "search_spaces": {
                "stock_pool": [
                    ("60", "00", "30"),
                    ("60", "00", "30", "688"),
                ],
            },
        },
    )

    with pytest.raises(ValueError, match="会搜索多个 stock_pool"):
        _resolve_profile_ranking_pool("dynamic_pool")


def test_candidate_cannot_reuse_ranks_from_a_different_pool():
    with pytest.raises(ValueError, match="禁止复用错误因子排名"):
        _validate_config_ranking_pool(
            {"stock_pool": ["60", "00", "30", "688"]},
            ("60", "00", "30"),
        )

    worker_args = (
        {"_ranking_pool_prefixes": ("60", "00", "30")},
        set(),
        [],
        [],
        {},
        [],
        {"stock_pool": ["60", "00", "30", "688"]},
        {},
        {},
        {},
    )
    with pytest.raises(ValueError, match="禁止复用错误因子排名"):
        _worker_evaluate(worker_args)


def test_resume_metadata_binds_cache_to_ranking_scope_and_pool():
    common = {
        "profile_name": "fixed_pool",
        "seed": 7,
        "sealed_holdout": False,
        "split_period_results": False,
        "training_objective": {},
    }
    metadata = _build_run_metadata(
        **common,
        ranking_pool=("60", "00", "30"),
    )

    assert metadata["factor_ranking_version"] == FACTOR_RANKING_VERSION
    assert metadata["factor_ranking_pool"] == ["00", "30", "60"]
    _validate_resume_metadata(
        metadata,
        **common,
        ranking_pool=("30", "60", "00"),
    )

    legacy_metadata = dict(metadata)
    legacy_metadata.pop("factor_ranking_version")
    with pytest.raises(ValueError, match="因子排名口径"):
        _validate_resume_metadata(
            legacy_metadata,
            **common,
            ranking_pool=("60", "00", "30"),
        )

    with pytest.raises(ValueError, match="factor_ranking_pool"):
        _validate_resume_metadata(
            metadata,
            **common,
            ranking_pool=("60", "00", "30", "688"),
        )
