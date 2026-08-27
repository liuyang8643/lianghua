from datetime import datetime
import json

import numpy as np

import testback.run_ga as run_ga
from core.ga import build_individual_config, get_profile
from core.scoring import FactorScoreMatrices
from core.strategy_config import load_strategy_config
from testback.run_ga import (
    SELECTION_SLEEVES_VERSION,
    _build_run_metadata,
    _validate_resume_metadata,
)


def test_v48_profile_is_a_fixed_single_change_from_v31():
    profile = get_profile("v48_fixed_core24_q60trend6")

    assert [factor.__name__ for factor in profile["factor_classes"]] == [
        "PreCloseMarketCap",
        "AmihudIlliquidityStrict",
        "TrendReversalPreCloseStrict",
        "VolumeCVStrict",
        "CompletedTrendConsistency60Strict",
    ]
    assert profile["weight_search_spaces"] is None
    assert profile["fixed_weights"] == {
        "PreCloseMarketCap": 1.0,
        "AmihudIlliquidityStrict": 0.4,
        "TrendReversalPreCloseStrict": 0.1,
        "VolumeCVStrict": 0.2,
        "CompletedTrendConsistency60Strict": 0.0,
    }
    assert profile["search_spaces"] == {"buy_n": [30]}
    assert profile["fixed_parameters"]["selection_sleeves"] == [
        {
            "name": "v31_core",
            "slots": 24,
            "weights": {
                "PreCloseMarketCap": 1.0,
                "AmihudIlliquidityStrict": 0.4,
                "TrendReversalPreCloseStrict": 0.1,
                "VolumeCVStrict": 0.2,
            },
        },
        {
            "name": "positive_q60_trend",
            "slots": 6,
            "weights": {"CompletedTrendConsistency60Strict": 1.0},
        },
    ]

    config = build_individual_config(
        buy_n=30,
        profile_name="v48_fixed_core24_q60trend6",
    )
    assert config["selection_sleeves"] == profile["fixed_parameters"][
        "selection_sleeves"
    ]
    assert config["sell_m"] == 30


def test_strategy_config_loads_factors_referenced_only_by_sleeves(tmp_path):
    path = tmp_path / "sleeve.json"
    path.write_text(
        json.dumps(
            {
                "ga_profile": "v48_fixed_core24_q60trend6",
                "individual_config": {
                    "weights": {"PreCloseMarketCap": 1.0},
                    "buy_n": 2,
                    "selection_sleeves": [
                        {
                            "name": "core",
                            "slots": 1,
                            "weights": {"PreCloseMarketCap": 1.0},
                        },
                        {
                            "name": "trend",
                            "slots": 1,
                            "weights": {
                                "CompletedTrendConsistency60Strict": 1.0
                            },
                        },
                    ],
                },
            }
        ),
        encoding="utf-8",
    )

    loaded = load_strategy_config(path)

    assert loaded["factor_names"] == [
        "PreCloseMarketCap",
        "CompletedTrendConsistency60Strict",
    ]


def test_ga_worker_keeps_sleeve_validity_out_of_common_filter(
    monkeypatch,
):
    rows, stocks = 3, 3
    matrices = {
        "Core": np.asarray(
            [[1.0, 0.5, 0.0], [1.0, 0.5, 0.0], [1.0, 0.5, 0.0]],
            dtype=np.float32,
        ),
        "Trend": np.asarray(
            [[0.0, 1.0, 0.5], [0.0, 1.0, 0.5], [0.0, 1.0, 0.5]],
            dtype=np.float32,
        ),
        "_factor_valid_Core": np.ones((rows, stocks), dtype=bool),
        "_factor_valid_Trend": np.asarray(
            [[False, True, False]] * rows,
            dtype=bool,
        ),
        "FilterST": np.ones((rows, stocks), dtype=bool),
    }
    for name in (
        "open",
        "close",
        "high",
        "low",
        "preClose",
        "volume",
        "amount",
        "total_share",
    ):
        matrices[name] = np.full((rows, stocks), 10.0)
    matrices["st_mask"] = np.zeros((rows, stocks), dtype=bool)
    matrices["issue_price"] = np.full(stocks, 10.0)
    matrices["stock_codes"] = np.asarray(
        ["600001.SH", "000001.SZ", "300001.SZ"]
    )
    matrices["trade_dates"] = np.asarray(
        ["2010-01-04", "2010-01-05", "2010-01-06"],
        dtype="datetime64[D]",
    )
    matrices["_market_open_index"] = np.ones(rows)
    monkeypatch.setattr(
        run_ga,
        "_worker_shm_cache",
        {name: (None, values) for name, values in matrices.items()},
    )
    monkeypatch.setattr(
        run_ga,
        "_compute_timing_multipliers",
        lambda *args, **kwargs: None,
    )
    timing_captured = {}

    def fake_timing(**kwargs):
        timing_captured["filter_masks"] = kwargs["filter_masks"]
        return None

    monkeypatch.setattr(
        "core.trend_timing.compute_configured_timing_multipliers",
        fake_timing,
    )

    captured = {}

    def fake_backtest(*args, **kwargs):
        captured["scores"] = args[1]
        captured["filter_masks"] = kwargs["filter_masks"]
        captured["selection_sleeves"] = kwargs["selection_sleeves"]
        return {
            "daily_returns": [0.1, 0.2, 0.3],
            "daily_exposures": [0.5, 0.5, 0.5],
            "total_return": 0.6,
            "cleared_positions_count": 0,
        }

    monkeypatch.setattr(run_ga, "_backtest_direct", fake_backtest)
    dates = [
        datetime(2010, 1, 4),
        datetime(2010, 1, 5),
        datetime(2010, 1, 6),
    ]
    sleeves = [
        {"name": "core", "slots": 2, "weights": {"Core": 1.0}},
        {"name": "trend", "slots": 1, "weights": {"Trend": 1.0}},
    ]
    config = {
        "weights": {"Core": 1.0, "Trend": 0.0},
        "buy_n": 3,
        "sell_m": 3,
        "stock_pool": ["60", "00", "30"],
        "holding_period": 1,
        "filter_factors": {"FilterST": True},
        "selection_sleeves": sleeves,
    }

    result = run_ga._worker_evaluate(
        (
            {"_ranking_pool_prefixes": ("60", "00", "30")},
            {"Core", "Trend"},
            dates,
            [0, 1, 2],
            {
                "600001.SH": 0,
                "000001.SZ": 1,
                "300001.SZ": 2,
            },
            ["600001.SH", "000001.SZ", "300001.SZ"],
            config,
            {},
            {},
            {"mode": "calmar"},
        )
    )

    assert isinstance(captured["scores"], FactorScoreMatrices)
    assert captured["scores"].factor_validity["Trend"].tolist() == [
        [False, True, False],
        [False, True, False],
        [False, True, False],
    ]
    assert set(captured["filter_masks"]) == {"FilterST"}
    assert set(timing_captured["filter_masks"]) == {
        "FilterST",
        "_active_factor_intersection",
    }
    assert timing_captured["filter_masks"][
        "_active_factor_intersection"
    ].all()
    assert captured["selection_sleeves"] == sleeves
    assert result["individual_config"] == config


def test_ga_resume_metadata_binds_fixed_sleeve_semantics():
    sleeves = get_profile("v48_fixed_core24_q60trend6")[
        "fixed_parameters"
    ]["selection_sleeves"]
    common = {
        "profile_name": "v48_fixed_core24_q60trend6",
        "seed": 20260727,
        "sealed_holdout": True,
        "split_period_results": False,
        "training_objective": get_profile(
            "v48_fixed_core24_q60trend6"
        )["training_objective"],
        "ranking_pool": ("60", "00", "30"),
        "selection_sleeves": sleeves,
    }
    metadata = _build_run_metadata(**common)

    assert metadata["selection_sleeves_version"] == (
        SELECTION_SLEEVES_VERSION
    )
    assert metadata["selection_sleeves"] == sleeves
    _validate_resume_metadata(metadata, **common)

    changed = [dict(sleeve) for sleeve in sleeves]
    changed[0] = {**changed[0], "slots": 23}
    try:
        _validate_resume_metadata(
            metadata,
            **{**common, "selection_sleeves": changed},
        )
    except ValueError as exc:
        assert "selection_sleeves" in str(exc)
    else:
        raise AssertionError("changed sleeve semantics must fail resume")
