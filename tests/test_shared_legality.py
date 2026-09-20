from __future__ import annotations

import numpy as np
import pytest

from env.legality import (
    classify_board_types,
    evaluate_buy_legality_mask,
    evaluate_trade_legality,
)
from env.action_schema import CORE_FACTOR_NAMES, CORE_FILTER_NAMES
from env.contracts import DayConfig
from env.planner import DayMarketData, DayPlanner
from env.quantity import buy_quantity_step, minimum_buy_quantity


def _config(*, limit_up_protection: bool = False) -> DayConfig:
    return DayConfig(
        factor_weights={
            name: 1.0 if index == 0 else 0.0
            for index, name in enumerate(CORE_FACTOR_NAMES)
        },
        factor_enabled={
            name: index == 0 for index, name in enumerate(CORE_FACTOR_NAMES)
        },
        filter_flags={name: False for name in CORE_FILTER_NAMES},
        buy_n=1,
        turnover_rate=1.0,
        limit_up_protection=limit_up_protection,
        rebalance_band_pct=0.0,
        single_buy_pct=1.0,
    )


def _market(
    *,
    decision_date: str,
    codes: tuple[str, ...],
    ages: np.ndarray,
    opens: np.ndarray,
    precloses: np.ndarray,
    issues: np.ndarray,
    st: np.ndarray,
) -> DayMarketData:
    size = len(codes)
    return DayMarketData(
        decision_date=decision_date,
        stock_codes=codes,
        factor_ranks={name: np.ones(size) for name in CORE_FACTOR_NAMES},
        factor_validity={
            name: np.ones(size, dtype=np.bool_) for name in CORE_FACTOR_NAMES
        },
        filter_masks={
            name: np.ones(size, dtype=np.bool_) for name in CORE_FILTER_NAMES
        },
        open_prices=opens,
        preclose_prices=precloses,
        issue_prices=issues,
        st_mask=st,
        delisted_mask=np.zeros(size, dtype=np.bool_),
        listing_age=ages,
    )


def test_trade_legality_returns_vector_masks_and_per_stock_reasons() -> None:
    codes = tuple(f"60000{index}.SH" for index in range(6))
    result = evaluate_trade_legality(
        decision_date="2026-08-20",
        stock_codes=codes,
        listing_age=np.asarray((10, 10, 10, 10, 10, -1), dtype=np.int32),
        open_prices=np.asarray((10.0, 11.0, 9.0, 10.0, np.nan, 10.0)),
        preclose_prices=np.asarray((10.0, 10.0, 10.0, np.nan, 10.0, 10.0)),
        issue_prices=np.full(6, 8.0),
        st_mask=np.zeros(6, dtype=np.bool_),
        delisted_mask=np.zeros(6, dtype=np.bool_),
        limit_up_protection=True,
    )

    np.testing.assert_array_equal(
        result.buy_allowed,
        np.asarray((True, False, True, False, False, False)),
    )
    np.testing.assert_array_equal(
        result.sell_allowed,
        np.asarray((True, False, False, False, False, False)),
    )
    assert result.buy_reasons.tolist() == [
        "ok",
        "limit_up",
        "ok",
        "missing_preclose",
        "suspended_or_missing_open",
        "not_listed",
    ]
    assert result.sell_reasons.tolist() == [
        "ok",
        "limit_up_protected",
        "limit_down",
        "missing_preclose",
        "suspended_or_missing_open",
        "not_listed",
    ]


def test_delisted_stock_with_stale_open_is_never_tradeable() -> None:
    result = evaluate_trade_legality(
        decision_date="2026-08-20",
        stock_codes=("600001.SH",),
        listing_age=np.asarray((100,), dtype=np.int32),
        open_prices=np.asarray((10.0,)),
        preclose_prices=np.asarray((10.0,)),
        issue_prices=np.asarray((8.0,)),
        st_mask=np.asarray((False,)),
        delisted_mask=np.asarray((True,)),
    )

    assert result.buy_allowed.tolist() == [False]
    assert result.sell_allowed.tolist() == [False]
    assert result.buy_reasons.tolist() == ["delisted"]
    assert result.sell_reasons.tolist() == ["delisted"]


def test_chunked_buy_mask_matches_complete_panel_without_diagnostics() -> None:
    codes = ("600000.SH", "300001.SZ", "688001.SH")
    dates = np.asarray(("2020-08-21", "2020-08-24", "2024-01-02"), dtype="datetime64[D]")
    opens = np.asarray(
        (
            (10.0, 10.5, 12.0),
            (11.0, 10.5, 20.0),
            (10.0, np.nan, 12.0),
        )
    )
    ages = np.asarray(((10, 10, 5), (10, 10, 4), (10, -1, 5)), dtype=np.int32)
    precloses = np.full(opens.shape, 10.0)
    issues = np.full(len(codes), 8.0)
    st = np.zeros(opens.shape, dtype=np.bool_)
    complete = evaluate_trade_legality(
        decision_date=dates,
        stock_codes=codes,
        listing_age=ages,
        open_prices=opens,
        preclose_prices=precloses,
        issue_prices=issues,
        st_mask=st,
        delisted_mask=np.zeros(opens.shape, dtype=np.bool_),
    )
    chunked = evaluate_buy_legality_mask(
        decision_date=dates,
        stock_codes=codes,
        listing_age=ages,
        open_prices=opens,
        preclose_prices=precloses,
        issue_prices=issues,
        st_mask=st,
        delisted_mask=np.zeros(opens.shape, dtype=np.bool_),
        chunk_rows=1,
    )
    lightweight = evaluate_trade_legality(
        decision_date=dates[0],
        stock_codes=codes,
        listing_age=ages[0],
        open_prices=opens[0],
        preclose_prices=precloses[0],
        issue_prices=issues,
        st_mask=st[0],
        delisted_mask=np.zeros(len(codes), dtype=np.bool_),
        diagnostics=False,
    )

    np.testing.assert_array_equal(chunked, complete.buy_allowed)
    assert lightweight.sell_allowed is None
    assert lightweight.buy_reasons is None
    assert lightweight.up_limit_prices is None


@pytest.mark.parametrize(
    ("decision_date", "code", "age", "open_price", "preclose", "issue", "st", "buy", "reason"),
    (
        ("2017-01-13", "603690.SH", 0, 12.0, np.nan, 10.0, False, False, "ipo_open_limit"),
        ("2017-01-13", "603690.SH", 0, 11.0, 10.0, np.nan, False, True, "ok"),
        ("2023-04-10", "603690.SH", 0, 30.0, np.nan, 10.0, False, True, "ok_no_daily_limit"),
        ("2020-08-21", "300001.SZ", 10, 10.5, 10.0, 8.0, True, False, "limit_up"),
        ("2020-08-24", "300001.SZ", 10, 10.5, 10.0, 8.0, True, True, "ok"),
        ("2024-01-02", "688001.SH", 4, 20.0, 10.0, 8.0, False, True, "ok_no_daily_limit"),
        ("2024-01-02", "688001.SH", 5, 12.0, 10.0, 8.0, False, False, "limit_up"),
        ("2024-01-02", "430001.BJ", 0, 20.0, 10.0, 8.0, False, True, "ok_no_daily_limit"),
        ("2024-01-02", "430001.BJ", 1, 13.0, 10.0, 8.0, False, False, "limit_up"),
        ("2013-12-31", "600000.SH", 0, 20.0, 10.0, 8.0, False, True, "ok_no_daily_limit"),
        ("2026-07-03", "600000.SH", 10, 10.5, 10.0, 8.0, True, False, "limit_up"),
        ("2026-07-06", "600000.SH", 10, 10.5, 10.0, 8.0, True, True, "ok"),
    ),
)
def test_board_ipo_st_and_regime_boundaries(
    decision_date,
    code,
    age,
    open_price,
    preclose,
    issue,
    st,
    buy,
    reason,
) -> None:
    result = evaluate_trade_legality(
        decision_date=decision_date,
        stock_codes=(code,),
        listing_age=np.asarray((age,), dtype=np.int32),
        open_prices=np.asarray((open_price,)),
        preclose_prices=np.asarray((preclose,)),
        issue_prices=np.asarray((issue,)),
        st_mask=np.asarray((st,)),
        delisted_mask=np.asarray((False,)),
    )

    assert bool(result.buy_allowed[0]) is buy
    assert result.buy_reasons[0] == reason


def test_689_cdr_codes_use_star_market_limits_and_freeze_price() -> None:
    np.testing.assert_array_equal(
        classify_board_types(("688981.SH", "689009.SH")),
        (2, 2),
    )
    result = evaluate_trade_legality(
        decision_date="2024-04-25",
        stock_codes=("689009.SH", "689009.SH"),
        listing_age=np.asarray((4, 5), dtype=np.int32),
        open_prices=np.asarray((50.0, 35.18)),
        preclose_prices=np.asarray((31.81, 31.81)),
        issue_prices=np.asarray((20.0, 20.0)),
        st_mask=np.asarray((False, False)),
        delisted_mask=np.asarray((False, False)),
    )

    np.testing.assert_array_equal(result.buy_allowed, (True, True))
    assert result.buy_reasons.tolist() == ["ok_no_daily_limit", "ok"]
    assert DayPlanner._freeze_price(35.18, 31.81, ratio=0.2) == pytest.approx(
        31.81 * 1.20
    )
    assert minimum_buy_quantity("689009.SH") == 200
    assert buy_quantity_step("689009.SH") == 1


def test_beijing_exchange_uses_100_share_minimum_and_one_share_step() -> None:
    assert minimum_buy_quantity("430001.BJ") == 100
    assert buy_quantity_step("430001.BJ") == 1


def test_conservative_cent_rounding_and_epsilon_boundary() -> None:
    common = {
        "decision_date": "2024-01-02",
        "stock_codes": ("600000.SH", "600001.SH"),
        "listing_age": np.asarray((10, 10), dtype=np.int32),
        "preclose_prices": np.asarray((10.05, 10.05)),
        "issue_prices": np.asarray((8.0, 8.0)),
        "st_mask": np.zeros(2, dtype=np.bool_),
        "delisted_mask": np.zeros(2, dtype=np.bool_),
    }
    result = evaluate_trade_legality(
        **common,
        open_prices=np.asarray((11.0491, 11.0489)),
    )

    np.testing.assert_allclose(result.up_limit_prices, (11.05, 11.05))
    np.testing.assert_array_equal(result.buy_allowed, (False, True))


def test_planner_delegates_to_identical_row_legality() -> None:
    codes = ("600000.SH", "300001.SZ", "688001.SH", "430001.BJ")
    dates = np.arange("2024-01-02", "2024-01-10", dtype="datetime64[D]")
    shape = (len(dates), len(codes))
    trade_idx = 5
    opens = np.full(shape, np.nan)
    opens[0:, 0] = 10.0
    opens[1:, 1] = 10.0
    opens[5:, 2] = 20.0
    opens[4:, 3] = 13.0
    precloses = np.full(shape, 10.0)
    issue = np.asarray((8.0, 8.0, 8.0, 8.0))
    st = np.zeros(shape, dtype=np.bool_)
    ages = np.asarray((5, 4, 0, 1), dtype=np.int32)
    market = _market(
        decision_date=str(dates[trade_idx]),
        codes=codes,
        ages=ages,
        opens=opens[trade_idx],
        precloses=precloses[trade_idx],
        issues=issue,
        st=st[trade_idx],
    )
    config = _config(limit_up_protection=True)
    canonical = evaluate_trade_legality(
        decision_date=str(dates[trade_idx]),
        stock_codes=codes,
        listing_age=ages,
        open_prices=opens[trade_idx],
        preclose_prices=precloses[trade_idx],
        issue_prices=issue,
        st_mask=st[trade_idx],
        delisted_mask=np.zeros(len(codes), dtype=np.bool_),
        limit_up_protection=True,
        precomputed_board_types=classify_board_types(codes),
    )
    actual = market.trade_legality(config.limit_up_protection)
    np.testing.assert_array_equal(actual.buy_allowed, canonical.buy_allowed)
    np.testing.assert_array_equal(actual.sell_allowed, canonical.sell_allowed)
    np.testing.assert_array_equal(actual.buy_reason_codes, canonical.buy_reason_codes)
    np.testing.assert_array_equal(actual.sell_reason_codes, canonical.sell_reason_codes)
