from __future__ import annotations

import numpy as np
import pandas as pd

from factor_db.factors import DividendQuality as module
from factor_db.factors.DividendQuality import CompletedDividendYield252PIT
from factor_db.factors.DividendQuality import (
    FilterDividendPaidEachOfLast3YearsPIT,
    FilterDividendYieldTop50Positive252PIT,
)
from factor_db.factors.MarketCapFloorFilter import FilterMarketCapTop50Pct


def test_dividend_activates_strictly_after_known_date_and_units(monkeypatch):
    events = pd.DataFrame(
        {
            "security_code": ["600001"],
            "known_date": [pd.Timestamp("2012-01-03")],
            "cash_per_ten": [10.0],
            "event_total_shares": [100_000_000.0],
            "total_cash_yuan": [100_000_000.0],
        }
    )
    monkeypatch.setattr(module, "_implemented_cash_dividends", lambda: events)
    dates = np.arange(
        np.datetime64("2011-01-01"),
        np.datetime64("2012-01-06"),
        dtype="datetime64[D]",
    )
    rows = len(dates)
    panel = {
        "trade_dates": dates,
        "stock_codes": np.array(["600001.SH"]),
        "preClose": np.full((rows, 1), 10.0),
        "total_share": np.full((rows, 1), 10_000.0),
    }
    raw = CompletedDividendYield252PIT.raw_yield(panel)
    known = int(np.searchsorted(dates, np.datetime64("2012-01-03")))
    assert raw[known, 0] == 0.0
    assert raw[known + 1, 0] == 0.1


def test_top_half_market_cap_filter_excludes_smaller_half():
    panel = {
        "preClose": np.ones((1, 4)),
        "total_share": np.array([[1.0, 2.0, 3.0, 4.0]]),
        "stock_codes": np.array(
            ["600001.SH", "000001.SZ", "300001.SZ", "600002.SH"]
        ),
    }
    result = FilterMarketCapTop50Pct().calc_batch(panel)
    assert np.isnan(result[0, :2]).all()
    assert np.array_equal(result[0, 2:], np.ones(2, dtype=np.float32))


def test_dividend_filter_requires_positive_yield_and_keeps_upper_half(
    monkeypatch,
):
    raw = np.array([[0.0, 0.01, 0.02, 0.03, 0.04]], dtype=np.float64)
    monkeypatch.setattr(
        CompletedDividendYield252PIT,
        "raw_yield",
        classmethod(lambda cls, panel: raw),
    )
    result = FilterDividendYieldTop50Positive252PIT().calc_batch({})
    assert np.isnan(result[0, :3]).all()
    assert np.array_equal(result[0, 3:], np.ones(2, dtype=np.float32))


def test_three_year_consistency_requires_one_known_event_per_slot(monkeypatch):
    events = pd.DataFrame(
        {
            "security_code": ["600001"] * 3,
            "ex_date": pd.to_datetime(
                ["2011-07-16", "2012-08-19", "2013-09-23"]
            ),
            "known_date": pd.to_datetime(
                ["2011-07-16", "2012-08-19", "2013-09-23"]
            ),
            "cash_per_ten": [1.0] * 3,
            "event_total_shares": [1.0] * 3,
            "total_cash_yuan": [0.1] * 3,
        }
    )
    monkeypatch.setattr(module, "_implemented_cash_dividends", lambda: events)
    dates = np.arange(
        np.datetime64("2010-01-01"),
        np.datetime64("2014-01-03"),
        dtype="datetime64[D]",
    )
    result = FilterDividendPaidEachOfLast3YearsPIT().calc_batch(
        {"trade_dates": dates, "stock_codes": np.array(["600001.SH"])}
    )
    target = int(np.searchsorted(dates, np.datetime64("2014-01-01")))
    assert result[target, 0] == 1.0
