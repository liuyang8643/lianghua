from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data.financial_pit import (
    build_pit_source_indices,
    materialize_pit_field,
    statutory_disclosure_deadlines,
)


def test_statutory_deadlines_cover_four_standard_report_periods():
    periods = pd.Series([20230331, 20230630, 20230930, 20231231])

    actual = statutory_disclosure_deadlines(periods)

    assert actual.tolist() == [
        np.datetime64("2023-04-30"),
        np.datetime64("2023-08-31"),
        np.datetime64("2023-10-31"),
        np.datetime64("2024-04-30"),
    ]


def test_nonstandard_report_period_fails_closed():
    with pytest.raises(ValueError, match="unexpected financial report periods"):
        statutory_disclosure_deadlines(pd.Series([20230531]))


def test_nonstandard_rows_are_excluded_from_pit_panel():
    rows = pd.DataFrame(
        {
            "stock_code": ["600000.SH", "600000.SH"],
            "report_period": [20230531, 20230630],
            "roe": [99.0, 12.0],
        }
    )
    dates = np.array(
        ["2023-06-01", "2023-09-01"],
        dtype="datetime64[D]",
    )

    source = build_pit_source_indices(
        rows,
        np.array(["600000.SH"]),
        dates,
    )
    actual = materialize_pit_field(rows, source, "roe")

    assert np.isnan(actual[0, 0])
    assert actual[1, 0] == 12.0


def test_pit_starts_after_deadline_and_newer_period_wins_same_deadline():
    rows = pd.DataFrame(
        {
            "stock_code": ["600000.SH", "600000.SH"],
            "report_period": [20221231, 20230331],
            "roe": [10.0, 20.0],
        }
    )
    dates = np.array(
        ["2023-04-28", "2023-05-04", "2023-05-05"],
        dtype="datetime64[D]",
    )
    source = build_pit_source_indices(
        rows,
        np.array(["600000.SH"]),
        dates,
    )

    actual = materialize_pit_field(rows, source, "roe")

    assert np.isnan(actual[0, 0])
    assert actual[1:, 0].tolist() == [20.0, 20.0]


def test_missing_value_in_new_report_does_not_reuse_old_report_field():
    rows = pd.DataFrame(
        {
            "stock_code": ["600000.SH", "600000.SH"],
            "report_period": [20230331, 20230630],
            "roe": [12.0, np.nan],
            "gross_margin": [30.0, 35.0],
        }
    )
    dates = np.array(
        ["2023-05-04", "2023-08-31", "2023-09-01"],
        dtype="datetime64[D]",
    )
    source = build_pit_source_indices(
        rows,
        np.array(["600000.SH"]),
        dates,
    )

    roe = materialize_pit_field(rows, source, "roe")
    gross_margin = materialize_pit_field(
        rows,
        source,
        "gross_margin",
    )

    assert roe[:, 0].tolist()[:2] == [12.0, 12.0]
    assert np.isnan(roe[2, 0])
    assert gross_margin[:, 0].tolist() == [30.0, 30.0, 35.0]
