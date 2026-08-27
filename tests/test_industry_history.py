from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data.update_industry_history import (
    build_industry_panel,
    normalize_industry_history,
)


def _memberships() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "stock_code": "000001.SZ",
                "start_date": "2020-01-02",
                "industry_code": "480101",
                "update_date": "2099-12-31",
            },
            {
                "stock_code": "000001.SZ",
                "start_date": "2020-01-06",
                "industry_code": "480301",
                "update_date": "2000-01-01",
            },
            {
                "stock_code": "600000.SH",
                "start_date": "2019-12-31",
                "industry_code": "440101",
                "update_date": "2020-01-01",
            },
        ]
    )


def test_event_activates_only_on_first_trade_strictly_after_start_date():
    dates = np.array(
        ["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07"],
        dtype="datetime64[D]",
    )

    panel = build_industry_panel(
        dates,
        ["000001.SZ", "600000.SH", "300001.SZ"],
        _memberships(),
        level="L3",
    )

    assert panel.dtype == np.int32
    np.testing.assert_array_equal(
        panel,
        np.array(
            [
                [-1, 440101, -1],
                [480101, 440101, -1],
                [480101, 440101, -1],
                [480301, 440101, -1],
            ],
            dtype=np.int32,
        ),
    )


@pytest.mark.parametrize(
    ("level", "expected_before", "expected_after"),
    [
        (1, 480000, 480000),
        ("L2", 480100, 480300),
        ("l3", 480101, 480301),
    ],
)
def test_levels_use_official_six_digit_padded_codes(
    level, expected_before, expected_after
):
    dates = np.array(["2020-01-03", "2020-01-07"], dtype="datetime64[D]")

    panel = build_industry_panel(
        dates, ["000001.SZ"], _memberships(), level=level
    )

    np.testing.assert_array_equal(
        panel[:, 0], np.array([expected_before, expected_after], dtype=np.int32)
    )


def test_update_date_never_controls_availability():
    events = _memberships().copy()
    dates = np.array(["2020-01-02", "2020-01-03"], dtype="datetime64[D]")

    first = build_industry_panel(dates, ["000001.SZ"], events)
    events["update_date"] = ["1900-01-01", "2200-01-01", "1800-01-01"]
    second = build_industry_panel(dates, ["000001.SZ"], events)

    np.testing.assert_array_equal(first, second)


def test_panel_builder_is_offline_and_does_not_require_saved_artifacts(
    monkeypatch, tmp_path
):
    import requests

    def reject_network(*args, **kwargs):
        raise AssertionError("offline panel builder attempted network access")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(requests, "get", reject_network)

    panel = build_industry_panel(
        np.array(["2020-01-03"], dtype="datetime64[D]"),
        ["000001.SZ"],
        _memberships(),
    )

    assert panel.tolist() == [[480101]]


def test_normalization_requires_official_schema_and_normalizes_codes():
    source = pd.DataFrame(
        {
            "股票代码": [1, "600000", "832317"],
            "计入日期": ["1991-04-03", "1999-11-10", "2024-01-01"],
            "行业代码": [480101, "440101.SI", "330101"],
            "更新日期": ["2021-07-30", "2021-07-30", "2024-01-01"],
        }
    )

    normalized = normalize_industry_history(source)

    assert normalized["stock_code"].tolist() == ["000001.SZ", "600000.SH"]
    assert normalized["l1_code"].tolist() == ["480000", "440000"]
    assert normalized["l2_code"].tolist() == ["480100", "440100"]
    assert normalized["l3_code"].tolist() == ["480101", "440101"]
    assert normalized.attrs["excluded_non_sh_sz_rows"] == 1
    with pytest.raises(RuntimeError, match="missing"):
        normalize_industry_history(source.drop(columns="计入日期"))


def test_panel_rejects_non_monotonic_dates_and_duplicate_events():
    with pytest.raises(ValueError, match="strictly increasing"):
        build_industry_panel(
            np.array(["2020-01-03", "2020-01-02"], dtype="datetime64[D]"),
            ["000001.SZ"],
            _memberships(),
        )

    duplicates = pd.concat([_memberships(), _memberships().iloc[[0]]])
    with pytest.raises(ValueError, match="duplicate"):
        build_industry_panel(
            np.array(["2020-01-03"], dtype="datetime64[D]"),
            ["000001.SZ"],
            duplicates,
        )
