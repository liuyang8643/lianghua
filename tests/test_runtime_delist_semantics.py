from datetime import date

import numpy as np
import pandas as pd
import pytest

from data.build_runtime import (
    KlineDailyEvidenceError,
    ListingDateAlignmentError,
    apply_baostock_daily_evidence,
    build_delisted_mask,
    build_listing_age,
    sanitize_first_bar_preclose,
    validate_listing_date_alignment,
    validate_delisted_kline_coverage,
    validate_delisted_kline_panel,
)
from data.db.delist import DelistStockInfo
from data.kline_mootdx import (
    BAOSTOCK_EVIDENCE_COLUMNS,
    BAOSTOCK_EVIDENCE_SCHEMA,
    UNAPPLIED_K_SHA256,
    _source_payload_sha256,
    _write_baostock_evidence_stage,
)
from env.legality import evaluate_trade_legality


def _write_baostock_evidence(
    path,
    *,
    code="000001.SZ",
    dates=("2020-01-02", "2020-01-03", "2020-01-06"),
    statuses=("0", "1", "1"),
    reference_opens=(10.0, 10.0, 11.0),
    listing_date="2020-01-02",
    out_date="2020-01-06",
    normalization_applied=None,
):
    executable = [
        status == "1" or (status == "" and float(open_value) > 0.0)
        for status, open_value in zip(statuses, reference_opens)
    ]
    executable_dates = [
        value for value, is_executable in zip(dates, executable) if is_executable
    ]
    if not executable_dates:
        raise ValueError("test evidence must contain an executable date")
    volumes = [100.0 if value else 0.0 for value in executable]
    if normalization_applied is None:
        normalization_applied = [False] * len(dates)
    symbol = f"{code[-2:].lower()}.{code[:6]}"
    frame = pd.DataFrame(
        {
            "stock_code": [code] * len(dates),
            "baostock_code": [symbol] * len(dates),
            "instrument_type": ["1"] * len(dates),
            "instrument_status": ["0" if out_date else "1"] * len(dates),
            "date": list(dates),
            "listing_date": [listing_date] * len(dates),
            "out_date": [out_date] * len(dates),
            "first_executable_date": [executable_dates[0]] * len(dates),
            "last_executable_date": [executable_dates[-1]] * len(dates),
            "source_tradestatus": statuses,
            "tradestatus": statuses,
            "status_correction": [""] * len(dates),
            "reference_open": reference_opens,
            "reference_volume": volumes,
            "reference_amount": [value * 10.0 for value in volumes],
            "direct_preclose": [
                np.nan
                if index == 0
                else max(float(reference_opens[index - 1]), 1.0)
                for index in range(len(dates))
            ],
            "normalization_applied": normalization_applied,
            "applied_k_sha256": [
                ("a" * 64 if applied else UNAPPLIED_K_SHA256)
                for applied in normalization_applied
            ],
            "source": ["baostock.adjustflag3.daily-reference"] * len(dates),
            "schema_version": [BAOSTOCK_EVIDENCE_SCHEMA] * len(dates),
        },
        columns=BAOSTOCK_EVIDENCE_COLUMNS,
    )
    frame["source_payload_sha256"] = [
        _source_payload_sha256(row) for _, row in frame.iterrows()
    ]
    frame = frame.loc[:, BAOSTOCK_EVIDENCE_COLUMNS]
    _write_baostock_evidence_stage(frame, path)


def _secondary_daily_frame(
    *,
    code="000001.SZ",
    dates=("2020-01-03",),
    statuses=("1",),
    reference_opens=(10.0,),
):
    return pd.DataFrame(
        {
            "stock_code": [code] * len(dates),
            "date": np.array(dates, dtype="datetime64[D]"),
            "reference_open": reference_opens,
            "tradestatus": statuses,
        }
    )


@pytest.fixture(autouse=True)
def _never_read_production_secondary_snapshot(monkeypatch):
    monkeypatch.setattr(
        "data.build_runtime._load_secondary_daily_evidence",
        lambda _path=None: _secondary_daily_frame(
            dates=(),
            statuses=(),
            reference_opens=(),
        ),
    )


def test_runtime_delisted_mask_turns_on_strictly_after_source_date(monkeypatch):
    monkeypatch.setattr(
        "data.db.delist.get_delist_stock_info",
        lambda: {
            "000001.SZ": DelistStockInfo(
                "退市样本",
                date(2020, 1, 1),
                date(2020, 1, 3),
            )
        },
    )
    dates = np.array(
        ["2020-01-02", "2020-01-03", "2020-01-06"],
        dtype="datetime64[D]",
    )

    mask = build_delisted_mask(
        np.array(["000001.SZ", "000002.SZ"]),
        dates,
    )

    np.testing.assert_array_equal(mask[:, 0], (False, False, True))
    np.testing.assert_array_equal(mask[:, 1], (False, False, False))


def test_runtime_listing_age_uses_full_axis_and_survives_missing_open_rows():
    ages = build_listing_age(
        np.array(
            [
                [10.0, np.nan],
                [np.nan, np.nan],
                [11.0, 5.0],
                [12.0, np.nan],
            ]
        )
    )

    np.testing.assert_array_equal(ages[:, 0], (0, 1, 2, 3))
    np.testing.assert_array_equal(ages[:, 1], (-1, -1, 0, 1))


def test_runtime_build_rejects_missing_delisted_stock_in_requested_history(monkeypatch):
    monkeypatch.setattr(
        "data.db.delist.get_delist_stock_info",
        lambda: {
            "000001.SZ": DelistStockInfo(
                "退市样本",
                date(2020, 1, 1),
                date(2020, 1, 3),
            )
        },
    )

    with pytest.raises(RuntimeError, match="幸存者偏差"):
        validate_delisted_kline_coverage(
            set(),
            np.array(["2020-01-02", "2020-01-03"], dtype="datetime64[D]"),
        )


def test_runtime_build_ignores_delist_history_outside_requested_range(monkeypatch):
    monkeypatch.setattr(
        "data.db.delist.get_delist_stock_info",
        lambda: {
            "000001.SZ": DelistStockInfo(
                "早期退市样本",
                date(2000, 1, 1),
                date(2005, 1, 1),
            )
        },
    )

    validate_delisted_kline_coverage(
        set(),
        np.array(["2020-01-02", "2020-01-03"], dtype="datetime64[D]"),
    )


def test_runtime_allows_primary_only_missing_delisted_tail(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(
        "data.db.delist.get_delist_stock_info",
        lambda: {
            "000001.SZ": DelistStockInfo(
                "截断样本",
                date(2020, 1, 2),
                date(2020, 1, 6),
            )
        },
    )
    dates = np.array(
        ["2020-01-02", "2020-01-03", "2020-01-06"],
        dtype="datetime64[D]",
    )
    evidence_path = tmp_path / "evidence.parquet"
    _write_baostock_evidence(evidence_path)

    validate_delisted_kline_panel(
        np.array(["000001.SZ"]),
        dates,
        {
            "open": np.array([[np.nan], [10.0], [np.nan]]),
            "preClose": np.array([[np.nan], [np.nan], [np.nan]]),
        },
        evidence_path=evidence_path,
    )


def test_runtime_accepts_a_share_bar_matching_one_source_listing_candidate(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(
        "data.db.delist.get_delist_stock_info",
        lambda: {
            "600190.SH": DelistStockInfo(
                "A/B日期样本",
                date(2020, 1, 2),
                date(2020, 1, 6),
                (date(2020, 1, 2), date(2020, 1, 3)),
            )
        },
    )
    monkeypatch.setattr(
        "data.build_runtime._load_listing_event_dates",
        lambda: {},
    )
    dates = np.array(
        ["2020-01-02", "2020-01-03", "2020-01-06"],
        dtype="datetime64[D]",
    )
    evidence_path = tmp_path / "evidence.parquet"
    _write_baostock_evidence(evidence_path, code="600190.SH")

    validate_delisted_kline_panel(
        np.array(["600190.SH"]),
        dates,
        {
            "open": np.array([[np.nan], [10.0], [11.0]]),
            "preClose": np.array([[np.nan], [np.nan], [10.0]]),
        },
        evidence_path=evidence_path,
    )


def test_first_bar_preclose_uses_issue_price_and_never_same_day_close():
    opens = np.array([[10.0, np.nan], [11.0, 20.0], [12.0, 21.0]])
    same_day_close_leak = np.array([[10.5, np.nan], [10.0, 20.5], [11.0, 20.0]])

    result = sanitize_first_bar_preclose(
        opens,
        same_day_close_leak,
        np.array([8.0, 15.0]),
        np.array(["2020-01-02", "2020-01-03", "2020-01-06"], dtype="datetime64[D]"),
        np.array(["2020-01-02", "2020-01-02"], dtype="datetime64[D]"),
    )

    assert result[0, 0] == 8.0
    # A valid-looking issue price with a mismatched listing date must not be
    # attached to a truncated first bar.
    assert np.isnan(result[1, 1])
    assert result[1, 0] == 10.0


def test_runtime_build_rejects_first_kline_misaligned_with_listing_evidence(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(
        "data.build_runtime._load_listing_event_dates",
        lambda: {"000001": (np.datetime64("2020-01-02", "D"),)},
    )

    with pytest.raises(
        ListingDateAlignmentError,
        match="首条 K 线与上市日期证据不一致",
    ) as caught:
        validate_listing_date_alignment(
            np.array(["000001.SZ"]),
            np.array(
                ["2020-01-02", "2020-01-03", "2020-01-06"],
                dtype="datetime64[D]",
            ),
            np.array([[np.nan], [10.0], [11.0]]),
            np.array(["2020-01-02"], dtype="datetime64[D]"),
            evidence_path=tmp_path / "absent.parquet",
        )

    assert caught.value.diagnostics == (
        {
            "code": "000001.SZ",
            "first_kline_date": "2020-01-03",
            "first_executable_date": "2020-01-03",
            "expected_listing_dates": ["2020-01-02"],
            "expected_first_executable_dates": ["2020-01-02"],
            "evidence": {
                "stock_name_events": ["2020-01-02"],
                "issue_date": "2020-01-02",
                "baostock_listing_date": None,
                "baostock_first_executable_date": None,
            },
        },
    )


def test_daily_evidence_masks_status_zero_placeholder_before_listing_age(tmp_path):
    evidence_path = tmp_path / "evidence.parquet"
    _write_baostock_evidence(evidence_path)
    arrays = {
        "open": np.array([[10.0], [10.0], [11.0]]),
        "high": np.array([[10.0], [10.2], [11.2]]),
        "low": np.array([[10.0], [9.8], [10.8]]),
        "close": np.array([[10.0], [10.0], [11.0]]),
        "volume": np.array([[0.0], [100.0], [100.0]]),
        "amount": np.array([[0.0], [1000.0], [1100.0]]),
        "preClose": np.array([[9.0], [10.0], [10.0]]),
    }
    dates = np.array(
        ["2020-01-02", "2020-01-03", "2020-01-06"],
        dtype="datetime64[D]",
    )

    apply_baostock_daily_evidence(
        arrays,
        np.array(["000001.SZ"]),
        dates,
        evidence_path=evidence_path,
    )

    assert np.isnan(arrays["open"][0, 0])
    assert np.isnan(arrays["preClose"][0, 0])
    assert arrays["volume"][0, 0] == 0.0
    np.testing.assert_array_equal(build_listing_age(arrays["open"])[:, 0], (-1, 0, 1))
    validate_listing_date_alignment(
        np.array(["000001.SZ"]),
        dates,
        arrays["open"],
        np.array(["2020-01-02"], dtype="datetime64[D]"),
        evidence_path=evidence_path,
    )


def test_daily_placeholder_without_exact_date_evidence_fails_closed(tmp_path):
    evidence_path = tmp_path / "evidence.parquet"
    _write_baostock_evidence(evidence_path)
    arrays = {
        field: np.array([[value], [10.0], [11.0]])
        for field, value in {
            "open": 10.0,
            "high": 10.0,
            "low": 10.0,
            "close": 10.0,
            "volume": 0.0,
            "amount": 0.0,
            "preClose": 9.0,
        }.items()
    }
    dates = np.array(
        ["2020-01-01", "2020-01-03", "2020-01-06"],
        dtype="datetime64[D]",
    )

    with pytest.raises(KlineDailyEvidenceError) as caught:
        apply_baostock_daily_evidence(
            arrays,
            np.array(["000001.SZ"]),
            dates,
            evidence_path=evidence_path,
        )

    assert caught.value.diagnostics[0] == {
        "code": "000001.SZ",
        "date": "2020-01-01",
        "kind": "missing_daily_evidence",
    }


def test_listing_guard_ignores_primary_only_earlier_executable_date(tmp_path):
    evidence_path = tmp_path / "evidence.parquet"
    _write_baostock_evidence(evidence_path)

    validate_listing_date_alignment(
        np.array(["000001.SZ"]),
        np.array(
            ["2020-01-02", "2020-01-03", "2020-01-06"],
            dtype="datetime64[D]",
        ),
        np.array([[np.nan], [np.nan], [11.0]]),
        np.array(["2020-01-02"], dtype="datetime64[D]"),
        evidence_path=evidence_path,
    )


def test_daily_execution_uses_status_and_reference_open_not_t_day_turnover(
    tmp_path,
):
    evidence_path = tmp_path / "evidence.parquet"
    dates = np.array(
        ["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07"],
        dtype="datetime64[D]",
    )
    _write_baostock_evidence(
        evidence_path,
        dates=tuple(str(value) for value in dates),
        statuses=("1", "0", "", ""),
        reference_opens=(0.0, 10.0, 11.0, 0.0),
        out_date="2020-01-07",
    )

    def make_arrays(t_volume, t_amount):
        return {
            "open": np.array([[10.0], [10.0], [11.0], [12.0]]),
            "high": np.array([[10.2], [10.2], [11.2], [12.2]]),
            "low": np.array([[9.8], [9.8], [10.8], [11.8]]),
            "close": np.array([[10.0], [10.0], [11.0], [12.0]]),
            "volume": np.array([[0.0], [100.0], [t_volume], [100.0]]),
            "amount": np.array([[0.0], [1000.0], [t_amount], [1200.0]]),
            "preClose": np.array([[9.0], [10.0], [10.0], [11.0]]),
        }

    baseline = make_arrays(0.0, 0.0)
    changed = make_arrays(1e30, 1e40)
    for arrays in (baseline, changed):
        apply_baostock_daily_evidence(
            arrays,
            np.array(["000001.SZ"]),
            dates,
            evidence_path=evidence_path,
        )

    # status=1 is executable even with zero source/runtime turnover; status=0
    # is never executable; empty status needs a positive independent open.
    np.testing.assert_array_equal(
        np.isfinite(baseline["open"][:, 0]),
        (True, False, True, False),
    )
    np.testing.assert_array_equal(baseline["open"], changed["open"])
    baseline_age = build_listing_age(baseline["open"])
    changed_age = build_listing_age(changed["open"])
    np.testing.assert_array_equal(baseline_age, changed_age)

    def legality(arrays, ages):
        return evaluate_trade_legality(
            decision_date=dates[2],
            stock_codes=["000001.SZ"],
            listing_age=ages[2],
            open_prices=arrays["open"][2],
            preclose_prices=arrays["preClose"][2],
            issue_prices=np.array([8.0]),
            st_mask=np.array([False]),
            delisted_mask=np.array([False]),
        )

    baseline_legality = legality(baseline, baseline_age)
    changed_legality = legality(changed, changed_age)
    np.testing.assert_array_equal(
        baseline_legality.buy_allowed,
        changed_legality.buy_allowed,
    )
    np.testing.assert_array_equal(
        baseline_legality.sell_allowed,
        changed_legality.sell_allowed,
    )


def test_listing_guard_fails_closed_without_any_listing_evidence(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr("data.build_runtime._load_listing_event_dates", lambda: {})

    with pytest.raises(ListingDateAlignmentError) as caught:
        validate_listing_date_alignment(
            np.array(["000001.SZ"]),
            np.array(["2020-01-02"], dtype="datetime64[D]"),
            np.array([[10.0]]),
            np.array(["NaT"], dtype="datetime64[D]"),
            evidence_path=tmp_path / "absent.parquet",
        )

    assert caught.value.diagnostics[0]["reason"] == "missing_listing_evidence"


def test_listing_guard_accepts_local_issue_evidence_without_baostock_download(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr("data.build_runtime._load_listing_event_dates", lambda: {})

    validate_listing_date_alignment(
        np.array(["000001.SZ"]),
        np.array(["2020-01-02", "2020-01-03"], dtype="datetime64[D]"),
        np.array([[10.0], [10.1]]),
        np.array(["2020-01-02"], dtype="datetime64[D]"),
        evidence_path=tmp_path / "absent.parquet",
    )


@pytest.mark.parametrize("missing_index", [1, 3])
def test_delisted_panel_ignores_missing_primary_only_executable_row(
    monkeypatch,
    tmp_path,
    missing_index,
):
    monkeypatch.setattr(
        "data.db.delist.get_delist_stock_info",
        lambda: {
            "000001.SZ": DelistStockInfo(
                "逐日覆盖样本",
                date(2020, 1, 2),
                date(2020, 1, 7),
            )
        },
    )
    date_text = ("2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07")
    dates = np.array(date_text, dtype="datetime64[D]")
    evidence_path = tmp_path / "evidence.parquet"
    _write_baostock_evidence(
        evidence_path,
        dates=date_text,
        statuses=("1", "1", "1", "1"),
        reference_opens=(10.0, 10.1, 10.2, 10.3),
        out_date="2020-01-07",
    )
    opens = np.array([[10.0], [10.1], [10.2], [10.3]])
    opens[missing_index, 0] = np.nan

    validate_delisted_kline_panel(
        np.array(["000001.SZ"]),
        dates,
        {
            "open": opens,
            "preClose": np.array([[np.nan], [10.0], [10.1], [10.2]]),
        },
        evidence_path=evidence_path,
    )


def test_delisted_panel_requires_preclose_on_each_later_executable_day(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(
        "data.db.delist.get_delist_stock_info",
        lambda: {
            "000001.SZ": DelistStockInfo(
                "前收逐日样本",
                date(2020, 1, 2),
                date(2020, 1, 6),
            )
        },
    )
    date_text = ("2020-01-02", "2020-01-03", "2020-01-06")
    evidence_path = tmp_path / "evidence.parquet"
    _write_baostock_evidence(
        evidence_path,
        dates=date_text,
        statuses=("1", "1", "1"),
        reference_opens=(10.0, 10.1, 10.2),
    )

    with pytest.raises(RuntimeError, match="preClose missing 2020-01-03"):
        validate_delisted_kline_panel(
            np.array(["000001.SZ"]),
            np.array(date_text, dtype="datetime64[D]"),
            {
                "open": np.array([[10.0], [10.1], [10.2]]),
                # The last preClose is valid, so the former np.any guard would
                # incorrectly accept the missing intermediate reference.
                "preClose": np.array([[np.nan], [np.nan], [10.1]]),
            },
            evidence_path=evidence_path,
        )


def test_same_code_predecessor_uses_only_exact_secondary_date_union(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr("data.build_runtime._load_listing_event_dates", lambda: {})
    monkeypatch.setattr(
        "data.db.delist.get_delist_stock_info",
        lambda: {
            "600018.SH": DelistStockInfo(
                "同代码前身样本",
                date(2000, 7, 19),
                date(2006, 10, 27),
            )
        },
    )
    secondary = _secondary_daily_frame(
        code="600018.SH",
        dates=("2000-07-19", "2000-07-20"),
        statuses=("1", "1"),
        reference_opens=(5.0, 5.1),
    )
    monkeypatch.setattr(
        "data.build_runtime._load_secondary_daily_evidence",
        lambda _path=None: secondary,
    )
    dates = np.array(
        ["2000-07-19", "2000-07-20", "2006-10-26", "2006-10-27"],
        dtype="datetime64[D]",
    )
    evidence_path = tmp_path / "evidence.parquet"
    _write_baostock_evidence(
        evidence_path,
        code="600018.SH",
        dates=("2006-10-26", "2006-10-27"),
        statuses=("1", "1"),
        reference_opens=(10.0, 11.0),
        listing_date="2006-10-26",
        out_date="2006-10-27",
    )
    opens = np.array([[5.0], [5.1], [10.0], [11.0]])

    validate_listing_date_alignment(
        np.array(["600018.SH"]),
        dates,
        opens,
        np.array(["2006-10-26"], dtype="datetime64[D]"),
        evidence_path=evidence_path,
        secondary_evidence_path=tmp_path / "secondary.parquet",
    )
    validate_delisted_kline_panel(
        np.array(["600018.SH"]),
        dates,
        {
            "open": opens,
            "preClose": np.array([[np.nan], [5.0], [5.1], [10.0]]),
        },
        evidence_path=evidence_path,
        secondary_evidence_path=tmp_path / "secondary.parquet",
    )


def test_same_code_predecessor_rejects_forged_date_inside_old_interval(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(
        "data.build_runtime._load_secondary_daily_evidence",
        lambda _path=None: _secondary_daily_frame(
            code="600018.SH",
            dates=("2000-07-19",),
            statuses=("1",),
            reference_opens=(5.0,),
        ),
    )
    evidence_path = tmp_path / "evidence.parquet"
    _write_baostock_evidence(
        evidence_path,
        code="600018.SH",
        dates=("2006-10-26",),
        statuses=("1",),
        reference_opens=(10.0,),
        listing_date="2006-10-26",
        out_date="",
    )
    dates = np.array(
        ["2000-07-19", "2000-07-20", "2006-10-26"],
        dtype="datetime64[D]",
    )
    arrays = {
        field: np.array([[5.0], [5.1], [10.0]])
        for field in ("open", "high", "low", "close", "preClose")
    }
    arrays["volume"] = np.ones((3, 1))
    arrays["amount"] = np.ones((3, 1))

    with pytest.raises(KlineDailyEvidenceError) as caught:
        apply_baostock_daily_evidence(
            arrays,
            np.array(["600018.SH"]),
            dates,
            evidence_path=evidence_path,
            secondary_evidence_path=tmp_path / "secondary.parquet",
        )

    assert caught.value.diagnostics == (
        {
            "code": "600018.SH",
            "date": "2000-07-20",
            "kind": "missing_daily_evidence",
        },
    )


def test_same_code_predecessor_rejects_missing_exact_secondary_date(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(
        "data.db.delist.get_delist_stock_info",
        lambda: {
            "600018.SH": DelistStockInfo(
                "同代码前身缺日样本",
                date(2000, 7, 19),
                date(2006, 10, 26),
            )
        },
    )
    monkeypatch.setattr(
        "data.build_runtime._load_secondary_daily_evidence",
        lambda _path=None: _secondary_daily_frame(
            code="600018.SH",
            dates=("2000-07-19", "2000-07-20"),
            statuses=("1", "1"),
            reference_opens=(5.0, 5.1),
        ),
    )
    evidence_path = tmp_path / "evidence.parquet"
    _write_baostock_evidence(
        evidence_path,
        code="600018.SH",
        dates=("2006-10-26",),
        statuses=("1",),
        reference_opens=(10.0,),
        listing_date="2006-10-26",
        out_date="2006-10-26",
    )

    with pytest.raises(RuntimeError, match="open missing 2000-07-20"):
        validate_delisted_kline_panel(
            np.array(["600018.SH"]),
            np.array(
                ["2000-07-19", "2000-07-20", "2006-10-26"],
                dtype="datetime64[D]",
            ),
            {
                "open": np.array([[5.0], [np.nan], [10.0]]),
                "preClose": np.array([[np.nan], [np.nan], [5.0]]),
            },
            evidence_path=evidence_path,
            secondary_evidence_path=tmp_path / "secondary.parquet",
        )


def test_600018_independently_executable_20010816_missing_local_k_fails_closed(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(
        "data.build_runtime._load_secondary_daily_evidence",
        lambda _path=None: _secondary_daily_frame(
            code="600018.SH",
            dates=("2001-08-16",),
            statuses=("1",),
            reference_opens=(19.4,),
        ),
    )
    evidence_path = tmp_path / "evidence.parquet"
    _write_baostock_evidence(
        evidence_path,
        code="600018.SH",
        dates=("2006-10-26",),
        statuses=("1",),
        reference_opens=(3.7,),
        listing_date="2006-10-26",
        out_date="",
    )
    arrays = {
        "open": np.array([[np.nan]]),
        "high": np.array([[np.nan]]),
        "low": np.array([[np.nan]]),
        "close": np.array([[np.nan]]),
        "volume": np.array([[0.0]]),
        "amount": np.array([[0.0]]),
        "preClose": np.array([[np.nan]]),
    }

    with pytest.raises(KlineDailyEvidenceError) as caught:
        apply_baostock_daily_evidence(
            arrays,
            np.array(["600018.SH"]),
            np.array(["2001-08-16"], dtype="datetime64[D]"),
            evidence_path=evidence_path,
            secondary_evidence_path=tmp_path / "secondary.parquet",
        )

    assert caught.value.diagnostics == (
        {
            "code": "600018.SH",
            "date": "2001-08-16",
            "kind": "missing_local_executable_history",
        },
    )


def test_secondary_t_open_evidence_strictly_unions_missing_executable_date(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(
        "data.db.delist.get_delist_stock_info",
        lambda: {
            "000001.SZ": DelistStockInfo(
                "次级逐日样本",
                date(2020, 1, 2),
                date(2020, 1, 6),
            )
        },
    )
    monkeypatch.setattr(
        "data.build_runtime._load_secondary_daily_evidence",
        lambda _path: _secondary_daily_frame(),
    )
    evidence_path = tmp_path / "evidence.parquet"
    _write_baostock_evidence(
        evidence_path,
        dates=("2020-01-02", "2020-01-06"),
        statuses=("1", "1"),
        reference_opens=(10.0, 10.2),
    )
    dates = np.array(
        ["2020-01-02", "2020-01-03", "2020-01-06"],
        dtype="datetime64[D]",
    )

    def make_arrays(t_volume, t_amount):
        return {
            "open": np.array([[10.0], [10.1], [10.2]]),
            "high": np.array([[10.1], [10.2], [10.3]]),
            "low": np.array([[9.9], [10.0], [10.1]]),
            "close": np.array([[10.0], [10.1], [10.2]]),
            "volume": np.array([[100.0], [t_volume], [100.0]]),
            "amount": np.array([[1000.0], [t_amount], [1020.0]]),
            "preClose": np.array([[np.nan], [10.0], [10.1]]),
        }

    baseline = make_arrays(0.0, 0.0)
    changed = make_arrays(1e30, 1e40)
    for arrays in (baseline, changed):
        apply_baostock_daily_evidence(
            arrays,
            np.array(["000001.SZ"]),
            dates,
            evidence_path=evidence_path,
            secondary_evidence_path=tmp_path / "secondary.parquet",
        )
    np.testing.assert_array_equal(baseline["open"], changed["open"])
    assert baseline["open"][1, 0] == 10.1

    validate_delisted_kline_panel(
        np.array(["000001.SZ"]),
        dates,
        baseline,
        evidence_path=evidence_path,
        secondary_evidence_path=tmp_path / "secondary.parquet",
    )


def test_secondary_executable_row_wins_union_over_primary_status_zero(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(
        "data.build_runtime._load_secondary_daily_evidence",
        lambda _path: _secondary_daily_frame(
            statuses=("1",),
            # Adjusted evidence may be negative at entity-change boundaries.
            # It proves only the date; canonical local K must remain untouched.
            reference_opens=(-8.44,),
        ),
    )
    evidence_path = tmp_path / "evidence.parquet"
    _write_baostock_evidence(
        evidence_path,
        statuses=("1", "0", "1"),
        reference_opens=(10.0, 10.1, 10.2),
        normalization_applied=(False, True, False),
    )
    arrays = {
        "open": np.array([[10.0], [10.1], [10.2]]),
        "high": np.array([[10.1], [10.2], [10.3]]),
        "low": np.array([[9.9], [10.0], [10.1]]),
        "close": np.array([[10.0], [10.1], [10.2]]),
        "volume": np.array([[100.0], [0.0], [100.0]]),
        "amount": np.array([[1000.0], [0.0], [1020.0]]),
        "preClose": np.array([[np.nan], [10.0], [10.1]]),
    }

    apply_baostock_daily_evidence(
        arrays,
        np.array(["000001.SZ"]),
        np.array(
            ["2020-01-02", "2020-01-03", "2020-01-06"],
            dtype="datetime64[D]",
        ),
        evidence_path=evidence_path,
        secondary_evidence_path=tmp_path / "secondary.parquet",
    )

    assert arrays["open"][1, 0] == 10.1


def test_audited_placeholder_candidate_requires_secondary_exact_override(
    tmp_path,
):
    evidence_path = tmp_path / "evidence.parquet"
    _write_baostock_evidence(
        evidence_path,
        statuses=("1", "1", "1"),
        normalization_applied=(False, True, False),
    )

    def make_arrays(volume, amount):
        return {
            "open": np.array([[10.0], [10.0], [11.0]]),
            "high": np.array([[10.1], [10.1], [11.1]]),
            "low": np.array([[9.9], [9.9], [10.9]]),
            "close": np.array([[10.0], [10.0], [11.0]]),
            "volume": np.array([[100.0], [volume], [100.0]]),
            "amount": np.array([[1000.0], [amount], [1100.0]]),
            "preClose": np.array([[np.nan], [10.0], [10.0]]),
        }

    low_turnover = make_arrays(0.0, 0.0)
    high_turnover = make_arrays(1e20, 1e30)
    dates = np.array(
        ["2020-01-02", "2020-01-03", "2020-01-06"],
        dtype="datetime64[D]",
    )
    for arrays in (low_turnover, high_turnover):
        apply_baostock_daily_evidence(
            arrays,
            np.array(["000001.SZ"]),
            dates,
            evidence_path=evidence_path,
            secondary_evidence_path=tmp_path / "absent-secondary.parquet",
        )

    assert np.isnan(low_turnover["open"][1, 0])
    assert np.isnan(high_turnover["open"][1, 0])
    assert low_turnover["volume"][1, 0] == 0.0
    assert high_turnover["volume"][1, 0] == 0.0


@pytest.mark.parametrize(
    ("tail_open", "tail_preclose", "message"),
    [
        (np.nan, np.nan, "open missing 2020-01-06"),
        (10.2, np.nan, "preClose missing 2020-01-06"),
    ],
)
def test_secondary_union_cannot_relax_delisted_tail_or_preclose(
    monkeypatch,
    tmp_path,
    tail_open,
    tail_preclose,
    message,
):
    monkeypatch.setattr(
        "data.db.delist.get_delist_stock_info",
        lambda: {
            "000001.SZ": DelistStockInfo(
                "次级尾部样本",
                date(2020, 1, 2),
                date(2020, 1, 6),
            )
        },
    )
    monkeypatch.setattr(
        "data.build_runtime._load_secondary_daily_evidence",
        lambda _path: _secondary_daily_frame(
            dates=("2020-01-06",),
            reference_opens=(10.2,),
        ),
    )
    evidence_path = tmp_path / "evidence.parquet"
    _write_baostock_evidence(
        evidence_path,
        dates=("2020-01-02", "2020-01-03"),
        statuses=("1", "1"),
        reference_opens=(10.0, 10.1),
        out_date="2020-01-06",
    )

    with pytest.raises(RuntimeError, match=message):
        validate_delisted_kline_panel(
            np.array(["000001.SZ"]),
            np.array(
                ["2020-01-02", "2020-01-03", "2020-01-06"],
                dtype="datetime64[D]",
            ),
            {
                "open": np.array([[10.0], [10.1], [tail_open]]),
                "preClose": np.array([[np.nan], [10.0], [tail_preclose]]),
            },
            evidence_path=evidence_path,
            secondary_evidence_path=tmp_path / "secondary.parquet",
        )


def test_secondary_executable_date_does_not_replace_listing_lifecycle_evidence(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr("data.build_runtime._load_listing_event_dates", lambda: {})
    monkeypatch.setattr(
        "data.build_runtime._load_secondary_daily_evidence",
        lambda _path: _secondary_daily_frame(),
    )

    with pytest.raises(ListingDateAlignmentError) as caught:
        validate_listing_date_alignment(
            np.array(["000001.SZ"]),
            np.array(["2020-01-03"], dtype="datetime64[D]"),
            np.array([[10.0]]),
            np.array(["NaT"], dtype="datetime64[D]"),
            evidence_path=tmp_path / "absent-primary.parquet",
            secondary_evidence_path=tmp_path / "secondary.parquet",
        )

    assert caught.value.diagnostics[0]["reason"] == "missing_listing_evidence"
