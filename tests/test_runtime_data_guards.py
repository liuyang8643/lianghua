import numpy as np
import pandas as pd
import pytest

from data import build_runtime


def _issue_reference(*rows) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "stock_code": code[:6],
                "issue_price": 10.0,
                "list_date": listed,
                "source": "test-sealed-source",
                "source_as_of": date,
            }
            for code, listed, date in rows
        ]
    )


def test_terminal_active_codes_excludes_only_proven_future_listing():
    terminal = pd.Timestamp("2026-08-28").date()
    reference = _issue_reference(
        ("000001.SZ", terminal, terminal),
        ("001234.SZ", pd.Timestamp("2026-09-01").date(), terminal),
    )

    result = build_runtime.resolve_terminal_active_codes(
        ("000001.SZ", "001234.SZ"),
        terminal,
        ("000001.SZ",),
        issue_reference=reference,
    )

    assert result == ("000001.SZ",)


def test_terminal_active_codes_treats_list_date_equal_to_t_as_active():
    terminal = pd.Timestamp("2026-08-28").date()
    reference = _issue_reference(("001234.SZ", terminal, terminal))

    assert build_runtime.resolve_terminal_active_codes(
        ("001234.SZ",), terminal, (), issue_reference=reference
    ) == ("001234.SZ",)


def test_terminal_active_codes_rejects_unknown_code_outside_k_axis():
    terminal = pd.Timestamp("2026-08-28").date()
    reference = _issue_reference(("000001.SZ", terminal, terminal))

    with pytest.raises(RuntimeError, match="轴外股票缺少封存上市日"):
        build_runtime.resolve_terminal_active_codes(
            ("000001.SZ", "001234.SZ"),
            terminal,
            ("000001.SZ",),
            issue_reference=reference,
        )


def test_terminal_active_codes_rejects_future_listing_with_existing_k():
    terminal = pd.Timestamp("2026-08-28").date()
    reference = _issue_reference(
        ("000001.SZ", terminal, terminal),
        ("001234.SZ", pd.Timestamp("2026-09-01").date(), terminal),
    )

    with pytest.raises(RuntimeError, match="预上市生命周期.*K.*冲突"):
        build_runtime.resolve_terminal_active_codes(
            ("000001.SZ", "001234.SZ"),
            terminal,
            ("000001.SZ", "001234.SZ"),
            issue_reference=reference,
        )


def test_offline_runtime_requires_terminal_evidence_for_every_current_stock(
    monkeypatch,
):
    import data.db.stock_list as stock_list

    monkeypatch.setattr(
        stock_list,
        "load_current_stock_codes",
        lambda: ("000001.SZ", "600001.SH"),
    )
    expected = build_runtime._resolve_expected_latest_codes(None)
    with pytest.raises(RuntimeError, match="末端.*覆盖不完整"):
        build_runtime.validate_current_kline_terminal_coverage(
            np.array(["000001.SZ", "600001.SH"]),
            np.array(["2024-01-02", "2024-01-03"], dtype="datetime64[D]"),
            np.array([[10.0, 20.0], [10.1, np.nan]]),
            expected,
        )


def test_terminal_coverage_can_validate_an_explicit_subset(
    monkeypatch,
):
    import data.db.stock_list as stock_list

    monkeypatch.setattr(
        stock_list,
        "load_current_stock_codes",
        lambda: ("000001.SZ", "600001.SH"),
    )
    expected = build_runtime._resolve_expected_latest_codes(["000001.SZ"])
    build_runtime.validate_current_kline_terminal_coverage(
        np.array(["000001.SZ", "600001.SH"]),
        np.array(["2024-01-02", "2024-01-03"], dtype="datetime64[D]"),
        np.array([[10.0, 20.0], [10.1, np.nan]]),
        expected,
    )


def test_live_partial_runtime_cannot_use_empty_or_unknown_bypass(monkeypatch):
    import data.db.stock_list as stock_list

    monkeypatch.setattr(
        stock_list,
        "load_current_stock_codes",
        lambda: ("000001.SZ",),
    )
    with pytest.raises(ValueError, match="不得为空"):
        build_runtime._resolve_expected_latest_codes([])
    with pytest.raises(ValueError, match="当前在市股票子集"):
        build_runtime._resolve_expected_latest_codes(["600001.SH"])


def test_live_open_overlay_fills_only_missing_open_and_preclose(tmp_path):
    path = tmp_path / "2024-01-03.parquet"
    pd.DataFrame(
        {
            "trade_date": ["2024-01-03", "2024-01-03"],
            "stock_code": ["000001.SZ", "600001.SH"],
            "open": [99.0, 0.0],
            "preClose": [9.9, 19.8],
        }
    ).to_parquet(path, index=False)
    arrays = {
        "open": np.array([[9.8, 19.7], [10.0, np.nan]]),
        "preClose": np.array([[9.7, 19.6], [9.9, np.nan]]),
        "close": np.array([[9.8, 19.7], [np.nan, np.nan]]),
    }

    build_runtime.apply_live_open_overlay(
        arrays,
        np.array(["000001.SZ", "600001.SH"]),
        np.array(["2024-01-02", "2024-01-03"], dtype="datetime64[D]"),
        path,
        ("000001.SZ", "600001.SH"),
    )

    assert arrays["open"][-1, 0] == 10.0
    assert np.isnan(arrays["open"][-1, 1])
    assert arrays["preClose"][-1].tolist() == [9.9, 19.8]
    assert np.isnan(arrays["close"][-1]).all()


def test_live_open_overlay_requires_full_current_axis_and_exact_date(tmp_path):
    path = tmp_path / "bad.parquet"
    pd.DataFrame(
        {
            "trade_date": ["2024-01-02"],
            "stock_code": ["000001.SZ"],
            "open": [10.0],
            "preClose": [9.9],
        }
    ).to_parquet(path, index=False)
    arrays = {
        "open": np.full((1, 2), np.nan),
        "preClose": np.full((1, 2), np.nan),
    }

    with pytest.raises(ValueError, match="严格覆盖"):
        build_runtime.apply_live_open_overlay(
            arrays,
            np.array(["000001.SZ", "600001.SH"]),
            np.array(["2024-01-03"], dtype="datetime64[D]"),
            path,
            ("000001.SZ", "600001.SH"),
        )


def test_full_runtime_accepts_suspension_only_with_terminal_preclose_evidence():
    build_runtime.validate_current_kline_terminal_coverage(
        np.array(["000001.SZ"]),
        np.array(["2024-01-03"], dtype="datetime64[D]"),
        np.array([[9.9]]),
        ("000001.SZ",),
    )
    with pytest.raises(RuntimeError, match="行情覆盖不完整"):
        build_runtime.validate_current_kline_terminal_coverage(
            np.array(["000001.SZ"]),
            np.array(["2024-01-03"], dtype="datetime64[D]"),
            np.array([[np.nan]]),
            ("000001.SZ",),
        )


def test_total_share_ignores_unversioned_derived_financial_values(
    tmp_path,
    monkeypatch,
):
    financial = tmp_path / "financial"
    financial.mkdir()
    pd.DataFrame(
        {
            "stock_code": ["000001.SZ"],
            "m_anntime": ["2024-01-02"],
            "cap_stk": [100.0],
        }
    ).to_parquet(financial / "balance.parquet", index=False)
    pd.DataFrame(
        {
            "stock_code": ["600001.SH"],
            "m_anntime": ["2024-01-02"],
            "cap_stk": [999.0],
        }
    ).to_parquet(financial / "balance_derived.parquet", index=False)
    import data.db.delist as delist

    monkeypatch.setattr(build_runtime, "DATA_DIR", tmp_path)
    monkeypatch.setattr(delist, "get_delist_stock_info", lambda: {})
    result = build_runtime.build_total_share(
        np.array(["000001.SZ", "600001.SH"]),
        np.array(["2024-01-02", "2024-01-03"], dtype="datetime64[D]"),
    )

    np.testing.assert_array_equal(result[:, 0], np.array([100.0, 100.0]))
    assert np.isnan(result[:, 1]).all()
