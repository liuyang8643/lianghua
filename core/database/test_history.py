"""Tests for mootdx K-line history data source.

Tests get_history_data(), get_history_data_after_download(), and
related utility functions (both mock-based unit tests and real network tests).
"""
import time
from datetime import datetime, date
import pytest
import pandas as pd
from core.database.history import (
    get_history_data,
    get_history_data_after_download,
    _to_mootdx_code,
    _mootdx_frequency,
    _convert_to_wbr,
    _build_reliable_bar_mask,
    _get_full_history_start,
    _get_expected_history_count,
)


# ==================== Unit Tests (no network) ====================


class TestMootdxCodeConversion:
    """Test code format conversion utilities."""

    def test_to_mootdx_code_sh(self):
        assert _to_mootdx_code("600000.SH") == "600000"

    def test_to_mootdx_code_sz(self):
        assert _to_mootdx_code("000001.SZ") == "000001"

    def test_mootdx_frequency_daily(self):
        assert _mootdx_frequency("1d") == 9

    def test_mootdx_frequency_minute(self):
        assert _mootdx_frequency("1m") == 8

    def test_mootdx_frequency_invalid(self):
        with pytest.raises(ValueError):
            _mootdx_frequency("5d")


class TestConvertToWbr:
    """Test the mootdx-to-WBR DataFrame conversion."""

    def test_convert_basic_dataframe(self):
        """Test conversion of a standard mootdx DataFrame."""
        df_in = pd.DataFrame({
            "open": [1.0],
            "high": [1.1],
            "low": [0.9],
            "close": [1.05],
            "vol": [10000],
            "amount": [100000.0],
        }, index=pd.DatetimeIndex([datetime(2024, 1, 1, 15, 0, 0)]))
        result = _convert_to_wbr(df_in)
        assert list(result.columns) == ["time", "open", "high", "low", "close", "volume", "amount", "preClose"]
        assert result["open"].iloc[0] == 1.0
        assert result["volume"].iloc[0] == 10000

    def test_convert_empty_dataframe(self):
        """Test conversion of an empty DataFrame."""
        df_in = pd.DataFrame()
        result = _convert_to_wbr(df_in)
        assert result.empty
        assert list(result.columns) == ["time", "open", "high", "low", "close", "volume", "amount", "preClose"]

    def test_convert_none_dataframe(self):
        """Test conversion of None."""
        result = _convert_to_wbr(None)
        assert result.empty


class TestBuildReliableBarMask:
    """Test the reliable bar mask builder."""

    def test_filters_zero_ohlc(self):
        """Bars with all-zero OHLC should be masked out."""
        df = pd.DataFrame({
            "time": [1000, 2000, 3000],
            "open": [1.0, 0.0, 2.0],
            "high": [1.1, 0.0, 2.1],
            "low": [0.9, 0.0, 1.9],
            "close": [1.0, 0.0, 2.0],
            "volume": [100, 0, 200],
            "amount": [100.0, 0.0, 200.0],
        })
        mask = _build_reliable_bar_mask(df)
        assert list(mask) == [True, False, True]

    def test_empty_dataframe(self):
        """Empty DataFrame should return None."""
        assert _build_reliable_bar_mask(pd.DataFrame()) is None

    def test_no_ohlc_columns(self):
        """DataFrame without OHLC columns should return all True."""
        df = pd.DataFrame({"time": [1000], "volume": [100]})
        mask = _build_reliable_bar_mask(df)
        assert mask is not None
        assert mask.all()

    def test_none_input(self):
        """None input should return None."""
        assert _build_reliable_bar_mask(None) is None


class TestGetFullHistoryStart:
    """Test _get_full_history_start (mocked)."""

    def test_daily_period(self, monkeypatch):
        monkeypatch.setattr(
            "core.database.history._get_stock_date_range",
            lambda code: (date(2025, 1, 1), None),
        )
        start = _get_full_history_start("000001.SZ", "1d")
        assert start == datetime(2025, 1, 1, 0, 0, 0)

    def test_minute_period(self):
        assert _get_full_history_start("000001.SZ", "1m") is None


class TestExpectedHistoryCount:
    """Test _get_expected_history_count (mocked)."""

    def test_normal_case(self, monkeypatch):
        monkeypatch.setattr(
            "core.database.history._get_stock_date_range",
            lambda code: (date(2026, 1, 1), None),
        )
        count = _get_expected_history_count("000001.SZ", datetime(2026, 4, 10, 15, 0, 0), "1d", 60)
        assert count > 0
        assert count <= 60

    def test_stock_too_new(self, monkeypatch):
        monkeypatch.setattr(
            "core.database.history._get_stock_date_range",
            lambda code: (date(2026, 12, 1), None),
        )
        count = _get_expected_history_count("000001.SZ", datetime(2026, 4, 10, 15, 0, 0), "1d", 60)
        assert count == 0

    def test_no_date_range(self, monkeypatch):
        monkeypatch.setattr(
            "core.database.history._get_stock_date_range",
            lambda code: None,
        )
        count = _get_expected_history_count("000001.SZ", datetime(2026, 4, 10, 15, 0, 0), "1d", 60)
        assert count == 60


class TestGetHistoryDataAfterDownload:
    """Test get_history_data_after_download (mocked)."""

    def test_passes_through(self, monkeypatch):
        data = pd.DataFrame({
            "time": [int(datetime(2026, 4, 10, 15, 0, 0).timestamp() * 1000)],
            "open": [1.0],
            "high": [1.1],
            "low": [0.9],
            "close": [1.0],
            "volume": [10000],
            "amount": [100000.0],
        })

        def mock_get(codes, count, base_time, period, div_type):
            return {"000001.SZ": data}

        monkeypatch.setattr("core.database.history.get_history_data", mock_get)
        result = get_history_data_after_download(
            ["000001.SZ"], 5, datetime(2026, 4, 10, 15, 0, 0), "1d", "back"
        )
        assert result["000001.SZ"] is data


# ==================== Network Integration Tests ====================


class TestGetHistoryDataNetwork:
    """Real network tests against mootdx (通达信)."""

    @pytest.mark.network
    @pytest.mark.mootdx
    def test_shanghai_stock_daily(self):
        """Test getting daily data for a Shanghai stock (600000.SH)."""
        result = get_history_data(
            ["600000.SH"],
            count=10,
            base_time=datetime.now(),
            period="1d",
        )
        assert "600000.SH" in result
        df = result["600000.SH"]
        assert df is not None
        assert not df.empty
        assert list(df.columns) == ["time", "open", "high", "low", "close", "volume", "amount", "preClose"]
        assert len(df) == 10, f"Expected 10 bars, got {len(df)}"

    @pytest.mark.network
    @pytest.mark.mootdx
    def test_shenzhen_stock_daily(self):
        """Test getting daily data for a Shenzhen stock (000001.SZ)."""
        result = get_history_data(
            ["000001.SZ"],
            count=10,
            base_time=datetime.now(),
            period="1d",
        )
        assert "000001.SZ" in result
        df = result["000001.SZ"]
        assert df is not None
        assert not df.empty
        assert len(df) == 10

    @pytest.mark.network
    @pytest.mark.mootdx
    def test_count_parameter(self):
        """Test with different count parameter values."""
        for count_value in [1, 5, 60]:
            result = get_history_data(
                ["600000.SH"],
                count=count_value,
                base_time=datetime.now(),
                period="1d",
            )
            df = result["600000.SH"]
            assert df is not None
            if not df.empty:
                assert len(df) <= count_value, f"Expected <= {count_value} bars, got {len(df)}"

    @pytest.mark.network
    @pytest.mark.mootdx
    def test_multiple_stocks(self):
        """Test requesting data for multiple stocks at once."""
        result = get_history_data(
            ["600000.SH", "000001.SZ"],
            count=5,
            base_time=datetime.now(),
            period="1d",
        )
        assert "600000.SH" in result
        assert "000001.SZ" in result
        for code in ["600000.SH", "000001.SZ"]:
            df = result[code]
            assert df is not None
            if not df.empty:
                assert len(df) > 0

    @pytest.mark.network
    @pytest.mark.mootdx
    def test_time_within_30_seconds(self):
        """Verify get_history_data() completes within 30 seconds."""
        start = time.time()
        get_history_data(
            ["600000.SH"],
            count=10,
            base_time=datetime.now(),
            period="1d",
        )
        elapsed = time.time() - start
        assert elapsed < 30, f"Took {elapsed:.1f}s, expected < 30s"

    @pytest.mark.network
    @pytest.mark.mootdx
    def test_result_structure(self):
        """Verify the result DataFrame structure and data types."""
        result = get_history_data(
            ["600000.SH"],
            count=5,
            base_time=datetime.now(),
            period="1d",
        )
        df = result["600000.SH"]
        if df is not None and not df.empty:
            # Check data types
            assert df["time"].dtype.kind in ("i", "u"), "time should be integer"
            assert df["open"].dtype.kind == "f", "open should be float"
            assert df["high"].dtype.kind == "f", "high should be float"
            assert df["low"].dtype.kind == "f", "low should be float"
            assert df["close"].dtype.kind == "f", "close should be float"
            # Basic sanity: high >= low
            assert (df["high"] >= df["low"]).all(), "high should be >= low"

    @pytest.mark.network
    @pytest.mark.mootdx
    def test_nonexistent_stock_returns_empty(self):
        """A nonexistent stock code should return an empty DataFrame, not crash."""
        result = get_history_data(
            ["999999.SH"],
            count=10,
            base_time=datetime.now(),
            period="1d",
        )
        assert "999999.SH" in result
        df = result["999999.SH"]
        # Should be an empty DataFrame with the right columns, not None
        assert df is not None
        assert list(df.columns) == ["time", "open", "high", "low", "close", "volume", "amount", "preClose"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
