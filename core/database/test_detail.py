"""Tests for stock detail data source (AkShare stock_individual_info_em).

Tests get_stock_detail() and internal _fetch_stock_detail_akshare().
"""
import time
import pytest
from core.database.detail import get_stock_detail


class TestGetStockDetail:
    """Test get_stock_detail() against real AkShare API."""

    @pytest.mark.network
    def test_shanghai_stock_600000(self):
        """Test retrieving detail for a well-known Shanghai stock (浦发银行)."""
        detail = get_stock_detail("600000.SH")
        assert detail is not None
        assert detail["InstrumentID"] == "600000"
        assert detail["ExchangeID"] == "SSE"
        assert "InstrumentName" in detail
        assert len(detail["InstrumentName"]) > 0
        assert "OpenDate" in detail

    @pytest.mark.network
    def test_shenzhen_stock_000001(self):
        """Test retrieving detail for a well-known Shenzhen stock (平安银行)."""
        detail = get_stock_detail("000001.SZ")
        assert detail is not None
        assert detail["InstrumentID"] == "000001"
        assert detail["ExchangeID"] == "SZE"
        assert "InstrumentName" in detail
        assert len(detail["InstrumentName"]) > 0

    @pytest.mark.network
    def test_invalid_stock_code_returns_none(self):
        """An invalid stock code should return None, not hang."""
        start = time.time()
        detail = get_stock_detail("999999.SH")
        elapsed = time.time() - start
        assert detail is None
        assert elapsed < 30, f"Took {elapsed:.1f}s, expected < 30s"

    @pytest.mark.network
    def test_nonexistent_stock_code_returns_none(self):
        """A clearly nonexistent stock code should return None."""
        detail = get_stock_detail("000000.SZ")
        assert detail is None

    @pytest.mark.network
    def test_total_volume_is_populated(self):
        """TotalVolume should be > 0 for a normal stock."""
        detail = get_stock_detail("600000.SH")
        assert detail is not None
        assert detail["TotalVolume"] > 0, "TotalVolume should be > 0"

    @pytest.mark.network
    def test_float_volume_is_populated(self):
        """FloatVolume should be > 0 for a normal stock."""
        detail = get_stock_detail("000001.SZ")
        assert detail is not None
        assert detail["FloatVolume"] > 0, "FloatVolume should be > 0"

    @pytest.mark.network
    def test_timeout_within_30_seconds(self):
        """Each call should complete within 30 seconds."""
        start = time.time()
        get_stock_detail("600000.SH")
        get_stock_detail("000001.SZ")
        elapsed = time.time() - start
        assert elapsed < 30, f"Took {elapsed:.1f}s, expected < 30s"

    @pytest.mark.network
    def test_kcb_stock_detail(self):
        """Test retrieving detail for a科创板 stock (688xxx)."""
        detail = get_stock_detail("688001.SH")
        if detail is not None:
            assert detail["ExchangeID"] == "SSE"
            assert "InstrumentName" in detail

    @pytest.mark.network
    def test_cyb_stock_detail(self):
        """Test retrieving detail for a创业板 stock (300xxx or 301xxx)."""
        detail = get_stock_detail("300001.SZ")
        if detail is not None:
            assert detail["ExchangeID"] == "SZE"

    @pytest.mark.network
    def test_open_date_is_reasonable(self):
        """OpenDate should be a non-zero 8-digit string in YYYYMMDD format."""
        detail = get_stock_detail("600000.SH")
        assert detail is not None
        open_date = detail["OpenDate"]
        assert len(open_date) == 8 and open_date.isdigit()
        assert open_date != "00000000"

    @pytest.mark.network
    def test_cache_works_across_calls(self):
        """Second call for the same stock should be faster (cache hit)."""
        # First call to warm cache
        detail1 = get_stock_detail("600000.SH")
        assert detail1 is not None
        # Second call should use shared memory cache
        start = time.time()
        detail2 = get_stock_detail("600000.SH")
        elapsed = time.time() - start
        assert detail2 is not None
        assert detail2["InstrumentID"] == detail1["InstrumentID"]
        # Cache hit should be very fast (under 1 second)
        assert elapsed < 1, f"Cache hit took {elapsed:.3f}s"

    @pytest.mark.network
    def test_instrument_status_defaults_to_zero(self):
        """Active stocks should have InstrumentStatus <= 0."""
        detail = get_stock_detail("600000.SH")
        assert detail is not None
        assert detail["InstrumentStatus"] <= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
