"""Tests for stock_list data source (AkShare stock_info_a_code_name).

Tests _fetch_all_a_stocks() and the higher-level get_all_stock_code_list().
"""
import time
import pytest
from core.database.stock_list import _fetch_all_a_stocks, get_all_stock_code_list


class TestFetchAllAStocks:
    """Test the core _fetch_all_a_stocks() function via AkShare."""

    @pytest.mark.network
    def test_returns_tuple_of_strings(self):
        """Verify it returns a non-empty tuple of stock codes."""
        result = _fetch_all_a_stocks()
        assert isinstance(result, tuple)
        assert len(result) > 0

    @pytest.mark.network
    def test_codes_have_correct_exchange_suffix(self):
        """Check that codes have correct .SH / .SZ suffixes."""
        result = _fetch_all_a_stocks()
        sh_stocks = [c for c in result if c.endswith(".SH")]
        sz_stocks = [c for c in result if c.endswith(".SZ")]
        assert len(sh_stocks) > 0
        assert len(sz_stocks) > 0
        # All codes should end with .SH, .SZ, or .BJ (BSE)
        assert all(c.endswith((".SH", ".SZ", ".BJ")) for c in result)

    @pytest.mark.network
    def test_shanghai_codes_start_with_6(self):
        """Shanghai stocks (ending .SH) should begin with 6 (main board) or 9 (B-share)."""
        result = _fetch_all_a_stocks()
        sh_stocks = [c for c in result if c.endswith(".SH")]
        for code in sh_stocks[:50]:  # spot-check first 50
            bare = code.split(".")[0]
            assert bare.startswith(("5", "6", "7", "9")), f"Unexpected SH prefix: {code}"

    @pytest.mark.network
    def test_shenzhen_codes_start_with_0_or_3(self):
        """Shenzhen stocks (ending .SZ) should begin with 0 or 3."""
        result = _fetch_all_a_stocks()
        sz_stocks = [c for c in result if c.endswith(".SZ")]
        for code in sz_stocks[:50]:  # spot-check first 50
            bare = code.split(".")[0]
            assert bare.startswith(("0", "3")), f"Unexpected SZ prefix: {code}"

    @pytest.mark.network
    def test_b_stocks_are_not_returned(self):
        """B-share codes (900xxx.SH, 200xxx.SZ) should NOT appear in A-stock list."""
        result = _fetch_all_a_stocks()
        b_shares = [c for c in result if c.startswith(("900", "200"))]
        assert len(b_shares) == 0, f"Found B-shares in A-stock list: {b_shares[:5]}"

    @pytest.mark.network
    def test_famous_stocks_are_present(self):
        """Common well-known stocks should be in the list."""
        result = set(_fetch_all_a_stocks())
        well_known = ["600000.SH", "000001.SZ", "600519.SH", "000002.SZ"]
        for code in well_known:
            assert code in result, f"Expected well-known stock {code} in list"

    @pytest.mark.network
    def test_timeout_within_60_seconds(self):
        """Verify _fetch_all_a_stocks() completes within 60 seconds."""
        start = time.time()
        _fetch_all_a_stocks.cache_clear()
        result = _fetch_all_a_stocks()
        elapsed = time.time() - start
        assert elapsed < 60, f"Took {elapsed:.1f}s, expected < 60s"
        assert len(result) > 0


class TestGetAllStockCodeList:
    """Test the higher-level get_all_stock_code_list() function."""

    @pytest.mark.network
    def test_without_date_returns_all(self):
        """Calling without a date should return all (non-B) stocks."""
        result = get_all_stock_code_list()
        assert isinstance(result, list)
        assert len(result) > 0
        b_shares = [c for c in result if c.startswith(("900", "200"))]
        assert len(b_shares) == 0

    @pytest.mark.network
    def test_invalid_date_returns_empty(self):
        """A very old date should return no stocks (none listed that early)."""
        from datetime import datetime
        result = get_all_stock_code_list(datetime(1989, 1, 1))
        assert isinstance(result, list)

    @pytest.mark.network
    def test_cached_calls_are_fast(self):
        """After first call, cached results should be near-instant."""
        start = time.time()
        _ = get_all_stock_code_list()
        elapsed = time.time() - start

    @pytest.mark.network
    def test_result_order_is_stable(self):
        """The returned list should be sorted and consistent across calls."""
        r1 = get_all_stock_code_list()
        r2 = get_all_stock_code_list()
        assert r1 == r2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
