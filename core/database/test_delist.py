"""Tests for delist data source (AkShare stock_info_sh_delist / stock_info_sz_delist).

Tests get_delist_stock_info() and the DelistStockInfo data structure.
"""
import time
import pytest
from datetime import date
from data.db.delist import get_delist_stock_info, DelistStockInfo


class TestGetDelistStockInfo:
    """Test get_delist_stock_info() via AkShare."""

    @pytest.mark.network
    def test_returns_dict(self):
        """Verify it returns a dictionary."""
        result = get_delist_stock_info()
        assert isinstance(result, dict)

    @pytest.mark.network
    def test_has_delisted_stocks(self):
        """There should be at least some delisted stocks in the result."""
        result = get_delist_stock_info()
        # There are usually many delisted stocks in both exchanges
        assert len(result) > 0, "Expected at least some delisted stocks"

    @pytest.mark.network
    def test_values_are_delist_stock_info(self):
        """Each value should be a DelistStockInfo named tuple."""
        result = get_delist_stock_info()
        for v in result.values():
            assert isinstance(v, DelistStockInfo)

    @pytest.mark.network
    def test_delist_stock_info_fields(self):
        """DelistStockInfo should have name, list_date, delist_date."""
        result = get_delist_stock_info()
        for v in result.values():
            assert isinstance(v.name, str), f"Expected str name, got {type(v.name)}"
            assert isinstance(v.list_date, date), f"Expected date list_date, got {type(v.list_date)}"
            assert isinstance(v.delist_date, date), f"Expected date delist_date, got {type(v.delist_date)}"

    @pytest.mark.network
    def test_delist_date_is_after_list_date(self):
        """For each stock, delist_date should be >= list_date."""
        result = get_delist_stock_info()
        for code, info in result.items():
            assert info.delist_date >= info.list_date, (
                f"{code}: delist_date {info.delist_date} < list_date {info.list_date}"
            )

    @pytest.mark.network
    def test_codes_have_correct_suffix(self):
        """Delisted stock codes should end with .SH or .SZ."""
        result = get_delist_stock_info()
        for code in result.keys():
            assert code.endswith((".SH", ".SZ")), f"Unexpected code suffix: {code}"

    @pytest.mark.network
    def test_shanghai_delisted_stocks_present(self):
        """There should be some Shanghai (.SH) delisted stocks."""
        result = get_delist_stock_info()
        sh_stocks = [c for c in result if c.endswith(".SH")]
        assert len(sh_stocks) > 0, "Expected at least some SH delisted stocks"

    @pytest.mark.network
    def test_shenzhen_delisted_stocks_present(self):
        """There should be some Shenzhen (.SZ) delisted stocks."""
        result = get_delist_stock_info()
        sz_stocks = [c for c in result if c.endswith(".SZ")]
        assert len(sz_stocks) > 0, "Expected at least some SZ delisted stocks"

    @pytest.mark.network
    def test_known_delisted_stock_sh(self):
        """Check for a known delisted Shanghai stock (e.g., 600532, 华阳新材 / formerly *ST华资)."""
        result = get_delist_stock_info()
        # 600532 is a known delisted SH stock
        if "600532.SH" in result:
            info = result["600532.SH"]
            assert info.list_date is not None
            assert info.delist_date is not None

    @pytest.mark.network
    def test_known_delisted_stock_sz(self):
        """Check for a known delisted Shenzhen stock."""
        result = get_delist_stock_info()
        # 300799 is a known delisted SZ stock
        if "300799.SZ" in result:
            info = result["300799.SZ"]
            assert info.list_date is not None
            assert info.delist_date is not None

    @pytest.mark.network
    def test_get_delist_stock_info(self):
        """读取退市股票信息并验证返回格式。"""
        result = get_delist_stock_info()
        assert isinstance(result, dict)
        assert len(result) > 0

    @pytest.mark.network
    def test_timeout_within_60_seconds(self):
        """Verify get_delist_stock_info() completes within 60 seconds."""
        start = time.time()
        result = get_delist_stock_info()
        elapsed = time.time() - start
        assert elapsed < 60, f"Took {elapsed:.1f}s, expected < 60s"
        assert len(result) > 0

    @pytest.mark.network
    def test_delist_stock_info_repr(self):
        """Verify the named tuple has a sensible string representation."""
        result = get_delist_stock_info()
        if result:
            code = next(iter(result))
            info = result[code]
            repr_str = repr(info)
            assert info.name in repr_str, f"Expected name in repr: {repr_str}"
            assert "list_date=" in repr_str, f"Expected list_date field in repr: {repr_str}"
            assert "delist_date=" in repr_str, f"Expected delist_date field in repr: {repr_str}"
            # Named tuple repr uses datetime.date(year, month, day) format
            date_repr = f"datetime.date({info.list_date.year}, {info.list_date.month}, {info.list_date.day})"
            assert date_repr in repr_str, f"Expected {date_repr} in repr: {repr_str}"

    @pytest.mark.network
    def test_delist_stock_info_equality(self):
        """Verify equality comparison of DelistStockInfo."""
        info1 = DelistStockInfo("Test", date(2020, 1, 1), date(2023, 1, 1))
        info2 = DelistStockInfo("Test", date(2020, 1, 1), date(2023, 1, 1))
        info3 = DelistStockInfo("Other", date(2020, 1, 1), date(2023, 1, 1))
        assert info1 == info2
        assert info1 != info3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
