"""Tests for local financial data source."""
import time
from datetime import date
import pytest

from core.database.financial.data import (
    get_financial_data,
    get_financial_indicator,
    get_financial_indicators,
    get_eps,
    get_roe,
)


def _data_available():
    from pathlib import Path
    p = Path(__file__).resolve().parents[2] / "data" / "financial" / "pershare_index.parquet"
    return p.exists()


class TestGetFinancialData:
    """Test get_financial_data() for PershareIndex from local parquet."""

    def test_pershare_index_for_600000(self):
        if not _data_available():
            pytest.skip("财务数据 parquet 不存在")
        df = get_financial_data("600000.SH", "PershareIndex")
        assert df is not None
        assert not df.empty
        assert "m_timetag" in df.columns
        assert "m_anntime" in df.columns
        assert "stock_code" in df.columns
        assert "s_fa_eps_basic" in df.columns

    def test_eps_is_accessible(self):
        if not _data_available():
            pytest.skip("财务数据 parquet 不存在")
        df = get_financial_data("600000.SH", "PershareIndex")
        assert df is not None
        eps_values = df["s_fa_eps_basic"].dropna()
        assert len(eps_values) > 0

    def test_roe_is_accessible(self):
        if not _data_available():
            pytest.skip("财务数据 parquet 不存在")
        df = get_financial_data("600000.SH", "PershareIndex")
        assert df is not None
        roe_values = df["du_return_on_equity"].dropna()
        assert len(roe_values) > 0

    def test_invalid_stock_returns_none(self):
        if not _data_available():
            pytest.skip("财务数据 parquet 不存在")
        result = get_financial_data("999999.SH", "PershareIndex")
        assert result is None or result.empty

    def test_invalid_table_name(self):
        if not _data_available():
            pytest.skip("财务数据 parquet 不存在")
        result = get_financial_data("600000.SH", "NonExistentTable")
        assert result is None

    def test_timeout_within_10_seconds(self):
        if not _data_available():
            pytest.skip("财务数据 parquet 不存在")
        start = time.time()
        get_financial_data("600000.SH", "PershareIndex")
        elapsed = time.time() - start
        assert elapsed < 10, f"Took {elapsed:.1f}s, expected < 10s"


class TestGetFinancialIndicator:
    """Test get_financial_indicator() and convenient wrappers."""

    def test_get_eps_by_indicator(self):
        if not _data_available():
            pytest.skip("财务数据 parquet 不存在")
        eps = get_financial_indicator("600000.SH", date(2024, 6, 30), "s_fa_eps_basic")
        assert eps is None or isinstance(eps, float)

    def test_get_roe_by_indicator(self):
        if not _data_available():
            pytest.skip("财务数据 parquet 不存在")
        roe = get_financial_indicator("600000.SH", date(2024, 6, 30), "du_return_on_equity")
        assert roe is None or isinstance(roe, float)

    def test_get_eps_convenience(self):
        if not _data_available():
            pytest.skip("财务数据 parquet 不存在")
        eps = get_eps("600000.SH", date(2024, 6, 30))
        assert eps is None or isinstance(eps, float)

    def test_get_roe_convenience(self):
        if not _data_available():
            pytest.skip("财务数据 parquet 不存在")
        roe = get_roe("600000.SH", date(2024, 6, 30))
        assert roe is None or isinstance(roe, float)

    def test_invalid_stock_indicator(self):
        if not _data_available():
            pytest.skip("财务数据 parquet 不存在")
        result = get_financial_indicator("999999.SH", date(2024, 6, 30), "s_fa_eps_basic")
        assert result is None

    def test_get_financial_indicators_batch(self):
        if not _data_available():
            pytest.skip("财务数据 parquet 不存在")
        result = get_financial_indicators(
            "600000.SH",
            date(2024, 6, 30),
            ["s_fa_eps_basic", "du_return_on_equity", "s_fa_bps"],
        )
        assert isinstance(result, dict)
        assert "s_fa_eps_basic" in result
        assert "du_return_on_equity" in result
        assert "s_fa_bps" in result

    def test_data_leakage_prevention(self):
        if not _data_available():
            pytest.skip("财务数据 parquet 不存在")
        result = get_financial_indicator(
            "600000.SH", date(2000, 1, 1), "s_fa_eps_basic", use_announce_date=True
        )
        assert result is None

    def test_timeout_within_10_seconds(self):
        if not _data_available():
            pytest.skip("财务数据 parquet 不存在")
        start = time.time()
        get_financial_indicator("600000.SH", date(2024, 6, 30), "s_fa_eps_basic")
        elapsed = time.time() - start
        assert elapsed < 10, f"Took {elapsed:.1f}s, expected < 10s"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
