"""Tests for local money flow data source.

Tests get_money_flow_data(), get_main_fund_net_inflow(),
get_retail_flow_amount(), and get_retail_net_flow().
"""
import time
from datetime import date, datetime
import pytest
from pathlib import Path

from core.database.money_flow.data import (
    get_money_flow_data,
    get_main_fund_net_inflow,
    get_retail_flow_amount,
    get_retail_net_flow,
)


def _data_available():
    p = Path(__file__).resolve().parents[3] / "data" / "money_flow" / "money_flow.parquet"
    return p.exists()


class TestGetMoneyFlowData:
    """Test get_money_flow_data() for a specific trading date."""

    def test_recent_trading_date(self):
        if not _data_available():
            pytest.skip("资金流向数据 parquet 不存在")
        df = get_money_flow_data(date(2024, 1, 8))
        assert df is not None
        assert not df.empty

    def test_data_has_required_columns(self):
        if not _data_available():
            pytest.skip("资金流向数据 parquet 不存在")
        df = get_money_flow_data(date(2024, 1, 8))
        if df is not None:
            assert "code" in df.columns
            assert "name" in df.columns
            assert "date" in df.columns
            money_cols = [c for c in df.columns if "金额" in c or "净额" in c]
            assert len(money_cols) > 0

    def test_data_has_reasonable_row_count(self):
        if not _data_available():
            pytest.skip("资金流向数据 parquet 不存在")
        df = get_money_flow_data(date(2024, 1, 8))
        if df is not None:
            assert len(df) > 100, f"Only got {len(df)} rows, expected > 100"

    def test_code_column_is_6_digit_string(self):
        if not _data_available():
            pytest.skip("资金流向数据 parquet 不存在")
        df = get_money_flow_data(date(2024, 1, 8))
        if df is not None:
            sample_codes = df["code"].head(10).tolist()
            for code in sample_codes:
                assert len(str(code)) == 6, f"Expected 6-digit code, got: {code}"
                assert str(code).isdigit(), f"Expected numeric code, got: {code}"

    def test_datetime_input_accepted(self):
        if not _data_available():
            pytest.skip("资金流向数据 parquet 不存在")
        df = get_money_flow_data(datetime(2024, 1, 8, 10, 30))
        assert df is not None

    def test_timeout_within_10_seconds(self):
        if not _data_available():
            pytest.skip("资金流向数据 parquet 不存在")
        start = time.time()
        get_money_flow_data(date(2024, 1, 8))
        elapsed = time.time() - start
        assert elapsed < 10, f"Took {elapsed:.1f}s, expected < 10s"


class TestGetMoneyFlowErrorHandling:
    """Test error handling for invalid inputs."""

    def test_future_date(self):
        if not _data_available():
            pytest.skip("资金流向数据 parquet 不存在")
        from datetime import timedelta
        future_date = date.today() + timedelta(days=365)
        result = get_money_flow_data(future_date)
        assert result is None

    def test_weekend_date(self):
        if not _data_available():
            pytest.skip("资金流向数据 parquet 不存在")
        result = get_money_flow_data(date(2024, 11, 30))
        assert result is None

    def test_very_old_date(self):
        if not _data_available():
            pytest.skip("资金流向数据 parquet 不存在")
        result = get_money_flow_data(date(2010, 1, 4))
        assert result is None


class TestGetMainFundNetInflow:
    """Test get_main_fund_net_inflow() for specific stocks."""

    def test_known_stock_main_fund_flow(self):
        if not _data_available():
            pytest.skip("资金流向数据 parquet 不存在")
        inflow = get_main_fund_net_inflow("600000.SH", date(2024, 1, 8))
        assert inflow is None or isinstance(inflow, float)

    def test_invalid_stock_returns_none(self):
        if not _data_available():
            pytest.skip("资金流向数据 parquet 不存在")
        inflow = get_main_fund_net_inflow("999999.SH", date(2024, 1, 8))
        assert inflow is None


class TestGetRetailFlow:
    """Test retail flow helper functions."""

    def test_get_retail_flow_amount(self):
        if not _data_available():
            pytest.skip("资金流向数据 parquet 不存在")
        amount = get_retail_flow_amount("600000.SH", date(2024, 1, 8))
        assert amount is None or isinstance(amount, float)

    def test_get_retail_net_flow(self):
        if not _data_available():
            pytest.skip("资金流向数据 parquet 不存在")
        net = get_retail_net_flow("600000.SH", date(2024, 1, 8))
        assert net is None or isinstance(net, float)

    def test_stock_code_without_suffix(self):
        if not _data_available():
            pytest.skip("资金流向数据 parquet 不存在")
        with_suffix = get_main_fund_net_inflow("600000", date(2024, 1, 8))
        without_suffix = get_main_fund_net_inflow("600000.SH", date(2024, 1, 8))
        assert with_suffix is None or isinstance(with_suffix, float)
        assert without_suffix is None or isinstance(without_suffix, float)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
