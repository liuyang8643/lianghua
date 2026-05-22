"""Tests for CNINFO stock name data source.

Tests get_stock_name_at_date(), is_st_at_date(), is_star_st_at_date(),
prefetch_stock_histories(), clear_stock_name_cache(), and build_st_mask().
"""
import time
from datetime import date
import pytest

from data.db.stock_name import (
    get_stock_name_at_date,
    is_st_at_date,
    is_star_st_at_date,
    clear_stock_name_cache,
)


def setup_function():
    """Clear cache before each test to ensure fresh results."""
    clear_stock_name_cache()


class TestGetStockNameAtDate:
    """Test historical stock name queries."""

    @pytest.mark.network
    @pytest.mark.cninfo
    def test_sh_stock_before_rename(self):
        """Test stock name before a rename event."""
        name = get_stock_name_at_date("600186.SH", date(2019, 4, 1))
        assert name is not None
        # Before 2019-04-29, 600186 was not *ST
        assert "ST" not in name.upper(), f"Expected no ST, got: {name}"

    @pytest.mark.network
    @pytest.mark.cninfo
    def test_sh_stock_during_st(self):
        """Test stock name during ST period."""
        name = get_stock_name_at_date("600186.SH", date(2019, 5, 1))
        assert name is not None
        # After 2019-04-29, 600186 was *ST莲花
        assert "ST" in name.upper(), f"Expected ST in name, got: {name}"

    @pytest.mark.network
    @pytest.mark.cninfo
    def test_sh_stock_after_st_removal(self):
        """Test stock name after ST removal."""
        name = get_stock_name_at_date("600186.SH", date(2021, 1, 1))
        assert name is not None
        assert "ST" not in name.upper(), f"Expected no ST after removal, got: {name}"

    @pytest.mark.network
    @pytest.mark.cninfo
    def test_no_rename_stock(self):
        """Test a stock that has never been renamed (贵州茅台)."""
        name = get_stock_name_at_date("600519.SH", date(2023, 1, 1))
        assert name is not None
        assert "茅台" in name

    @pytest.mark.network
    @pytest.mark.cninfo
    def test_unknown_stock_returns_none(self):
        """An unknown stock code should return None."""
        name = get_stock_name_at_date("999999.SH", date(2023, 1, 1))
        assert name is None

    @pytest.mark.network
    @pytest.mark.cninfo
    def test_timeout_within_30_seconds(self):
        """Verify the function completes within 30 seconds (CNINFO has 15s internal timeout)."""
        start = time.time()
        get_stock_name_at_date("600519.SH", date(2023, 1, 1))
        elapsed = time.time() - start
        assert elapsed < 30, f"Took {elapsed:.1f}s, expected < 30s"

    @pytest.mark.network
    @pytest.mark.cninfo
    def test_current_name_works(self):
        """Querying for today's date should return the current name."""
        name = get_stock_name_at_date("600519.SH", date.today())
        assert name is not None


class TestIsSTAtDate:
    """Test historical ST status detection."""

    @pytest.mark.network
    @pytest.mark.cninfo
    def test_600186_not_st_early(self):
        """Test that 600186 was not ST before 2019."""
        assert is_st_at_date("600186.SH", date(2018, 1, 1)) is False

    @pytest.mark.network
    @pytest.mark.cninfo
    def test_600186_is_st(self):
        """Test that 600186 was ST in mid-2019."""
        assert is_st_at_date("600186.SH", date(2019, 6, 1)) is True

    @pytest.mark.network
    @pytest.mark.cninfo
    def test_600186_st_removed_later(self):
        """Test that 600186 had ST removed by 2021."""
        assert is_st_at_date("600186.SH", date(2021, 1, 1)) is False

    @pytest.mark.network
    @pytest.mark.cninfo
    def test_normal_stock_never_st(self):
        """Test that a normal stock (600519) is never ST."""
        assert is_st_at_date("600519.SH", date(2023, 1, 1)) is False

    @pytest.mark.network
    @pytest.mark.cninfo
    def test_datetime_input_accepted(self):
        """The function should accept datetime objects as well."""
        from datetime import datetime
        result = is_st_at_date("600519.SH", datetime(2023, 6, 1, 10, 30))
        assert result is False

    @pytest.mark.network
    @pytest.mark.cninfo
    def test_timeout_within_30_seconds(self):
        """Verify ST detection completes within 30 seconds."""
        start = time.time()
        is_st_at_date("600186.SH", date(2019, 6, 1))
        elapsed = time.time() - start
        assert elapsed < 30, f"Took {elapsed:.1f}s, expected < 30s"


class TestIsStarSTAtDate:
    """Test *ST / delisting status detection."""

    @pytest.mark.network
    @pytest.mark.cninfo
    def test_600186_was_star_st(self):
        """600186 was *ST (pix*) so is_star should be True during that period."""
        assert is_star_st_at_date("600186.SH", date(2019, 6, 1)) is True

    @pytest.mark.network
    @pytest.mark.cninfo
    def test_normal_stock_not_star(self):
        """A normal stock should not be *ST."""
        assert is_star_st_at_date("600519.SH", date(2023, 1, 1)) is False

    @pytest.mark.network
    @pytest.mark.cninfo
    def test_before_st_not_star(self):
        """Before ST period, is_star should be False."""
        assert is_star_st_at_date("600186.SH", date(2018, 1, 1)) is False


class TestBuildStMask:
    """Test build_st_mask() for batch ST status panel construction."""

    @pytest.mark.network
    @pytest.mark.cninfo
    def test_small_stock_list(self):
        """Test build_st_mask() with a small list of stocks and dates."""
        from data.db.stock_name import build_st_mask

        trade_dates = [
            date(2018, 1, 1),
            date(2019, 6, 1),
            date(2021, 1, 1),
        ]
        stock_codes = ["600186.SH", "600519.SH"]
        mask = build_st_mask(stock_codes, trade_dates)

        assert mask is not None
        import pandas as pd
        assert isinstance(mask, pd.DataFrame)
        assert list(mask.index) == trade_dates
        assert list(mask.columns) == stock_codes

        # 600186: not ST in 2018, ST in 2019, not ST in 2021
        assert mask.loc[date(2018, 1, 1), "600186.SH"] == False
        assert mask.loc[date(2019, 6, 1), "600186.SH"] == True
        assert mask.loc[date(2021, 1, 1), "600186.SH"] == False

        # 600519: never ST
        assert mask.loc[date(2018, 1, 1), "600519.SH"] == False
        assert mask.loc[date(2019, 6, 1), "600519.SH"] == False
        assert mask.loc[date(2021, 1, 1), "600519.SH"] == False

    @pytest.mark.network
    @pytest.mark.cninfo
    def test_empty_inputs(self):
        """Empty input lists should return an empty DataFrame."""
        from data.db.stock_name import build_st_mask
        import pandas as pd

        mask1 = build_st_mask([], [date(2023, 1, 1)])
        assert isinstance(mask1, pd.DataFrame) and mask1.empty

        mask2 = build_st_mask(["600519.SH"], [])
        assert isinstance(mask2, pd.DataFrame) and mask2.empty

    @pytest.mark.network
    @pytest.mark.cninfo
    def test_single_stock_single_date(self):
        """Minimal case: one stock, one date."""
        from data.db.stock_name import build_st_mask

        mask = build_st_mask(["600519.SH"], [date(2023, 1, 1)])
        assert mask is not None
        assert mask.shape == (1, 1)
        assert mask.iloc[0, 0] == False

    @pytest.mark.network
    @pytest.mark.cninfo
    def test_timeout_within_60_seconds(self):
        """build_st_mask for a small set should complete in under 60 seconds."""
        from data.db.stock_name import build_st_mask

        start = time.time()
        trade_dates = [
            date(2018, 1, 1),
            date(2019, 6, 1),
            date(2021, 1, 1),
        ]
        build_st_mask(["600186.SH", "600519.SH"], trade_dates)
        elapsed = time.time() - start
        assert elapsed < 60, f"Took {elapsed:.1f}s, expected < 60s"


class TestPrefetch:
    """Test prefetch_stock_histories() for batch loading."""

    @pytest.mark.network
    @pytest.mark.cninfo
    def test_prefetch_small_set(self):
        """Test prefetching a small set of stock histories."""
        from data.db.stock_name import prefetch_stock_histories

        count = prefetch_stock_histories(["600186.SH", "600519.SH"])
        assert isinstance(count, int)
        # Should either have fetched from network or used cache
        assert count >= 0

    @pytest.mark.network
    @pytest.mark.cninfo
    def test_prefetch_empty_list(self):
        """Prefetching an empty list should return 0."""
        from data.db.stock_name import prefetch_stock_histories

        count = prefetch_stock_histories([])
        assert count == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
