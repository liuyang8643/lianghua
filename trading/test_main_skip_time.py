from datetime import datetime

from trading.main import _parse_skip_datetime
from trading.scheduler import PREPARE_REBALANCE_START


def test_skip_1500_minute_aligns_to_rebalance_second():
    assert _parse_skip_datetime("202606221500") == datetime.combine(
        datetime(2026, 6, 22).date(), PREPARE_REBALANCE_START)


def test_skip_accepts_explicit_seconds():
    assert _parse_skip_datetime("20260622150030") == datetime(2026, 6, 22, 15, 0, 30)
