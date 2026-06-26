from datetime import datetime

from trading.main import _parse_skip_datetime


def test_skip_0925_minute_aligns_to_rebalance_second():
    assert _parse_skip_datetime("202606220925") == datetime(2026, 6, 22, 9, 25, 10)


def test_skip_accepts_explicit_seconds():
    assert _parse_skip_datetime("20260622092510") == datetime(2026, 6, 22, 9, 25, 10)
