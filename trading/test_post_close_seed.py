from datetime import date

import pandas as pd

from trading.post_close import _extract_live_seed


def _summary(prev_date, cash):
    return pd.DataFrame([{'date': prev_date, 'cash': cash, 'total_asset': cash}])


def test_extract_live_seed_basic():
    prev = date(2026, 5, 29)
    pos_df = pd.DataFrame([
        {'code': '600000.SH', 'volume': 1000, 'avg_price': 10.0, 'last_price': 11.0},
        {'code': '000001.SZ', 'volume': 500, 'avg_price': 20.0, 'last_price': 19.0},
        {'code': '300001.SZ', 'volume': 0, 'avg_price': 5.0, 'last_price': 5.0},  # 空仓行应被忽略
    ])
    seed = _extract_live_seed(pos_df, _summary(prev, 123_456.0), prev)
    assert seed is not None
    seed_cash, seed_positions, y_eod = seed

    assert seed_cash == 123_456.0
    assert set(seed_positions) == {'600000.SH', '000001.SZ'}
    assert seed_positions['600000.SH'] == {'volume': 1000, 'avg_price': 10.0}

    # T-1 收盘市值用 last_price 估
    mv = {p['code']: p['current_value'] for p in y_eod}
    assert mv['600000.SH'] == 1000 * 11.0
    assert mv['000001.SZ'] == 500 * 19.0


def test_extract_live_seed_returns_none_when_summary_missing_prev_date():
    prev = date(2026, 5, 29)
    pos_df = pd.DataFrame([
        {'code': '600000.SH', 'volume': 1000, 'avg_price': 10.0, 'last_price': 11.0},
    ])
    # daily_summary 没有 prev_date 行 → 无法确定种子现金
    other = _summary(date(2026, 5, 28), 100.0)
    assert _extract_live_seed(pos_df, other, prev) is None


def test_extract_live_seed_returns_none_when_no_positions():
    prev = date(2026, 5, 29)
    empty_pos = pd.DataFrame(columns=['code', 'volume', 'avg_price', 'last_price'])
    assert _extract_live_seed(empty_pos, _summary(prev, 100.0), prev) is None


def test_extract_live_seed_returns_none_when_all_positions_zero():
    prev = date(2026, 5, 29)
    pos_df = pd.DataFrame([
        {'code': '600000.SH', 'volume': 0, 'avg_price': 10.0, 'last_price': 11.0},
    ])
    assert _extract_live_seed(pos_df, _summary(prev, 100.0), prev) is None
