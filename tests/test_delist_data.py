from datetime import date, datetime

import numpy as np
import pandas as pd

from core.backtest import _backtest_direct
from data.db.delist import DelistStockInfo, _parse_delist_frame
from data.update_all import _derive_missing_total_share


def test_parse_delist_frame_keeps_shanghai_and_shenzhen_rows():
    frame = pd.DataFrame([
        {'exchange': 'SH', '公司代码': '600001', '公司简称': '上证退市', '上市日期': '1992-01-01', '暂停上市日期': '2024-01-02'},
        {'exchange': 'SZ', '证券代码': '000001', '证券简称': '深证退市', '上市日期': '1991-01-01', '终止上市日期': '2024-01-03'},
    ])

    result = _parse_delist_frame(frame)

    assert set(result) == {'600001.SH', '000001.SZ'}
    assert result['000001.SZ'].delist_date == date(2024, 1, 3)


def test_derive_missing_total_share_is_pit_and_preserves_official_codes():
    deep = pd.DataFrame([
        {'stock_code': '600001.SH', 'report_period': 20230930, 'net_profit': 100.0, 'eps': 2.0},
        {'stock_code': '600000.SH', 'report_period': 20230930, 'net_profit': 100.0, 'eps': 2.0},
    ])

    result = _derive_missing_total_share(deep, {'600000.SH'})

    assert result.to_dict('records') == [{
        'stock_code': '600001.SH', 'm_anntime': pd.Timestamp('2024-01-28'), 'cap_stk': 0.005,
    }]


def test_backtest_writes_off_seeded_position_after_delist(monkeypatch):
    code = '600001.SH'
    data = {
        'stock_codes': np.array([code]),
        'stock_names': np.array(['退市样本']),
        'trade_dates': np.array(['2024-01-03'], dtype='datetime64[D]'),
        'open': np.array([[10.0]]),
        'high': np.array([[10.0]]),
        'low': np.array([[10.0]]),
        'close': np.array([[10.0]]),
        'preClose': np.array([[10.0]]),
        'st_mask': np.array([[False]]),
        'issue_price': np.array([10.0]),
    }
    monkeypatch.setattr(
        'core.backtest.get_delist_stock_info',
        lambda: {code: DelistStockInfo('退市样本', date(1992, 1, 1), date(2024, 1, 2))},
    )

    result = _backtest_direct(
        data=data,
        all_scores={'F': np.array([[1.0]])},
        valid_dates=[datetime(2024, 1, 3)],
        date_indices=[0],
        valid_stocks=[code],
        stock_indices={code: 0},
        weights={'F': 1.0},
        buy_n=1,
        sell_m=1,
        init_cash=1.0,
        init_total_asset=1001.0,
        init_positions={code: {'volume': 100, 'avg_price': 10.0}},
        filter_masks={'Filter': np.array([[False]])},
        list_dates_map={code: date(1992, 1, 1)},
    )

    assert result['delist_count'] == 1
    assert result['delist_events'][0]['code'] == code
    assert result['current_positions_count'] == 0
    assert result['final_asset'] == 1.0
