from datetime import date

import pandas as pd

from data.db.delist import _parse_delist_frame


def test_parse_delist_frame_keeps_shanghai_and_shenzhen_rows():
    frame = pd.DataFrame([
        {'exchange': 'SH', '公司代码': '600001', '公司简称': '上证退市', '上市日期': '1992-01-01', '暂停上市日期': '2024-01-02'},
        {'exchange': 'SZ', '证券代码': '000001', '证券简称': '深证退市', '上市日期': '1991-01-01', '终止上市日期': '2024-01-03'},
    ])

    result = _parse_delist_frame(frame)

    assert set(result) == {'600001.SH', '000001.SZ'}
    assert result['000001.SZ'].delist_date == date(2024, 1, 3)


def test_parse_delist_frame_preserves_duplicate_shanghai_listing_dates():
    frame = pd.DataFrame([
        {'exchange': 'SH', '公司代码': '600190', '公司简称': '样本', '上市日期': '1998-05-19', '暂停上市日期': '2025-07-28'},
        {'exchange': 'SH', '公司代码': '600190', '公司简称': '样本', '上市日期': '1999-06-09', '暂停上市日期': '2025-07-28'},
    ])

    info = _parse_delist_frame(frame)['600190.SH']

    assert info.list_date == date(1998, 5, 19)
    assert info.list_date_candidates == (
        date(1998, 5, 19),
        date(1999, 6, 9),
    )
