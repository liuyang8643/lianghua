"""deep_value._aligned 股票维度对齐测试。"""
import numpy as np

from factor_db.factors.deep_value import _aligned


def test_aligned_reindexes_stocks():
    panel = {
        'trade_dates': [np.datetime64('2020-01-02'), np.datetime64('2020-01-03')],
        'stock_codes': ['000001.SZ', '600000.SH', '999999.SZ'],
    }
    fake_cache = {
        'dates': np.array(['2019-12-31', '2020-01-02', '2020-01-03'], dtype='datetime64[D]'),
        'stock_codes': ['600000.SH', '000001.SZ'],
        'bps': np.array([
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ], dtype=np.float64),
    }
    import factor_db.factors.deep_value as dv
    dv._cache.clear()
    dv._cache.update(fake_cache)

    out = _aligned(panel, 'bps')
    assert out.shape == (2, 3)
    assert out[0, 0] == 4.0   # 000001.SZ (aux col 1)
    assert out[0, 1] == 3.0   # 600000.SH (aux col 0)
    assert np.isnan(out[0, 2])  # 999999.SZ 不在 aux

    dv._cache.clear()
