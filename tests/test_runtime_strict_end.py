from datetime import datetime

import numpy as np

import core.runtime as runtime_module


def test_load_runtime_npz_strict_end_excludes_future_rows(
    tmp_path,
    monkeypatch,
):
    trade_dates = np.array(
        ['2018-12-20', '2018-12-27', '2018-12-28', '2019-01-02'],
        dtype='datetime64[D]',
    )
    np.savez(
        tmp_path / 'runtime_2018-12-20_2019-01-02.npz',
        trade_dates=trade_dates,
        stock_codes=np.array(['000001.SZ']),
        open=np.ones((4, 1), dtype=float),
    )
    monkeypatch.setattr(runtime_module, '_RUNTIME_DIR', tmp_path)
    requested = [
        datetime(2018, 12, 27),
        datetime(2018, 12, 28),
    ]

    strict = runtime_module.load_runtime_npz(
        requested,
        max_lookback=10,
        strict_end=True,
    )
    strict_without_lookback = runtime_module.load_runtime_npz(
        requested,
        strict_end=True,
    )
    buffered = runtime_module.load_runtime_npz(
        requested,
        max_lookback=10,
    )

    assert strict is not None
    assert strict_without_lookback is not None
    assert buffered is not None
    assert strict['trade_dates'][-1] == np.datetime64('2018-12-28')
    assert (
        strict_without_lookback['trade_dates'][-1]
        == np.datetime64('2018-12-28')
    )
    assert strict['open'].shape[0] == len(strict['trade_dates'])
    assert strict['trade_dates'].base is None
    assert strict['open'].base is None
    assert buffered['trade_dates'][-1] == np.datetime64('2019-01-02')
