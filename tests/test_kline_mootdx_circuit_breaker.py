from pathlib import Path

import pandas as pd
import pytest


class _EmptyMdx:
    def __init__(self):
        self.calls = []
        self.closed = False

    def bars(self, **kwargs):
        self.calls.append(kwargs['symbol'])
        return pd.DataFrame()

    def xdxr(self, **kwargs):
        return pd.DataFrame()

    def close(self):
        self.closed = True


class _ProbeOnlyMdx(_EmptyMdx):
    def bars(self, **kwargs):
        self.calls.append(kwargs['symbol'])
        return pd.DataFrame({'open': [1.0]})


def test_update_recent_stops_when_health_probe_is_empty(tmp_path, monkeypatch):
    import data.kline_mootdx as km

    mdx = _EmptyMdx()
    monkeypatch.setattr(km, 'RAW_DIR', Path(tmp_path))
    monkeypatch.setattr(km, '_connect_mootdx', lambda: mdx)
    monkeypatch.setattr(km, 'resolve_recent_range',
                        lambda days, anchor: ('20260903', '20260903', pd.Timestamp('2026-09-03').date()))
    codes = ['000001.SZ', '600000.SH', '000002.SZ', '000004.SZ']
    for code in codes:
        (tmp_path / f'{code}.parquet').touch()

    with pytest.raises(RuntimeError, match='健康探测 3 只全部无数据'):
        km.update_recent(1, codes=codes)

    assert mdx.calls == ['000001', '600000', '000002']
    assert mdx.closed is True


def test_update_recent_probes_even_when_all_codes_are_new(tmp_path, monkeypatch):
    import data.kline_mootdx as km

    mdx = _EmptyMdx()
    monkeypatch.setattr(km, 'RAW_DIR', Path(tmp_path))
    monkeypatch.setattr(km, '_connect_mootdx', lambda: mdx)
    monkeypatch.setattr(km, 'resolve_recent_range',
                        lambda days, anchor: ('20260903', '20260903', pd.Timestamp('2026-09-03').date()))

    with pytest.raises(RuntimeError, match='健康探测 3 只全部无数据'):
        km.update_recent(1, codes=['000001.SZ', '600000.SH', '000002.SZ'])

    assert mdx.calls == ['000001', '600000', '000002']


def test_new_code_full_download_circuit_breaks_after_consecutive_empty(
        tmp_path, monkeypatch):
    import data.kline_mootdx as km

    mdx = _ProbeOnlyMdx()
    monkeypatch.setattr(km, 'RAW_DIR', Path(tmp_path))
    monkeypatch.setattr(km, '_connect_mootdx', lambda: mdx)
    monkeypatch.setattr(km, 'resolve_recent_range',
                        lambda days, anchor: ('20260903', '20260903', pd.Timestamp('2026-09-03').date()))
    monkeypatch.setattr(km, '_fetch_bars_all', lambda mdx, code: pd.DataFrame())
    codes = [f'{i:06d}.SZ' for i in range(1, 20)]

    with pytest.raises(RuntimeError, match='已在全量下载 12/19 熔断'):
        km.update_recent(1, codes=codes)


def test_connect_mootdx_fails_over_until_server_returns_bars(monkeypatch):
    import data.kline_mootdx as km
    import mootdx.quotes

    made = []

    class _Quote:
        def __init__(self, works):
            self.works = works
            self.closed = False

        def bars(self, **kwargs):
            return pd.DataFrame({'open': [1.0]}) if self.works else pd.DataFrame()

        def close(self):
            self.closed = True

    def factory(**kwargs):
        quote = _Quote(works=len(made) == 1)
        made.append((kwargs, quote))
        return quote

    monkeypatch.setattr(mootdx.quotes.Quotes, 'factory', factory)
    result = km._connect_mootdx()

    assert result is made[1][1]
    assert made[0][1].closed is True
    assert made[0][0]['server'] == km.MOOTDX_SERVERS[0]
    assert made[1][0]['server'] == km.MOOTDX_SERVERS[1]
    assert made[1][0]['auto_retry'] is False


def test_recent_update_uses_ex_dividend_reference(tmp_path, monkeypatch):
    import data.kline_mootdx as km

    class Quote:
        def bars(self, **kwargs):
            return pd.DataFrame({
                'datetime': pd.to_datetime(['2026-09-10', '2026-09-11']),
                'open': [10., 9.], 'close': [10., 9.],
                'high': [10., 9.], 'low': [10., 9.],
                'volume': [100., 100.], 'amount': [1000., 900.],
            })

        def xdxr(self, **kwargs):
            return pd.DataFrame([dict(year=2026, month=9, day=11, category=1,
                                      fenhong=10., songzhuangu=0., peigu=0., peigujia=0.)])

    path = tmp_path / '000001.SZ.parquet'
    path.touch()
    monkeypatch.setattr(km, 'RAW_DIR', tmp_path)
    monkeypatch.setattr(km, 'resolve_recent_range',
                        lambda days, anchor: ('20260911', '20260911', pd.Timestamp('2026-09-11').date()))
    km._update_recent_connected(Quote(), 1, codes=['000001.SZ'])
    saved = pd.read_parquet(path)
    assert saved['preClose'].tolist() == [9.]


@pytest.mark.parametrize('returns_none', [False, True])
def test_xdxr_failure_does_not_write_kline(tmp_path, monkeypatch, returns_none):
    import data.kline_mootdx as km

    class Quote:
        def xdxr(self, **kwargs):
            if returns_none:
                return None
            raise RuntimeError('xdxr unavailable')

    monkeypatch.setattr(km, 'RAW_DIR', tmp_path)
    with pytest.raises(RuntimeError):
        km.download(Quote(), ['000001.SZ'], '19900101', '20260911')
    assert not list(tmp_path.glob('*.parquet'))


@pytest.mark.parametrize('category', [9, 11, 12, 15])
def test_unsupported_adjustment_has_no_executable_reference(category):
    import numpy as np
    import data.kline_mootdx as km

    times = pd.to_datetime(['2026-09-10', '2026-09-11']).as_unit('ms').astype('int64').to_numpy()
    events = pd.DataFrame([dict(year=2026, month=9, day=11, category=category)])
    result = km._compute_preclose(np.array([10., 9.]), times, events)
    assert np.isnan(result[1])
