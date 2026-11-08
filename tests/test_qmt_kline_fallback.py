import pandas as pd
import pytest

from data import update_kline as source


def test_qmt_dates_and_units_are_preserved_without_raw_timestamp_shift():
    frame = pd.DataFrame({
        'time': [1788969600000, 1789056000000],
        'open': [10., 11.], 'high': [11., 12.], 'low': [9., 10.],
        'close': [10., 11.], 'preClose': [9., 10.],
        'volume': [100., 200.], 'amount': [100000., 220000.],
    }, index=['20260910', '20260911'])
    raw = source._normalize_qmt(frame, '20260910', '20260910')
    assert pd.to_datetime(raw.time, unit='ms').tolist() == [pd.Timestamp('2026-09-10 15:00')]
    assert raw.volume.tolist() == [100.]
    assert raw.preClose.tolist() == [9.]


@pytest.mark.parametrize('error', [RuntimeError('no nodes'), source.TdxFunctionCallError('bad response')])
def test_fallback_keeps_historical_cutoff(monkeypatch, error):
    calls = []
    def fail(*a, **kw):
        raise error
    monkeypatch.setattr(source.mdx, 'update_recent', fail)
    monkeypatch.setattr(source.mdx, 'resolve_recent_range', lambda *a: ('20260901', '20260911', None))
    monkeypatch.setattr(source, '_qmt_update', lambda *a: calls.append(a) or {})
    source.update_recent(8, anchor_date=pd.Timestamp('2026-09-11').date(), codes=['000001.SZ'])
    assert calls == [('20260901', '20260911', ['000001.SZ'], False)]


def test_invalid_qmt_preclose_is_rejected():
    frame = pd.DataFrame({field: [1.] for field in source.FIELDS}, index=['20260911'])
    frame['preClose'] = float('nan')
    with pytest.raises(ValueError):
        source._normalize_qmt(frame, '20260901', '20260911')


def test_nonempty_but_incomplete_history_is_rejected():
    frame = pd.DataFrame({'close': [10.]}, index=['20260910'])
    with pytest.raises(RuntimeError, match='缺少 1 个交易日'):
        source._validate_qmt_dates(frame, pd.to_datetime(['20260910', '20260911']), '000001.SZ')


@pytest.mark.parametrize('field,value', [('volume', float('nan')), ('amount', -1.)])
def test_invalid_activity_is_rejected(field, value):
    frame = pd.DataFrame({field: [1.] for field in source.FIELDS}, index=['20260911'])
    frame[field] = value
    with pytest.raises(ValueError, match='成交量/金额'):
        source._normalize_qmt(frame, '20260901', '20260911')


def test_zero_price_on_normal_session_is_rejected():
    frame = pd.DataFrame({field: [0.] for field in source.FIELDS}, index=['20260911'])
    frame['suspendFlag'] = 0
    with pytest.raises(ValueError, match='非停牌日'):
        source._normalize_qmt(frame, '20260901', '20260911')


def test_suspension_exemption_requires_matching_code_and_date():
    missing = pd.to_datetime(['20260910', '20260911'])
    rows = [dict(SECUCODE='000001.SZ', SUSPEND_START_TIME='2026-09-10 09:30',
                 SUSPEND_END_TIME='2026-09-10 15:00')]
    cache = {'2026-09-10': rows, '2026-09-11': rows}
    assert source._suspended_dates('000001.SZ', missing, cache).tolist() == [missing[0]]
    assert source._suspended_dates('600000.SH', missing, cache).empty


@pytest.mark.parametrize('first,last', [('2026-09-10 13:00', '2026-09-10 15:00'),
                                       ('2026-09-09 09:30', '2026-09-10 09:30')])
def test_partial_session_suspension_does_not_exempt_missing_daily_bar(first, last):
    row = dict(SECUCODE='000001.SZ', SUSPEND_START_TIME=first, SUSPEND_END_TIME=last)
    assert source._suspended_dates('000001.SZ', pd.to_datetime(['20260910']),
                                   {'2026-09-10': [row]}).empty


def test_ipo_missing_bars_allowed_only_before_the_download_end():
    pending = dict(LISTING_DATE=None, CONTINUOUS_1WORD_NUM='待上市')
    scheduled = dict(LISTING_DATE='2026-09-22 00:00:00', CONTINUOUS_1WORD_NUM=None)
    announced = dict(LISTING_DATE=None, CONTINUOUS_1WORD_NUM=None)
    assert source._ipo_missing_bars_allowed(pending, '20260921')
    assert source._ipo_missing_bars_allowed(scheduled, '20260921')
    assert not source._ipo_missing_bars_allowed(scheduled, '20260922')
    assert not source._ipo_missing_bars_allowed(announced, '20260921')


@pytest.mark.parametrize('pending', [True, False])
def test_empty_new_listing_needs_explicit_ipo_confirmation(tmp_path, monkeypatch, pending):
    from xtquant import xtdata
    from types import SimpleNamespace
    from data.db import delist

    def detail(code):
        return {'OpenDate': '19700101' if code == '301686.SZ' else '19910403'}
    def frames(fields, codes, **kwargs):
        code = codes[0]
        if code == '301686.SZ':
            return {code: pd.DataFrame(columns=source.FIELDS)}
        frame = pd.DataFrame({field: [1.] for field in source.FIELDS}, index=['20260911'])
        frame['suspendFlag'] = 0
        return {code: frame}

    monkeypatch.setattr(source.mdx, 'RAW_DIR', tmp_path)
    monkeypatch.setattr(delist, 'get_delist_stock_info', lambda: {})
    monkeypatch.setattr(xtdata, 'get_client', lambda: SimpleNamespace(is_connected=lambda: True))
    monkeypatch.setattr(xtdata, 'get_instrument_detail', detail)
    monkeypatch.setattr(xtdata, 'download_history_data', lambda *a, **kw: None)
    monkeypatch.setattr(xtdata, 'get_market_data_ex', frames)
    monkeypatch.setattr(source, '_reference_rows', lambda *a: [dict(
        SECURITY_CODE='301686', LISTING_DATE=None, CONTINUOUS_1WORD_NUM='待上市')] if pending else [])
    if pending:
        assert source._qmt_update('20260911', '20260911', codes=['301686.SZ']) == {}
        assert not list(tmp_path.glob('*.parquet'))
    else:
        with pytest.raises(RuntimeError):
            source._qmt_update('20260911', '20260911', codes=['301686.SZ'])


def _patch_qmt(monkeypatch, tmp_path, ipo_row, bars_for_new=False):
    from types import SimpleNamespace
    from xtquant import xtdata
    from data.db import delist

    def detail(code):
        return {'OpenDate': '19700101' if code == '301686.SZ' else '19910403'}

    def frames(fields, codes, **kwargs):
        code = codes[0]
        if code == '301686.SZ' and not bars_for_new:
            return {code: pd.DataFrame(columns=source.FIELDS)}
        frame = pd.DataFrame({field: [1.] for field in source.FIELDS}, index=['20260911'])
        frame['suspendFlag'] = 0
        return {code: frame}

    monkeypatch.setattr(source.mdx, 'RAW_DIR', tmp_path)
    monkeypatch.setattr(delist, 'get_delist_stock_info', lambda: {})
    monkeypatch.setattr(xtdata, 'get_client', lambda: SimpleNamespace(is_connected=lambda: True))
    monkeypatch.setattr(xtdata, 'get_instrument_detail', detail)
    monkeypatch.setattr(xtdata, 'download_history_data', lambda *a, **kw: None)
    monkeypatch.setattr(xtdata, 'get_market_data_ex', frames)
    monkeypatch.setattr(source, '_reference_rows', lambda *a: [ipo_row])


def test_listing_after_download_end_skips_empty_bars(tmp_path, monkeypatch):
    _patch_qmt(monkeypatch, tmp_path, dict(
        SECURITY_CODE='301686', LISTING_DATE='2026-09-14 00:00:00', CONTINUOUS_1WORD_NUM=None))
    assert source._qmt_update('20260911', '20260911', codes=['301686.SZ']) == {}
    assert not list(tmp_path.glob('*.parquet'))


def test_listing_day_without_bars_still_fails(tmp_path, monkeypatch):
    _patch_qmt(monkeypatch, tmp_path, dict(
        SECURITY_CODE='301686', LISTING_DATE='2026-09-11 00:00:00', CONTINUOUS_1WORD_NUM=None))
    with pytest.raises(RuntimeError, match='无上市日期'):
        source._qmt_update('20260911', '20260911', codes=['301686.SZ'])


def test_listing_day_bar_uses_ipo_date_when_qmt_open_date_missing(tmp_path, monkeypatch):
    _patch_qmt(monkeypatch, tmp_path, dict(
        SECURITY_CODE='301686', LISTING_DATE='2026-09-11 00:00:00', CONTINUOUS_1WORD_NUM=None),
        bars_for_new=True)
    assert source._qmt_update('20260911', '20260911', codes=['301686.SZ']) == {}
    assert len(pd.read_parquet(tmp_path / '301686.SZ.parquet')) == 1


def test_preserved_delisted_history_accepts_explicit_missing_but_rejects_zero(tmp_path):
    frame = pd.DataFrame({field: [1., float('nan')] for field in source.FIELDS})
    frame['time'] = [1000, 2000]
    path = tmp_path / 'old.parquet'
    frame.to_parquet(path, index=False)
    source._validate_preserved_history(path)
    frame.loc[0, ['open', 'high', 'low', 'close']] = 0.
    frame.to_parquet(path, index=False)
    with pytest.raises(ValueError):
        source._validate_preserved_history(path)
