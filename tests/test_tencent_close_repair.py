from datetime import date, datetime
import json
import pandas as pd
import pytest

from data.update_tencent_close import parse_close_packet


def packet(*, symbol='sz000001', volume=1000, amount=1000000, stamp='20261102161500', previous=10):
    fields = [''] * 40
    for key, value in {2: symbol[2:], 3: 10, 4: previous, 5: 10.5, 6: volume,
                       30: stamp, 31: 10 - previous, 32: round((10 / previous - 1) * 100, 2),
                       33: 11, 34: 9, 35: f'10/{volume}/{amount}', 36: volume, 37: amount / 10000}.items():
        fields[key] = str(value)
    return 'v_' + symbol + '="' + '~'.join(fields) + '";'


def test_closing_volume_is_normalized_to_lots_by_consistent_turnover():
    day = date(2026, 11, 2)
    normal = parse_close_packet(packet(), {'sz000001'}, day)['sz000001']
    shares = parse_close_packet(packet(symbol='sh688092', volume=100000), {'sh688092'}, day)['sh688092']
    assert normal['volume'] == shares['volume'] == 1000
    assert normal['preClose'] == 10
    assert pd.Timestamp(normal['time'], unit='ms') == pd.Timestamp('2026-11-02 15:00')


@pytest.mark.parametrize('value', [packet(stamp='20261103093000'), packet(stamp='20261102140000'),
    packet(amount=1), packet(volume=float('nan')), packet()+packet()])
def test_bad_or_wrong_date_quote_is_rejected(value):
    with pytest.raises(ValueError):
        parse_close_packet(value, {'sz000001'}, date(2026, 11, 2))


def test_old_quote_is_not_relabelled_as_target_session():
    rows = parse_close_packet(packet(stamp='20261030161500'), {'sz000001'}, date(2026, 11, 2))
    assert rows['sz000001'] is None


@pytest.mark.parametrize('minute,allowed', [(34, False), (35, True)])
def test_closing_source_obeys_the_1535_post_close_boundary(tmp_path, monkeypatch, minute, allowed):
    import data.update_tencent_close as module
    class Clock(datetime):
        @classmethod
        def now(cls):
            return cls(2026, 11, 2, 15, minute)
    monkeypatch.setattr(module, 'datetime', Clock)
    monkeypatch.setattr(module.mdx, 'DATA_DIR', tmp_path)
    cache = tmp_path / 'market_status/tencent_close_2026-11-02.json'
    cache.parent.mkdir()
    cache.write_text(json.dumps(dict(date='2026-11-02', symbols=['sz000001'],
        packets=[dict(symbols=['sz000001'], text=packet(stamp='20261102153500'))])), encoding='utf-8')
    if allowed:
        assert module.fetch_closed_quotes(['000001.SZ'], date(2026, 11, 2))['000001.SZ']['close'] == 10
    else:
        with pytest.raises(ValueError, match='15:35'):
            module.fetch_closed_quotes(['000001.SZ'], date(2026, 11, 2))


def test_preclose_must_match_quote_change_fields():
    text = packet().replace('~0~0.0~', '~1~10~')
    if text == packet():
        fields = packet().split('~')
        fields[31] = '1'
        text = '~'.join(fields)
    with pytest.raises(ValueError, match='reference price'):
        parse_close_packet(text, {'sz000001'}, date(2026, 11, 2))


def test_sina_keeps_unadjusted_prices_and_reuses_corporate_action_reference(monkeypatch, tmp_path):
    import data.update_sina_history as module
    raw = pd.DataFrame([
        dict(date='2026-10-30T00:00:00Z', prevclose=9., open=10., high=11., low=9., close=10., volume=10000., amount=100000.),
        dict(date='2026-11-02T00:00:00Z', prevclose=None, open=10., high=12., low=9., close=11., volume=20000., amount=220000.)])
    monkeypatch.setattr(module, 'fetch_raw_history', lambda code: raw)
    monkeypatch.setattr(module, 'DATA_DIR', tmp_path)
    result = module.fetch_unadjusted_history('000001.SZ', date(2026, 10, 30), date(2026, 11, 2), pd.DataFrame())
    assert result['preClose'].tolist() == [9., 10.]
    assert result['volume'].tolist() == [100., 200.]
    assert pd.to_datetime(result['time'], unit='ms').dt.hour.tolist() == [15, 15]
