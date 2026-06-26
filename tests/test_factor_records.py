"""factor_db.records 单元测试：append-only 明细记录读写。"""
import sqlite3

import pytest

from factor_db import records


@pytest.fixture()
def temp_records(tmp_path, monkeypatch):
    monkeypatch.setattr(records, '_DB_PATH', tmp_path / 'registry.db')
    records.init_records()
    return records


def _add(rec, name, dates, rets, topn, **kw):
    return rec.add_run(
        name, bt_start='19930101', bt_end='20181231', buy_n=20,
        dates=dates, daily_returns=rets, topn=topn, **kw,
    )


def test_add_and_get_detail_roundtrip(temp_records):
    pk = _add(temp_records, 'F1', ['2020-01-01', '2020-01-02'], [1.0, -0.5],
              [['600000', '000001'], ['600000']], sharpe=1.23, annualized=42.0)
    assert pk == 1
    assert temp_records.has_run('F1') is True
    assert temp_records.has_run('Nope') is False

    summary = temp_records.get_latest_run('F1')
    assert summary['sharpe'] == 1.23
    assert summary['n_days'] == 2
    assert 'dates' not in summary  # detail=False 不解压

    detail = temp_records.get_latest_run('F1', detail=True)
    assert detail['dates'] == ['2020-01-01', '2020-01-02']
    assert detail['daily_returns'] == [1.0, -0.5]
    assert detail['topn'] == [['600000', '000001'], ['600000']]


def test_latest_run_picks_newest(temp_records):
    _add(temp_records, 'F', ['2020-01-01'], [1.0], [['a']], sharpe=0.1)
    _add(temp_records, 'F', ['2020-01-01'], [2.0], [['b']], sharpe=0.9)
    assert temp_records.get_latest_run('F')['sharpe'] == 0.9
    assert len(temp_records.list_runs()) == 2


def test_append_only_triggers(temp_records):
    _add(temp_records, 'F', ['2020-01-01'], [1.0], [['a']])
    with sqlite3.connect(temp_records._DB_PATH) as conn:
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("UPDATE factor_runs SET sharpe=9 WHERE factor_name='F'")
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("DELETE FROM factor_runs WHERE factor_name='F'")
        assert conn.execute('SELECT COUNT(*) FROM factor_runs').fetchone()[0] == 1


def test_no_update_delete_api(temp_records):
    for forbidden in ('update_run', 'delete_run', 'remove_run', 'update', 'delete'):
        assert not hasattr(records, forbidden)
