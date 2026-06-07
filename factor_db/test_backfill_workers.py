"""backfill_records 串行路径 smoke test。"""
from unittest.mock import patch

from factor_db.backfill_records import _max_lookback, backfill


class _FakeCls:
    hist_days = 60


def test_max_lookback_picks_max():
    classes = {'A': _FakeCls(), 'B': type('C', (), {'hist_days': 120})()}
    assert _max_lookback(classes, ['A', 'B']) == 120
    assert _max_lookback(classes, ['missing']) is None


def test_backfill_serial_with_preloaded_panel(tmp_path, monkeypatch):
    db_path = tmp_path / 'registry.db'
    monkeypatch.setattr('factor_db.db._DB_PATH', db_path)
    monkeypatch.setattr('factor_db.records._DB_PATH', db_path)

    from factor_db import db, records

    db.init_db()
    records.init_records()
    db.add_factor(
        name='TestFactor', file_path='factor_db/factors/deep_value.py',
        op='seed', generation=0, params_count=0, status='active',
    )

    fake_metrics = {
        'sharpe': 1.23, 'annualized': 15.0, 'max_dd': -10.0, 'n_trades': 42,
        'dates': ['2018-01-02'], 'daily_returns': [0.01], 'topn': [['000001.SZ']],
    }

    with patch('factor_db.backfill_records.get_all_factor_classes') as mock_cls, \
         patch('factor_db.backfill_records.load_runtime_npz', return_value={'open': []}), \
         patch('factor_db.backfill_records.evaluator.build_universe', return_value=([], [])), \
         patch('factor_db.backfill_records.evaluator.evaluate_detailed', return_value=fake_metrics) as mock_eval:
        mock_cls.return_value = {'TestFactor': _FakeCls}
        backfill('20050101', '20181231', 20, 'all_A', force=True, workers=1)

    mock_eval.assert_called_once()
    run = records.get_latest_run('TestFactor')
    assert run is not None
    assert run['sharpe'] == 1.23
