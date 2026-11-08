from datetime import date, datetime
import json
import os
from pathlib import Path

import pandas as pd
import pytest

from trading.bridge_wire import atomic_json, open_database
from trading.holding_log import holding_comparison
from trading.replay_day import prepare_replay


def test_holding_table_preserves_unmanaged_and_nets_same_stock_rows():
    doc = dict(buy_priority=['688021.SH'], targets={
        '688021.SH': dict(shares=2399), '688528.SH': dict(shares=4797)})
    positions = [dict(stock_code='688021.SH', volume=2609),
                 dict(stock_code='688528.SH', volume=4597),
                 dict(stock_code='158905.SZ', volume=100000)]
    rows = {r['code']: r for r in holding_comparison(doc, positions)}
    assert rows['688021.SH']['difference'] == -210
    assert rows['688528.SH']['difference'] == 200
    assert rows['158905.SZ']['managed'] is False
    assert rows['158905.SZ']['difference'] == 0


def test_historical_input_uses_bound_preopen_receipts_and_original_plan(tmp_path):
    root = tmp_path
    (root / 'data/live_file').mkdir(parents=True)
    (root / 'data/live_bridge').mkdir()
    (root / 'data/live_trades').mkdir()
    (root / 'data/k-line').mkdir()
    atomic_json(root / 'data/live_file/config.json', {'account': 'SIMULATED'})
    db = open_database(root / 'data/live_bridge/bridge.sqlite3')
    asset = dict(account_id='SIMULATED', cash=1000, total_asset=2000, market_value=1000)
    positions = [dict(account_id='SIMULATED', stock_code='600001.SH', volume=100,
                      can_use_volume=100, market_value=1000)]
    for index, (operation, result) in enumerate([('asset', asset), ('positions', positions)]):
        stamp = datetime(2026, 11, 2, 9, 25, 17).timestamp() + index * .1
        db.execute('INSERT INTO requests VALUES (?,?,?,?,?,?,?,?,?)',
                   (str(index), stamp, stamp + 10, 1, 'instance', operation,
                    json.dumps({'account_id': 'SIMULATED'}), 'done', json.dumps({'ok': True, 'result': result})))
    db.commit()
    db.close()
    plan_path = root / 'data/live_trades/plan_2026-11-02.parquet'
    pd.DataFrame([dict(date=date(2026, 11, 2), code='600001.SH', name='样本', direction='buy',
                       est_price=10., est_volume=100, plan_seq=1, factor_score=99999)]).to_parquet(plan_path)
    stamp = datetime(2026, 11, 2, 9, 25, 29).timestamp()
    os.utime(plan_path, (stamp, stamp))
    pd.DataFrame({'time': [pd.Timestamp('2026-10-30 15:00').value // 1000000],
                  'close': [10.]}).to_parquet(root / 'data/k-line/600001.SH.parquet')
    path = prepare_replay(root, date(2026, 11, 2))
    data = json.loads(path.read_text())
    assert data['document']['targets']['600001.SH']['shares'] == 200
    assert data['source_receipts'] == ['0', '1']
    assert data['account_captured_at'] == datetime(2026, 11, 2, 9, 25, 17).timestamp() + .1
    assert 'factor_score' not in path.read_text()
    assert data['simulation_only'] is True
    assert prepare_replay(root, date(2026, 11, 2)) == path
    with pytest.raises((FileNotFoundError, ValueError, RuntimeError)):
        prepare_replay(root, date(2026, 11, 3))


def test_preopen_monitor_submits_sells_before_buys_even_with_sufficient_cash():
    from trading import execution_engine as module
    from trading.app_simulation_worker import run_case
    original_time = module.time
    try:
        result = run_case('preopen_sell_before_buy', date.today())
        assert [call['side'] for call in result['fake_submissions']] == [24, 23]
        assert result['final_positions']['600001.SH'] == 300
        assert result['final_positions']['600002.SH'] == 0
    finally:
        module.time = original_time
