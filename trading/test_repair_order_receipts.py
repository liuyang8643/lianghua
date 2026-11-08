import copy
from datetime import datetime

import pytest

from trading.repair_order_receipts import reconcile


def facts():
    remark = 'wbr:abcdefabcdef:000000'
    request = dict(remark=remark, direction='BUY', code='600001.SH', shares=100,
                   price=10., order_id=None, submission_state='not_submitted', rejection_reason='old zero result')
    state = dict(account='test', id='original', pending={'trade_date':'2026-11-03'},
                 target_document={'immutable':True}, target_sha256='frozen', requests=[request])
    order = dict(account_id='test', order_remark=remark, stock_code='600001.SH', order_type=23,
                 order_volume=100, order_time=datetime(2026,11,3,9,30).timestamp(),
                 strategy_name='original', order_id=123, price=10.)
    snapshot = dict(account='test', trade_date='2026-11-03', captured_at=order['order_time'], orders=[order])
    spec = dict(account_id='test', trade_date='2026-11-03', stock_code='600001.SH', order_type=23,
                volume=100, strategy_name='original', order_remark=remark, price=10., price_type=11)
    return state, snapshot, {remark:spec}


def test_reconciliation_preserves_original_plan_and_audits_wrong_marker():
    state, snapshot, specs = facts()
    before = copy.deepcopy(state)
    result, updates = reconcile(state, snapshot, specs, 'original')
    assert state == before
    for field in ('account','id','pending','target_document','target_sha256'):
        assert result[field] == before[field]
    assert updates == [(123, state['requests'][0]['remark'])]
    row = result['requests'][0]
    assert row['order_id'] == 123 and 'submission_state' not in row
    assert row['reconciliation_history'][0]['previous_rejection_reason'] == 'old zero result'


@pytest.mark.parametrize('change', [dict(order_volume=200),dict(order_type=24),dict(stock_code='600002.SH'),
    dict(account_id='other'),dict(strategy_name='manual'),dict(order_id=0),dict(price=11.),
    dict(order_time=datetime(2026,11,2,9,30).timestamp())])
def test_contradictory_broker_fact_cannot_repair(change):
    state, snapshot, specs = facts()
    snapshot['orders'][0].update(change)
    with pytest.raises(ValueError):
        reconcile(state, snapshot, specs, 'original')


def test_duplicate_or_missing_broker_fact_cannot_repair():
    for count in (0,2):
        state, snapshot, specs = facts()
        snapshot['orders'] *= count
        with pytest.raises(ValueError):
            reconcile(state, snapshot, specs, 'original')


@pytest.mark.parametrize('broken', [None, 'positions', 'pending_order', 'not_completed'])
def test_completed_audit_only_writes_a_fully_settled_plan(tmp_path, monkeypatch, broken):
    import hashlib
    import json
    import sqlite3
    import trading.repair_order_receipts as module
    from trading.bridge_wire import open_database
    now = datetime(2026,11,3,9,55)
    class Clock(datetime):
        @classmethod
        def now(cls):
            return now
    monkeypatch.setattr(module, 'datetime', Clock)
    monkeypatch.setattr(module.time, 'time', lambda: now.timestamp())
    state, snapshot, specs = facts()
    state['completed'] = broken != 'not_completed'
    state['target_document'] = {'targets': {'600001.SH': {'shares':100}}}
    target = json.dumps(state['target_document']).encode()
    state['target_sha256'] = hashlib.sha256(target).hexdigest()
    snapshot.update(captured_at=now.timestamp(), positions=[{'stock_code':'600001.SH','volume':99 if broken=='positions' else 100}])
    snapshot['orders'][0]['order_status'] = 50 if broken=='pending_order' else 56
    directory = tmp_path / 'data/live_file'; (directory/'journal').mkdir(parents=True)
    def write(name, value):
        (directory/name).write_text(json.dumps(value))
    write('config.json', dict(live_enabled=True, account='test',strategy_name='original',price_types=[11,44]))
    write('status.json', dict(live_enabled=True, stage='completed',updated=now.timestamp()))
    write('account_latest.json', snapshot)
    write('journal/execution_2026-11-03.json', state)
    (directory/'targets_2026-11-03.json').write_bytes(target)
    db=open_database(directory/'execution.sqlite3')
    for remark,spec in specs.items():
        db.execute('INSERT INTO order_intents VALUES (?,?,?,NULL)',(remark,json.dumps(spec),'executing'))
    db.commit(); db.close()
    if broken:
        with pytest.raises(RuntimeError, match='Completed audit'):
            module.repair(tmp_path, apply=True, completed_audit=True)
    else:
        assert module.repair(tmp_path, apply=True, completed_audit=True)['matched']==1
    with sqlite3.connect(directory/'execution.sqlite3') as db:
        assert db.execute('SELECT state,result FROM order_intents').fetchone()==(('executing',None) if broken else ('done',123))
