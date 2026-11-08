import json
from types import SimpleNamespace

import pytest

from trading.target_file import diff_positions, reconcile_file
from trading.test_bridge import bridge


def position(code, shares, available=None):
    return SimpleNamespace(stock_code=code, volume=shares,
                           can_use_volume=shares if available is None else available,
                           market_value=1000.0)


def setup_target(server, broker, root, targets):
    broker.query_stock_positions = lambda account: [position('688092.SH', 200)]
    path = root / 'data/live_bridge/targets/test.json'
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({'account': 'test-account', 'trade_date': '2026-11-02',
                               'expires': server.clock() + 300,
                               'batch_id': '123456abcdef', 'targets': targets}))
    return str(path)


def test_preview_reads_file_and_queries_real_adapter_without_submitting(bridge):
    server, broker, root = bridge
    path = setup_target(server, broker, root, {'688092.SH': {'shares': 400, 'price': 20}})
    result = server.dispatch('preview_targets', {'account_id': 'test-account', 'filename': path}, 1)
    assert result['diff'][0]['shares'] == 200
    assert result['diff'][0]['reason'] == 'buy'
    assert result['dry_run'] is True
    assert broker.submissions == []


def test_filled_targets_are_zero_and_out_of_scope_positions_untouched():
    positions = {'688092.SH': {'volume': 400, 'available': 200},
                 '600001.SH': {'volume': 1000, 'available': 1000}}
    result = diff_positions({'688092.SH': {'shares': 400, 'price': 20}}, positions, [])
    assert len(result) == 1
    assert result[0]['shares'] == 0


@pytest.mark.parametrize('status', [48, 49, 50, 51, 52, 55])
def test_manual_inflight_and_cancel_pending_block_new_orders(status):
    order = SimpleNamespace(stock_code='603208.SH', order_status=status,
                            order_volume=4200, traded_volume=400)
    result = diff_positions({'603208.SH': {'shares': 0, 'price': 9.8}},
                            {'603208.SH': {'volume': 3800, 'available': 0}}, [order])
    assert result[0]['pending_shares'] == 3800
    assert result[0]['shares'] == 0


def test_sell_uses_available_once_and_preserves_t1_restriction():
    result = diff_positions({'688092.SH': {'shares': 0, 'price': 20}},
                            {'688092.SH': {'volume': 1000, 'available': 210}}, [])
    assert result[0]['shares'] == 210


def test_execute_is_disabled_by_default(bridge):
    server, broker, root = bridge
    path = setup_target(server, broker, root, {'688092.SH': {'shares': 400, 'price': 20}})
    with pytest.raises(RuntimeError, match='not been enabled'):
        reconcile_file(server, path, 1, execute=True)
    assert broker.submissions == []


def test_buy_sell_and_retry_use_durable_order_intents(bridge):
    server, broker, root = bridge
    server.config['target_file_execution_enabled'] = True
    path = setup_target(server, broker, root, {'688092.SH': {'shares': 0, 'price': 20},
                                             '600001.SH': {'shares': 100, 'price': 10}})
    result = reconcile_file(server, path, 1, execute=True)
    assert [row[1:3] for row in broker.submissions] == [(24, 200), (23, 100)]
    assert result['stage'] == 'submitted_once_wait_for_broker_fills'
    # Simulate an already filled buy whose holdings cache has not caught up.
    reconcile_file(server, path, 1, execute=True)
    assert len(broker.submissions) == 2


def test_lost_submission_ack_never_resends(bridge):
    server, broker, root = bridge
    server.config['target_file_execution_enabled'] = True
    path = setup_target(server, broker, root, {'688092.SH': {'shares': 400, 'price': 20}})
    broker.raise_after_submit = True
    with pytest.raises(RuntimeError, match='Disconnected'):
        reconcile_file(server, path, 1, execute=True)
    broker.raise_after_submit = False
    with pytest.raises(RuntimeError, match='unknown'):
        reconcile_file(server, path, 1, execute=True)
    assert len(broker.submissions) == 1


def test_expired_or_other_account_file_cannot_submit(bridge):
    server, broker, root = bridge
    path = setup_target(server, broker, root, {'688092.SH': {'shares': 400, 'price': 20}})
    target = json.loads(open(path).read())
    target['account'] = 'another-account'
    with open(path, 'w') as stream:
        json.dump(target, stream)
    with pytest.raises(ValueError):
        reconcile_file(server, path, 1, execute=True)
    assert broker.submissions == []


def test_partial_position_cache_never_becomes_empty_buy_baseline(bridge):
    server, broker, root = bridge
    path = setup_target(server, broker, root, {'688092.SH': {'shares': 400, 'price': 20}})
    broker.query_stock_positions = lambda account: []
    with pytest.raises(ValueError, match='partial snapshot'):
        reconcile_file(server, path, 1)
    assert broker.submissions == []


def test_nonterminal_fully_traded_order_blocks_lagging_holdings():
    order = SimpleNamespace(stock_code='688092.SH', order_status=55,
                            order_volume=200, traded_volume=200)
    result = diff_positions({'688092.SH': {'shares': 400, 'price': 20}},
                            {'688092.SH': {'volume': 200, 'available': 200}}, [order])
    assert result[0]['reason'] == 'existing_order_wait'
    assert result[0]['shares'] == 0


@pytest.mark.parametrize('status', [53, 54, 56, 57])
def test_existing_terminal_fill_blocks_new_batch_when_holdings_may_lag(status):
    order = SimpleNamespace(stock_code='688092.SH', order_status=status,
                            order_volume=200, traded_volume=200)
    result = diff_positions({'688092.SH': {'shares': 400, 'price': 20}},
                            {'688092.SH': {'volume': 200, 'available': 200}}, [order])
    assert result[0]['reason'] == 'existing_fill_reconciliation_required'
    assert result[0]['shares'] == 0


def test_file_expiring_during_query_never_reaches_order_api(bridge):
    server, broker, root = bridge
    path = setup_target(server, broker, root, {'688092.SH': {'shares': 400, 'price': 20}})
    server.config['target_file_execution_enabled'] = True
    initial = server.clock()
    calls = []
    def query(account):
        calls.append(1)
        server.clock = lambda: initial + 301
        return [position('688092.SH', 200)]
    broker.query_stock_positions = query
    with pytest.raises(RuntimeError, match='expired'):
        reconcile_file(server, path, 1, execute=True)
    assert server.request_deadline is None
    assert broker.submissions == []


def test_new_manual_trade_between_snapshot_and_submit_blocks_batch(bridge):
    server, broker, root = bridge
    path = setup_target(server, broker, root, {'688092.SH': {'shares': 400, 'price': 20}})
    server.config['target_file_execution_enabled'] = True
    calls = []
    def query(account):
        calls.append(1)
        return [position('688092.SH', 200 if len(calls) <= 2 else 400)]
    broker.query_stock_positions = query
    with pytest.raises(RuntimeError, match='Holdings changed'):
        reconcile_file(server, path, 1, execute=True)
    assert broker.submissions == []


def test_waiting_for_existing_order_does_not_freeze_empty_batch(bridge):
    server, broker, root = bridge
    path = setup_target(server, broker, root, {'688092.SH': {'shares': 400, 'price': 20}})
    server.config['target_file_execution_enabled'] = True
    broker.orders = [SimpleNamespace(stock_code='688092.SH', order_status=50,
                                    order_volume=200, traded_volume=0)]
    with pytest.raises(RuntimeError, match='before freezing'):
        reconcile_file(server, path, 1, execute=True)
    assert server.database.execute('SELECT count(*) FROM target_batches').fetchone()[0] == 0
    broker.orders = []
    reconcile_file(server, path, 1, execute=True)
    assert len(broker.submissions) == 1


def test_already_at_target_does_not_consume_batch(bridge):
    server, broker, root = bridge
    path = setup_target(server, broker, root, {'688092.SH': {'shares': 200, 'price': 20}})
    server.config['target_file_execution_enabled'] = True
    assert reconcile_file(server, path, 1, execute=True)['stage'] == 'at_target'
    assert server.database.execute('SELECT count(*) FROM target_batches').fetchone()[0] == 0
    assert broker.submissions == []
