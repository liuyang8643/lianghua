from types import SimpleNamespace as NS

import pytest

from trading.bridge_acceptance_watch import execution_coverage, matched_broker_facts


def facts(status=56):
    journal = {'requests': [dict(remark='wbr:test', code='688092.SH', shares=200, direction='BUY')]}
    order = NS(order_id=1, order_remark='wbr:test', stock_code='688092.SH',
               order_volume=200, order_type=23, order_status=status)
    trade = NS(order_id=1, stock_code='688092.SH', order_type=23, traded_volume=100,
               traded_price=20.0, traded_id='real-fill-id', traded_time=123, traded_amount=2000.0)
    return journal, order, trade


def test_only_reported_matching_broker_orders_and_real_fills_count():
    journal, order, trade = facts()
    accepted, fills = matched_broker_facts(journal, [order], [trade, trade])
    assert len(accepted) == 1 and len(fills) == 1
    assert fills[0]['traded_volume'] == 100


@pytest.mark.parametrize('status', [48, 49, 57])
def test_sdk_pending_and_rejected_orders_are_not_evidence_of_acceptance(status):
    journal, order, trade = facts(status)
    assert matched_broker_facts(journal, [order], [trade]) == ([], [])


def test_unrelated_manual_orders_do_not_establish_strategy_success():
    journal, order, trade = facts()
    order.order_remark = 'manual'
    assert matched_broker_facts(journal, [order], [trade]) == ([], [])


def test_intent_mismatch_or_no_submission_contradiction_cannot_pass():
    journal, order, trade = facts()
    journal['requests'][0]['submission_state'] = 'not_submitted'
    with pytest.raises(ValueError, match='contradicts'):
        matched_broker_facts(journal, [order], [trade])
    journal['requests'][0].pop('submission_state')
    order.order_volume = 300
    with pytest.raises(ValueError, match='differs'):
        matched_broker_facts(journal, [order], [trade])


def test_one_fill_is_not_proof_that_the_whole_plan_is_complete():
    journal, order, trade = facts()
    journal['pending'] = dict(buy_allocations={'688092.SH': 200, '600000.SH': 100},
                              sell_orders=[['000001.SZ', 100]])
    _, fills = matched_broker_facts(journal, [order], [trade])
    coverage = execution_coverage(journal, fills)
    assert not coverage['complete']
    assert sum(item['remaining'] for item in coverage['targets']) == 300


def test_exact_real_target_coverage_passes_and_overfill_does_not():
    journal = dict(pending=dict(buy_allocations={'688092.SH': 200}, sell_orders=[]))
    fill = dict(stock_code='688092.SH', order_type=23, traded_volume=200)
    assert execution_coverage(journal, [fill])['complete']
    fill['traded_volume'] = 201
    with pytest.raises(ValueError, match='exceed'):
        execution_coverage(journal, [fill])


@pytest.mark.parametrize('with_original_target', [True, False])
def test_unplanned_fills_never_count_as_full_execution(with_original_target):
    journal = dict(pending=dict(buy_allocations={'688092.SH': 200}
                               if with_original_target else {}, sell_orders=[]))
    fills = [dict(stock_code='600000.SH', order_type=23, traded_volume=100)]
    if with_original_target:
        fills.append(dict(stock_code='688092.SH', order_type=23, traded_volume=200))
    with pytest.raises(ValueError, match='outside'):
        execution_coverage(journal, fills)
