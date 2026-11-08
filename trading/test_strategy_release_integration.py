from datetime import date
from types import SimpleNamespace as NS
import json
import numpy as np
import pytest

from trading.execution_engine import OrderMonitor


def buy_monitor(holdings, *, reserved=(), unmanaged=()):
    monitor=OrderMonitor.__new__(OrderMonitor)
    monitor.max_positions=40
    monitor.unmanaged_codes=set(unmanaged)
    monitor.buy_seq=['600999.SH','600998.SH']
    monitor.retry_after={}
    monitor.trader=NS(query_account_snapshot=lambda:(None,[NS(stock_code=c,volume=100) for c in holdings]),
                      query_orders=lambda:[])
    monitor._can_submit_now=lambda now:True
    monitor._available_cash=lambda:100000
    monitor._has_open_order=lambda c:c in reserved
    monitor._filled_inflight=lambda c:(0,100 if c in reserved else 0)
    monitor._has_unhandled_terminal=lambda c:False
    monitor._remaining=lambda c:100
    monitor._unit_cost=lambda c:10
    submitted=[]
    monitor._submit=lambda side,c,q:(submitted.append((side,c,q)) or True)
    return monitor,submitted


def test_unsold_holding_does_not_release_slot_even_with_cash():
    monitor,submitted=buy_monitor(['600%03d.SH'%i for i in range(40)])
    assert not monitor._submit_affordable_buys()
    assert submitted==[]


def test_one_actual_free_slot_admits_only_one_buy_and_reserves_it():
    monitor,submitted=buy_monitor(['600%03d.SH'%i for i in range(39)])
    assert monitor._submit_affordable_buys()
    assert submitted==[('BUY','600999.SH',100)]


def test_inflight_buy_uses_slot_and_unmanaged_asset_does_not():
    holdings=['600%03d.SH'%i for i in range(39)]+['158905.SZ']
    monitor,submitted=buy_monitor(holdings,reserved=['600999.SH'],unmanaged=['158905.SZ'])
    assert not monitor._submit_affordable_buys()
    assert submitted==[]


def test_manual_inflight_buy_reserves_capacity_without_cancelling_it():
    monitor,submitted=buy_monitor(['600%03d.SH'%i for i in range(39)])
    monitor.trader.query_orders=lambda:[NS(stock_code='600997.SH',order_type=23,order_status=50)]
    assert not monitor._submit_affordable_buys()
    assert submitted==[]


def test_filled_manual_buy_waits_for_account_and_completed_roundtrip_releases_slot():
    monitor,submitted=buy_monitor(['600%03d.SH'%i for i in range(39)])
    orders=[NS(stock_code='600997.SH',order_type=23,order_status=56,traded_volume=100)]
    monitor.trader.query_orders=lambda:orders
    assert not monitor._submit_affordable_buys()
    orders.append(NS(stock_code='600997.SH',order_type=24,order_status=56,traded_volume=100))
    assert monitor._submit_affordable_buys()
    assert submitted==[('BUY','600999.SH',100)]


def test_research_and_release_share_selection_and_quantity_authorities():
    import env.qlib_strategy as research
    import strategy_release.selection as release
    import env.quantity as quantity
    import utils.stock.quantity as shared
    from utils.stock.info import floor_buy_shares,floor_sell_shares
    assert research.select_rank_buffer is release.select_rank_buffer
    assert research.allocate_rank_buffer is release.allocate_rank_buffer
    assert quantity.floor_quantity is shared.floor_quantity
    assert floor_buy_shares('920001.BJ',201)==201
    assert floor_buy_shares('689001.SH',201)==201
    assert floor_buy_shares('600001.SH',201)==200
    assert floor_sell_shares('600001.SH',37,37,37)==37


def test_account_history_does_not_guess_cost_based_acquisition_dates(tmp_path):
    from trading.position_age import rebuild_history,holding_days
    days=[date(2026,11,2),date(2026,11,3)]
    for day,qty in zip(days,[100,200]):
        from datetime import datetime
        record=dict(account='test',trade_date=day.isoformat(),captured_at=datetime.combine(day,datetime.min.time()).replace(hour=16).timestamp(),
            asset=dict(cash=1000,total_asset=1000+qty*10,market_value=qty*10),
            positions=[dict(stock_code='600001.SH',volume=qty,can_use_volume=qty,market_value=qty*10)],
            trades_available=True,trades=[])
        (tmp_path/('account_eod_'+day.isoformat()+'.json')).write_text(json.dumps(record))
    with pytest.raises(ValueError,match='reconciliation'):
        rebuild_history(tmp_path,'test',date(2026,11,4),days)


def test_missing_frozen_identity_is_not_accepted_as_new_target():
    from trading.target_protocol import validate_target
    with pytest.raises(KeyError):
        validate_target({},'test',date(2026,11,9),0)


def test_one_percentile_rule_preserves_missing_ties_and_binary_features():
    from strategy_release.normalization import percentile
    from ai.supervised.feature_data import rank_features
    raw=np.array([[1.,1.,3.,np.nan],[4.,2.,8.,7.]])
    eligible=np.array([[True,True,True,True],[False,False,False,False]])
    expected=np.array([[1/3,1/3,5/6,np.nan],[np.nan]*4],dtype=np.float32)
    np.testing.assert_array_equal(percentile(raw,eligible),expected)
    np.testing.assert_array_equal(rank_features(raw,eligible),expected)
    np.testing.assert_array_equal(percentile(raw[0],eligible[0],binary=True),raw[0].astype(np.float32))


def test_day_plan_json_is_absolute_and_rereading_cannot_add_again():
    from datetime import datetime
    from trading.plan_publisher import document_from_day_plan
    from trading.target_protocol import targets_to_pending
    positions={'600001.SH':NS(volume=300,can_use_volume=300),
               '158905.SZ':NS(volume=1000,can_use_volume=0)}
    plan=NS(sell_orders=[('600001.SH',100)],buy_orders={'600002.SH':200},
            prices={'600001.SH':10.,'600002.SH':20.},limit_prices={'600001.SH':11.,'600002.SH':22.},
            buy_n_stocks=['600002.SH'])
    stamp=datetime(2026,11,9,9,25,10).timestamp()
    document=document_from_day_plan(plan,positions,'test',date(2026,11,9),stamp,stamp)
    document.update(max_positions=40,unmanaged_codes=['158905.SZ'])
    assert document['targets']['600001.SH']['shares']==200
    assert document['targets']['600002.SH']['shares']==200
    assert '158905.SZ' not in document['targets']
    first=targets_to_pending(document,{c:p.volume for c,p in positions.items()})
    assert first['sell_orders']==plan.sell_orders and first['buy_allocations']==plan.buy_orders
    second=targets_to_pending(document,{'600001.SH':200,'600002.SH':200,'158905.SZ':1000})
    assert second['sell_orders']==[] and second['buy_allocations']=={}


def test_independent_date_evaluation_preserves_generated_factor_formulae():
    import importlib
    from ai.supervised.feature_data import _independent_generated
    from strategy_release import load_default
    rng=np.random.default_rng(431)
    panel={name:rng.uniform(.5,20,(12,30)) for name in (
        'open','close','preClose','high','low','volume','amount','eps','bps','roe',
        'profit_yoy','revenue_yoy','operating_cf_ps','gross_margin','total_share')}
    panel.update(issue_price=rng.uniform(2,20,30),st_mask=np.zeros((12,30),bool),
                 delisted_mask=np.zeros((12,30),bool),listing_age=np.ones((12,30),int))
    selected=np.array([2,7,11])
    short={k:(v[selected] if v.ndim==2 else v) for k,v in panel.items()}
    checked=0
    for name in load_default().names:
        if not _independent_generated(name):continue
        cls=getattr(importlib.import_module('factor_db.factors.'+name),name)
        np.testing.assert_array_equal(cls().calc_batch(panel)[selected],cls().calc_batch(short))
        checked+=1
    assert checked==31
