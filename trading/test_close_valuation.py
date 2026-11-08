import copy
from datetime import date
from types import SimpleNamespace

import pandas as pd
import pytest

from trading.close_valuation import value_at_close


def snapshot():
    return dict(account='test',trade_date='2026-11-02',captured_at=1,
        asset=dict(total_asset=1200,cash=100,market_value=1000,frozen_cash=0),
        positions=[dict(stock_code='600001.SH',volume=100,market_value=1000,last_price=10)])


def test_stale_broker_values_repriced_without_changing_raw(tmp_path):
    raw=snapshot(); original=copy.deepcopy(raw)
    pd.DataFrame([dict(time=pd.Timestamp('2026-11-02 15:00'),close=11)]).to_parquet(tmp_path/'600001.SH.parquet')
    valued,proof=value_at_close(raw,tmp_path)
    assert raw==original
    assert valued['positions'][0]['market_value']==1100
    assert proof['raw_asset_component_gap']==100 and proof['asset_component_gap']==0
    assert proof['rows'][0]['valuation_date']=='2026-11-02' and proof['rows'][0]['sha256']


def test_wrong_date_and_unreconciled_valuation_rejected(tmp_path):
    path=tmp_path/'600001.SH.parquet'
    pd.DataFrame([dict(time=pd.Timestamp('2026-10-30 15:00'),close=10)]).to_parquet(path)
    with pytest.raises(ValueError,match='Missing'):value_at_close(snapshot(),tmp_path)
    pd.DataFrame([dict(time=pd.Timestamp('2026-11-02 15:00'),close=10)]).to_parquet(path)
    with pytest.raises(ValueError,match='reconcile'):value_at_close(snapshot(),tmp_path)


def test_single_unquoted_holding_is_explicit_and_requires_account_crosscheck(tmp_path):
    raw=snapshot();raw['asset']['total_asset']=1100
    _,proof=value_at_close(raw,tmp_path)
    assert proof['unpriced_holdings']==['600001.SH']
    assert not proof['rows'][0]['independent_price_verified']
    raw['positions'].append(dict(stock_code='159001.SZ',volume=1,market_value=1))
    with pytest.raises(ValueError,match='Multiple'):value_at_close(raw,tmp_path)


def test_yesterday_withdrawal_does_not_change_today_and_broker_match_not_counted_twice(tmp_path,monkeypatch):
    from trading import persistence
    monkeypatch.setattr(persistence,'_TRADE_DIR',tmp_path)
    recorder=persistence.LiveTradeRecorder()
    for _ in range(2):
        recorder.record_cash_flow(-20000,'user_confirmed','user says 11/2',date(2026,11,2),reference='withdrawal-1102')
    assert recorder.get_today_cash_flows(date(2026,11,2))==-20000
    assert recorder.get_today_cash_flows(date(2026,11,3))==0
    trader=SimpleNamespace(query_bank_transfers=lambda *a:[SimpleNamespace(success=True,balance=20000,
        transfer_direction='2',transfer_no='broker123',date='20261102',time='',bank_name='',remark='')])
    for _ in range(2):recorder.sync_bank_transfers_from_qmt(trader,date(2026,11,3))
    frame=pd.read_parquet(tmp_path/'cash_flows.parquet')
    assert len(frame)==1 and frame.iloc[0]['type']=='qmt_sync'
    assert recorder.get_today_cash_flows(date(2026,11,2))==-20000
