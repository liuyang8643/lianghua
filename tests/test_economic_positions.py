import pytest
from testback.economic_positions import EconomicPositionLedger

def account(cash,q,price):return {'cash':cash,'positions':{'A':q} if q else {},'last_prices':{'A':price} if q else {},'nav':cash+q*price}
def fill(side,q,price,fee=0):return {'code':'A','side':side,'quantity':q,'price':price,'fee':fee}
def step(ledger,before,after,fills,day,**extra):ledger.observe(before,after,{'decision_date':f'2020-01-{day:02}'},fills,{'next_decision_date':f'2020-01-{day+1:02}',**extra})

def test_topups_partial_sales_split_and_cash_residual():
    l=EconomicPositionLedger();a=account(10000,0,0);b=account(8999,100,10)
    step(l,a,b,[fill('buy',100,10,1)],1)
    c=account(8438,150,11);step(l,b,c,[fill('buy',50,11,11)],2)
    d=account(9028,100,12);step(l,c,d,[fill('sell',50,12,10)],3);assert not l.closed
    e=account(9028.5,200,6);step(l,d,e,[],4,corporate_action_adjustments={'A':{'quantity_before':100,'synthetic_quantity_after_integer_rebase':200,'rounding_cash_residual':.5}})
    f=account(10418.5,0,0);step(l,e,f,[fill('sell',200,7,10)],5)
    assert len(l.closed)==1;assert l.closed[0]['income']==pytest.approx(418.5)
    assert l.closed[0]['cost']==1562;assert l.summary(f)['account_pnl']==pytest.approx(418.5)
    assert l.enrich_report({'positions':[]},f)['metrics']['max_loss']==0
    assert not l.daily['2020-01-02']['entered_stocks']
    assert not l.daily['2020-01-03']['exited_stocks']
    assert l.daily['2020-01-05']['exited_stocks']==['A']

def test_delisting_is_closed_without_fictitious_sale():
    l=EconomicPositionLedger();a=account(1000,0,0);b=account(499,100,5);step(l,a,b,[fill('buy',100,5,1)],1)
    c=account(499,0,0);step(l,b,c,[],2,delist_write_offs=[{'code':'A'}])
    assert l.closed[0]['income']==-501 and l.closed[0]['clear_price'] is None
    report=l.enrich_report({'positions':[]},c);assert report['position_lot_analytics_available'] and not report['kline_position_state_available']
    assert report['metrics']['max_profit']==0
    assert report['round_trip_count']==0 and report['cleared_positions_count']==1

def test_unexplained_disappearance_rejected():
    l=EconomicPositionLedger();a=account(1000,0,0);b=account(500,100,5);step(l,a,b,[fill('buy',100,5)],1)
    with pytest.raises(ValueError,match='Unexplained'):step(l,b,account(500,0,0),[],2)

def test_starting_positions_do_not_invent_history():
    with pytest.raises(ValueError,match='Starting positions'):step(EconomicPositionLedger(),account(500,100,5),account(500,100,5),[],1)
