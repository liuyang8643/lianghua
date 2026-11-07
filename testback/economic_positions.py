"""Zero-to-zero position episodes under the native economic-share account.

Observe actual transitions only. Partial sales, topups, corporate cash residuals
and delisting remain in the same episode until actual account quantity is zero.
This is not broker FIFO accounting or an alternate execution engine.
"""
from collections.abc import Mapping
import math

def value(obj,name,default=None):
    return obj.get(name,default) if isinstance(obj,Mapping) else getattr(obj,name,default)

class EconomicPositionLedger:
    def __init__(self):
        self.open={};self.closed=[];self.steps=0;self.initial_nav=None;self.daily={};self.fill_count=0

    def __call__(self,before,transition):
        step=value(transition,'step_result');self.observe(before,value(step,'account_state'),value(transition,'order_plan'),value(step,'fills'),value(step,'diagnostics'))

    def observe(self,before,after,plan,fills,diagnostics):
        positions=dict(value(before,'positions',{}));ending=dict(value(after,'positions',{}));date=value(plan,'decision_date');next_date=diagnostics['next_decision_date']
        if self.initial_nav is None:
            if any(positions.values()):raise ValueError('Starting positions require an explicit opening ledger; history must not be invented')
            self.initial_nav=float(value(before,'nav'))
        assert {c:e['quantity'] for c,e in self.open.items()}=={c:int(q) for c,q in positions.items() if q>0},'ledger/account quantity discontinuity'
        cash_change=0.
        for fill in fills:
            code=value(fill,'code');side=value(fill,'side');q=int(value(fill,'quantity'));price=float(value(fill,'price'));fee=float(value(fill,'fee'));gross=q*price
            event_id=self.fill_count;self.fill_count+=1
            if side=='buy':
                if code not in self.open:self.open[code]={'code':code,'buy_date':date,'entry_index':self.steps,'first_event_id':event_id,'quantity':0,'buy_quantity':0,'sell_quantity':0,'buy_gross':0.,'sell_gross':0.,'buy_fees':0.,'sell_fees':0.,'corporate_cash':0.}
                e=self.open[code];e['quantity']+=q;e['buy_quantity']+=q;e['buy_gross']+=gross;e['buy_fees']+=fee;cash_change-=gross+fee
                e['last_event_id']=event_id
            elif side=='sell':
                e=self.open[code];assert 0<q<=e['quantity'];e['quantity']-=q;e['sell_quantity']+=q;e['sell_gross']+=gross;e['sell_fees']+=fee;cash_change+=gross-fee
                e['last_event_id']=event_id
                if e['quantity']==0:self._close(code,date,self.steps,'实际卖出归零')
            else:raise ValueError('Unknown fill side')
        for code,adjustment in diagnostics.get('corporate_action_adjustments',{}).items():
            e=self.open[code];assert e['quantity']==int(adjustment['quantity_before'])
            residual=float(adjustment['rounding_cash_residual']);e['corporate_cash']+=residual;cash_change+=residual;e['quantity']=int(adjustment['synthetic_quantity_after_integer_rebase'])
        events=diagnostics.get('delist_write_offs',[])
        delisted={value(e,'code') for e in events} if not isinstance(events,Mapping) else set(events)
        for code in list(self.open):
            if ending.get(code,0)==0:
                if self.open[code]['quantity']!=0 and code not in delisted and not any(value(e,'code')==code and value(e,'type')=='delist_write_off' for e in diagnostics.get('account_events',[])):
                    raise ValueError('Unexplained account disappearance: '+code)
                self.open[code]['quantity']=0;self._close(code,next_date,self.steps+1,'退市归零' if code in delisted else '权益或账户事件归零')
            else:assert self.open[code]['quantity']==int(ending[code]),code
        assert set(self.open)=={c for c,q in ending.items() if q>0}
        tolerance=max(1e-5,abs(float(value(after,'nav')))*1e-10)
        if not math.isclose(float(value(after,'cash'))-float(value(before,'cash')),cash_change,rel_tol=1e-10,abs_tol=tolerance):raise ValueError('Position cash flows do not reconcile')
        old_members={c for c,q in positions.items() if q>0};new_members={c for c,q in ending.items() if q>0}
        self.daily[date]={'entered_stocks':sorted(new_members-old_members),'exited_stocks':sorted(old_members-new_members),'position_changes_available':True}
        plan_diagnostics=value(plan,'diagnostics',{})
        if 'buy_n_stocks' in plan_diagnostics:self.daily[date]['buy_n_list']=list(plan_diagnostics['buy_n_stocks'])
        self.steps+=1;self.summary(after)

    def _close(self,code,date,index,reason):
        e=self.open.pop(code);cost=e['buy_gross']+e['buy_fees'];income=e['sell_gross']-e['sell_fees']+e['corporate_cash']-cost
        self.closed.append({**e,'clear_date':date,'clear_signal_date':date if reason=='实际卖出归零' else '—','holding_days':index-e['entry_index'],'volume':e['buy_quantity'],'avg_price':e['buy_gross']/e['buy_quantity'],'clear_price':e['sell_gross']/e['sell_quantity'] if e['sell_quantity'] else None,'cost':cost,'income':income,'income_pct':100*income/cost,'clear_reason':reason,'price_field':'open','accounting_basis':'economic_position_episode','return_definition':'net_episode_profit / cumulative_buy_cost_including_fees'})

    def summary(self,account):
        marks=value(account,'last_prices',{});unrealized=sum(e['sell_gross']-e['sell_fees']+e['corporate_cash']-e['buy_gross']-e['buy_fees']+e['quantity']*marks[c] for c,e in self.open.items());realized=sum(e['income'] for e in self.closed)
        pnl=float(value(account,'nav'))-self.initial_nav
        if not math.isclose(realized+unrealized,pnl,rel_tol=1e-10,abs_tol=max(1e-5,abs(float(value(account,'nav')))*1e-10)):raise ValueError('Closed plus open episode PnL does not reconcile to NAV')
        return {'closed_pnl':realized,'open_pnl':unrealized,'account_pnl':pnl,'closed_episodes':len(self.closed),'open_episodes':len(self.open),'accounting_basis':'economic_position_episode','broker_exact':False}

    def enrich_report(self,report,account):
        summary=self.summary(account);closed=list(self.closed);profits=[e['income'] for e in closed];wins=[p for p in profits if p>0];losses=[p for p in profits if p<=0]
        round_trips=sum(e['clear_reason']=='实际卖出归零' for e in closed)
        ages=[e['holding_days'] for e in closed]+[self.steps-e['entry_index'] for e in self.open.values()]
        metrics=report.setdefault('metrics',{})
        metrics.update(round_trip_count=round_trips,cleared_positions_count=len(closed),wins=len(wins),losses=len(losses),win_rate=100*len(wins)/len(closed) if closed else 0.,avg_profit=sum(profits)/len(profits) if profits else 0.,avg_win=sum(wins)/len(wins) if wins else 0.,avg_loss=sum(losses)/len(losses) if losses else 0.,max_profit=max([0.,*profits]),max_loss=min([0.,*profits]),average_holding_days=sum(ages)/len(ages) if ages else 0.)
        report.update(cleared_positions=closed,position_lot_analytics_available=True,holding_period_available=True,kline_position_state_available=False,position_changes_available=True,position_accounting_basis='economic_position_episode',position_ledger_summary=summary,round_trip_count=round_trips,cleared_positions_count=len(closed))
        report['holding_stats']={'average_holding_days':metrics['average_holding_days'],'average_current_holding_days':sum(self.steps-e['entry_index'] for e in self.open.values())/len(self.open) if self.open else 0.,'average_cleared_holding_days':sum(e['holding_days'] for e in closed)/len(closed) if closed else 0.}
        report['holding_stats'].update(max_holding_days=max(ages,default=0),min_holding_days=min(ages,default=0))
        for row in report.get('positions',[]):
            e=self.open[row['code']];row['holding_days']=self.steps-e['entry_index'];row['buy_date']=e['buy_date'];row['first_event_id']=e['first_event_id']
        for row in report.get('daily_snapshots',[]):row.update(self.daily[row['trade_date']])
        return report
