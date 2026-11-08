"""Bind all native report tables to saved observed accounts; no browser use."""
import gzip,json,re
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
import numpy as np
from ai.supervised.periods import observation_splits
from utils.atomic_file import atomic_write_json,file_sha256
from delivery_inputs import admit,bind,verify,read

def plain(cell):return unescape(re.sub(r'<[^>]*>','',cell['html']))

def number(cell,value,shown=None):
    if value is None:assert cell['sort']=='' and plain(cell) in ('N/A','—','')
    else:assert float(cell['sort'])==float(value),(cell,value)
    if shown is not None:assert plain(cell)==shown,(plain(cell),shown)

def text_cell(cell,value):assert cell['sort']==str(value) and plain(cell)==str(value),(cell,value)

def stock_cell(cell,code,event=None):
    assert plain(cell).split(' ',1)[0]==code
    if event is not None:assert f'data-kline-code="{code}"' in cell['html'] and f'data-kline-event="{event}"' in cell['html']

def pct(v):return ('+' if v>0 else '')+f'{v:.2f}%'

def check_tables(data,payload,ledger):
    tables=data['tables'];trades=payload['trade_log'];assert len(tables['trades']['rows'])==len(trades)
    for i,(t,row) in enumerate(zip(trades,tables['trades']['rows'])):
        text_cell(row[0],t['signal_date']);text_cell(row[1],t['trade_date']);stock_cell(row[2],t['code'],i);text_cell(row[3],'买入' if t['action']=='buy' else '卖出');text_cell(row[4],t['price_field'])
        number(row[5],t['price'],f"{t['price']:.4f}");number(row[6],t['volume'],f"{t['volume']:,}");number(row[7],t['amount'],f"{t['amount']:,.2f}")
        for col,key in [(8,'broker_commission'),(9,'transfer_fee'),(10,'stamp_tax'),(11,'slippage'),(12,'total_fee')]:number(row[col],t[key],f"{t[key]:.2f}")
        number(row[13],t.get('income'),'—' if t.get('income') is None else f"¥{t['income']:,.2f}");text_cell(row[14],t.get('reason') or '')
        np.testing.assert_allclose(t['price']*t['volume'],t['amount'],rtol=1e-12,atol=1e-8)
        np.testing.assert_allclose(sum(t[k] for k in ('broker_commission','transfer_fee','stamp_tax','slippage')),t['total_fee'],rtol=1e-12,atol=1e-8)
    assert payload['cleared_positions']==ledger['closed'] and len(tables['cleared']['rows'])==len(ledger['closed'])
    for item,row in zip(ledger['closed'],tables['cleared']['rows']):
        stock_cell(row[0],item['code'],item['first_event_id']);text_cell(row[1],item['buy_date']);text_cell(row[2],item['buy_date']);text_cell(row[3],item['clear_signal_date']);text_cell(row[4],item['clear_date']);text_cell(row[5],'open')
        number(row[6],item['holding_days'],str(item['holding_days']));number(row[7],item['volume'],f"{item['volume']:,}");number(row[8],item['avg_price'],f"{item['avg_price']:.4f}")
        number(row[9],item['clear_price'],'N/A' if item['clear_price'] is None else f"{item['clear_price']:.4f}")
        number(row[10],item['cost'],f"{item['cost']:,.2f}");number(row[11],item['income'],f"¥{item['income']:,.2f}");number(row[12],item['income_pct'],pct(item['income_pct']));text_cell(row[13],item['clear_reason'])
    assert len(tables['holdings']['rows'])==len(payload['positions'])==len(ledger['open'])
    for p,row in zip(payload['positions'],tables['holdings']['rows']):
        e=ledger['open'][p['code']];assert p['volume']==e['quantity'];stock_cell(row[0],p['code'],e['first_event_id']);text_cell(row[1],e['buy_date']);text_cell(row[2],e['buy_date']);text_cell(row[3],'open')
        number(row[4],p['holding_days'],str(p['holding_days']));number(row[5],p['volume'],f"{p['volume']:,}")
        for col,key,digits in [(6,'avg_price',4),(7,'cost',2),(8,'current_price',4),(9,'current_value',2)]:number(row[col],p[key],f"{p[key]:,.2f}" if digits==2 else f"{p[key]:.4f}")
        income=p['current_value']-p['cost'];ratio=income/p['cost']*100 if p['cost']>0 else 0
        number(row[10],income,f'¥{income:,.2f}');number(row[11],ratio,pct(ratio))
    assert len(tables['daily']['rows'])==len(payload['daily_snapshots'])
    for s,row in zip(payload['daily_snapshots'],tables['daily']['rows']):
        text_cell(row[0],s['signal_date']);text_cell(row[1],s['trade_date']);text_cell(row[2],s['settlement_date']);text_cell(row[3],s.get('price_field') or 'open')
        for col,key in [(4,'cash'),(5,'market_value'),(6,'total_asset')]:number(row[col],s[key],f"{s[key]:,.2f}")
        for col,key in [(7,'daily_return_pct'),(8,'cumulative_return_pct')]:number(row[col],s[key],pct(s[key]))
        ratio=s['market_value']/s['total_asset']*100;number(row[9],ratio,f'{ratio:.1f}%')
        for key,col in [('executed_buy_list',11),('executed_sell_list',11),('entered_stocks',12),('exited_stocks',12)]:
            for code in s[key]:assert code in (row[col].get('title') or '')
        if 'buy_n_list' not in s:assert plain(row[10])=='N/A'
        else:
            for code in s['buy_n_list']:assert code in row[10]['sort']
    events=payload.get('delist_events',[]);assert len(tables['delist']['rows'])==len(events)
    for event,row in zip(events,tables['delist']['rows']):
        q=int(event.get('quantity') or 0);avg=float(event.get('average_cost') or 0);cost=q*avg;income=float(event.get('proceeds') or 0)-cost;ratio=income/cost*100 if cost>0 else 0
        stock_cell(row[0],event['code']);text_cell(row[1],event['effective_date']);number(row[2],q,f'{q:,}');number(row[3],avg,f'{avg:.4f}');number(row[4],cost,f'{cost:,.2f}');number(row[5],income,f'¥{income:,.2f}');number(row[6],ratio,pct(ratio))
    chart=data['charts']['equity'];months=[d[:7] for d in chart['trade_dates']];unique=sorted(set(months));assert len(tables['monthly']['rows'])==len(unique)
    for month,row in zip(unique,tables['monthly']['rows']):
        indices=[i for i,m in enumerate(months) if m==month];first,last=indices[0],indices[-1];curve=chart['strategy_nav'];before=curve[first-1] if first else 1.;after=curve[last]
        mr=round((after/before-1)*100,2);cum=round((after-1)*100,2);ret=[chart['daily_returns_pct'][i] for i in indices];wins=sum(v>0 for v in ret);losses=sum(v<0 for v in ret);rate=round(wins/len(ret)*100,1)
        text_cell(row[0],month);number(row[1],mr,pct(mr));number(row[2],cum,pct(cum));number(row[3],len(ret),str(len(ret)));number(row[4],wins,str(wins));number(row[5],losses,str(losses));number(row[6],rate,f'{rate:.1f}%')

class Anchors(HTMLParser):
    def __init__(self):super().__init__();self.values=[];self.current=None
    def handle_starttag(self,tag,attrs):
        if tag=='a':self.current=[dict(attrs).get('href'),[]]
    def handle_data(self,data):
        if self.current is not None:self.current[1].append(data)
    def handle_endtag(self,tag):
        if tag=='a' and self.current is not None:self.values.append((self.current[0],''.join(self.current[1])));self.current=None

def check_navigation(markup,payload):
    parser=Anchors();parser.feed(markup)
    for nav in payload['report_navigation']:assert (nav['href'],nav['label']) in parser.values,nav

def check_rendered_average(markup,payload):
    pattern=r'<div class="metric-card"><div class="metric-label">平均实际仓位.*?<div class="metric-value[^\"]*">([^<]+)</div></div>'
    match=re.search(pattern,markup,re.S);assert match
    snapshots=payload['daily_snapshots'];expected=sum(s['exposure'] for s in snapshots)/len(snapshots) if snapshots else 0.
    assert match.group(1)==f'{expected*100:.2f}%'
    return match.span(1)

root=Path(__file__).resolve().parent
config,bindings,_=admit(root);bind(bindings,Path(__file__));rows=[]
for strategy in ('ml','static'):
    stage=root/('final' if strategy=='ml' else 'static_baseline')
    observations=json.loads((stage/'complete.json').read_text(encoding='utf8'))
    for split in observation_splits(config):
        folder=root/'reports'/f'{strategy}_{split}';path=folder/'single_report.html';html=path.read_text(encoding='utf8')
        receipt=read(folder/'complete.json');assert receipt['state']=='complete';bind(bindings,folder/'complete.json');bind(bindings,path,receipt['report_sha256']);bind(bindings,folder/'position_ledger.json',receipt['ledger_sha256']);bind(bindings,folder/'report_payload.json.gz',receipt['payload_sha256'])
        marker='<script id="report-data" type="application/json">'
        data=json.loads(html.split(marker,1)[1].split('</script>',1)[0]);s=data['summary'];tables=data['tables']
        ledger=json.loads((folder/'position_ledger.json').read_text(encoding='utf8'))
        with gzip.open(folder/'report_payload.json.gz','rt',encoding='utf8') as f:payload=json.load(f)
        m=next(r['metrics'] for r in observations['results'] if r['split']==split)
        for key,value in [('annualized_return_pct',m['annualized_return']*100),('max_drawdown_pct',-m['max_drawdown']*100),('calmar_ratio',m['calmar']),('total_fees',m['total_fees'])]:np.testing.assert_allclose(s[key],value,rtol=1e-12,atol=1e-8)
        with np.load(stage/f'{split}.npz') as z:
            np.testing.assert_allclose(s['final_asset'],z['nav'][-1],rtol=1e-12,atol=1e-8)
            np.testing.assert_allclose(data['charts']['equity']['strategy_nav'],z['nav'][1:]/z['nav'][0],rtol=1e-12,atol=1e-6)
            np.testing.assert_array_equal(data['charts']['equity']['trade_dates'],z['next_dates'].astype('datetime64[D]').astype(str))
            assert len(tables['daily']['rows'])==s['total_days']==len(z['returns'])
            np.testing.assert_allclose(ledger['summary']['account_pnl'],z['nav'][-1]-z['nav'][0],rtol=1e-10,atol=1e-5)
            np.testing.assert_array_equal([d['cash'] for d in payload['daily_snapshots']],z['cash']);np.testing.assert_array_equal([d['total_asset'] for d in payload['daily_snapshots']],z['nav'][1:])
            np.testing.assert_array_equal([d['signal_date'] for d in payload['daily_snapshots']],z['decision_dates'].astype(str));np.testing.assert_array_equal([d['trade_date'] for d in payload['daily_snapshots']],z['decision_dates'].astype(str))
            np.testing.assert_array_equal([d['settlement_date'] for d in payload['daily_snapshots']],z['next_dates'].astype(str))
            np.testing.assert_allclose([d['daily_return_pct'] for d in payload['daily_snapshots']],100*z['returns'],rtol=1e-12,atol=1e-10)
            np.testing.assert_allclose([d['cumulative_return_pct'] for d in payload['daily_snapshots']],100*(z['nav'][1:]/z['nav'][0]-1),rtol=1e-12,atol=1e-8)
            np.testing.assert_array_equal([d['market_value'] for d in payload['daily_snapshots']],z['nav'][1:]-z['cash'])
            np.testing.assert_array_equal(data['charts']['equity']['execution_dates'],z['decision_dates'].astype(str));np.testing.assert_allclose(data['charts']['equity']['daily_returns_pct'],100*z['returns'],rtol=0,atol=5.1e-7)
            np.testing.assert_array_equal(payload['daily_exposures'],z['exposure'])
            np.testing.assert_allclose(data['charts']['equity']['exposure_pct'],(z['nav'][1:]-z['cash'])/z['nav'][1:]*100,rtol=0,atol=5.1e-7)
            np.testing.assert_allclose(payload['metrics']['average_exposure'],z['exposure'].mean(),rtol=1e-12)
        assert s['position_lot_analytics_available'] is True and s['kline_position_state_available'] is False
        assert s['position_accounting_basis']=='economic_position_episode'
        assert len(tables['cleared']['rows'])==len(ledger['closed'])==s['wins']+s['losses']
        assert sum(data['charts']['winloss'].values())==len(ledger['closed'])
        wins=sum(e['income']>0 for e in ledger['closed']);losses=len(ledger['closed'])-wins
        assert s['wins']==wins and s['losses']==losses
        assert data['charts']['winloss']==({'wins':wins,'losses':losses} if ledger['closed'] else {})
        np.testing.assert_allclose(ledger['summary']['closed_pnl']+ledger['summary']['open_pnl'],ledger['summary']['account_pnl'],rtol=1e-10,atol=1e-5)
        assert s['round_trips']==sum(e['clear_reason']=='实际卖出归零' for e in ledger['closed'])
        assert s['total_trades']==len(tables['trades']['rows'])==len(payload['trade_log'])
        assert len(tables['holdings']['rows'])==len(ledger['open'])
        for item,row in zip(ledger['closed'],tables['cleared']['rows']):
            event=item['first_event_id'];assert f'data-kline-event="{event}"' in row[0]['html']
            fill=payload['trade_log'][event];assert fill['code']==item['code'] and fill['action']=='buy'
        for tab in ('monthly','trades','holdings','cleared','daily','delist'):
            assert f'data-tab="{tab}"' in html and f'id="panel-{tab}"' in html
        check_tables(data,payload,ledger);check_navigation(html,payload);check_rendered_average(html,payload)
        assert {n['href'] for n in payload['report_navigation']}=={f'../{st}_{sp}/single_report.html' for st in ('ml','static') for sp in observation_splits(config)}
        for nav in payload['report_navigation']:assert (folder/nav['href']).resolve().exists()
        rows.append({'strategy':strategy,'split':split,'closed_episodes':len(ledger['closed']),'actual_fills':s['total_trades'],'days':s['total_days'],'report_sha256':file_sha256(path),'ledger_sha256':file_sha256(folder/'position_ledger.json'),'six_original_tabs':True,'native_metrics_and_saved_trace_match':True})
verify(bindings)
atomic_write_json(root/'report_binding_review.json',{'state':'pass','reports':rows,'inputs':bindings,'script_sha256':file_sha256(__file__),'HTML_browser_rendered':False,'HTML_browser_clicked':False,'scope':'All six table financial cells and visible numeric formatting bound to saved payload/ledger; daily assets/cash/returns/calendar and both exposure meanings checked; actual navigation anchors matched. Not a browser visual test.'})
print({'state':'pass','reports':len(rows)},flush=True)
