"""Refresh dated financial inputs through local public HTTP sources, never QMT.

Sina supplies consolidated statements, publication dates and capital events.
Eastmoney's paginated announcement index identifies changed financial reports.
Original responses and every publication version remain auditable.
"""
from datetime import date,datetime,timedelta
import gzip,json,math,time
from pathlib import Path
import numpy as np
import pandas as pd
import requests

from offline_data.financial_store import read_tables
from offline_data.sources.sina_research_history import SHARES_URL,parse_share_structure,sina_symbol
from utils.atomic_file import atomic_write_json,file_sha256
from utils.eastmoney_request import throttled_get

REPORT_URL='https://quotes.sina.cn/cn/api/openapi.php/CompanyFinanceService.getFinanceReport2022'
INDEX_URL='https://datacenter-web.eastmoney.com/api/data/v1/get'
TABLES={
 'Income':('lrb',{'net_profit_excl_min_int_inc':'归属于母公司所有者的净利润',
    'oper_profit':'营业利润','revenue_inc':'营业收入','revenue':'营业总收入','total_expense':'营业成本'}),
 'Balance':('fzb',{'tot_shrhldr_eqy_excl_min_int':'归属于母公司股东权益合计',
    'inventories':'存货','other_payable':'其他应付款','account_receivable':'应收账款',
    'taxes_surcharges_payable':'应交税费','other_current_liability':'其他流动负债','tot_assets':'资产总计'}),
 'CashFlow':('llb',{'stot_cash_outflows_oper_act':'经营活动现金流出小计',
    'net_cash_flows_oper_act':'经营活动产生的现金流量净额','goods_sale_and_service_render_cash':'销售商品、提供劳务收到的现金'}),
 'PershareIndex':('gjzb',{'s_fa_eps_basic':'基本每股收益','s_fa_bps':'摊薄每股净资产_期末股数',
    's_fa_ocfps':'每股经营现金流','net_roe':'摊薄净资产收益率','equity_roe':'净资产收益率(ROE)',
    'inc_revenue_rate':'营业总收入增长率','sales_gross_profit':'毛利率'}),
}
EM_FIELDS={
 'Income':('GINCOME',{'net_profit_excl_min_int_inc':'PARENT_NETPROFIT','oper_profit':'OPERATE_PROFIT',
                     'revenue_inc':'OPERATE_INCOME','revenue':'TOTAL_OPERATE_INCOME','total_expense':'OPERATE_COST'}),
 'Balance':('GBALANCE',{'tot_shrhldr_eqy_excl_min_int':'TOTAL_PARENT_EQUITY','inventories':'INVENTORY',
    'other_payable':'OTHER_PAYABLE','account_receivable':'ACCOUNTS_RECE','taxes_surcharges_payable':'TAX_PAYABLE',
    'other_current_liability':'OTHER_CURRENT_LIAB','tot_assets':'TOTAL_ASSETS'}),
 'CashFlow':('GCASHFLOW',{'stot_cash_outflows_oper_act':'TOTAL_OPERATE_OUTFLOW',
    'net_cash_flows_oper_act':'NETCASH_OPERATE','goods_sale_and_service_render_cash':'SALES_SERVICES'}),
}


def number(value):
    return np.nan if value is None else float(value)


def parse_em_reports(code,payloads,after,through):
    output={name:[] for name in TABLES};by_period={}
    for table,(suffix,mapping) in EM_FIELDS.items():
        rows=payloads[suffix]
        by_period[table]={r['REPORT_DATE'][:10]:r for r in rows}
        for r in rows:
            published=date.fromisoformat(r['NOTICE_DATE'][:10])
            revision=date.fromisoformat(r['UPDATE_DATE'][:10])
            if not after<max(published,revision)<=through:continue
            if r['SECUCODE']!=code or r['CURRENCY']!='CNY':
                raise ValueError('Fallback statement identity/currency differs')
            row=dict(stock_code=code,m_timetag=r['REPORT_DATE'][:10].replace('-',''),
                     m_anntime=published.strftime('%Y%m%d'),known_after_date=max(published,revision).isoformat())
            row.update({field:number(r.get(key)) for field,key in mapping.items()})
            output[table].append(row)
    main={r['REPORT_DATE'][:10]:r for r in payloads['MAINFINADATA']}
    for period,income in by_period['Income'].items():
        balance=by_period['Balance'].get(period);cash=by_period['CashFlow'].get(period);metric=main.get(period)
        if balance is None or cash is None:continue
        available=max(date.fromisoformat(r['UPDATE_DATE'][:10]) for r in (income,balance,cash))
        published=max(date.fromisoformat(r['NOTICE_DATE'][:10]) for r in (income,balance,cash))
        if not after<max(published,available)<=through:continue
        equity=number(balance.get('TOTAL_PARENT_EQUITY'));profit=number(income.get('PARENT_NETPROFIT'))
        metrics_current=metric is not None and date.fromisoformat(metric['UPDATE_DATE'][:10])>=available
        def reported(key):return number(metric.get(key)) if metrics_current else np.nan
        row=dict(stock_code=code,m_timetag=period.replace('-',''),m_anntime=published.strftime('%Y%m%d'),
            known_after_date=max(published,available).isoformat(),
            s_fa_eps_basic=number(income.get('BASIC_EPS')),s_fa_bps=reported('BPS'),s_fa_ocfps=reported('MGJYXJJE'),
            equity_roe=reported('ROEJQ'),net_roe=profit/equity*100 if math.isfinite(equity) and equity!=0 else np.nan,
            du_profit_rate=number(income.get('NETPROFIT_YOY')),inc_revenue_rate=number(income.get('TOTAL_OPERATE_INCOME_YOY')))
        revenue=number(income.get('OPERATE_INCOME'));cost=number(income.get('OPERATE_COST'))
        row['sales_gross_profit']=(revenue-cost)/revenue*100 if math.isfinite(revenue) and revenue!=0 else np.nan
        row.update({k:round(v,4) for k,v in row.items() if isinstance(v,float) and math.isfinite(v)})
        output['PershareIndex'].append(row)
    return {table:pd.DataFrame(rows) for table,rows in output.items()}


def item_value(report,title):
    values=[x['item_value'] for x in report['data'] if x['item_title']==title and x.get('item_value') not in (None,'','--')]
    if not values:return np.nan
    values=[float(str(v).replace(',','')) for v in values]
    if any(v!=values[0] for v in values):
        raise ValueError('Conflicting financial item: '+title)
    return values[0]


def parse_reports(code,payloads,after,through):
    result={name:[] for name in TABLES}
    reports={source:payload['result']['data']['report_list'] for source,payload in payloads.items()}
    for table,(source,mapping) in TABLES.items():
        for period,report in reports[source].items():
            announced=datetime.strptime(report['publish_date'],'%Y%m%d').date()
            if not after<announced<=through:continue
            if report['rCurrency']!='CNY' or not str(report['rType']).startswith('合并'):
                raise ValueError('Financial statement currency/consolidation differs: '+code)
            row=dict(stock_code=code,m_timetag=period,m_anntime=announced.strftime('%Y%m%d'))
            row.update({field:item_value(report,title) for field,title in mapping.items()})
            if table=='PershareIndex':
                current=reports['lrb'].get(period)
                prior=reports['lrb'].get(str(int(period)-10000))
                growth=np.nan
                if current is not None and prior is not None:
                    known=max(announced,datetime.strptime(current['publish_date'],'%Y%m%d').date(),
                              datetime.strptime(prior['publish_date'],'%Y%m%d').date())
                    if known<=through:
                        before=item_value(prior,'净利润');now=item_value(current,'净利润')
                        if math.isfinite(before) and before!=0 and math.isfinite(now):
                            growth=(now/before-1)*100
                            row['m_anntime']=known.strftime('%Y%m%d')
                row['du_profit_rate']=growth
                row.update({k:round(v,4) for k,v in row.items() if isinstance(v,float) and math.isfinite(v)})
            result[table].append(row)
    return {name:pd.DataFrame(rows) for name,rows in result.items()}


class PublicFinancialSource:
    def __init__(self,directory):
        self.directory=Path(directory);self.directory.mkdir(parents=True,exist_ok=True)
        self.session=requests.Session();self.session.headers['User-Agent']='Mozilla/5.0'
        self.last_request=0.
        self.receipts=[]

    def get(self,url,params,label,*,eastmoney=False):
        if eastmoney:
            response,self.last_request=throttled_get(self.session,url=url,params=params,
                last_request_time=self.last_request,minimum_interval=1.5,maximum_jitter=.3)
        else:
            response=self.session.get(url,params=params,timeout=15)
        path=self.directory/(label+'.raw.gz')
        path.write_bytes(gzip.compress(response.content))
        self.receipts.append(dict(path=str(path),sha256=file_sha256(path),url=response.url,
                                 captured_at=time.time(),status_code=response.status_code))
        response.raise_for_status()
        return response

    def index(self,report_name,after,through,columns,*,url=INDEX_URL):
        params=dict(reportName=report_name,columns=columns,pageSize=500,pageNumber=1,
                    sortColumns='NOTICE_DATE,SECURITY_CODE',sortTypes='-1,-1')
        # Positive control distinguishes an explicit no-record result from an
        # unavailable endpoint, invalid schema, or malformed filter.
        health={**params,'pageSize':1,'filter':"(NOTICE_DATE>='2000-01-01')"}
        check=self.get(url,health,'health_'+report_name,eastmoney=True).json()
        if check['success'] is not True or check['result']['count']<1:
            raise ValueError('Financial index health check failed: '+report_name)
        params['filter']="(NOTICE_DATE>='%s')(NOTICE_DATE<='%s')"%(after.isoformat(),through.isoformat())
        response=self.get(url,params,'index_'+report_name+'_1',eastmoney=True).json()
        if response.get('code')==9201 and response.get('message')=='返回数据为空':
            return []
        if response['success'] is not True or response['result'] is None:
            raise ValueError('Financial announcement index failed: '+str(response))
        total=response['result']['count'];pages=response['result']['pages'];rows=[]
        for page in range(1,pages+1):
            if page>1:
                params['pageNumber']=page
                response=self.get(url,params,'index_'+report_name+'_'+str(page),eastmoney=True).json()
            if response['success'] is not True or response['result']['count']!=total:
                raise ValueError('Financial index changed during pagination')
            rows.extend(response['result']['data'])
        if len(rows)!=total:raise ValueError('Incomplete financial index')
        return rows

    def announcements(self,after,through):
        changed={}
        for table in ('INCOME','BALANCE','CASHFLOW'):
            rows=self.index('RPT_DMSK_FN_'+table,after,through,'SECUCODE,NOTICE_DATE,REPORT_DATE')
            for row in rows:
                announced=date.fromisoformat(row['NOTICE_DATE'][:10])
                if after<announced<=through:
                    changed.setdefault(row['SECUCODE'],set()).add(announced)
        return changed

    def capital_changes(self,after,through):
        rows=self.index('RPT_F10_EH_EQUITY',after,through,
            'SECUCODE,SECURITY_CODE,NOTICE_DATE,END_DATE,TOTAL_SHARES,LISTED_A_SHARES',
            url='https://datacenter.eastmoney.com/securities/api/data/v1/get')
        return pd.DataFrame([dict(stock_code=r['SECUCODE'],m_timetag=r['END_DATE'][:10].replace('-',''),
            m_anntime=r['NOTICE_DATE'][:10].replace('-',''),total_capital=float(r['TOTAL_SHARES']),
            float_a_shares=float(r['LISTED_A_SHARES']) if r['LISTED_A_SHARES'] is not None else np.nan)
            for r in rows if after<date.fromisoformat(r['NOTICE_DATE'][:10])<=through],
            columns=['stock_code','m_timetag','m_anntime','total_capital','float_a_shares'])

    def reports(self,code):
        return {source:self.get(REPORT_URL,dict(paperCode=sina_symbol(code),source=source,type=0,page=1,num=200),
                    code+'_'+source).json() for source,_ in TABLES.values()}

    def em_reports(self,code):
        result={}
        for suffix in ('GINCOME','GBALANCE','GCASHFLOW','MAINFINADATA'):
            params=dict(reportName='RPT_F10_FINANCE_'+suffix,columns='ALL',filter='(SECUCODE="'+code+'")',
                        pageNumber=1,pageSize=200,sortColumns='REPORT_DATE',sortTypes='-1',source='HSF10',client='PC')
            rows=[];page=1;pages=1;count=None
            while page<=pages:
                params['pageNumber']=page
                payload=self.get('https://datacenter.eastmoney.com/securities/api/data/v1/get',params,
                    code+'_'+suffix+'_'+str(page),eastmoney=True).json()
                if payload['success'] is not True or payload['result'] is None:
                    raise ValueError('Financial fallback failed: '+code+' '+suffix)
                item=payload['result'];pages=item['pages'];count=item['count'];rows.extend(item['data']);page+=1
            if not rows or len(rows)!=count:raise ValueError('Incomplete financial fallback')
            result[suffix]=rows
        return result

    def capital(self,code):
        raw=self.get(SHARES_URL.format(code=code[:6]),None,code+'_capital').content
        events=parse_share_structure(raw,code)
        return pd.DataFrame([dict(stock_code=code,m_timetag=e['effective_date'].replace('-',''),
            m_anntime=e['announcement_date'].replace('-',''),total_capital=e['total_shares'])
            for e in events if e['announcement_date'] is not None])


def observation_boundary(now=None):
    now=now or datetime.now()
    # Information collected before the open is available to that day's
    # decision; later collection must never rewrite that morning's inputs.
    boundary=now.date() if (now.hour,now.minute,now.second)>=(9,25,10) else now.date()-timedelta(days=1)
    return boundary.isoformat()


def publish_tables(root,frames,metadata):
    root=Path(root);folder=root/'financial_events'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    folder.mkdir(parents=True,exist_ok=False)
    tables={}
    for name,frame in frames.items():
        path=folder/(name+'.parquet');frame.to_parquet(path,index=False)
        tables[name]=dict(path=path.relative_to(root).as_posix(),sha256=file_sha256(path))
    atomic_write_json(root/'financial_manifest.json',dict(version=1,tables=tables,**metadata))


def seed(root,codes):
    """One-time format migration; preserves original source dates and hashes."""
    from offline_data.research_extension import read_financial_chain
    root=Path(root)
    if (root/'financial_manifest.json').exists():raise FileExistsError('Financial store already exists')
    legacy=json.loads((root/'manifest.json').read_text(encoding='utf8'))
    frames,hashes=read_financial_chain(legacy['financial_chain'],codes)
    publish_tables(root,frames,dict(verified_through=legacy['verified_through'],capital_verified_through=legacy['verified_through'],
        captured_at=None,seed_source_hashes=hashes,source='frozen research archive; not a new collection'))


def update(root,through,*,refresh_capital=True):
    from data.db.stock_list import load_current_stock_codes
    root=Path(root);frames,manifest=read_tables(root)
    after=date.fromisoformat(manifest['verified_through'])
    if through<after:raise ValueError('Cannot move financial evidence backwards')
    if through>=date.today():raise ValueError('Only completed calendar days can be certified')
    active=set(load_current_stock_codes())
    source=PublicFinancialSource(root/'public_financial_raw'/datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    changes=source.announcements(after,through)
    for code,announcements in sorted(changes.items()):
        if code not in active:continue
        try:
            parsed=parse_reports(code,source.reports(code),after,through)
            if any(parsed[name].empty for name in TABLES):
                raise ValueError('Primary statements have not caught up with announcements: '+code)
        except (requests.RequestException,ValueError,KeyError,TypeError) as error:
            print('Financial source fallback',code,type(error).__name__,flush=True)
            parsed=parse_em_reports(code,source.em_reports(code),after,through)
            if any(parsed[name].empty for name in TABLES):
                raise ValueError('Neither financial source covers announced reports: '+code) from error
        for name,rows in parsed.items():
            if not rows.empty:
                rows=rows.copy()
                observed=pd.Timestamp(observation_boundary())
                prior=pd.to_datetime(rows['known_after_date']) if 'known_after_date' in rows else pd.Series(pd.NaT,index=rows.index)
                rows['known_after_date']=prior.where(prior>=observed,observed).dt.strftime('%Y-%m-%d')
                frames[name]=pd.concat([frames[name],rows],ignore_index=True)
    capital_through=manifest['capital_verified_through']
    if refresh_capital:
        additions=source.capital_changes(date.fromisoformat(capital_through),through)
        if not additions.empty:
            additions=additions[additions.stock_code.isin(active)].copy()
            additions['known_after_date']=observation_boundary()
            frames['Capital']=pd.concat([frames['Capital'],additions],ignore_index=True)
        capital_through=through.isoformat()
    publish_tables(root,frames,dict(verified_through=through.isoformat(),capital_verified_through=capital_through,
        captured_at=time.time(),source='Sina public statements/capital; Eastmoney announcement index',
        receipts=source.receipts,seed_source_hashes=manifest.get('seed_source_hashes',{})))


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',type=Path,default=Path('data/strategy_inputs'))
    p.add_argument('--through',type=date.fromisoformat,required=True)
    args=p.parse_args();update(args.directory,args.through)
