"""Public Sina historical money-flow and circulating-share research inputs.

No authentication, no current-value backfill, no replacement of unknowns by zero.
Raw bytes and provider definitions must accompany any derived research result.
"""
from __future__ import annotations
from io import StringIO
import json
import re
import numpy as np
import pandas as pd

FLOW_URL='https://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/MoneyFlow.ssl_qsfx_lscjfb'
SHARES_URL='https://vip.stock.finance.sina.com.cn/corp/go.php/vCI_StockStructure/stockid/{code}.phtml'
HEADERS={'User-Agent':'Mozilla/5.0','Referer':'https://vip.stock.finance.sina.com.cn/moneyflow/'}

def sina_symbol(code: str) -> str:
    if len(code)!=9 or code[6:] not in ('.SH','.SZ') or not code[:6].isdigit():
        raise ValueError('Expected an explicit six-digit SH/SZ A-share ticker')
    return code[7:].lower()+code[:6]

def parse_flow(raw: bytes, code: str) -> dict:
    data=json.loads(raw.decode('utf-8'))
    if not isinstance(data,list):raise ValueError('Sina flow response is not a list')
    keys=('trade','turnover','r0','r1','r2','r3','r0_net','r1_net','r2_net','r3_net')
    frame=pd.DataFrame(data)
    if not len(frame):
        return dict(code=code,fields=list(keys),dates=np.array([],dtype='datetime64[D]'),values=np.empty((0,len(keys))))
    frame['date']=pd.to_datetime(frame['opendate'],format='%Y-%m-%d',errors='raise')
    frame=frame.sort_values('date')
    if frame.date.duplicated().any():raise ValueError('Duplicate historical flow date')
    numeric=frame.reindex(columns=keys).replace({'':np.nan,'--':np.nan,'-':np.nan})
    numeric=numeric.apply(pd.to_numeric,errors='raise')
    return dict(code=code,fields=list(keys),dates=frame.date.to_numpy(dtype='datetime64[D]'),
                values=numeric.to_numpy(dtype=np.float64))

def parse_share_structure(raw: bytes, code: str) -> list[dict]:
    text=raw.decode('gb18030',errors='strict')
    tables=pd.read_html(StringIO(text))
    events=[]
    for table in tables:
        labels=table.iloc[:,0].astype(str)
        change=labels.str.contains('变动日期',regex=False)
        if not change.any():continue
        announce=labels.str.contains('公告日期',regex=False)
        floating=labels.str.startswith('流通A股')
        total=labels.str.contains('总股本',regex=False)
        if not (announce.any() and floating.any() and total.any()):
            raise ValueError('Historical capital table is missing required rows')
        for col in table.columns[1:]:
            day=str(table.loc[change,col].iloc[0]).removesuffix('.0')
            if not re.fullmatch(r'\d{8}',day):continue
            effective=pd.to_datetime(day,format='%Y%m%d',errors='raise').strftime('%Y-%m-%d')
            ad=str(table.loc[announce,col].iloc[0]).removesuffix('.0')
            publication=None
            if re.fullmatch(r'\d{8}',ad) and ad!='19000101':
                publication=pd.to_datetime(ad,format='%Y%m%d',errors='raise').strftime('%Y-%m-%d')
            def amount(mask):
                value=str(table.loc[mask,col].iloc[0])
                if value in ('--','nan','-','None'):return None
                if '万股' not in value:raise ValueError('Unrecognized share unit: '+value)
                n=float(value.replace('万股','').replace(',','').strip())*10000
                return n if np.isfinite(n) and n>=0 else None
            events.append(dict(code=code,effective_date=effective,announcement_date=publication,
                               float_a_shares=amount(floating),total_shares=amount(total)))
    # A removed/unsupported listing must not silently receive current shares.
    if not events:raise ValueError('No usable historical capital table')
    unique={}
    for e in events:
        key=(e['effective_date'],e['announcement_date'] or '')
        if key in unique and unique[key]!=e:raise ValueError('Conflicting capital events')
        unique[key]=e
    return [unique[k] for k in sorted(unique)]

def align_float_shares(events: list[dict], dates: np.ndarray) -> tuple[np.ndarray,np.ndarray]:
    """Last known share state: announcement < day and effective <= day.

    Later archive rows cannot truncate earlier knowledge. For records known
    on the same day, the latest effective event wins, then its latest known
    version. Unknown announcement dates never replace a proven state.
    """
    days=np.asarray(dates,dtype='datetime64[D]')
    if days.ndim!=1 or np.isnat(days).any() or np.any(days[1:]<=days[:-1]):
        raise ValueError('share replay needs a strictly increasing date axis')
    values=np.full(len(days),np.nan)
    proven=np.zeros(len(days),bool)
    changes=[]
    for event in events:
        if not event['effective_date'] or not event['announcement_date']:continue
        effective=np.datetime64(event['effective_date'],'D');announced=np.datetime64(event['announcement_date'],'D')
        if np.isnat(effective) or np.isnat(announced):continue
        at=max(int(np.searchsorted(days,effective)),int(np.searchsorted(days,announced,side='right')))
        value=event['float_a_shares']
        value=float(value) if value is not None and np.isfinite(value) and value>0 else np.nan
        changes.append((at,effective,announced,value))
    latest=(np.datetime64('0001-01-01'),np.datetime64('0001-01-01'))
    for at,effective,announced,value in sorted(changes,key=lambda x:x[:3]):
        if at>=len(days) or (effective,announced)<latest:continue
        latest=(effective,announced)
        values[at:]=value;proven[at:]=np.isfinite(value)
    return values,proven

def align_main_flow(flow: dict, dates: np.ndarray, local_volume: np.ndarray,local_amount: np.ndarray) -> tuple[np.ndarray,np.ndarray,dict]:
    """Map r0 only and reject false-zero distributions/invalid amount identities."""
    dates=np.asarray(dates,dtype='datetime64[D]')
    net=np.full(len(dates),np.nan);gross=np.full(len(dates),np.nan)
    sd=flow['dates'];v=flow['values'];fields={n:i for i,n in enumerate(flow['fields'])}
    ix=np.searchsorted(dates,sd)
    hit=(ix<len(dates))&(dates[np.minimum(ix,len(dates)-1)]==sd)
    src=np.flatnonzero(hit);dest=ix[hit]
    allgross=v[src][:,[fields[k] for k in ('r0','r1','r2','r3')]]
    total=np.sum(allgross,axis=1)
    g=v[src,fields['r0']];n=v[src,fields['r0_net']]
    valid=np.isfinite(allgross).all(axis=1)&(allgross>=0).all(axis=1)&np.isfinite(n)
    local_trades=(np.isfinite(local_amount[dest])&(local_amount[dest]>0))|(np.isfinite(local_volume[dest])&(local_volume[dest]>0))
    placeholder=(total==0)&local_trades
    identity=np.abs(n)<=g+.02
    valid &= ~placeholder&identity
    net[dest[valid]]=n[valid];gross[dest[valid]]=g[valid]
    # An explicitly recorded zero-volume AND zero-amount market day has no
    # trading cash flow. Missing market fields never justify this inference.
    contradicts_zero=np.zeros(len(dates),bool)
    allnet=v[src][:,[fields[k] for k in ('r0_net','r1_net','r2_net','r3_net')]]
    contradicts_zero[dest]=(total>0)|np.any(np.isfinite(allnet)&(allnet!=0),axis=1)
    market_zero=np.isfinite(local_volume)&(local_volume==0)&np.isfinite(local_amount)&(local_amount==0)
    conflict=market_zero&contradicts_zero
    net[conflict]=np.nan;gross[conflict]=np.nan
    zero=market_zero&~contradicts_zero
    net[zero]=0.;gross[zero]=0.
    return net,gross,dict(aligned_rows=int(hit.sum()),placeholder_rows=int(placeholder.sum()),invalid_gross_net_rows=int((~identity).sum()),explicit_zero_activity_rows=int(zero.sum()),cross_provider_zero_conflicts=int(conflict.sum()),cross_provider_zero_conflict_dates=dates[conflict].astype(str).tolist())
