"""Dated industry and dividend source validation and point-in-time materialization."""
import numpy as np
import pandas as pd


def industry_panel(history,dates,codes,member):
    dates=np.asarray(dates,dtype='datetime64[D]');out=np.full((len(dates),len(codes)),-1,dtype=np.int32)
    lookup={c:i for i,c in enumerate(codes)}
    # Industry IDs are identifiers, never numerical factor magnitudes.
    levels={value:i for i,value in enumerate(sorted(history['l1_code'].dropna().astype(str).unique()))}
    for code,group in history.groupby('stock_code',sort=False):
        if code not in lookup:continue
        group=group.sort_values('start_date');col=lookup[code]
        for row in group.itertuples():
            if pd.isna(row.start_date) or pd.isna(row.l1_code):continue
            at=int(np.searchsorted(dates,np.datetime64(row.start_date.date()),side='right'))
            out[at:,col]=levels[str(row.l1_code)]
    out[~member]=-1
    return out


def implemented_dividends(frame):
    data=frame.copy();data['stock_code']=data['SECUCODE'].astype(str)
    columns=['EX_DIVIDEND_DATE','NOTICE_DATE','PUBLISH_DATE']
    for name in columns:data[name]=pd.to_datetime(data[name],errors='coerce')
    data['ex_date']=data['EX_DIVIDEND_DATE'];data['known_date']=data[columns].max(axis=1)
    cash=pd.to_numeric(data['PRETAX_BONUS_RMB'],errors='coerce');shares=pd.to_numeric(data['TOTAL_SHARES'],errors='coerce')
    keep=data[columns].notna().all(axis=1)&data['ASSIGN_PROGRESS'].eq('实施分配')&(cash>0)&(shares>0)
    data['total_cash_yuan']=cash/10*shares
    data=data[keep].copy()
    # Distinct cash amounts for one report/ex-date are an ambiguous revision,
    # not multiple dividends. Exclude conflicts instead of choosing a future one.
    keys=['stock_code','EX_DIVIDEND_DATE','REPORT_DATE']
    unique=data.groupby(keys,dropna=False)['total_cash_yuan'].transform('nunique')
    conflicts=int((unique>1).sum());data=data[unique==1]
    data=data.sort_values('known_date').drop_duplicates(keys,keep='last')
    return data[['stock_code','ex_date','known_date','total_cash_yuan']],{'source_rows':len(frame),'usable_events':len(data),'conflicting_rows_excluded':conflicts,'total_shares_unit':'shares','missing_dates_excluded':int(frame.shape[0]-keep.sum())}
