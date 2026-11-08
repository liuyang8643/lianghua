"""Construct the canonical T-open view without writing intraday market data."""
from dataclasses import replace
import json
from pathlib import Path
import numpy as np


def financial_panels(codes, dates, directory, output_directory):
    """Replay the same dated financial archive used by the frozen research model."""
    from offline_data.financial_versions import FINANCIAL_FIELDS, normalize_financial_events, iter_financial_fields
    from offline_data.financial_research import PERSHARE_FIELDS,replay_latest_fields
    from offline_data.financial_store import read_tables
    frames,manifest=read_tables(directory)
    if np.datetime64(manifest['verified_through'])<dates[-1]:
        raise ValueError('Dated strategy input collection is stale; refusing an incomplete runtime')
    core={table:normalize_financial_events(frames[table],tuple(map(str,codes)),fields)
          for table,fields in FINANCIAL_FIELDS.items()}
    output={}
    def array(name):
        if name not in output:
            output[name]=np.lib.format.open_memmap(Path(output_directory)/(name+'.npy'),
                mode='w+',dtype=np.float64,shape=(len(dates),len(codes)))
        return output[name]
    for row,(_,values) in enumerate(iter_financial_fields(dates,len(codes),core)):
        for name,value in values.items():
            array(name)[row]=value
    events=normalize_financial_events(frames['PershareIndex'],tuple(map(str,codes)),tuple(PERSHARE_FIELDS.values()))
    for row,values in enumerate(replay_latest_fields(dates,len(codes),events)):
        for column,name in enumerate(PERSHARE_FIELDS):
            array('pit_'+name)[row]=values[:,column]
    return output


def with_open_overlay(history, trade_date, overlay, inputs_directory):
    from data.db.stock_name import build_st_mask
    from data.build_runtime import build_delisted_mask
    from offline_data.financial_versions import FINANCIAL_FIELDS, normalize_financial_events, iter_financial_fields
    from offline_data.financial_research import PERSHARE_FIELDS, replay_latest_fields
    from offline_data.financial_store import read_tables
    from offline_data.capital_history import apply_capital_events
    today = np.datetime64(trade_date,'D')
    if history.trade_dates[-1] >= today:
        raise ValueError('T-open history must end strictly before the decision day')
    codes = history.stock_codes
    dates = np.r_[history.trade_dates, today]
    count = len(dates)-1
    data = {name:(values.copy() if values.ndim==1 else np.concatenate([values,values[-1:]],axis=0))
            for name,values in history.data.items()}
    for name in ('open','high','low','close','volume','amount','preClose'):
        data[name][-1] = np.nan
    for index,code in enumerate(codes):
        if code in overlay:
            for name in ('open','preClose'):
                data[name][-1,index] = float(overlay[code][name][-1])
    if not np.any(np.isfinite(data['open'][-1]) & (data['open'][-1]>0)):
        raise ValueError('Opening quotes are not ready')
    data['st_mask'][-1] = build_st_mask(np.asarray(codes),np.asarray([today]))[0]
    data['delisted_mask'][-1] = build_delisted_mask(np.asarray(codes),np.asarray([today]))[0]
    age = data['listing_age'][-2]
    data['listing_age'][-1] = np.where(age>=0,age+1,
        np.where(data['issue_date']==today,0,-1))
    root = Path(inputs_directory)
    frames,manifest=read_tables(root)
    if np.datetime64(manifest['verified_through']) < today-np.timedelta64(1,'D'):
        raise ValueError('Strategy financial/monthly/event inputs have not been refreshed through T-1')
    core = {table:normalize_financial_events(frames[table],codes,fields) for table,fields in FINANCIAL_FIELDS.items()}
    _,financial = next(iter_financial_fields(np.asarray([today]),len(codes),core))
    for name,values in financial.items():
        if name in data:
            data[name][-1] = values
    events = normalize_financial_events(frames['PershareIndex'],codes,tuple(PERSHARE_FIELDS.values()))
    values = next(replay_latest_fields(np.asarray([today]),len(codes),events))
    for column,name in enumerate(PERSHARE_FIELDS):
        data['pit_'+name][-1] = values[:,column]
    apply_capital_events(data['total_share'],dates,codes,count,len(codes),frames['Capital'],data['issue_date'])
    dates.flags.writeable=False
    for values in data.values():
        values.flags.writeable=False
    return replace(history,trade_dates=dates,data=data,decision_start=count,decision_stop=count+1)


def replace_open_quotes(prepared,overlay):
    """Replace only the decision row; all prepared non-quote inputs stay bound."""
    data=dict(prepared.data)
    for name in ('open','preClose'):
        values=data[name].copy();values[-1]=np.nan
        for column,code in enumerate(prepared.stock_codes):
            if code in overlay:values[-1,column]=float(overlay[code][name][-1])
        values.flags.writeable=False;data[name]=values
    return replace(prepared,data=data)
