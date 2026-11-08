"""Extend the verified research Runtime with completed local sessions only."""
from datetime import date,timedelta
import hashlib,json,tempfile
from pathlib import Path
import numpy as np
import pandas as pd

from utils.atomic_file import atomic_write_json,file_sha256


def build_runtime(*,end_date=None):
    from offline_data import latest_runtime_npz_path
    from data.build_runtime import save_runtime_npz_atomic,build_issue_data,build_delisted_mask
    from data.db.stock_name import build_st_mask
    from data.db.stock_list import load_current_stock_codes
    from data.strategy_inputs import financial_panels
    from offline_data.financial_store import read_tables
    from offline_data.capital_history import apply_capital_events
    from utils.stock.time import get_last_trading_day,get_trading_date_span
    root=Path(__file__).resolve().parents[1];inputs=root/'data/strategy_inputs'
    source=json.loads((inputs/'manifest.json').read_text(encoding='utf8'))
    seed=(root/source['runtime_seed']).resolve()
    if not seed.is_relative_to(root/'data/runtime') or file_sha256(seed)!=source['runtime_seed_sha256']:
        raise ValueError('Frozen historical Runtime seed differs')
    base_path=latest_runtime_npz_path()
    through=get_last_trading_day(end_date or date.today())
    with np.load(base_path,allow_pickle=False) as base:
        old_dates=base['trade_dates'].astype('datetime64[D]');old_codes=tuple(base['stock_codes'].astype(str))
        if old_dates[-1]>np.datetime64(through):
            raise ValueError('Refuse to replace a newer Runtime with an older one')
        active=set(load_current_stock_codes())
        added=tuple(sorted(active-set(old_codes)))
        new_dates=np.asarray(get_trading_date_span(old_dates[-1].item()+timedelta(days=1),through),dtype='datetime64[D]')
        if not len(new_dates) and not added:return base_path
        codes=old_codes+added;dates=np.r_[old_dates,new_dates];n0=len(old_dates);s0=len(old_codes)
        issue_price,issue_dates=build_issue_data(np.asarray(codes))
        for index,code in enumerate(added,start=s0):
            if np.isnat(issue_dates[index]) or issue_dates[index]<=old_dates[-1]:
                raise ValueError('A historical stock is missing from the frozen universe: '+code)
        output=root/'data/runtime'/('runtime_'+str(dates[0])+'_'+str(dates[-1])+'.npz')
        with tempfile.TemporaryDirectory(prefix='complete_runtime_',dir=output.parent) as temporary:
            arrays={}
            for name in base.files:
                if name in ('trade_dates','stock_codes'):continue
                original=base[name]
                shape=(len(dates),len(codes)) if original.ndim==2 else (len(codes),)
                values=np.lib.format.open_memmap(Path(temporary)/(name+'.npy'),mode='w+',dtype=original.dtype,shape=shape)
                if values.dtype.kind=='f':values[:]=np.nan
                elif values.dtype.kind=='M':values[:]=np.datetime64('NaT')
                elif name=='listing_age':values[:]=-1
                else:values[:]=False
                if original.ndim==2:values[:n0,:s0]=original
                else:values[:s0]=original
                arrays[name]=values
            arrays['issue_price'][s0:]=issue_price[s0:];arrays['issue_date'][s0:]=issue_dates[s0:]
            evidence={};raw_fields=('open','high','low','close','volume','amount','preClose')
            for column,code in enumerate(codes):
                path=root/'data/k-line'/(code+'.parquet')
                if not path.exists():
                    if not np.isnat(arrays['issue_date'][column]) and arrays['issue_date'][column]>dates[-1]:continue
                    raise FileNotFoundError('Missing local history: '+code)
                frame=pd.read_parquet(path,columns=['time',*raw_fields])
                times=frame['time']
                days=(pd.to_datetime(times,unit='ms') if pd.api.types.is_numeric_dtype(times) else pd.to_datetime(times)).to_numpy(dtype='datetime64[D]')
                selected=(days>old_dates[-1])&(days<=dates[-1])
                if np.unique(days[selected]).size!=selected.sum():raise ValueError('Duplicate daily bars: '+code)
                for at,row in zip(np.searchsorted(dates,days[selected]),frame.loc[selected].to_dict('records')):
                    for name in raw_fields:arrays[name][at,column]=row[name]
                if selected.any():evidence[code]=file_sha256(path)
            if len(new_dates):
                arrays['st_mask'][n0:]=build_st_mask(np.asarray(codes),new_dates)
                arrays['delisted_mask'][n0:]=build_delisted_mask(np.asarray(codes),new_dates)
                prior_age=arrays['listing_age'][n0-1,:s0]
                arrays['listing_age'][n0:,:s0]=np.where(prior_age>=0,prior_age+np.arange(1,len(new_dates)+1)[:,None],-1)
                for col in range(s0,len(codes)):
                    first=np.searchsorted(dates,arrays['issue_date'][col])
                    arrays['listing_age'][first:,col]=np.arange(len(dates)-first)
                fin_dir=Path(temporary)/'financial';fin_dir.mkdir()
                financial=financial_panels(codes,new_dates,inputs,fin_dir)
                for name,values in financial.items():
                    if name not in arrays:raise ValueError('Unexpected financial Runtime field: '+name)
                    arrays[name][n0:]=values
                    values._mmap.close()
                frames,_=read_tables(inputs)
                seed_end=np.datetime64(source['runtime_seed_end'])
                seed_rows=int(np.searchsorted(dates,seed_end,side='right'))
                apply_capital_events(arrays['total_share'],dates,codes,seed_rows,source['runtime_seed_stocks'],
                    frames['Capital'],arrays['issue_date'])
                from data.build_runtime import validate_current_kline_terminal_coverage
                expected={c for j,c in enumerate(codes) if c in active and arrays['listing_age'][-1,j]>=0}
                validate_current_kline_terminal_coverage(np.asarray(codes),dates,arrays['preClose'],expected)
            save_runtime_npz_atomic(output,stock_codes=np.asarray(codes),trade_dates=dates,**arrays)
            for values in arrays.values():values._mmap.close()
    atomic_write_json(output.with_suffix('.sources.json'),dict(seed_sha256=source['runtime_seed_sha256'],
        previous_runtime=str(base_path),completed_through=str(dates[-1]),kline_sha256=evidence,
        financial_manifest_sha256=file_sha256(inputs/'financial_manifest.json'),intraday_data=False))
    return output
