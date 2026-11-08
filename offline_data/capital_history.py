"""Versioned capital replay: announcement known and effective date reached."""
import numpy as np
import pandas as pd


def apply_capital_events(output,dates,codes,base_rows,base_columns,frame,issue_dates=None):
    dates=np.asarray(dates,dtype='datetime64[D]');cutoff=dates[base_rows-1];lookup={c:i for i,c in enumerate(codes)}
    capital=frame.copy()
    capital['ann']=pd.to_datetime(capital['m_anntime'].astype(str),format='%Y%m%d',errors='coerce')
    capital['effective']=pd.to_datetime(capital['m_timetag'].astype(str),format='%Y%m%d',errors='coerce')
    capital['value']=pd.to_numeric(capital['total_capital'],errors='coerce')/10000.
    ambiguous=(capital['ann']==capital['effective'])&capital['effective'].dt.is_quarter_end
    if 'known_after_date' in capital:
        observed=pd.to_datetime(capital['known_after_date'],errors='coerce')
        capital['ann']=capital['ann'].where(observed.isna()|(capital['ann']>=observed),observed)
    capital=capital[~ambiguous&capital['ann'].notna()&capital['effective'].notna()&np.isfinite(capital['value'])&(capital['value']>0)].copy()
    capital['activation']=np.maximum(np.searchsorted(dates,capital['ann'].to_numpy(dtype='datetime64[D]'),side='right'),
                                     np.searchsorted(dates,capital['effective'].to_numpy(dtype='datetime64[D]'),side='left'))
    capital=capital.sort_values(['stock_code','activation','effective','ann'])
    output[base_rows:,:base_columns]=output[base_rows-1,:base_columns]
    output[:,base_columns:]=np.nan
    mismatches=[];applied=[];discarded_old=0;processed=set()
    for code,group in capital.groupby('stock_code',sort=False):
        if code not in lookup:continue
        processed.add(code)
        col=lookup[code];latest=np.datetime64('0001-01-01')
        if col<base_columns:
            known=group[(group['ann']<=pd.Timestamp(str(cutoff)))&(group['effective']<=pd.Timestamp(str(cutoff)))]
            base_value=float(output[base_rows-1,col])
            if len(known):
                seed=known.sort_values(['effective','ann']).iloc[-1]
                if np.isfinite(base_value) and np.isclose(base_value,seed['value'],rtol=1e-7,atol=1e-8):
                    latest=np.datetime64(seed['effective'].date())
                elif not np.isfinite(base_value):
                    latest=np.datetime64(seed['effective'].date());output[base_rows:,col]=seed['value']
                    mismatches.append({'code':code,'policy':'fill_missing_tail_from_known_seed','seed':float(seed['value'])})
                else:
                    latest=cutoff;mismatches.append({'code':code,'policy':'preserve_base_with_cutoff_date_floor','base':base_value,'source_seed':float(seed['value'])})
            else:
                latest=cutoff;mismatches.append({'code':code,'policy':'preserve_base_no_prior_source_seed','base':base_value if np.isfinite(base_value) else None})
        for row in group.itertuples():
            at=int(row.activation)
            if col<base_columns and at<base_rows:continue
            effective=np.datetime64(row.effective.date())
            if effective<latest:discarded_old+=1;continue
            if at>=len(dates):continue
            latest=effective;at=max(at,base_rows) if col<base_columns else at
            output[at:,col]=row.value
            applied.append({'code':code,'first_date':str(dates[at]),'effective':str(effective),'announcement':str(row.ann.date()),'wan_shares':float(row.value)})
    if issue_dates is not None:
        for col in range(base_columns,len(codes)):output[dates<issue_dates[col],col]=np.nan
    for col,code in enumerate(codes[:base_columns]):
        if code not in processed:
            value=float(output[base_rows-1,col])
            mismatches.append({'code':code,'policy':'preserve_base_no_eligible_capital_rows','base':value if np.isfinite(value) else None})
    return {'ambiguous_quarter_end_rows':int(ambiguous.sum()),'boundary_policies':mismatches,
            'old_effective_records_ignored':discarded_old,'applied_events':applied,
            'contract':'announcement<T and effective<=T; latest effective date wins, then announcement version; native WAN units'}
