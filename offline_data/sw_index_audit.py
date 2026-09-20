"""Audit sealed SW index prices against an independently stored trading calendar."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd


def audit(prices: Path, calendar: Path, output: Path):
    p=pd.read_parquet(prices)
    with np.load(calendar,allow_pickle=False) as r:
        days=pd.DatetimeIndex(r['trade_dates'])
    days=days[(days>='2012-01-01')&(days<=p.date.max())]
    per_code=[]
    for code,g in p.groupby('code',sort=True):
        expected=days[(days>=g.date.min())&(days<=g.date.max())]
        missing=expected.difference(g.date)
        per_code.append({'code':code,'expected_rows':len(expected),'missing_dates':missing.strftime('%Y-%m-%d').tolist()})
    result={'prices_sha256':hashlib.sha256(prices.read_bytes()).hexdigest(),
            'calendar_path':str(calendar),'calendar_dates_sha256':hashlib.sha256(days.to_numpy(dtype='datetime64[D]').tobytes()).hexdigest(),
            'audit_start':str(days.min().date()),'audit_end':str(days.max().date()),
            'expected_days':len(days),'all_industries_missing_dates':days.difference(p.date).strftime('%Y-%m-%d').tolist(),
            'missing_rows_after_each_series_start':sum(len(x['missing_dates']) for x in per_code),
            'series':per_code,'pass':all(not x['missing_dates'] for x in per_code),
            'interpretation':'Absence before each series first source date is separately a historical coverage limitation, not certified PIT availability. No imputation or calendar compression.'}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='series'},ensure_ascii=False,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prices',type=Path,required=True)
    parser.add_argument('--calendar',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    audit(args.prices,args.calendar,args.output)
