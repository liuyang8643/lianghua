"""Apply only independently cross-checked missing open/close research rows."""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd


def repair(primary: Path, supplements: list[Path], output: Path, bounded_precision: list[Path] | None = None):
    p=pd.read_parquet(primary)[['code','date','open','close']].copy()
    p['date']=pd.to_datetime(p.date)
    original=p.copy()
    evidence=[]
    bounded_paths = {path.resolve() for path in (bounded_precision or [])}
    if not bounded_paths.issubset({path.resolve() for path in supplements}):
        raise ValueError('bounded precision paths must also be supplements')
    for path in supplements:
        s=pd.read_parquet(path)
        if 'ts_code' in s:
            s['code']=s.ts_code.str.split('.').str[0]
            s['date']=pd.to_datetime(s.trade_date.astype(str),format='%Y%m%d')
        s=s[['code','date','open','close']].copy()
        s['date']=pd.to_datetime(s.date)
        s=s[s.code.isin(original.code)]
        if s.duplicated(['code','date']).any():
            raise ValueError(f'duplicate supplement rows: {path}')
        if not np.isfinite(s[['open','close']]).all().all() or (s[['open','close']]<=0).any().any():
            raise ValueError(f'invalid supplement prices: {path}')
        overlap=s.merge(original,on=['code','date'],suffixes=('_new','_original'))
        delta=np.abs(overlap[['open_new','close_new']].to_numpy()-overlap[['open_original','close_original']].to_numpy())
        relative = delta / overlap[['open_original','close_original']].to_numpy()
        bounded = path.resolve() in bounded_paths
        consistent = len(overlap)>0 and (bool((delta<=0.050000001).all() and (relative<=7e-6).all()) if bounded else bool((delta==0).all()))
        if not consistent:
            raise ValueError(f'supplement price convention differs: {path}')
        existing=pd.MultiIndex.from_frame(p[['code','date']])
        missing=s[~pd.MultiIndex.from_frame(s[['code','date']]).isin(existing)].copy()
        bounds=original.groupby('code').date.agg(['min','max'])
        missing=missing.join(bounds,on='code')
        missing=missing[(missing.date>=missing['min'])&(missing.date<=missing['max'])].drop(columns=['min','max'])
        counts=overlap.groupby('code').size()
        if any(counts.get(code,0)<5 for code in missing.code.unique()):
            raise ValueError(f'need >=5 overlapping dates per patched industry: {path}')
        evidence.append({'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'overlap_rows':len(overlap),'comparison_contract':'explicit research precision bound: <=0.05 index points AND <=7e-6 relative; not exact, not proof of rounding' if bounded else 'exact','nonexact_price_cells':int((delta!=0).sum()),'max_relative_open_close_difference':float(relative.max()),'max_absolute_open_close_difference':float(delta.max()),'added_rows':len(missing),'added_keys':[{'code':r.code,'date':str(r.date.date())} for r in missing.itertuples()]})
        p=pd.concat([p,missing],ignore_index=True)
    output.mkdir(parents=True,exist_ok=True)
    p=p.sort_values(['date','code']).reset_index(drop=True)
    p.to_parquet(output/'repaired_indices.parquet',index=False)
    manifest={'schema':'sw-explicit-research-gap-patches-v2','primary':str(primary),'primary_sha256':hashlib.sha256(primary.read_bytes()).hexdigest(),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'supplements':evidence,'output_sha256':hashlib.sha256((output/'repaired_indices.parquet').read_bytes()).hexdigest(),'semantics':'Original rows unchanged; only missing open/close rows within each original series date bounds added. No price imputation. Overlap comparison is convention evidence, not full PIT certification. Explicit bounded-precision sources are not exact replications. Calendar audit must pass separately.'}
    (output/'repair_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(evidence,ensure_ascii=False,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--primary',type=Path,required=True)
    parser.add_argument('--supplement',type=Path,action='append',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--bounded-precision',type=Path,action='append',default=[],help='Explicitly named supplement with <=0.05 point AND <=7e-6 relative overlap differences; research only')
    args=parser.parse_args()
    repair(args.primary,args.supplement,args.output,args.bounded_precision)
