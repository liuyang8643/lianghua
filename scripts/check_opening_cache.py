"""Verify the pre-open cache against the already checked full-factor result."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[key]='1'
import argparse,json,time,hashlib
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from offline_data import load_runtime_slice
from offline_data.supplemental_fields import implemented_dividends
from strategy_release.runtime import FrozenStrategyRuntime,FeatureInputs
from data.strategy_inputs import replace_open_quotes


def main():
    p=argparse.ArgumentParser();p.add_argument('--research-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--cache-dir',type=Path,required=True)
    p.add_argument('--reference-scores',type=Path,required=True);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False);root=args.research_root
    read=lambda p:json.loads(p.read_text(encoding='utf8'))
    active=read(root/'active_data.json');runtime=load_runtime_slice(active['runtime_path'],
        '2026-10-30','2026-10-30',preload_rows=100001,cache_dir=args.cache_dir)
    bound=tuple(row['name'] for row in read(root/'bound_preflight.json')['results'])
    with np.load(root/'monthly/monthly.npz') as payload:monthly={k:payload[k] for k in payload.files}
    inputs=FeatureInputs(bound,pd.read_parquet(root.parents[2]/'data/industry_history/sw_industry_history.parquet'),
        implemented_dividends(pd.read_parquet(root/'dividend_source/events.parquet'))[0],monthly,
        hashlib.sha256((root/'features/identity.json').read_bytes()).hexdigest())
    assembly=FrozenStrategyRuntime();index=runtime.decision_start
    started=time.monotonic();cache=assembly.prepare_open(runtime,index,inputs)
    preparation=time.monotonic()-started
    print('Completed pre-open factor preparation',len(cache.columns),preparation,flush=True)
    quote={code:dict(open=np.asarray([runtime.field('open')[index,col]]),
                     preClose=np.asarray([runtime.field('preClose')[index,col]]))
           for col,code in enumerate(runtime.stock_codes)}
    actual=replace_open_quotes(runtime,quote)
    dates=np.load(root/'features/dates.npy')
    reference_row=int(np.flatnonzero(dates==runtime.trade_dates[index])[0])
    inventory=read(root/'features/inventory.json')
    normalized_reference=np.column_stack([np.load(inventory[name]['path'],mmap_mode='r')[reference_row]
                                          for name in assembly.strategy.names])
    checked=[]
    def check_features(row,raw,normalized,heads,fused,eligible):
        np.testing.assert_array_equal(normalized,normalized_reference)
        checked.append(len(assembly.strategy.names))
    started=time.monotonic();scores=assembly.predict(actual,[index],inputs,opening_cache=cache,
                                                     on_prediction=check_features)[index]
    opening_seconds=time.monotonic()-started
    np.testing.assert_array_equal(scores,np.load(args.reference_scores))
    np.save(args.output/'scores.npy',scores)
    result=dict(state='pass',scores_exact=True,precomputed_factors=len(cache.columns),
                total_factors=len(assembly.strategy.names),preparation_seconds=preparation,
                normalized_feature_columns_exact=checked[0],
                opening_compute_seconds=opening_seconds,scope='same frozen research inputs; no target publication')
    (args.output/'complete.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print(result,flush=True)


if __name__=='__main__':main()
