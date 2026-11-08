"""Regression of the integrated account engine against sealed research traces."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[name]='1'
import argparse,json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from offline_data import load_runtime_slice
from ai.supervised.feature_data import rank_features
from strategy_release.runtime import FrozenStrategyRuntime
from utils.atomic_file import file_sha256


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--research-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--cache-dir',type=Path,required=True)
    args=parser.parse_args();root=args.research_root;out=args.output
    out.mkdir(parents=True,exist_ok=False)
    read=lambda p:json.loads(p.read_text(encoding='utf8'))
    protocol=read(root/'protocol.json');active=read(root/'active_data.json')
    if file_sha256(active['runtime_path'])!=active['runtime_sha256']:
        raise ValueError('Frozen historical Runtime changed')
    done=read(root/'models/final/complete.json')
    if file_sha256(root/'models/final/predictions.npy')!=done['predictions_sha256']:
        raise ValueError('Saved predictions changed')
    assembly=FrozenStrategyRuntime()
    if file_sha256(assembly.strategy.root/'model.txt')!=done['model_sha256']:
        raise ValueError('Model changed')
    dates=np.load(root/'features/dates.npy');codes=np.load(root/'features/codes.npy').astype(str)
    predictions=np.load(root/'models/final/predictions.npy',mmap_mode='r')
    eligible=np.load(root/'features/eligible.npy',mmap_mode='r')
    results=[]
    for split in ('validation','train'):
        print('Replaying integrated engine',split,flush=True)
        rt=load_runtime_slice(active['runtime_path'],*protocol[split],preload_rows=252,cache_dir=args.cache_dir)
        np.testing.assert_array_equal(rt.stock_codes,codes)
        at=np.searchsorted(dates,rt.trade_dates);valid=at<len(dates)
        valid[valid]&=dates[at[valid]]==rt.trade_dates[valid]
        scores=np.full((rt.n_dates,rt.n_stocks),np.nan)
        rows=at[valid];raw=predictions[rows];pool=eligible[rows]&np.isfinite(raw).all(axis=2)
        ranks=np.stack([rank_features(raw[:,:,k],pool) for k in range(5)],axis=2)
        scores[valid]=np.sum(ranks*np.asarray(assembly.strategy.manifest['weights']),axis=2)
        prepared=assembly.prepare(rt,scores,initial_cash=protocol['initial_cash'],split=split)
        trace,metrics,_=prepared.evaluate(assembly.strategy.manifest['parameters'],details=True)
        reference=read(root/'final'/(split+'.json'))
        if file_sha256(root/'final'/(split+'.npz'))!=reference['trace_sha256']:
            raise ValueError('Reference trace changed')
        with np.load(root/'final'/(split+'.npz')) as baseline:
            for key,array in [('nav',trace.nav),('returns',trace.portfolio_returns),('cash',trace.cash),
                              ('exposure',trace.exposure),('decision_dates',trace.decision_dates),('next_dates',trace.next_decision_dates)]:
                np.testing.assert_array_equal(array,baseline[key])
        fills=pd.DataFrame([dict(date=str(d),code=f.code,side=f.side,quantity=f.quantity,price=f.price,fee=f.fee)
                            for d,day in zip(trace.decision_dates,trace.fills) for f in day])
        old=pd.read_csv(root/'final'/(split+'_fills.csv'))[list(fills.columns)]
        pd.testing.assert_frame_equal(fills,old,check_dtype=False,rtol=1e-12,atol=1e-10)
        fills.to_csv(out/(split+'_fills.csv'),index=False)
        np.savez_compressed(out/(split+'.npz'),nav=trace.nav,returns=trace.portfolio_returns,
            cash=trace.cash,exposure=trace.exposure,decision_dates=trace.decision_dates,next_dates=trace.next_decision_dates)
        results.append(dict(split=split,arrays_exact=True,fills_equal=True,fill_count=len(fills),metrics=metrics))
        print('Exact match',split,len(fills),'fills',flush=True)
    (out/'complete.json').write_text(json.dumps(dict(state='pass',results=results,
        scope='Integrated source account replay; predictions are hash-verified frozen research inputs'),indent=2),encoding='utf8')


if __name__=='__main__':main()
