"""Backtest the sole deployed frozen strategy using its canonical runtime."""
import argparse
import json
import os
for _thread_setting in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[_thread_setting]='1'
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from offline_data import latest_runtime_npz_path,load_runtime_slice
from strategy_release.runtime import FrozenStrategyRuntime,FeatureInputs
from ai.supervised.reporting import write_detailed_observation
from env.simulator import DaySimulator
from testback.economic_positions import EconomicPositionLedger


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start-date',required=True)
    parser.add_argument('--end-date',required=True)
    parser.add_argument('--runtime',type=Path)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--initial-cash',type=float,default=1_000_000.)
    parser.add_argument('--inputs',type=Path,default=Path('data/strategy_inputs'))
    args=parser.parse_args()
    out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=False)
    rt=load_runtime_slice(args.runtime or latest_runtime_npz_path(),args.start_date,args.end_date,
        preload_rows=100001,cache_dir=Path('data/runtime_cache'))
    assembly=FrozenStrategyRuntime()
    scores=assembly.predict(rt,np.arange(rt.decision_start,rt.decision_stop),FeatureInputs.load(args.inputs))
    prepared=assembly.prepare(rt,scores,initial_cash=args.initial_cash,split='requested')
    ledger=EconomicPositionLedger()
    original=DaySimulator.step
    def observe(simulator,account,plan,opening,next_opening,**kwargs):
        result=original(simulator,account,plan,opening,next_opening,**kwargs)
        ledger.observe(account,result.account_state,plan,result.fills,result.diagnostics)
        return result
    DaySimulator.step=observe
    try:
        trace,metrics,_=prepared.evaluate(assembly.strategy.manifest['parameters'],details=True)
    finally:
        DaySimulator.step=original
    (out/'protocol.json').write_text(json.dumps(dict(initial_cash=args.initial_cash,observation_splits=['requested'])),encoding='utf8')
    (out/'active_data.json').write_text(json.dumps(dict(manifest_path=str(out/'manifest.json'))),encoding='utf8')
    config={**assembly.strategy.manifest['parameters'],'weights':{'MultiHorizonScore':1.}}
    path=write_detailed_observation(out,'ml','requested',rt,trace,prepared.final_account,config,ledger)
    np.savez_compressed(out/'trace.npz',nav=trace.nav,returns=trace.portfolio_returns,cash=trace.cash,
        exposure=trace.exposure,decision_dates=trace.decision_dates,next_dates=trace.next_decision_dates)
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf8')
    print(path)


if __name__=='__main__':main()
