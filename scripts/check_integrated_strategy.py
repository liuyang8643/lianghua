"""Offline check of the imported source using its sealed research inputs."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[name]='1'
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from offline_data import load_runtime_slice
from strategy_release.runtime import FrozenStrategyRuntime,FeatureInputs


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--research-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--cache-dir',type=Path,default=Path('data/runtime_cache'))
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    root=args.research_root
    active=json.loads((root/'active_data.json').read_text())
    rt=load_runtime_slice(active['runtime_path'],'2026-10-30','2026-10-30',preload_rows=100001,
                          cache_dir=args.cache_dir)
    runner=FrozenStrategyRuntime()
    inputs=FeatureInputs.load('data/strategy_inputs')
    print('Computing 284 raw features through integrated canonical source',flush=True)
    scores=runner.predict(rt,[rt.decision_start],inputs)
    dates=np.load(root/'features/dates.npy')
    row=int(np.flatnonzero(dates==np.datetime64('2026-10-30'))[0])
    inventory=json.loads((root/'features/inventory.json').read_text())
    normalized=np.column_stack([np.load(inventory[name]['path'],mmap_mode='r')[row] for name in runner.strategy.names])
    eligible=np.load(root/'features/eligible.npy',mmap_mode='r')[row]
    raw,reference=runner.strategy.predict(normalized,feature_names=runner.strategy.names,
        codes=rt.stock_codes,eligible=eligible,complete_universe=True)
    np.testing.assert_array_equal(scores[rt.decision_start],reference)
    np.save(args.output/'scores.npy',scores[rt.decision_start])
    (args.output/'complete.json').write_text(json.dumps(dict(state='pass',date='2026-10-30',
        raw_features_recomputed=284,normalized_reference_source=str(root/'features/inventory.json'),
        scores_exact=True,model_retrained=False),indent=2),encoding='utf8')
    print('Integrated raw features -> normalize -> model scores exact',flush=True)


if __name__=='__main__':main()
