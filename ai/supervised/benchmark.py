"""Benchmark the shared account engine without opening validation/test returns."""
import argparse
from dataclasses import replace
import hashlib
import os
from pathlib import Path
import time

if __name__ == '__main__':
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):
        os.environ[key]='1'
import numpy as np
from offline_data import load_runtime_slice
from factor import FactorDefinition,FactorMetadata,precompute_factors
from env.action_schema import ActionSchema
from env.backtest import EpisodeSession,PreparedEpisode,run_day_config_episode
from env.fees import DEFAULT_FEE_SCHEDULE
from utils.atomic_file import atomic_write_json


class BenchmarkOpenPrice:
    def calc_batch(self,panel):return -np.asarray(panel['open'],dtype=np.float64)


def run(root,runtime_path):
    from scripts.run_research_guarded import require_guarded_job
    require_guarded_job()
    begun=time.monotonic()
    runtime=load_runtime_slice(runtime_path,'2000-01-01','2017-12-31',preload_rows=1,cache_dir=root/'benchmark_runtime_cache')
    definition=FactorDefinition(FactorMetadata('BenchmarkOpenPrice','timing-only-v1',0,('open',),
        hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),BenchmarkOpenPrice)
    factors=precompute_factors(runtime,definitions=(definition,))
    schema=ActionSchema(factor_names=factors.factor_names,filter_names=factors.filter_names,
                        fixed_buy_n=20,fixed_sell_m=50,schema_version='timing-only-research-v1')
    episode=PreparedEpisode.build(runtime,factors,encode_observations=False,action_schema=schema)
    config=schema.decode(np.zeros(schema.action_dim))
    prepare=time.monotonic()-begun
    warm=replace(episode,decision_stop=episode.decision_start+11)
    begun=time.monotonic();run_day_config_episode(EpisodeSession(warm,action_schema=schema),lambda _:config,record_details=False)
    jit=time.monotonic()-begun
    measurements=[]
    for _ in range(2):
        session=EpisodeSession(episode,action_schema=schema)
        begun=time.monotonic();trace=run_day_config_episode(session,lambda _:config,record_details=False)
        measurements.append(time.monotonic()-begun)
    # Intentionally do not inspect or publish profit metrics for this timing
    # fixture. It is not a trained model or a candidate chosen for returns.
    atomic_write_json(root/'account_benchmark.json',{
        'period':['2000-01-01','2017-12-31'],'actual_dates':[str(runtime.decision_dates[0]),str(runtime.decision_dates[-1])],
        'transitions':episode.transition_count,'stock_axis':runtime.n_stocks,'prepare_seconds':prepare,
        'jit_warmup_10_steps_seconds':jit,'pure_replay_seconds':measurements,
        'seconds_per_1000_days':float(np.mean(measurements))*1000/episode.transition_count,
        'fee_schedule':DEFAULT_FEE_SCHEDULE.__dict__,
        'scope':'canonical fixed-score account timing only; does not include ML prediction or daily MVO solver; no validation/test replay',
    },allow_nan=False)
    print({'state':'complete','pure_replay_seconds':measurements,'transitions':episode.transition_count},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--runtime',type=Path,required=True)
    a=p.parse_args();run(a.root,a.runtime)
