"""Refresh the frozen local static benchmark after factor numerical repair."""
import argparse,json
from pathlib import Path
from dataclasses import asdict
import numpy as np
from offline_data import load_runtime_slice
from .periods import data_end,observation_splits,oof_interval
from .reporting import write_detailed_observation
from testback.economic_positions import EconomicPositionLedger
from factor import precompute_factors
from env.action_schema import ActionSchema
from env.backtest import PreparedEpisode,EpisodeSession,run_day_config_episode
from env.fees import DEFAULT_FEE_SCHEDULE
from env.prefilter import prefilter_n_from_config
from ai.ga.config import canonicalize_individual_config
from utils.atomic_file import atomic_write_json,file_sha256

def run(root):
    from scripts.run_research_guarded import require_guarded_job
    require_guarded_job();out=root/'static_baseline';out.mkdir(exist_ok=False)
    c=json.loads((root/'protocol.json').read_text(encoding='utf8'));a=json.loads((root/'active_data.json').read_text());assert asdict(DEFAULT_FEE_SCHEDULE)==c['fees']
    parent=Path(c.get('numerical_repair_parent',root));path=Path(c['static_config_path']) if c.get('static_config_path') else parent/'source/configs/config.json'
    payload=json.loads(path.read_text(encoding='utf8'));(out/'config.json').write_bytes(path.read_bytes())
    rt=load_runtime_slice(a['runtime_path'],c['train'][0],data_end(c),preload_rows=100001,cache_dir=Path(c['feature_runtime_cache']))
    assert file_sha256(a['runtime_path'])==a['runtime_sha256']
    factors=precompute_factors(rt);schema=ActionSchema.from_config(payload,factor_names=factors.factor_names,filter_names=factors.filter_names)
    canonical,day=canonicalize_individual_config(payload,action_schema=schema);results=[]
    atomic_write_json(out/'identity.json',{'config_sha256':file_sha256(path),'runtime_sha256':a['runtime_sha256'],'canonical':canonical,'factor_metadata':[x.as_dict() for x in factors.factor_metadata]})
    for split in observation_splits(c):
        interval=oof_interval(c) if split=='oof' else c[split]
        lo=int(np.searchsorted(rt.trade_dates,np.datetime64(interval[0])));hi=int(np.searchsorted(rt.trade_dates,np.datetime64(interval[1]),side='right'))
        episode=PreparedEpisode.build(rt,factors,decision_start=lo,decision_stop=hi,prefilter_n=prefilter_n_from_config(payload),encode_observations=False,action_schema=schema)
        session=EpisodeSession(episode,action_schema=schema,initial_cash=c['initial_cash'],fees=DEFAULT_FEE_SCHEDULE)
        ledger=EconomicPositionLedger()
        trace=run_day_config_episode(session,lambda _:day,record_details=True,on_transition=ledger);m=trace.metrics.as_dict();m['mean_exposure']=float(trace.exposure.mean())
        write_detailed_observation(root,'static',split,rt,trace,session.current_account,canonical,ledger)
        m['total_fees']=float(sum(f.fee for day_fills in trace.fills for f in day_fills))
        np.testing.assert_allclose(trace.portfolio_returns,trace.nav[1:]/trace.nav[:-1]-1,rtol=1e-10,atol=1e-12)
        output=out/f'{split}.npz';np.savez_compressed(output,nav=trace.nav,returns=trace.portfolio_returns,exposure=trace.exposure,cash=trace.cash,decision_dates=trace.decision_dates,next_dates=trace.next_decision_dates)
        result={'split':split,'metrics':m,'trace_sha256':file_sha256(output)};results.append(result);atomic_write_json(out/f'{split}.json',result)
        print({'static_split':split,'metrics':m},flush=True)
    atomic_write_json(out/'complete.json',{'state':'complete','results':results,'note':'fresh static after independent-window repair; original config retained'})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);run(p.parse_args().root)
