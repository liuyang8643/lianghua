"""Train-period-only portfolio search; holdouts are observation outputs only."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import gc
import itertools
import json
import os
from pathlib import Path
import random
import time

if __name__=='__main__':
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[name]='1'
import numpy as np
from scipy.stats import rankdata
from env.diagonal_mvo import diagonal_variance
from env.forward_returns import open_forward_returns
from env.fees import DEFAULT_FEE_SCHEDULE
from .portfolio import PreparedModelPortfolio
from .search import ga_proposals,key,rank_trial,portfolio_methods
from utils.atomic_file import atomic_write_json,file_sha256

TOPK_SPACE={'topk':[20,30,50,100],'n_drop':[1,2,5,10],'hold_thresh':[1,3,5,10]}
POOL_CHOICES=[20,40,80]
MVO_PARAMETER_ORDER=('pool_size','risk_aversion','turnover_limit','ridge','lookback','shrinkage')
RISK_MULTIPLIERS=[.001,.01,.1,1.,10.,100.]
RIDGE_MULTIPLIERS=[0.,.001,.01,.1,1.,10.]


def scaled(value,multipliers):
    return [float(f'{value*factor:.12g}') for factor in multipliers]


def invalid_trial_state(error):
    """Failed native execution has no fitness; unrelated errors still abort."""
    message=str(error)
    if 'MVO solve failed' in message:return 'invalid_solver'
    if (message.startswith('planner violated the mandatory full-investment contract: ')
            and message.endswith(': cash_sweep_incomplete')):
        return 'invalid_execution_contract'
    raise error


def derive_training_space(prepared,protocol):
    if prepared.split!='train':raise ValueError('search-space statistics may only use training period')
    rt=prepared.runtime;episode=prepared.episode
    rows=np.unique(np.linspace(episode.decision_start,episode.decision_stop-1,min(32,episode.decision_stop-episode.decision_start),dtype=int))
    returns=open_forward_returns(rt.data,1);evidence=[]
    for index in rows:
        day=str(rt.trade_dates[index]);assert protocol['train'][0]<=day<=protocol['train'][1]
        market=episode.market_at(int(index));scores=prepared.prediction[index]
        valid=np.isfinite(scores)&(market.listing_age>=0)&~market.delisted_mask&~market.st_mask&np.isfinite(market.open_prices)&(market.open_prices>=2)
        buyable=valid&market.trade_legality(True).buy_allowed
        if buyable.sum()<30:continue
        percentiles=np.full(len(scores),np.nan);percentiles[valid]=(rankdata(scores[valid],method='average')-.5)/valid.sum()
        expected=prepared.calibration.predict(percentiles[buyable])
        order=np.argsort(-np.where(np.isfinite(scores),scores,-np.inf),kind='stable');pool=order[buyable[order]][:max(POOL_CHOICES)]
        variance,counts=diagonal_variance(returns[max(0,index-120):index,pool],horizon=prepared.horizon)
        evidence.append({'date':day,'return_spread':float(np.std(expected)),'variance_scale':float(np.median(variance)),
            'buyable_stocks':int(buyable.sum()),'risk_assets':len(pool),'risk_window_last_open':day})
    if not evidence:raise ValueError('no usable training-period scale observations')
    signal=max(float(np.median([row['return_spread'] for row in evidence])),1e-6)
    variance=max(float(np.median([row['variance_scale'] for row in evidence])),1e-10)
    reference_count=float(np.median(POOL_CHOICES))
    risk_anchor=reference_count*signal/variance;ridge_anchor=reference_count*signal
    space={'pool_size':POOL_CHOICES,'risk_aversion':scaled(risk_anchor,RISK_MULTIPLIERS),
        'turnover_limit':[.02,.05,.1,.2,.4,.8,1.,2.],'ridge':scaled(ridge_anchor,RIDGE_MULTIPLIERS),
        'lookback':[60,120,252],'shrinkage':[.1,.5,.9,1.]}
    receipt={'selection_split':'train','training_period':protocol['train'],'sample_dates_and_statistics':evidence,
        'signal_scale':signal,'variance_scale':variance,'reference_count':reference_count,
        'risk_anchor':risk_anchor,'ridge_anchor':ridge_anchor,'space':space,
        'rule':'fixed log-scale multipliers around training-only signal/risk units; no validation outcomes or prior winners used'}
    return space,receipt


def run(root):
    from scripts.run_research_guarded import require_guarded_job
    require_guarded_job();root=Path(root).resolve();started=time.monotonic()
    protocol=json.loads((root/'protocol.json').read_text(encoding='utf-8'))
    methods=portfolio_methods(protocol)
    if protocol.get('selection_split')!='train':raise ValueError('all optimization must use the training period')
    if (root/'final').exists():raise ValueError('cannot tune after holdout observation artifacts exist')
    if asdict(DEFAULT_FEE_SCHEDULE)!=protocol['fees']:raise ValueError('native fees differ from study contract')
    stage=root/'search';stage.mkdir(exist_ok=True);winners=[]
    budget_cfg=protocol.get('search_budget',{'mvo_max_trials':200,'mvo_min_trials':80,'mvo_soft_wall_seconds':5400})
    protected={str(p):file_sha256(p) for folder in ['env','ai/supervised'] for p in sorted(Path(folder).glob('*.py'))}
    for horizon in (1,5,10):
        prepared=PreparedModelPortfolio(root,horizon,'train')
        if prepared.split!='train':raise AssertionError('nontraining portfolio escaped admission')
        space_path=stage/f'n{horizon}_training_space.json'
        model=root/'models'/f'n{horizon}'
        input_identity={'model':file_sha256(model/'model.txt'),'predictions':file_sha256(model/'predictions.npy'),
            'calibration':file_sha256(model/'calibration.json'),'data_manifest':file_sha256(root/'features/data_manifest.json'),
            'runtime':json.loads((root/'active_data.json').read_text(encoding='utf-8'))['runtime_sha256'],
            'protocol':file_sha256(root/'protocol.json'),'source':protected,'selection_split':'train','period':protocol['train'],'horizon':horizon}
        if 'mvo' not in methods:
            mvo_space={}
        elif space_path.exists():
            saved=json.loads(space_path.read_text(encoding='utf-8'))
            if saved['input_identity']!=input_identity:raise ValueError('training-space identity changed')
            mvo_space=saved['space']
        else:
            mvo_space,saved=derive_training_space(prepared,protocol);saved['input_identity']=input_identity;atomic_write_json(space_path,saved)
        # JSON key sorting must not change RNG draw-to-parameter mapping on resume.
        if mvo_space:mvo_space={name:mvo_space[name] for name in MVO_PARAMETER_ORDER}
        for method in methods:
            arm=stage/f'n{horizon}_{method}';arm.mkdir(exist_ok=True)
            space=TOPK_SPACE if method=='topk' else mvo_space
            identity={**input_identity,'method':method,'space':space,'budget':budget_cfg if method=='mvo' else 'exhaustive',
                'seed':20261105+horizon*1009+(0 if method=='topk' else 1)}
            identity_path=arm/'identity.json'
            if identity_path.exists() and json.loads(identity_path.read_text(encoding='utf-8'))!=identity:raise ValueError('training search identity changed')
            atomic_write_json(identity_path,identity);identity_sha=file_sha256(identity_path)
            ledger=arm/'trials.jsonl';trials=[json.loads(line) for line in ledger.read_text(encoding='utf-8').splitlines()] if ledger.exists() else []
            if any(row.get('selection_split')!='train' or row.get('evaluation_identity_sha256')!=identity_sha for row in trials):
                raise ValueError('foreign or nontraining trial cannot enter this search')
            done={key(row['parameters']) for row in trials}
            if len(done)!=len(trials):raise ValueError('duplicate search parameters')
            if (arm/'complete.json').exists():
                receipt=json.loads((arm/'complete.json').read_text(encoding='utf-8'))
                if file_sha256(ledger)!=receipt['ledger_sha256'] or file_sha256(arm/'winner.json')!=receipt['winner_sha256']:raise ValueError('completed training-search output changed')
                winners.append(json.loads((arm/'winner.json').read_text(encoding='utf-8')));continue
            rng=random.Random(identity['seed'])
            if trials:
                def tuples(x):return tuple(map(tuples,x)) if isinstance(x,list) else x
                rng.setstate(tuples(trials[-1]['rng_after_proposal']))
            grid=[dict(zip(TOPK_SPACE,values)) for values in itertools.product(*TOPK_SPACE.values())] if method=='topk' else None
            maximum=len(grid) if grid is not None else budget_cfg['mvo_max_trials']
            while len(trials)<maximum:
                if (stage/'STOP').exists():raise RuntimeError('training search STOP')
                if method=='mvo' and len(trials)>=budget_cfg['mvo_min_trials'] and sum(row['seconds'] for row in trials)>=budget_cfg['mvo_soft_wall_seconds']:break
                if grid is not None:parameters=next(p for p in grid if key(p) not in done);proposal='exhaustive_grid'
                elif not trials:
                    parameters={'pool_size':40,'risk_aversion':space['risk_aversion'][3],'turnover_limit':.2,
                        'ridge':space['ridge'][3],'lookback':120,'shrinkage':.5};proposal='training_scale_reference'
                else:
                    attempts=0
                    while True:
                        parameters=ga_proposals(space,trials,rng);attempts+=1
                        if attempts>1000:parameters={name:rng.choice(values) for name,values in space.items()}
                        if key(parameters) not in done:break
                    proposal='GA_train_fitness'
                begin=time.monotonic()
                try:
                    _,metrics,_=prepared.evaluate(method,parameters,details=False)
                    row={'state':'complete','trial':len(trials),'parameters':parameters,'metrics':metrics,'seconds':time.monotonic()-begin}
                except RuntimeError as exc:
                    row={'state':invalid_trial_state(exc),'trial':len(trials),'parameters':parameters,'error':str(exc),'seconds':time.monotonic()-begin}
                row.update(selection_split='train',evaluation_identity_sha256=identity_sha,rng_after_proposal=rng.getstate(),proposal_method=proposal)
                with ledger.open('a',encoding='utf-8') as stream:stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())
                trials.append(row);done.add(key(parameters))
                status={'state':'searching','selection_split':'train','horizon':horizon,'method':method,'completed':len(trials),'max_trials':maximum,'elapsed_seconds':time.monotonic()-started}
                atomic_write_json(arm/'status.json',status);print(status,flush=True)
            valid=[row for row in trials if row['state']=='complete']
            if not valid:raise RuntimeError('no complete training-period candidate')
            best=max(valid,key=rank_trial)
            winner={'horizon':horizon,'method':method,'selection_split':'train','parameters':best['parameters'],'train':best['metrics'],
                'trials':len(trials),'valid_trials':len(valid),'selection_rule':'training net Calmar, then training CAGR/drawdown/turnover'}
            atomic_write_json(arm/'winner.json',winner,allow_nan=False)
            atomic_write_json(arm/'complete.json',{'state':'complete','ledger_sha256':file_sha256(ledger),'winner_sha256':file_sha256(arm/'winner.json')})
            atomic_write_json(arm/'status.json',{'state':'complete','selection_split':'train','completed':len(trials)});winners.append(winner)
        del prepared;gc.collect()
    if protected!={str(p):file_sha256(p) for folder in ['env','ai/supervised'] for p in sorted(Path(folder).glob('*.py'))}:raise ValueError('source changed during training-only search')
    chosen=max(winners,key=lambda row:rank_trial({'metrics':row['train']}))
    atomic_write_json(stage/'frozen_selection.json',{'state':'frozen_before_test','selection_split':'train','winner':chosen,
        'portfolio_methods':list(methods),'arm_winners':winners,'holdout_role':'observation for user only; no selection or refit feedback','elapsed_seconds':time.monotonic()-started},allow_nan=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);run(parser.parse_args().root)
