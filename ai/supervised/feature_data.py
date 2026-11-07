"""Frozen selected factor matrices through the requested final test open."""
import argparse
import json
import os
from pathlib import Path
import time
import gc
import hashlib

if __name__ == '__main__':
    for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):
        os.environ[k]='1'
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from offline_data import load_runtime_slice
from offline_data.supplemental_fields import industry_panel,implemented_dividends
from factor.research import plain_catalog,bound_groups,financial_adaptations
from factor.library.local_variants import Inputs,candidates,variant_name
from factor.library.qlib_open import CATALOG,ExpressionPanel,price_fields
from factor.library.research_supplements import definitions as supplement_definitions,cash_dividend_scores,monthly_pulse
from env.forward_returns import open_forward_returns
from .local_preflight import panel_for
from .monthly_data import verified_monthly_path
from .ic import daily_ic,summary
from env.forward_returns import label_split_mask
from env.legality import investment_universe_mask
from factor_db import require_allowed_factor_names
from utils.atomic_file import atomic_write_json,file_sha256,replace_file


def rank_features(raw,eligible,binary=False):
    values=np.where(eligible&np.isfinite(raw),raw,np.nan)
    if binary:return values.astype(np.float32)
    ranks=rankdata(values,axis=1,method='average',nan_policy='omit')
    counts=np.isfinite(values).sum(axis=1)[:,None]
    return np.divide(ranks-.5,counts,out=np.full(values.shape,np.nan),where=counts>0).astype(np.float32)


def materialize_factor_values(rt, names, save, *, bound_names=(), industry_history=None,
                              dividend_events=None, monthly_payload=None):
    """Compute selected factor formulae once through the shared research adapters.

    The callback owns storage or diagnostics. No labels, fitting, or source
    downloads occur here; callers supply the immutable input snapshot.
    """
    needed=set(names);emitted=set()
    def emit(name,raw):
        if name in emitted:return
        save(name,raw);emitted.add(name)
    qlib={r['name']:r for r in CATALOG['features']}
    if needed&set(qlib):
        evaluator=ExpressionPanel(price_fields(rt.data))
        for name in sorted(needed&set(qlib)):emit(name,evaluator.evaluate(qlib[name]['expression']))
        del evaluator;gc.collect()
    pit={k:rt.field('pit_'+k) for k in ('eps','bps','roe','profit_yoy','revenue_yoy','operating_cf_ps','gross_margin')}
    plain,_=plain_catalog(Path('.'),set(rt.data));plain={d.metadata.name:d for d in [*plain,*financial_adaptations()]}
    bound_names=set(bound_names)
    for name in sorted(needed&set(plain)-bound_names):
        d=plain[name];emit(name,d.implementation().calc_batch(panel_for(rt,d,pit)))
    if needed&bound_names:
        for group in bound_groups(rt):
            for d in group:
                if d.metadata.name in needed:emit(d.metadata.name,d.implementation().calc_batch(panel_for(rt,d,pit)))
    variants={variant_name(n,p):(n,p) for n,p in candidates()}
    if needed&set(variants):
        inputs=Inputs(rt.data)
        for name in sorted(needed&set(variants)):
            n,p=variants[name];emit(name,inputs.score(n,p))
        del inputs;gc.collect()
    supplements={d.metadata.name:d for d in supplement_definitions()}
    panel={**rt.data,'trade_dates':rt.trade_dates,'stock_codes':rt.stock_codes}
    if any(n.startswith('IndustryPeer') for n in needed):
        if industry_history is None:raise ValueError('selected industry factors require dated membership history')
        panel['industry_id']=industry_panel(industry_history,rt.trade_dates,rt.stock_codes,(rt.field('listing_age')>=0)&~rt.field('delisted_mask'))
    for name in sorted(needed&set(supplements)):emit(name,supplements[name].implementation().calc_batch(panel))
    if any(n.startswith('CompletedDividend') for n in needed):
        if dividend_events is None:raise ValueError('selected dividend factors require dated events')
        for name,raw in cash_dividend_scores(panel,dividend_events).items():
            if name in needed:emit(name,raw)
    if monthly_payload is not None:
        for name in monthly_payload['names'].astype(str):
            if name in needed:emit(name,monthly_pulse(rt.trade_dates,rt.stock_codes,monthly_payload,name))
    if emitted!=needed:raise ValueError('unresolved selected features: '+str(needed-emitted))


def run(root):
    from scripts.run_research_guarded import require_guarded_job
    require_guarded_job();started=time.monotonic()
    config=json.loads((root/'protocol.json').read_text(encoding='utf-8'));data=json.loads((root/'active_data.json').read_text())
    monthly_path=verified_monthly_path(config)
    if json.loads(monthly_path.with_name('manifest.json').read_text(encoding='utf-8'))['runtime_sha256']!=data['runtime_sha256']:
        raise ValueError('monthly signals and model inputs use different runtime snapshots')
    if data.get('state')!='admitted_market_readback':raise ValueError('repaired runtime is not admitted')
    if file_sha256(data['manifest_path'])!=data['manifest_sha256']:raise ValueError('active manifest changed')
    manifest=json.loads(Path(data['manifest_path']).read_text())
    if manifest['runtime_sha256']!=data['runtime_sha256'] or file_sha256(data['runtime_path'])!=data['runtime_sha256']:raise ValueError('actual runtime differs from admitted data')
    binding=json.loads((root/'selection/binding_audit.json').read_text())
    if binding['state']!='pass':raise ValueError('selection input/result binding is not admitted')
    for name,sha in binding['artifacts'].items():
        if file_sha256(root/'selection'/name)!=sha:raise ValueError('bound selection artifact changed: '+name)
    selected={};horizons=config['model_horizons']
    for h in horizons:
        choice=json.loads((root/'selection'/f'selected_{h}d.json').read_text(encoding='utf-8'))
        selected.update({r['name']:r for r in choice['selected']})
    require_allowed_factor_names(selected)
    stage=root/'features';stage.mkdir(exist_ok=True);(stage/'columns').mkdir(exist_ok=True);(stage/'records').mkdir(exist_ok=True)
    rt=load_runtime_slice(data['runtime_path'],config['train'][0],config.get('data_end') or config['test'][1],preload_rows=100001,cache_dir=root/'extended_cache')
    if rt.manifest.source_sha256!=data['runtime_sha256']:raise ValueError('loaded runtime identity differs')
    a,b=rt.decision_start,rt.decision_stop;dates=rt.trade_dates[a:b]
    eligible=investment_universe_mask(rt.field('open')[a:b],rt.field('listing_age')[a:b],rt.field('st_mask')[a:b],rt.field('delisted_mask')[a:b])
    ic_eligible=eligible
    if config.get('ic_market_buyable',False):
        from .labels import market_buyable_mask
        _,buyable=market_buyable_mask(rt)
        ic_eligible=buyable[a:b]
    dependencies=[*Path('factor').rglob('*.py'),*Path('factor_db/factors').glob('*.py'),*Path('offline_data').glob('*.py'),
                  Path('ai/supervised/local_preflight.py'),
                  Path('factor/library/qlib_catalog.json'),Path('factor_db/__init__.py'),Path('env/forward_returns.py'),Path('utils/numeric_panel.py'),
                  Path('data/industry_history/sw_industry_history.parquet'),root/'dividend_source/events.parquet',
                  monthly_path,monthly_path.with_name('manifest.json')]
    identity={'runtime':data['runtime_sha256'],'active_manifest':data['manifest_sha256'],'historical_values_reused':bool(config.get('verified_feature_reuse_root')),
              'protocol_sha256':file_sha256(root/'protocol.json'),
              'selection_binding_sha256':file_sha256(root/'selection/binding_audit.json'),
              'selections':{str(h):file_sha256(root/'selection'/f'selected_{h}d.json') for h in horizons},
              'source':file_sha256(Path(__file__)),'factor_sources':{str(p):file_sha256(p) for p in sorted(dependencies)}}
    identity_sha=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    path=stage/'identity.json'
    if path.exists() and json.loads(path.read_text())!=identity:raise ValueError('feature matrix identity changed')
    atomic_write_json(path,identity)
    np.save(stage/'dates.npy',dates);np.save(stage/'codes.npy',np.asarray(rt.stock_codes));np.save(stage/'eligible.npy',eligible)
    if config.get('ic_market_buyable',False):np.save(stage/'label_eligible.npy',ic_eligible)
    for h in horizons:
        path=stage/f'labels_{h}d.npy'
        actual=open_forward_returns(rt.data,h)[a:b]
        if path.exists():np.testing.assert_allclose(np.load(path,mmap_mode='r'),actual,rtol=1e-12,atol=1e-12,equal_nan=True)
        else:
            temp=path.with_suffix('.tmp')
            with temp.open('wb') as stream:np.save(stream,actual,allow_pickle=False)
            replace_file(temp,path)
        del actual
    labels={h:np.load(stage/f'labels_{h}d.npy',mmap_mode='r') for h in horizons}
    if config.get('ic_market_buyable',False):
        for h in horizons:np.save(stage/f'label_rank_{h}d.npy',rank_features(labels[h],ic_eligible))
    (stage/'daily_ic').mkdir(exist_ok=True)
    completed={}
    for name in selected:
        receipt=stage/'records'/f'{name}.json';path=stage/'columns'/f'{name}.npy'
        if receipt.exists():
            record=json.loads(receipt.read_text())
            if file_sha256(path)!=record['sha256']:raise ValueError('cached feature column changed')
            array=np.load(path,mmap_mode='r')
            if array.shape!=eligible.shape or array.dtype!=np.float32 or record.get('generation_identity_sha256')!=identity_sha or record['name']!=name:raise ValueError('feature column axes or generation identity differ')
            for h in horizons:
                ic=record['daily_ic'][str(h)]
                if file_sha256(ic['path'])!=ic['sha256']:raise ValueError('cached factor IC changed')
            completed[name]=record
    reuse={};reuse_ic={}
    if config.get('verified_feature_reuse_root'):
        old=Path(config['verified_feature_reuse_root'])
        receipt=json.loads((old/'features/identity.json').read_text(encoding='utf8'))
        original=Path(config['verified_feature_formula_root'])
        original_id=json.loads((original/'features/identity.json').read_text(encoding='utf8'))
        if receipt['runtime']!=data['runtime_sha256'] or original_id['runtime']!=data['runtime_sha256']:raise ValueError('reuse runtime differs')
        for p,sha in original_id['factor_sources'].items():
            if file_sha256(p)!=sha:raise ValueError('reuse formula dependency changed: '+p)
        np.testing.assert_array_equal(np.load(old/'features/dates.npy'),dates)
        np.testing.assert_array_equal(np.load(old/'features/eligible.npy'),eligible)
        np.testing.assert_array_equal(np.load(old/'features/label_eligible.npy'),ic_eligible)
        reuse=json.loads((old/'features/inventory.json').read_text(encoding='utf8'))
        reuse_ic={Path(k).as_posix():v for k,v in receipt.get('sources',{}).items()}
        if reuse_ic.get('ai/supervised/ic.py')!=file_sha256('ai/supervised/ic.py'):reuse_ic={}
        if config.get('verified_monthly_unit_repair'):
            # Reuse is limited to identical non-monthly columns and unchanged
            # monthly rows; this never reuses selections or model fitting.
            old_config=json.loads((old/'protocol.json').read_text(encoding='utf8'))
            for key in ('train','model_horizons','ic_market_buyable','ic_normalized_input','ic_train_only','minimum_pairs'):
                assert old_config.get(key)==config.get(key),key
            assert file_sha256(old/'features/inventory.json')==config['reuse_parent_inventory_sha256']
            assert file_sha256(old/'features/identity.json')==config['reuse_parent_identity_sha256']
            old_monthly=Path(old_config['monthly_signal_path'])
            changed=set()
            with np.load(old_monthly) as previous,np.load(monthly_path) as current:
                for key in ('names','stock_codes','signal_dates','eligible'):np.testing.assert_array_equal(previous[key],current[key])
                for k,name in enumerate(current['names'].astype(str)):
                    if not np.array_equal(previous['scores'][k],current['scores'][k],equal_nan=True):changed.add(name)
            excluded=set(config['verified_feature_reuse_exclude'])
            assert changed==excluded=={'LowFloatTurnover20_10'},changed
            for h in horizons:
                for prefix in ('labels','label_rank'):
                    filename=f'{prefix}_{h}d.npy'
                    assert file_sha256(stage/filename)==file_sha256(old/'features'/filename),filename
            old_sources={Path(k).as_posix():v for k,v in json.loads((old/'stages/selection/inputs.json').read_text(encoding='utf8')).items()}
            assert old_sources['ai/supervised/ic.py']==file_sha256('ai/supervised/ic.py')
            for h in horizons:assert file_sha256(root/'selection'/f'selected_{h}d.json')==file_sha256(old/'selection'/f'selected_{h}d.json')
            for name in excluded:reuse.pop(name,None)
            reuse_ic={'ai/supervised/ic.py':file_sha256('ai/supervised/ic.py')}
            atomic_write_json(stage/'reuse_validation.json',{'state':'pass_identical_unaffected_inputs','parent':str(old),'parent_inventory_sha256':config['reuse_parent_inventory_sha256'],'parent_identity_sha256':config['reuse_parent_identity_sha256'],'old_monthly_sha256':file_sha256(old_monthly),'new_monthly_sha256':file_sha256(monthly_path),'recomputed_factors':sorted(excluded),'reused_factor_count':len(reuse),'all_labels_equal':True,'all_ic_settings_equal':True,'no_selection_or_model_reuse':True})
    def save(name,raw=None):
        if name not in selected:return
        record=selected[name];target=stage/'columns'/f'{name}.npy';receipt=stage/'records'/f'{name}.json'
        if receipt.exists():
            item=json.loads(receipt.read_text());
            if file_sha256(target)!=item['sha256']:raise ValueError('feature column changed')
            completed[name]=item;return
        binary=record.get('metadata',{}).get('score_semantics')=='binary'
        inherited=None
        if raw is None:
            inherited=reuse[name]
            if file_sha256(inherited['path'])!=inherited['sha256']:raise ValueError('reuse column changed')
            os.link(inherited['path'],target)
        else:
            actual=np.asarray(raw)[a:b]
            temp=target.with_suffix('.tmp.npy');mapped=np.lib.format.open_memmap(temp,mode='w+',dtype=np.float32,shape=eligible.shape);mapped[:]=np.nan
            mapped[:]=rank_features(actual,eligible,binary)
            mapped.flush();mapped._mmap.close();replace_file(temp,target)
        if config.get('ic_normalized_input',False):actual=np.load(target,mmap_mode='r')
        ic_records={}
        for h in horizons:
            ic_path=stage/'daily_ic'/f'{name}_{h}d.npy'
            cached=inherited.get('daily_ic',{}).get(str(h)) if inherited and reuse_ic else None
            if cached:
                if file_sha256(cached['path'])!=cached['sha256']:raise ValueError('reuse IC changed')
                os.link(cached['path'],ic_path);metrics=np.load(ic_path)
            elif config.get('ic_train_only',False):
                mask=label_split_mask(dates,*config['train'],max(horizons));metrics=np.full((len(dates),3),np.nan)
                metrics[mask]=daily_ic(actual[mask],labels[h][mask],ic_eligible[mask],config.get('minimum_pairs',30));np.save(ic_path,metrics)
            else:
                metrics=daily_ic(actual,labels[h],ic_eligible,config.get('minimum_pairs',30));np.save(ic_path,metrics)
            ic_records[str(h)]={'path':str(ic_path),'sha256':file_sha256(ic_path),
                'reused':bool(cached),'train':summary(metrics,label_split_mask(dates,*config['train'],max(horizons) if config.get('ic_train_only',False) else h))}
        item={'name':name,'path':str(target),'sha256':file_sha256(target),'normalization':'binary' if binary else 'daily_PIT_percentile',
              'generation_identity_sha256':identity_sha,'shape':list(eligible.shape),'dtype':'float32',
              'historical_values_reused':inherited is not None,'daily_ic':ic_records}
        atomic_write_json(receipt,item);completed[name]=item
        atomic_write_json(stage/'status.json',{'state':'computing','completed':len(completed),'total':len(selected),'name':name,'elapsed_seconds':time.monotonic()-started})
        print({'completed':len(completed),'total':len(selected),'name':name},flush=True);gc.collect()
    for name in sorted(set(selected)&set(reuse)-set(completed)):save(name)
    needed=set(selected)-set(completed)
    bound_names={row['name'] for row in json.loads((root/'bound_preflight.json').read_text())['results']}
    industry=pd.read_parquet('data/industry_history/sw_industry_history.parquet') if any(n.startswith('IndustryPeer') for n in needed) else None
    events=implemented_dividends(pd.read_parquet(root/'dividend_source/events.parquet'))[0] if any(n.startswith('CompletedDividend') for n in needed) else None
    with np.load(monthly_path,allow_pickle=False) as monthly:
        materialize_factor_values(rt,needed,save,bound_names=bound_names,industry_history=industry,dividend_events=events,monthly_payload=monthly)
    if set(completed)!=set(selected):raise ValueError('unresolved selected features: '+str(set(selected)-set(completed)))
    atomic_write_json(stage/'inventory.json',completed)
    atomic_write_json(stage/'data_manifest.json',{p.name:file_sha256(p) for p in sorted(stage.glob('*.npy'))})
    atomic_write_json(stage/'status.json',{'state':'complete','completed':len(completed),'elapsed_seconds':time.monotonic()-started})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);run(p.parse_args().root)
