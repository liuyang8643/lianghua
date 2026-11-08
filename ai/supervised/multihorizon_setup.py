"""Freeze the approved multi-horizon experiment and enumerate all local factors."""
import argparse,json,shutil,platform,sys
from pathlib import Path
from dataclasses import asdict
from utils.atomic_file import atomic_write_json,file_sha256
from .multihorizon_admission import rejected_factors

def setup(root):
    from scripts.run_research_guarded import require_guarded_job
    require_guarded_job()
    import numpy as np
    from offline_data import load_runtime_slice
    from factor.research import plain_catalog,bound_groups,financial_adaptations
    from factor.library.local_variants import definitions
    from factor.library.research_supplements import definitions as supplements
    from factor.library.qlib_open import CATALOG
    from factor_db import require_allowed_factor_names
    parent=Path('artifacts/research/qlib_buyable_rank_buffer50_20261106')
    root.mkdir(parents=True,exist_ok=False)
    protocol=json.loads((parent/'protocol.json').read_text(encoding='utf8'))
    protocol.update(model_horizons=[3,7,15,30,60],ic_horizons=[3,7,15,30,60],
        fold_years=list(range(2011,2018)),maximum_label_horizon=60,minimum_ic_days=60,correlation_threshold=.8,
        ic_market_buyable=True,ic_normalized_input=True,ic_train_only=True,
        verified_feature_reuse_root=str(parent.resolve()),verified_feature_formula_root=str(Path(protocol['feature_parent']).resolve()),
        feature_runtime_cache=str(Path(protocol['feature_runtime_cache']).resolve()),
        selection='per-fold fit-only; positive RankIC and abs mean daily Spearman <.8; five horizon union',
        model='single categorical-horizon LightGBM; five-task/equal-date weighted MSE',
        model_retrain='7 folds and final fixed full-development refit; median fold best iteration',
        selection_split='OOF2011-2017 continuous native account; all outer periods observation only',
        ga={'population':32,'generations':60,'patience':10,'min_improvement':.0001,'seed':20261106,
            'buy_n':[20,30,40],'sell_m':[40,60,80],'hold_thresh':[3,7,15],'max_replacements':[1,2,3,5,10],
            'weight_units':10,'total_space':135135},
        compute={'threads':8,'max_rounds':1000,'early_stopping_rounds':60,'seed':20261106},
        status='approved_protocol',known_limitations=protocol['known_limitations'][:3]+[
            'OOF search evaluates annual model changes; final holdouts evaluate one frozen model.',
            'Historical factor families and outer periods were already studied; not new blind evidence.'])
    repaired=Path('artifacts/research/multihorizon_oof_ga_20261106_windowfix')
    if (repaired/'features/status.json').exists() and json.loads((repaired/'features/status.json').read_text()).get('state')=='complete':
        proof=json.loads((repaired/'features/identity.json').read_text(encoding='utf8'))
        compatible=all(Path(p).exists() and file_sha256(p)==sha for p,sha in proof['factor_sources'].items())
        if compatible:
            protocol['verified_feature_reuse_root']=str(repaired.resolve())
            protocol['verified_feature_formula_root']=str(repaired.resolve())
    proof=json.loads((Path(protocol['verified_feature_formula_root'])/'features/identity.json').read_text(encoding='utf8'))
    mismatches=[p for p,sha in proof['factor_sources'].items() if not Path(p).exists() or file_sha256(p)!=sha]
    if mismatches:
        protocol.pop('verified_feature_reuse_root');protocol.pop('verified_feature_formula_root')
        protocol['cache_reuse_rejected_changed_sources']=mismatches
    atomic_write_json(root/'protocol.json',protocol);shutil.copy2(parent/'active_data.json',root/'active_data.json')
    old=Path(protocol['feature_parent']);(root/'dividend_source').mkdir()
    shutil.copy2(old/'dividend_source/events.parquet',root/'dividend_source/events.parquet')
    active=json.loads((root/'active_data.json').read_text())
    rt=load_runtime_slice(active['runtime_path'],protocol['train'][0],protocol['test'][1],preload_rows=100001,cache_dir=Path(protocol['feature_runtime_cache']))
    catalog={};excluded=[]
    historic=Path('artifacts/research/qlib_open_rank_20261105')
    for stage in ['qlib_factors_v2','local_factors_v2','supplement_factors']:
        for r in json.loads((historic/stage/'inventory.json').read_text(encoding='utf8')):catalog[r['name']]=r
    plain,ex=plain_catalog(Path('.'),set(rt.data));excluded+=ex
    for d in [*plain,*financial_adaptations(),*definitions(),*supplements()]:catalog[d.metadata.name]={'name':d.metadata.name,'metadata':asdict(d.metadata)}
    bound=[]
    for group in bound_groups(rt):
        for d in group:
            catalog[d.metadata.name]={'name':d.metadata.name,'metadata':asdict(d.metadata)};bound.append({'name':d.metadata.name})
    for r in CATALOG['features']:catalog[r['name']]=r
    banned=rejected_factors(catalog.values())
    for name in banned:excluded.append({'name':name,'reason':banned[name]});del catalog[name]
    for name in list(catalog):
        try:require_allowed_factor_names([name])
        except ValueError as e:excluded.append({'name':name,'reason':str(e)});del catalog[name]
    atomic_write_json(root/'bound_preflight.json',{'state':'catalog_enumerated','results':bound})
    (root/'selection').mkdir()
    for h in protocol['model_horizons']:atomic_write_json(root/'selection'/f'selected_{h}d.json',{'horizon':h,'role':'candidate enumeration only; no statistical selection yet','selected':list(catalog.values())})
    atomic_write_json(root/'selection/binding_audit.json',{'state':'pass','scope':'candidate-file identity only, not IC selection',
        'artifacts':{p.name:file_sha256(p) for p in (root/'selection').glob('selected*json')}})
    atomic_write_json(root/'catalog.json',{'candidates':list(catalog),'count':len(catalog),'exclusions':excluded})
    shutil.copytree(Path(protocol['feature_runtime_cache']),root/'extended_cache',dirs_exist_ok=False,copy_function=lambda a,b: __import__('os').link(a,b))
    # Only immutable source data/cache are hard-linked; manifests explain provenance.
    files=[p for d in ['ai/supervised','env','factor','factor_db','offline_data','utils'] for p in Path(d).rglob('*.py')]
    files += [Path('scripts/run_research_guarded.py'),Path('configs/config.json')]
    for p in files:
        dest=root/'source'/p;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    atomic_write_json(root/'source_manifest_initial.json',{str(p):file_sha256(p) for p in files})
    import importlib.metadata as md
    atomic_write_json(root/'environment.json',{'python':sys.version,'executable':sys.executable,'platform':platform.platform(),
        'packages':{n:md.version(n) for n in ['numpy','scipy','pandas','lightgbm','numba']}})
    print({'state':'prepared','candidate_factors':len(catalog),'root':str(root)},flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);setup(p.parse_args().root)
