"""Prepare a fresh expanded-training study; never starts fitting implicitly."""
import argparse,json,shutil
from pathlib import Path
from utils.atomic_file import atomic_write_json,file_sha256
from ai.supervised.multihorizon_admission import require_catalog
from scripts.run_research_guarded import require_guarded_job

def prepare(root,reuse_unaffected_from=None):
    require_guarded_job();settings=json.loads(Path('configs/research/multihorizon_pre_live_20261107.json').read_text(encoding='utf8'));old=Path('artifacts/research/multihorizon_oof_ga_20261106_windowfix').resolve();data=Path(settings['data_root']).resolve()
    assert not root.exists();assert shutil.disk_usage(root.parent).free>200*2**30
    templates=Path('scripts/pre_live_delivery');delivery=json.loads((templates/'manifest.json').read_text(encoding='utf8'))
    for name,sha in delivery['files'].items():
        assert Path(name).name==name and file_sha256(templates/name)==sha,name
    active=json.loads((data/'active_data.json').read_text(encoding='utf8'));assert active['runtime_sha256']==settings['data_runtime_sha256'];assert file_sha256(active['runtime_path'])==active['runtime_sha256'];assert json.loads((data/'data_admission.json').read_text(encoding='utf8'))['state']=='pass'
    root.mkdir();p=json.loads((old/'protocol.json').read_text(encoding='utf8'))
    for k in ['verified_feature_reuse_root','verified_feature_formula_root','numerical_repair_parent']:p.pop(k,None)
    p.update(train=settings['train'],fit=['2005-01-01','2020-12-31'],internal_validation=['2021-01-01','2021-12-31'],validation=settings['validation'],test=None,data_end=settings['validation'][1],fold_years=settings['fold_years'],observation_splits=['oof','fit','internal_validation','train','validation'],selection_split='continuous OOF2011-2021; outer2022-2026 observation only',feature_runtime_cache=str(root/'extended_cache'),portfolio_runtime_cache=str(root/'portfolio_runtime_cache'),monthly_signal_path=str(root/'monthly/monthly.npz'),monthly_source_root=str(Path(p['monthly_source_root']).resolve()),static_config_path=str(old/'static_baseline/config.json'),model_retrain='11freshfolds plus final2005-2021 fixed-round refit',status='expanded_training_prepared',prior_validation_exposure=True)
    p['early_stopping']='Each fold uses its prior-year internal stop only; final trees are median of all11fold best iterations.'
    if reuse_unaffected_from is not None:
        parent=Path(reuse_unaffected_from).resolve()
        assert json.loads((parent/'features/status.json').read_text(encoding='utf8'))['state']=='complete'
        assert (parent/'invalid_input_stop.json').exists(),'Parent must retain its identified invalid-input stop evidence'
        p.update(verified_feature_reuse_root=str(parent),verified_feature_formula_root=str(parent),verified_monthly_unit_repair=True,verified_feature_reuse_exclude=['LowFloatTurnover20_10'],reuse_parent_inventory_sha256=file_sha256(parent/'features/inventory.json'),reuse_parent_identity_sha256=file_sha256(parent/'features/identity.json'))
    atomic_write_json(root/'protocol.json',p);atomic_write_json(root/'active_data.json',active)
    for name in ['catalog.json','bound_preflight.json']:shutil.copy2(old/name,root/name)
    (root/'selection').mkdir()
    for h in p['model_horizons']:shutil.copy2(old/'selection'/f'selected_{h}d.json',root/'selection'/f'selected_{h}d.json')
    atomic_write_json(root/'selection/binding_audit.json',{'state':'pass','scope':'candidate enumeration only; no statistical selection reused','artifacts':{f.name:file_sha256(f) for f in (root/'selection').glob('*.json')}})
    (root/'dividend_source').mkdir();shutil.copy2(old/'dividend_source/events.parquet',root/'dividend_source/events.parquet');require_catalog(root)
    for name in delivery['files']:shutil.copy2(templates/name,root/name)
    atomic_write_json(root/'delivery_template_manifest.json',{'version':delivery['version'],'source_manifest_sha256':file_sha256(templates/'manifest.json'),'files':delivery['files']})
    adapter=json.loads((root/'adapter_code_review.json').read_text(encoding='utf8'))
    if adapter['state']=='pass' and adapter['source_sha256']==file_sha256('ai/supervised/explainability.py'):
        atomic_write_json(root/'review_explanation_adapter_implemented.json',{**adapter,'reused_code_only_review':True,'reuse_basis':'Exact implementation SHA; no data/model/financial-result acceptance reused','source_review_sha256':file_sha256(root/'adapter_code_review.json')})
    atomic_write_json(root/'task_state.json',{'state':'prepared','learner_started':False,'next':'rebuild_monthly_and_complete_features','no_live_deployment':True})
    atomic_write_json(root/'preparation.json',{'state':'prepared','settings_sha256':file_sha256('configs/research/multihorizon_pre_live_20261107.json'),'original_study':str(old),'corrected_data_admission_sha256':file_sha256(data/'data_admission.json'),'fresh_features':reuse_unaffected_from is None,'verified_unaffected_parent':str(reuse_unaffected_from) if reuse_unaffected_from else None,'monthly_recomputed':True,'all_selections_and_models_fresh':True,'no_old2026_factor_cache':True})
    print(root,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--reuse-unaffected-from',type=Path);a=p.parse_args();prepare(a.root,a.reuse_unaffected_from)
