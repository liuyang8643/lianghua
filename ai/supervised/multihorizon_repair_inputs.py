"""Audited numerical-window repair: preserve healthy columns and rebuild affected ones."""
import argparse,ast,gc,hashlib,json,os,shutil,time
from pathlib import Path
from dataclasses import asdict
import numpy as np
from offline_data import load_runtime_slice
from factor.research import plain_catalog
from factor.registry import get_factor_definition
from factor.library.local_variants import candidates,variant_name,Inputs
from env.forward_returns import label_split_mask
from .feature_data import rank_features
from .ic import daily_ic,summary
from .multihorizon_admission import require_catalog
from utils.atomic_file import atomic_write_json,file_sha256

def run(parent,root):
    from scripts.run_research_guarded import require_guarded_job
    require_guarded_job();started=time.monotonic()
    assert json.loads((parent/'features/status.json').read_text())['state']=='complete'
    events=[json.loads(line) for line in (parent/'data_logs/guard.jsonl').read_text().splitlines()]
    assert events[-1]['event']=='child_exit' and events[-1]['returncode']==0,'original producer must be terminal'
    manifest=json.loads((parent/'repair_candidate/manifest.json').read_text())
    for name,r in manifest.items():assert file_sha256(name)==r['candidate'],'repair not installed exactly: '+name
    old_identity=json.loads((parent/'features/identity.json').read_text(encoding='utf8'))
    changed={Path(n).as_posix() for n in manifest}
    for name,sha in old_identity['factor_sources'].items():
        assert file_sha256(name)==sha or Path(name).as_posix() in changed,'unexpected dependency change: '+name
    root.mkdir(parents=True,exist_ok=False);source=root/'features';source.mkdir();(source/'columns').mkdir();(source/'daily_ic').mkdir();(source/'records').mkdir()
    config=json.loads((parent/'protocol.json').read_text(encoding='utf8'))
    config['numerical_repair_parent']=str(parent.resolve())
    config['early_stopping']='each fold uses its preceding year; final refit fixes median7 best rounds'
    config['portfolio_config_role']='historical reference only; actual parameters come from approved GA space'
    atomic_write_json(root/'protocol.json',config)
    for name in ['active_data.json','bound_preflight.json','catalog.json','environment.json','package_versions.json']:
        if (parent/name).exists():shutil.copy2(parent/name,root/name)
    (root/'dividend_source').mkdir();os.link(parent/'dividend_source/events.parquet',root/'dividend_source/events.parquet')
    old_manifest=json.loads((parent/'features/data_manifest.json').read_text())
    for name,sha in old_manifest.items():
        assert file_sha256(parent/'features'/name)==sha
        os.link(parent/'features'/name,source/name)
    atomic_write_json(source/'data_manifest.json',old_manifest)
    active=json.loads((root/'active_data.json').read_text());assert file_sha256(active['runtime_path'])==active['runtime_sha256']
    rt=load_runtime_slice(active['runtime_path'],config['train'][0],config['test'][1],preload_rows=100001,cache_dir=Path(config['feature_runtime_cache']))
    a,b=rt.decision_start,rt.decision_stop;dates=np.load(source/'dates.npy');eligible=np.load(source/'eligible.npy',mmap_mode='r');buyable=np.load(source/'label_eligible.npy',mmap_mode='r')
    np.testing.assert_array_equal(rt.trade_dates[a:b],dates);np.testing.assert_array_equal(rt.stock_codes,np.load(source/'codes.npy').astype(str))
    old_inventory=json.loads((parent/'features/inventory.json').read_text(encoding='utf8'))
    affected=[n for n in old_inventory if n in ['AmihudIlliquidityStrict','DownsideAmihudIlliquidityStrict','CompletedAmihudIlliquidity20'] or n.startswith('CompletedAmihudIlliquidity20__window_')]
    assert len(affected)==9
    dependencies={p:file_sha256(p) for p in old_identity['factor_sources']}
    identity={'runtime':active['runtime_sha256'],'active_manifest':active['manifest_sha256'],
        'source_path':'ai/supervised/multihorizon_repair_inputs.py','source':file_sha256(__file__),
        'parent_identity_sha256':file_sha256(parent/'features/identity.json'),'factor_sources':dependencies,
        'sources':{'ai/supervised/ic.py':file_sha256('ai/supervised/ic.py')},'protocol_sha256':file_sha256(root/'protocol.json'),
        'label_pool':'current market buyable','repair_names':affected,'repair_manifest':manifest,
        'unaffected_contract':'exact parent bytes; only these three implementations and six window variants changed'}
    atomic_write_json(source/'identity.json',identity);generation=file_sha256(source/'identity.json')
    inventory={};changes=[];mask=label_split_mask(dates,*config['train'],60)
    plain,_=plain_catalog(Path('.'),set(rt.data));defs={d.metadata.name:d for d in plain};defs['CompletedAmihudIlliquidity20']=get_factor_definition('CompletedAmihudIlliquidity20')
    variants={variant_name(n,p):(n,p) for n,p in candidates()};inputs=Inputs(rt.data)
    for name,r in old_inventory.items():
        assert file_sha256(r['path'])==r['sha256'],name
        path=source/'columns'/f'{name}.npy';record={**r,'path':str(path),'parent_column_sha256':r['sha256'],'daily_ic':{}}
        if name not in affected:
            os.link(r['path'],path);record['historical_values_reused']=True
            for h in config['model_horizons']:
                old=r['daily_ic'][str(h)];assert file_sha256(old['path'])==old['sha256'];dest=source/'daily_ic'/f'{name}_{h}d.npy';os.link(old['path'],dest)
                record['daily_ic'][str(h)]={**old,'path':str(dest),'reused':True}
        else:
            if name in variants:
                family,params=variants[name];raw=inputs.score(family,params)
            else:raw=defs[name].implementation().calc_batch(rt.data)
            normalized=rank_features(raw[a:b],eligible);np.save(path,normalized)
            old_column=np.load(r['path'],mmap_mode='r')
            changes.append({'name':name,'changed_finite_cells':int((np.isfinite(normalized)&np.isfinite(old_column)&(normalized!=old_column)).sum()),'validity_changes':int((np.isfinite(normalized)!=np.isfinite(old_column)).sum())})
            record.update(sha256=file_sha256(path),historical_values_reused=False,generation_identity_sha256=generation,repair='independent completed-window sums')
            for h in config['model_horizons']:
                labels=np.load(source/f'labels_{h}d.npy',mmap_mode='r');ic=np.full((len(dates),3),np.nan);ic[mask]=daily_ic(normalized[mask],labels[mask],buyable[mask])
                dest=source/'daily_ic'/f'{name}_{h}d.npy';np.save(dest,ic)
                record['daily_ic'][str(h)]={'path':str(dest),'sha256':file_sha256(dest),'reused':False,'train':summary(ic,mask)}
            del raw,normalized,old_column;gc.collect()
        inventory[name]=record;atomic_write_json(source/'records'/f'{name}.json',record)
        atomic_write_json(source/'status.json',{'state':'repairing','completed':len(inventory),'total':len(old_inventory),'name':name})
    (root/'selection').mkdir()
    for h in config['model_horizons']:
        x=json.loads((parent/'selection'/f'selected_{h}d.json').read_text(encoding='utf8'))
        for row in x['selected']:
            if row['name'] in defs and row['name'] in affected:row['metadata']=asdict(defs[row['name']].metadata)
        atomic_write_json(root/'selection'/f'selected_{h}d.json',x)
    atomic_write_json(root/'selection/binding_audit.json',{'state':'pass','scope':'candidate identity only','artifacts':{p.name:file_sha256(p) for p in (root/'selection').glob('selected*json')}})
    require_catalog(root)
    atomic_write_json(source/'inventory.json',inventory)
    for p,sha in dependencies.items():assert file_sha256(p)==sha,p
    atomic_write_json(root/'repair_receipt.json',{'state':'complete','parent':str(parent.resolve()),'unchanged_columns':len(inventory)-len(affected),'rebuilt':changes,'labels_unchanged':True,'old_producer_preserved':True,'seconds':time.monotonic()-started})
    atomic_write_json(source/'status.json',{'state':'complete','completed':len(inventory),'repaired_columns':len(affected),'seconds':time.monotonic()-started})
    print({'state':'complete','repaired':affected,'unchanged':len(inventory)-len(affected)},flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--parent',type=Path,required=True);p.add_argument('--root',type=Path,required=True);a=p.parse_args();run(a.parent,a.root)
