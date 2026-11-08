"""Append-date inference inputs for a frozen shared-horizon model, without labels."""
import argparse,gc,json,os,time
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from offline_data import load_runtime_slice
from offline_data.supplemental_fields import implemented_dividends
from env.legality import investment_universe_mask
from .feature_data import materialize_factor_values,rank_features
from .multihorizon_model import predict
from utils.atomic_file import atomic_write_json,file_sha256


def industry_vintages(old,new,cutoff):
    """Keep frozen membership history; admit new records only after availability."""
    keys=['stock_code','start_date','industry_code']
    known=set(old[keys].itertuples(index=False,name=None))
    appended=new[[x not in known for x in new[keys].itertuples(index=False,name=None)]].copy()
    if len(appended):
        appended['start_date']=pd.concat([pd.to_datetime(appended['start_date']),pd.to_datetime(appended['update_date']),pd.Series(pd.Timestamp(cutoff),index=appended.index)],axis=1).max(axis=1)
    return pd.concat([old,appended],ignore_index=True)


def source_inputs(root,parent):
    protocol=json.loads((parent/'protocol.json').read_text(encoding='utf8'))
    dividend_old=parent/'dividend_source/events.parquet'
    if not dividend_old.exists():dividend_old=Path(protocol['feature_parent'])/'dividend_source/events.parquet'
    raw=pd.concat([pd.read_parquet(dividend_old),pd.read_parquet(root/'dividend_tail/events.parquet')],ignore_index=True)
    # The current source was independently checked to contain no duplicate
    # event keys. Refuse future ambiguous revisions instead of rewriting a
    # historical signal through the legacy all-table conflict branch.
    usable=raw[raw['ASSIGN_PROGRESS'].eq('实施分配')]
    if usable.duplicated(['SECUCODE','EX_DIVIDEND_DATE','REPORT_DATE']).any():
        raise ValueError('unadmitted dividend revisions require a versioned event audit')
    dividends,_=implemented_dividends(raw)
    old=pd.read_parquet('data/industry_history/sw_industry_history.parquet')
    fresh=pd.read_parquet(root/'industry_source/sw_industry_history.parquet')
    industry=industry_vintages(old,fresh,'2026-10-30')
    return industry,dividends,[dividend_old,root/'dividend_tail/events.parquet',Path('data/industry_history/sw_industry_history.parquet'),root/'industry_source/sw_industry_history.parquet']


def run(root):
    from scripts.run_research_guarded import require_guarded_job
    require_guarded_job();begun=time.monotonic();task=json.loads((root/'task.json').read_text());parent=Path(task['parent'])
    active=json.loads((root/'active_data.json').read_text());assert active['state']=='admitted_market_readback'
    assert file_sha256(active['manifest_path'])==active['manifest_sha256']
    model_path=parent/'models/final/model.txt';assert file_sha256(model_path)==task['model_sha256']
    meta=json.loads((parent/'models/final/model.json').read_text());names=meta['feature_names'];horizons=meta['horizons']
    records={x['name']:x for x in json.loads((parent/'selection/selected_3d.json').read_text(encoding='utf8'))['selected']}
    bound={x['name'] for x in json.loads((parent/'bound_preflight.json').read_text())['results']}
    stage=root/'features';stage.mkdir(exist_ok=False);(stage/'columns').mkdir();(stage/'raw_columns').mkdir()
    rt=load_runtime_slice(active['runtime_path'],task['requested_start'],task['requested_end'],preload_rows=100001,cache_dir=root/'runtime_cache',reuse_cache_dir=task.get('runtime_reuse_cache_dir'))
    assert rt.manifest.source_sha256==active['runtime_sha256'];a,b=rt.decision_start,rt.decision_stop;dates=rt.trade_dates[a:b]
    eligible=investment_universe_mask(rt.field('open')[a:b],rt.field('listing_age')[a:b],rt.field('st_mask')[a:b],rt.field('delisted_mask')[a:b])
    np.save(stage/'dates.npy',dates);np.save(stage/'codes.npy',np.asarray(rt.stock_codes));np.save(stage/'eligible.npy',eligible)
    industry,dividends,source_paths=source_inputs(root,parent)
    industry.to_parquet(stage/'industry_vintages.parquet',index=False);dividends.to_parquet(stage/'dividend_events.parquet',index=False)
    sources={str(p):file_sha256(p) for d in ['factor','factor_db','offline_data','utils'] for p in Path(d).rglob('*.py')}
    for p in [Path(__file__),Path('ai/supervised/feature_data.py'),Path('ai/supervised/local_preflight.py'),Path('factor/library/qlib_catalog.json'),*source_paths]:sources[str(p)]=file_sha256(p)
    identity={'model_sha256':task['model_sha256'],'runtime_sha256':active['runtime_sha256'],'names':names,'horizons':horizons,'sources':sources,'data_manifest':rt.manifest.as_dict(),'feature_range':[str(dates[0]),str(dates[-1])],'labels_used':False}
    atomic_write_json(stage/'identity.json',identity)
    old_dates=np.load(parent/'features/dates.npy');old_codes=np.load(parent/'features/codes.npy').astype(str);np.testing.assert_array_equal(np.asarray(rt.stock_codes[:len(old_codes)]),old_codes)
    old_inventory=json.loads((parent/'features/inventory.json').read_text(encoding='utf8'));old_rows=np.searchsorted(old_dates,dates);valid=old_rows<len(old_dates)
    valid[valid]&=old_dates[old_rows[valid]]==dates[valid];inventory={};parity=[]
    if task.get('expected_history_equal_through'):valid&=dates<=np.datetime64(task['expected_history_equal_through'])
    reuse_names=set(task.get('unchanged_feature_names',[]))
    if reuse_names:
        previous=Path(task['feature_reuse_root'])/'features';prior=json.loads((previous/'inventory.json').read_text())
        np.testing.assert_array_equal(np.load(previous/'dates.npy'),dates);np.testing.assert_array_equal(np.load(previous/'codes.npy').astype(str),rt.stock_codes);np.testing.assert_array_equal(np.load(previous/'eligible.npy'),eligible)
        for name in sorted(reuse_names):
            assert name in names and 'preClose' not in records[name].get('metadata',{}).get('required_fields',[])
            source=prior[name]
            for filename,key,sha in [('columns','path','sha256'),('raw_columns','raw_path','raw_sha256')]:
                assert file_sha256(source[key])==source[sha];os.link(source[key],stage/filename/f'{name}.npy')
            inventory[name]={'path':str(stage/'columns'/f'{name}.npy'),'sha256':source['sha256'],'raw_path':str(stage/'raw_columns'/f'{name}.npy'),'raw_sha256':source['raw_sha256'],'unchanged_row_local_input_reuse':str(previous)}
    def save(name,raw):
        raw_values=np.asarray(raw)[a:b];np.save(stage/'raw_columns'/f'{name}.npy',raw_values)
        values=rank_features(raw_values,eligible,records[name].get('metadata',{}).get('score_semantics')=='binary')
        path=stage/'columns'/f'{name}.npy';np.save(path,values)
        reference=np.load(old_inventory[name]['path'],mmap_mode='r')[old_rows[valid]];new=values[valid,:len(old_codes)]
        finite=np.isfinite(reference)&np.isfinite(new);mask_changes=int(np.count_nonzero(np.isfinite(reference)!=np.isfinite(new)))
        diff=np.abs(np.where(finite,new-reference,0));changed=int(np.count_nonzero(diff>0))
        parity.append({'name':name,'finite_mask_changes':mask_changes,'changed_values':changed,'max_abs_difference':float(diff.max(initial=0))})
        inventory[name]={'path':str(path),'sha256':file_sha256(path),'raw_path':str(stage/'raw_columns'/f'{name}.npy'),'raw_sha256':file_sha256(stage/'raw_columns'/f'{name}.npy')}
        atomic_write_json(stage/'status.json',{'state':'computing','completed':len(inventory),'total':len(names),'last':name,'seconds':time.monotonic()-begun})
        if len(inventory)%10==0:print({'features':len(inventory),'total':len(names),'last':name},flush=True)
        gc.collect()
    materialize_factor_values(rt,set(names)-reuse_names,save,bound_names=bound,industry_history=industry,dividend_events=dividends)
    atomic_write_json(stage/'inventory.json',inventory);atomic_write_json(stage/'historical_parity.json',parity)
    mismatches=[x for x in parity if x['finite_mask_changes'] or x['changed_values']]
    if mismatches:raise ValueError('historical factor prefix changed; inspect historical_parity.json before forecasting')
    columns=[np.load(inventory[n]['path'],mmap_mode='r') for n in names];model=lgb.Booster(model_file=str(model_path))
    predictions=predict(model,columns,eligible,np.arange(len(dates)),horizons);np.save(stage/'predictions.npy',predictions)
    previous=np.load(parent/'models/final/predictions.npy',mmap_mode='r')[old_rows[valid]]
    np.testing.assert_allclose(predictions[valid,:len(old_codes)],previous,rtol=1e-12,atol=1e-12,equal_nan=True)
    pool=eligible&np.isfinite(predictions).all(axis=2);ranks=np.stack([rank_features(predictions[:,:,k],pool) for k in range(len(horizons))],axis=2)
    scores=np.sum(ranks*(np.asarray(task['parameters']['weights'])/10),axis=2);np.save(stage/'head_ranks.npy',ranks);np.save(stage/'scores.npy',scores)
    assert file_sha256(model_path)==task['model_sha256']
    for p,sha in sources.items():assert file_sha256(p)==sha,p
    atomic_write_json(stage/'complete.json',{'state':'complete','feature_count':len(names),'dates':[str(dates[0]),str(dates[-1])],'historical_factor_prefix_exact':True if valid.any() else None,'historical_prediction_prefix_equal':True if valid.any() else None,'prefix_checked_dates':int(valid.sum()),'prefix_check_end':task.get('expected_history_equal_through',str(old_dates[-1])),'unchanged_row_local_factors_reused':sorted(reuse_names),'seconds':time.monotonic()-begun,'sha256':{p.name:file_sha256(p) for p in stage.glob('*.npy')},'identity_sha256':file_sha256(stage/'identity.json')})
    print({'state':'features_and_predictions_complete','seconds':time.monotonic()-begun},flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);run(p.parse_args().root)
