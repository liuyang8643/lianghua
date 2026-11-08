"""Current-market-buyable ranking targets over immutable causal features."""
from __future__ import annotations
import argparse,json,os,time
from pathlib import Path
if __name__=='__main__':
    for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[k]='1'
import numpy as np
from offline_data import load_runtime_slice
from env.legality import investment_universe_mask,evaluate_buy_legality_mask
from env.forward_returns import label_split_mask
from .feature_data import rank_features
from .ic import daily_ic,summary
from factor_db import require_allowed_factor_names
from utils.atomic_file import atomic_write_json,file_sha256


def market_buyable_mask(runtime):
    """Identical price inputs to build_day_market; no HLC/volume/future flags."""
    opening=runtime.field('open');base=investment_universe_mask(opening,runtime.field('listing_age'),runtime.field('st_mask'),runtime.field('delisted_mask'))
    issue=np.where(runtime.trade_dates[:,None]==runtime.field('issue_date')[None,:],runtime.field('issue_price')[None,:],np.nan)
    legal=evaluate_buy_legality_mask(decision_date=runtime.trade_dates,stock_codes=runtime.stock_codes,
        listing_age=runtime.field('listing_age'),open_prices=opening,preclose_prices=runtime.field('preClose'),
        issue_prices=issue,st_mask=runtime.field('st_mask'),delisted_mask=runtime.field('delisted_mask'))
    return base,base&legal


def run(root):
    from scripts.run_research_guarded import require_guarded_job
    require_guarded_job();root=Path(root);started=time.monotonic();protocol=json.loads((root/'protocol.json').read_text(encoding='utf8'))
    parent=Path(protocol['feature_parent']);old=parent/'features';stage=root/'features';stage.mkdir(exist_ok=False)
    assert json.loads((old/'status.json').read_text())['state']=='complete'
    active=json.loads((root/'active_data.json').read_text());old_identity=json.loads((old/'identity.json').read_text(encoding='utf8'))
    assert old_identity['runtime']==active['runtime_sha256'] and file_sha256(active['runtime_path'])==active['runtime_sha256']
    assert file_sha256(active['manifest_path'])==active['manifest_sha256']
    rt=load_runtime_slice(active['runtime_path'],protocol['train'][0],protocol['test'][1],preload_rows=100001,cache_dir=Path(protocol['feature_runtime_cache']))
    assert rt.manifest.source_sha256==active['runtime_sha256'];a,b=rt.decision_start,rt.decision_stop;dates=rt.trade_dates[a:b]
    before=json.loads((old/'data_manifest.json').read_text())
    for name,sha in before.items():assert file_sha256(old/name)==sha,name
    np.testing.assert_array_equal(dates,np.load(old/'dates.npy'));np.testing.assert_array_equal(rt.stock_codes,np.load(old/'codes.npy').astype(str))
    base,buyable=market_buyable_mask(rt);base=base[a:b];buyable=buyable[a:b]
    np.testing.assert_array_equal(base,np.load(old/'eligible.npy'))
    # Original broad-PIT feature transformation remains causal and unchanged.
    # It also supplies scores for held stocks that cannot currently be bought.
    for name in ('dates.npy','codes.npy','eligible.npy',*[f'labels_{h}d.npy' for h in protocol['model_horizons']]):
        os.link(old/name,stage/name)
    np.save(stage/'label_eligible.npy',buyable,allow_pickle=False)
    labels={}
    for h in protocol['model_horizons']:
        raw=np.load(stage/f'labels_{h}d.npy',mmap_mode='r');ranked=rank_features(raw,buyable)
        np.save(stage/f'label_rank_{h}d.npy',ranked,allow_pickle=False);labels[h]=raw
        assert np.isnan(ranked[~buyable]).all()
    del ranked,rt
    inputs=json.loads((old/'inventory.json').read_text(encoding='utf8'));require_allowed_factor_names(inputs)
    for name,r in inputs.items():assert file_sha256(r['path'])==r['sha256'],name
    selected={r['name'] for r in json.loads((root/'selection/selected_1d.json').read_text(encoding='utf8'))['selected']}
    assert set(inputs)==selected
    for h in protocol['model_horizons']:
        assert {r['name'] for r in json.loads((root/'selection'/f'selected_{h}d.json').read_text(encoding='utf8'))['selected']}==selected
    paths=[Path(__file__),Path('env/legality.py'),Path('ai/supervised/ic.py'),Path('ai/supervised/feature_data.py')]
    identity={'runtime':active['runtime_sha256'],'active_manifest':active['manifest_sha256'],
        'protocol_sha256':file_sha256(root/'protocol.json'),'feature_reuse':'entire immutable columns, full SHA verified',
        'parent_feature_identity_sha256':file_sha256(old/'identity.json'),'parent_inventory_sha256':file_sha256(old/'inventory.json'),
        'label_pool':'base PIT universe intersect native current-day market buy legality',
        'prediction_pool':'base PIT, including held stocks currently not buyable',
        'no_future_sale_filter':True,'sources':{str(p):file_sha256(p) for p in paths}}
    atomic_write_json(stage/'identity.json',identity);(stage/'daily_ic').mkdir();(stage/'records').mkdir();inventory={}
    for name,r in inputs.items():
        values=np.load(r['path'],mmap_mode='r');ic={}
        for h in protocol['model_horizons']:
            d=daily_ic(values,labels[h],buyable,protocol.get('minimum_pairs',30));path=stage/'daily_ic'/f'{name}_{h}d.npy';np.save(path,d)
            ic[str(h)]={'path':str(path),'sha256':file_sha256(path),'train':summary(d,label_split_mask(dates,*protocol['train'],h)),
                'scope':'current-buyable pool; Pearson uses normalized model input, Spearman is invariant'}
        record={**r,'daily_ic':ic,'historical_values_reused':True,'reused_input_sha256':r['sha256'],
            'reuse_scope':'entire verified column from same repaired runtime; not old pre-repair scores',
            'label_generation_identity_sha256':file_sha256(stage/'identity.json')}
        inventory[name]=record;atomic_write_json(stage/'records'/f'{name}.json',record)
        status={'state':'computing','completed':len(inventory),'total':len(inputs),'name':name,'seconds':time.monotonic()-started}
        atomic_write_json(stage/'status.json',status);print(status,flush=True)
    atomic_write_json(stage/'inventory.json',inventory)
    atomic_write_json(stage/'data_manifest.json',{p.name:file_sha256(p) for p in sorted(stage.glob('*.npy'))})
    counts={}
    for split in ('fit','internal_validation','train','validation','test'):
        mask=(dates>=np.datetime64(protocol[split][0]))&(dates<=np.datetime64(protocol[split][1]))
        counts[split]={'base_stock_days':int(base[mask].sum()),'buyable_stock_days':int(buyable[mask].sum())}
    atomic_write_json(stage/'label_audit.json',{'state':'complete','counts':counts,'label_definition':'N-day gross economic open return percentile among T-buyable stocks with observed outcomes',
        'no_future_sale_filter':True,'old_columns_unchanged':True,'new_rank_label_files':[f'label_rank_{h}d.npy' for h in labels]})
    for p in paths:assert file_sha256(p)==identity['sources'][str(p)],str(p)
    atomic_write_json(stage/'status.json',{'state':'complete','completed':len(inputs),'elapsed_seconds':time.monotonic()-started})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);run(p.parse_args().root)
