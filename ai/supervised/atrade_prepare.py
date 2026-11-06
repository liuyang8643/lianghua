"""Prepare only the missing 60-day model, reusing the frozen causal features."""
from __future__ import annotations
import argparse,json,os,shutil
from pathlib import Path
import numpy as np
from offline_data import load_runtime_slice
from env.forward_returns import open_forward_returns
from .feature_data import rank_features
from utils.atomic_file import atomic_write_json,file_sha256


def prepare(root:Path,parent:Path):
    from scripts.run_research_guarded import require_guarded_job
    require_guarded_job()
    root.mkdir(parents=True,exist_ok=False)
    protocol=json.loads((parent/'protocol.json').read_text(encoding='utf8'))
    protocol.update(model_horizons=[60],selection='same frozen262 feature names; no new selection',
        study_parent=str(parent.resolve()),portfolio_methods=['atrade_three_horizon'],
        selection_split='none',test_policy='fixed rules; all external periods observation only')
    atomic_write_json(root/'protocol.json',protocol)
    shutil.copy2(parent/'active_data.json',root/'active_data.json')
    source=parent/'features';stage=root/'features';stage.mkdir();(root/'selection').mkdir()
    for name in ['dates.npy','codes.npy','eligible.npy','label_eligible.npy']:
        os.link(source/name,stage/name)
    for name in ['inventory.json','identity.json']:
        shutil.copy2(source/name,stage/name)
    shutil.copy2(parent/'selection/selected_5d.json',root/'selection/selected_60d.json')
    active=json.loads((root/'active_data.json').read_text())
    assert file_sha256(active['runtime_path'])==active['runtime_sha256']
    rt=load_runtime_slice(active['runtime_path'],*['2005-01-01',protocol['test'][1]],
                          preload_rows=100001,cache_dir=Path(protocol['feature_runtime_cache']))
    a,b=rt.decision_start,rt.decision_stop
    np.testing.assert_array_equal(rt.trade_dates[a:b],np.load(stage/'dates.npy'))
    np.testing.assert_array_equal(rt.stock_codes,np.load(stage/'codes.npy').astype(str))
    raw=open_forward_returns(rt.data,60)[a:b]
    eligible=np.load(stage/'label_eligible.npy',mmap_mode='r')
    ranked=rank_features(raw,eligible)
    assert np.isnan(ranked[~eligible]).all()
    np.save(stage/'labels_60d.npy',raw);np.save(stage/'label_rank_60d.npy',ranked)
    atomic_write_json(stage/'data_manifest.json',{p.name:file_sha256(p) for p in sorted(stage.glob('*.npy'))})
    atomic_write_json(stage/'status.json',{'state':'complete','new_label_horizon':60,'features_reselected':False})
    atomic_write_json(root/'model_sources.json',{'5':str(parent.resolve()),'20':str(parent.resolve()),'60':str(root.resolve())})
    atomic_write_json(root/'preparation.json',{'state':'complete','runtime_sha256':active['runtime_sha256'],
        'parent_manifest_sha256':file_sha256(source/'data_manifest.json'),
        'reused_models':{str(h):file_sha256(parent/f'models/n{h}/complete.json') for h in [5,20]},
        'new_model_horizon':60,'label':'economic open[T+60]/open[T]-1 ranked within T-buyable pool',
        'upstream_commit':'03bda3a1c8aebb5f0cde83aae341a8fca615afae',
        'adaptation':'ATrade portfolio rules on local fixed open-time models, not full upstream rolling-training replication'})
    print('Prepared N60 labels and frozen feature references',flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--parent',type=Path,required=True)
    a=ap.parse_args();prepare(a.root,a.parent)
