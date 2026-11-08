"""Read-only admission of sealed study outputs for delivery consumers."""
import json
from pathlib import Path
import numpy as np
from ai.supervised.periods import observation_splits,interval
from utils.atomic_file import file_sha256

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def bind(bindings,path,expected=None):
    path=Path(path).resolve();actual=file_sha256(path)
    if expected is not None:assert actual==expected,str(path)
    key=str(path)
    if key in bindings:assert bindings[key]==actual,key
    bindings[key]=actual;return path

def verify(bindings):
    for path,expected in bindings.items():assert file_sha256(path)==expected,path

def admit(root,*,regression=True):
    root=Path(root).resolve();bindings={};bind(bindings,Path(__file__))
    gate=read(root/'prelaunch_review.json');assert gate['state']=='pass';bind(bindings,root/'prelaunch_review.json')
    bind(bindings,root/'protocol.json',gate['protocol_sha256']);config=read(root/'protocol.json')
    guard=root/'workflow_logs/guard.jsonl';events=[json.loads(line) for line in guard.read_text(encoding='utf8').splitlines()]
    assert events[-1]['event']=='child_exit' and events[-1]['returncode']==0,'Primary financial workflow must finish first'
    bind(bindings,guard)
    for stage in ('selection','fold_models','search','final_model','observations','static_baseline'):
        path=root/'stages'/stage/'complete.json';receipt=read(path);assert receipt['state']=='complete';bind(bindings,path)
        bind(bindings,root/'stages'/stage/'inputs.json',receipt['inputs_sha256'])
        output=Path(receipt['output_receipt']).resolve();assert output.is_relative_to(root);bind(bindings,output,receipt['output_sha256'])
    active=read(root/'active_data.json');bind(bindings,root/'active_data.json');bind(bindings,active['manifest_path'],active['manifest_sha256'])
    assert read(active['manifest_path'])['runtime_sha256']==active['runtime_sha256']
    assert Path(read(active['manifest_path'])['runtime_path']).resolve()==Path(active['runtime_path']).resolve()
    manifest=read(root/'features/data_manifest.json');bind(bindings,root/'features/data_manifest.json')
    for name in ('dates.npy','codes.npy','eligible.npy'):bind(bindings,root/'features'/name,manifest[name])
    dates=np.load(root/'features/dates.npy').astype('datetime64[D]');splits=observation_splits(config)
    assert splits==['oof','fit','internal_validation','train','validation'] and config.get('test') is None
    final=read(root/'models/final/complete.json');bind(bindings,root/'models/final/model.txt',final['model_sha256'])
    bind(bindings,root/'models/final/model.json');bind(bindings,root/'models/final/identity.json',final['identity_sha256'])
    identity=read(root/'models/final/identity.json');meta=read(root/'models/final/model.json')
    assert identity['protocol']==file_sha256(root/'protocol.json') and identity['data_manifest']==file_sha256(root/'features/data_manifest.json')
    assert meta['parameters']==identity['parameters']
    bind(bindings,root/'fold_selection/final/complete.json',identity['selection'])
    assert meta['feature_names']==read(root/'fold_selection/final/complete.json')['features']
    assert meta['horizons']==config['model_horizons'] and meta['best_iteration']==final['trees']==identity['fixed_rounds']
    trees=[]
    for year in config['fold_years']:
        p=root/f'models/fold{year}/complete.json';bind(bindings,p);d=read(p);assert d['state']=='complete';trees.append(d['trees'])
    assert final['trees']==int(np.median(trees))
    observations={};calendar={}
    for strategy,folder in [('ml','final'),('static','static_baseline')]:
        done=read(root/folder/'complete.json');assert done['state']=='complete';results=done['results']
        assert len(results)==len(splits) and {x['split'] for x in results}==set(splits)
        if strategy=='ml':assert done['winner_sha256']==file_sha256(root/'search/winner.json')
        observations[strategy]={row['split']:row for row in results}
        for split in splits:
            path=root/folder/f'{split}.npz';bind(bindings,path,observations[strategy][split]['trace_sha256'])
            start,end=interval(config,split);expected=dates[(dates>=np.datetime64(start))&(dates<=np.datetime64(end))]
            with np.load(path) as z:
                d=z['decision_dates'].astype('datetime64[D]');n=z['next_dates'].astype('datetime64[D]')
                np.testing.assert_array_equal(d,expected[:-1]);np.testing.assert_array_equal(n,expected[1:])
                assert len(z['nav'])==len(d)+1 and len(d)>0 and np.isfinite(z['nav']).all() and (z['nav']>0).all()
                np.testing.assert_allclose(z['returns'],z['nav'][1:]/z['nav'][:-1]-1,rtol=1e-10,atol=1e-12)
                if strategy=='ml':calendar[split]=(d.copy(),n.copy())
                else:
                    np.testing.assert_array_equal(d,calendar[split][0]);np.testing.assert_array_equal(n,calendar[split][1])
    if regression:
        path=root/'regression/complete.json';done=read(path);assert done['state']=='complete';bind(bindings,path);bind(bindings,root/'regression/inputs.json')
        assert done['source_sha256']==read(root/'regression/inputs.json')['source_sha256']
        expected={(s,n) for s in ('ml','static') for n in splits};records=done['results'];assert len(records)==10 and {(r['strategy'],r['split']) for r in records}==expected
        for r in records:
            assert r['all_six_trace_arrays_exact'] is True and not any(r['trade_audit'].values()),r
            assert r['prior_trace_sha256']==observations[r['strategy']][r['split']]['trace_sha256']
            ledger=r['ledger'];np.testing.assert_allclose(ledger['closed_pnl']+ledger['open_pnl'],ledger['account_pnl'],rtol=1e-10,atol=1e-5)
            if r['strategy']=='ml':
                p=root/'regression'/f"ml_{r['split']}"/'trade_audit.json';a=read(p);assert a['state']=='pass' and a['counts']==r['trade_audit'];bind(bindings,p)
    return config,bindings,observations

def validation_2026(root,bindings):
    root=Path(root);path=root/'regression/ml_validation/daily_attribution.json';bind(bindings,path);all_days=read(path)
    trace=root/'final/validation.npz';bind(bindings,trace)
    with np.load(trace) as z:
        decision=z['decision_dates'].astype('datetime64[D]').astype(str);settlement=z['next_dates'].astype('datetime64[D]').astype(str)
        assert len(all_days)==len(decision)
        np.testing.assert_array_equal([d['decision_date'] for d in all_days],decision);np.testing.assert_array_equal([d['settlement_date'] for d in all_days],settlement)
        np.testing.assert_allclose([d['pretrade_nav'] for d in all_days],z['nav'][:-1],rtol=1e-12,atol=1e-6)
        np.testing.assert_array_equal([d['ending_nav'] for d in all_days],z['nav'][1:])
        np.testing.assert_allclose([d['return_pct'] for d in all_days],100*z['returns'],rtol=1e-12,atol=1e-10)
        indices=np.flatnonzero(np.char.startswith(decision,'2026-'));assert len(indices)>0 and np.all(np.diff(indices)==1)
        days=[all_days[i] for i in indices]
        np.testing.assert_array_equal([d['decision_date'] for d in days],decision[indices])
    return days
