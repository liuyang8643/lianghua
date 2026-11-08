import json
import numpy as np
from pathlib import Path
from test_supervised_ic_driver import study
from offline_data import load_runtime_slice
from factor.library.qlib_open import CATALOG,ExpressionPanel,price_fields
from ai.supervised import feature_data
from utils.atomic_file import file_sha256


def test_selected_features_recompute_repaired_history_not_stale_scores(study):
    root,source,protocol=study
    with np.load(source,allow_pickle=False) as z:data={k:z[k] for k in z.files}
    for name in ['eps','bps','roe','profit_yoy','revenue_yoy','operating_cf_ps','gross_margin']:data['pit_'+name]=np.ones(data['open'].shape)
    extended=root/'extended.npz';np.savez_compressed(extended,**data)
    rt=load_runtime_slice(extended,str(data['trade_dates'][0]),str(data['trade_dates'][-1]),preload_rows=100001)
    name='Alpha360_OPEN1';expression=next(r['expression'] for r in CATALOG['features'] if r['name']==name)
    raw=ExpressionPanel(price_fields(rt.data)).evaluate(expression)
    legacy=Path(protocol['legacy_ic_root']);dates=rt.trade_dates;cut=50
    np.save(legacy/'dates.npy',dates[:cut]);np.save(legacy/'eligible.npy',np.ones((cut,40),bool))
    (legacy/'protocol.json').write_text(json.dumps({'runtime':str(source),'runtime_sha256':file_sha256(source)}),encoding='utf-8')
    score=root/'historical_score.npy';np.save(score,np.full_like(raw[:cut],999.))
    monthly=root/'monthly';monthly.mkdir();np.savez(monthly/'monthly.npz',names=np.array([],dtype=str))
    (monthly/'manifest.json').write_text(json.dumps({'state':'complete','runtime_sha256':file_sha256(extended),'output_sha256':file_sha256(monthly/'monthly.npz'),
        'implementation_sha256':{p:file_sha256(p) for p in ('offline_data/sources/sina_research_history.py','factor/library/bilibili_money_flow.py','offline_data/volume_units.py','ai/supervised/monthly_data.py')}}))
    protocol['monthly_signal_path']=str(monthly/'monthly.npz');protocol['model_horizons']=[1,5,10,20,30]
    protocol['test']=[str(dates[cut]),str(dates[-1])];(root/'protocol.json').write_text(json.dumps(protocol),encoding='utf-8')
    manifest=root/'extended_manifest.json';manifest.write_text(json.dumps({'runtime_sha256':file_sha256(extended)}))
    (root/'active_data.json').write_text(json.dumps({'runtime_path':str(extended),'runtime_sha256':file_sha256(extended),
        'state':'admitted_market_readback','manifest_path':str(manifest),'manifest_sha256':file_sha256(manifest)}),encoding='utf-8')
    (root/'dividend_source').mkdir();(root/'dividend_source/events.parquet').write_bytes(b'unused source fixture')
    (root/'bound_preflight.json').write_text('{"results":[]}',encoding='utf-8')
    (root/'selection').mkdir()
    record={'name':name,'scores_path':str(score),'scores_sha256':file_sha256(score),'metadata':{}}
    for h in protocol['model_horizons']:(root/'selection'/f'selected_{h}d.json').write_text(json.dumps({'selected':[record]}),encoding='utf-8')
    (root/'selection/binding_audit.json').write_text(json.dumps({'state':'pass','artifacts':{f'selected_{h}d.json':file_sha256(root/'selection'/f'selected_{h}d.json') for h in protocol['model_horizons']}}))
    feature_data.run(root)
    result=np.load(root/'features/columns'/f'{name}.npy')
    expected=feature_data.rank_features(raw,np.ones(raw.shape,bool))
    np.testing.assert_allclose(result,expected,equal_nan=True)
    assert np.isnan(np.load(root/'features/labels_10d.npy')[-10:]).all()
    assert np.isnan(np.load(root/'features/labels_30d.npy')[-30:]).all()
    feature_data.run(root)
