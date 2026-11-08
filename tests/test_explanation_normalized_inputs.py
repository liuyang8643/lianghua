"""Optional raw-input contract; only one tiny synthetic LightGBM is fitted."""
import json,shutil
from pathlib import Path
import numpy as np
import pytest
import lightgbm as lgb
from ai.supervised.explainability import explain
from utils.atomic_file import file_sha256

@pytest.fixture(scope='module')
def tiny_model(tmp_path_factory):
    directory=tmp_path_factory.mktemp('normalized_explanation_model')
    x=np.asarray([[i/200,i%5] for i in range(200)],float);y=.2*x[:,0]+.1*x[:,1]
    model=lgb.train({'objective':'regression','verbosity':-1,'num_threads':1,'num_leaves':7,'min_data_in_leaf':2,'seed':4,'deterministic':True},lgb.Dataset(x,label=y,feature_name=['factor','horizon'],categorical_feature=['horizon']),num_boost_round=3)
    path=directory/'model.txt';model.save_model(str(path));return path

def case(base,tiny_model,monkeypatch,flag=None,raw=False):
    from scripts import run_research_guarded
    monkeypatch.setattr(run_research_guarded,'require_guarded_job',lambda:None)
    parent=base/'parent';model_dir=parent/'models/final';model_dir.mkdir(parents=True);shutil.copy2(tiny_model,model_dir/'model.txt')
    horizons=[3,7,15,30,60];(model_dir/'model.json').write_text(json.dumps({'feature_names':['VolumeCV'],'horizons':horizons}),encoding='utf8')
    (parent/'selection').mkdir();(parent/'selection/selected_3d.json').write_text(json.dumps({'selected':[{'name':'VolumeCV','metadata':{}}]}),encoding='utf8')
    root=base/'case';stage=root/'features';stage.mkdir(parents=True);(root/'replay').mkdir()
    values=np.arange(1,7,dtype=float).reshape(2,3)/10;norm=stage/'column.npy';np.save(norm,values)
    record={'path':str(norm),'sha256':file_sha256(norm)}
    if raw:
        p=stage/'raw.npy';np.save(p,values+100);record.update(raw_path=str(p),raw_sha256=file_sha256(p))
    (stage/'inventory.json').write_text(json.dumps({'VolumeCV':record}),encoding='utf8');(stage/'complete.json').write_text('{"state":"complete"}',encoding='utf8')
    dates=np.array(['2026-01-05','2026-01-06'],dtype='datetime64[D]');codes=np.array(['000001.SZ','000002.SZ','000003.SZ']);np.save(stage/'dates.npy',dates);np.save(stage/'codes.npy',codes);np.save(stage/'eligible.npy',np.ones((2,3),bool))
    model=lgb.Booster(model_file=str(tiny_model));prediction=np.stack([model.predict(np.column_stack([values.ravel(),np.full(values.size,k)]),num_threads=1).reshape(2,3) for k in range(5)],axis=2)
    np.save(stage/'predictions.npy',prediction);np.save(stage/'head_ranks.npy',np.full((2,3,5),.5));np.save(stage/'scores.npy',np.full((2,3),.5))
    days=[{'decision_date':str(d),'settlement_date':str(d+1),'stocks':[{'code':str(codes[i]),'open_weight':.5,'bought':False,'sold':False,'net_pnl':1.,'nav_contribution_pct':.1}]} for i,d in enumerate(dates)]
    (root/'replay/daily_attribution.json').write_text(json.dumps(days),encoding='utf8')
    task={'parent':str(parent),'model_sha256':file_sha256(model_dir/'model.txt'),'parameters':{'weights':[10,0,0,0,0]},'observation_scope':'synthetic contract fixture'}
    if flag is not None:task['normalized_inputs_only']=flag
    (root/'task.json').write_text(json.dumps(task),encoding='utf8');return root

@pytest.mark.parametrize('flag',[None,False,'true',1])
def test_missing_raw_requires_exact_true(tmp_path,tiny_model,monkeypatch,flag):
    root=case(tmp_path,tiny_model,monkeypatch,flag=flag)
    with pytest.raises(ValueError,match='raw'):explain(root)

def test_normalized_only_and_legacy_predictions_are_identical(tmp_path,tiny_model,monkeypatch):
    legacy=case(tmp_path/'legacy',tiny_model,monkeypatch,raw=True)
    normalized=case(tmp_path/'normalized',tiny_model,monkeypatch,flag=True)
    a=explain(legacy);b=explain(normalized)
    assert a['raw_values_available'] is True and b['raw_values_available'] is False
    assert b['observation_scope']=='synthetic contract fixture'
    for day in b['daily_stocks']:
        for stock in day:
            assert stock['top_factors'][0]['raw_value'] is None and stock['top_factors'][0]['normalized_value']>0
    np.testing.assert_array_equal(np.load(legacy/'explanations/contributions_3d.npy'),np.load(normalized/'explanations/contributions_3d.npy'))

@pytest.mark.parametrize('field',['sha256','raw_sha256'])
def test_provided_column_hash_never_bypassed(tmp_path,tiny_model,monkeypatch,field):
    root=case(tmp_path,tiny_model,monkeypatch,flag=True,raw=True)
    p=root/'features/inventory.json';data=json.loads(p.read_text());data['VolumeCV'][field]='0'*64;p.write_text(json.dumps(data),encoding='utf8')
    with pytest.raises(AssertionError):explain(root)
