"""Explain the frozen final model on the actual continuous-account 2026 slice."""
import csv,gc,inspect,json,os,time
from pathlib import Path
import numpy as np
import lightgbm as lgb
from ai.supervised.explainability import explain
from ai.supervised.multihorizon_search import load_forecasts
from utils.atomic_file import atomic_write_json,file_sha256
from scripts.run_research_guarded import require_guarded_job
from delivery_inputs import admit,bind,verify,validation_2026,read

require_guarded_job();root=Path(__file__).resolve().parent;start=time.monotonic()
config,bindings,_=admit(root)
adapter=read(root/'review_explanation_adapter_implemented.json');assert adapter['state']=='pass'
bind(bindings,root/'review_explanation_adapter_implemented.json');bind(bindings,'ai/supervised/explainability.py',adapter['source_sha256'])
model_dir=root/'models/final';done=json.loads((model_dir/'complete.json').read_text(encoding='utf8'));assert done['state']=='complete'
for filename,key in [('model.txt','model_sha256'),('predictions.npy','predictions_sha256'),('rows.npy','rows_sha256')]:assert file_sha256(model_dir/filename)==done[key]
winner=json.loads((root/'search/winner.json').read_text(encoding='utf8'));params=winner['winner']['parameters']
meta=json.loads((model_dir/'model.json').read_text(encoding='utf8'));features=root/'features';dates=np.load(features/'dates.npy');codes=np.load(features/'codes.npy')
assert meta['horizons']==config['model_horizons']==[3,7,15,30,60]
assert len(params['weights'])==5 and sum(params['weights'])==10
assert file_sha256(root/'search/winner.json')==json.loads((root/'stages/search/complete.json').read_text(encoding='utf8'))['output_sha256']
assert file_sha256(model_dir/'complete.json')==json.loads((root/'stages/final_model/complete.json').read_text(encoding='utf8'))['output_sha256']
np.testing.assert_array_equal(np.load(model_dir/'rows.npy'),np.arange(len(dates)))
days=validation_2026(root,bindings)
lookup={str(d):i for i,d in enumerate(dates)};rows=np.asarray([lookup[d['decision_date']] for d in days]);assert np.all(np.diff(rows)>0)
out=root/'explain_2026';out.mkdir(exist_ok=False);stage=out/'features';stage.mkdir();(out/'replay').mkdir()
inputs=[features/'inventory.json',Path('ai/supervised/multihorizon_search.py'),Path(__file__)]
for path in inputs:bind(bindings,path)
atomic_write_json(out/'inputs.json',bindings)
for name in ('dates.npy','codes.npy','eligible.npy'):os.link(features/name,stage/name)
os.link(model_dir/'predictions.npy',stage/'predictions.npy')
inventory=json.loads((features/'inventory.json').read_text(encoding='utf8'));atomic_write_json(stage/'inventory.json',{n:inventory[n] for n in meta['feature_names']})
rt,cube,provenance=load_forecasts(root,'validation');np.testing.assert_array_equal(np.asarray(rt.stock_codes),codes.astype(str))
rr=np.asarray([rt.index_of(d['decision_date']) for d in days]);weights=np.asarray(params['weights'],float)/10
rank_path=stage/'head_ranks.npy';score_path=stage/'scores.npy'
ranks=np.lib.format.open_memmap(rank_path,mode='w+',dtype=np.float32,shape=(len(dates),len(codes),5));ranks[:]=np.nan;ranks[rows]=cube[rr]
np.testing.assert_array_equal(ranks[rows].astype(float),cube[rr])
scores=np.lib.format.open_memmap(score_path,mode='w+',dtype=np.float64,shape=(len(dates),len(codes)));scores[:]=np.nan;scores[rows]=np.sum(cube[rr]*weights,axis=2)
ranks.flush();scores.flush();ranks._mmap.close();scores._mmap.close();del cube,rt,ranks,scores;gc.collect()
atomic_write_json(stage/'complete.json',{'state':'complete','scope':'Only actual2026decision rows have derived ranks/scores in full feature-axis containers. Other rows are deliberately missing. No factor recalculation or new model fit.','normalized_inputs_only':True,'derived_rows':rows.tolist(),'source_models':provenance,'head_ranks_sha256':file_sha256(rank_path),'scores_sha256':file_sha256(score_path)})
atomic_write_json(out/'replay/daily_attribution.json',days)
atomic_write_json(out/'task.json',{'parent':str(root),'model_sha256':done['model_sha256'],'parameters':params,'normalized_inputs_only':True,'observation_scope':'2026 decision-date slice of the continuous2022-start validation account; not a fresh2026cash account and not a new blind test; unavailable raw factor values remain null.'})
result=explain(out)
model=lgb.Booster(model_file=str(model_dir/'model.txt'));gain=model.feature_importance(importance_type='gain');splits=model.feature_importance(importance_type='split');names=[*meta['feature_names'],'horizon']
with (out/'explanations/model_gain_importance.csv').open('w',encoding='utf-8-sig',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=['feature','split_count','gain','gain_fraction']);writer.writeheader();writer.writerows({'feature':names[k],'split_count':int(splits[k]),'gain':float(gain[k]),'gain_fraction':float(gain[k]/gain.sum()) if gain.sum()>0 else None} for k in np.argsort(-gain))
verify(bindings)
atomic_write_json(out/'complete.json',{'state':'complete','model_sha256':done['model_sha256'],'winner_sha256':file_sha256(root/'search/winner.json'),'days':len(days),'decision_start':days[0]['decision_date'],'decision_end':days[-1]['decision_date'],'last_valuation':days[-1]['settlement_date'],'explanation_receipt_sha256':file_sha256(out/'explanations/complete.json'),'gain_importance_sha256':file_sha256(out/'explanations/model_gain_importance.csv'),'training_calls':0,'strategy_selection_calls':0,'actual_account_replays':0,'raw_factor_values_available':False,'seconds':time.monotonic()-start,'scope':'Native TreeSHAP of the new frozen model; raw-score additivity and cached prediction parity checked. Not attribution of nonlinear final ranks or causal investment returns.'})
print({'state':'complete','days':len(days),'seconds':time.monotonic()-start},flush=True)
