"""Export delivered interval comparisons; never chooses models or parameters."""
import csv,json
from pathlib import Path
import numpy as np
from ai.supervised.periods import observation_splits,interval
from utils.atomic_file import atomic_write_json,file_sha256
from delivery_inputs import admit,bind,verify

root=Path(__file__).resolve().parent
config,bindings,results=admit(root);bind(bindings,Path(__file__))
roles={'oof':'OOF：模型用于选参；静态仅同区间冻结对照','fit':'最终模型样本内：2005—2020','internal_validation':'最终模型样本内：2021','train':'最终模型完整样本内：2005—2021','validation':'外部观察：2022—2026；未参与本轮选择，但历史已被观察'}
rows=[]
for split in observation_splits(config):
    row={'split':split,'interval':interval(config,split),'role':roles[split]}
    for s,d in [('ml','final'),('static','static_baseline')]:
        m=results[s][split]['metrics']
        with np.load(root/d/f'{split}.npz') as z:
            row[s]={'annualized_return':m['annualized_return'],'max_drawdown':m['max_drawdown'],'calmar':m['calmar'],'total_fees':m['total_fees'],'final_nav':float(z['nav'][-1]),'first_decision':str(z['decision_dates'][0]),'last_decision':str(z['decision_dates'][-1]),'last_valuation':str(z['next_dates'][-1])}
            valuation=((z['nav'][1:]-z['cash'])/z['nav'][1:]).tolist()
            row[s].update(mean_post_fill_exposure=float(z['exposure'].mean()),mean_valuation_exposure=sum(valuation)/len(valuation))
        row[s+'_cell']=f"{m['annualized_return']:.2%} / {m['max_drawdown']:.2%} / {m['calmar']:.3f}"
    rows.append(row)

old=Path('artifacts/research/multihorizon_all_periods_preclose_20261107')
old_model=Path('artifacts/research/multihorizon_oof_ga_20261106_windowfix/models/final/model.json')
bind(bindings,old_model)
for strategy in ('ml','static'):
    path=old/f'results/{strategy}_test.json';bind(bindings,path);receipt=json.loads(path.read_text(encoding='utf8'))
    bind(bindings,old/f'results/{strategy}_test.npz',receipt['trace_sha256'])
assert 'LowFloatTurnover20_10' not in json.loads(old_model.read_text(encoding='utf8'))['feature_names']
with np.load(root/'static_baseline/validation.npz') as new,np.load(old/'results/static_test.npz') as previous:
    for name in ('nav','returns','cash','exposure','decision_dates','next_dates'):np.testing.assert_array_equal(new[name],previous[name])
with np.load(root/'final/validation.npz') as new,np.load(old/'results/ml_test.npz') as previous:
    for name in ('decision_dates','next_dates'):np.testing.assert_array_equal(new[name],previous[name])
previous=json.loads((old/'results/ml_test.json').read_text(encoding='utf8'))
verify(bindings)
atomic_write_json(root/'final/delivery_comparison.json',{'state':'exported_observations_only','rows':rows,'previous_repaired_model_same_external_interval':previous,'static_external_all_six_arrays_exact_to_previously_repaired_baseline':True,'all_new_ML_static_date_axes_equal':True,'no_independent_test':True,'no_result_based_selection':True,'old_model_does_not_use_repaired_monthly_factor':True,'inputs':bindings,'script_sha256':file_sha256(__file__)})
with (root/'final/delivery_comparison.csv').open('w',encoding='utf-8-sig',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=['split','interval','role','ml_cell','static_cell']);writer.writeheader();writer.writerows({k:r[k] for k in writer.fieldnames} for r in rows)
print({'intervals':len(rows),'static_external_exact':True},flush=True)
