"""Export actual GA convergence and coverage; never creates candidates."""
import collections,csv,json
from pathlib import Path
from utils.atomic_file import atomic_write_json,file_sha256
root=Path(__file__).resolve().parent;stage=root/'search';history=[];sources={}
for p in sorted(stage.glob('generation_*.json')):
    r=json.loads(p.read_text(encoding='utf8'));m=r['best']['metrics'];params=r['best']['parameters'];sources[p.name]=file_sha256(p)
    history.append({'generation':r['generation'],'unique_trials':r['unique_trials'],'stale':r['stale'],
        'best_calmar':m['calmar'],'best_cagr':m['annualized_return'],'best_drawdown':m['max_drawdown'],
        **{f'weight_{h}d':v/10 for h,v in zip([3,7,15,30,60],params['weights'])},
        **{k:params[k] for k in ['buy_n','sell_m','hold_thresh','max_replacements']}})
if history:
    with (stage/'history.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(history[0]));w.writeheader();w.writerows(history)
trials=[json.loads(p.read_text(encoding='utf8')) for p in (stage/'trials').glob('*.json')]
params=[r['parameters'] for r in trials];keys=['buy_n','sell_m','hold_thresh','max_replacements']
coverage={'unique_completed_candidates':len(trials),'declared_joint_space':135135,
    'unique_weight_vectors':len({tuple(p['weights']) for p in params}),'declared_weight_space':1001,
    'unique_trading_parameter_vectors':len({tuple(p[k] for k in keys) for p in params}),'declared_trading_space':135,
    'parameter_counts':{k:dict(collections.Counter(str(p[k]) for p in params)) for k in keys},
    'complete_generations':len(history),'winner_sealed':(stage/'winner.json').exists(),
    'scope':'Actual sampled candidates only; not exhaustive coverage or proof of global optimum.'}
atomic_write_json(stage/'coverage.json',coverage)
atomic_write_json(root/'search_export_receipt.json',{'state':'exported_actual_snapshot','generation_sources':sources,'coverage_sha256':file_sha256(stage/'coverage.json'),'history_sha256':file_sha256(stage/'history.csv') if history else None})
print(coverage)
