"""Export completed model receipts only; no training or selection decisions."""
import csv,json,zipfile
from pathlib import Path
import numpy as np
from utils.atomic_file import atomic_write_json,file_sha256
root=Path(__file__).resolve().parent;rows=[];sources={}
def sample_count(path):
    with zipfile.ZipFile(path) as z,z.open('flat.npy') as f:
        version=np.lib.format.read_magic(f)
        shape,_,_=np.lib.format.read_array_header_1_0(f) if version==(1,0) else np.lib.format.read_array_header_2_0(f)
        return int(shape[0])
for p in sorted((root/'models').glob('*/complete.json'),key=lambda p:(p.parent.name=='final',p.parent.name)):
    done=json.loads(p.read_text(encoding='utf8'));folder=p.parent
    if done.get('state')!='complete':continue
    meta=json.loads((folder/'model.json').read_text(encoding='utf8'));identity=json.loads((folder/'identity.json').read_text(encoding='utf8'))
    sources[str(p.relative_to(root))]=file_sha256(p)
    r={'model':folder.name,'fit_start':done['spec']['fit'][0],'fit_end':done['spec']['fit'][1],
       'last_fit_label_end':identity['last_fit_label_end'],'factors':done['feature_count'],'trees':done['trees'],
       'expanded_task_samples':sample_count(folder/'samples.npz'),'fit_seconds':meta['fit_seconds'],
       'fit_and_prediction_seconds':done['fit_and_prediction_seconds'],'reload_parity':done['reload_parity']}
    for h in ['3','7','15','30','60']:
        d=done.get('validation_by_horizon',{}).get(h)
        if d is None:continue
        r[f'valid_{h}d_mse']=d['weighted_date_mse'];r[f'valid_{h}d_rankic']=d['rank_ic']
    rows.append(r)
if rows:
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with (root/'models/summary.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
    atomic_write_json(root/'model_export_receipt.json',{'state':'exported_completed_models','count':len(rows),'sources':sources,'summary_sha256':file_sha256(root/'models/summary.csv')})
print({'completed_models':len(rows),'trees':{r['model']:r['trees'] for r in rows}})
