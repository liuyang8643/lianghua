"""Training-side availability diagnostics; no external outcome inspection."""
import csv,json
from pathlib import Path
import numpy as np
from env.forward_returns import label_split_mask
from ai.supervised.multihorizon_select import fold_specs
from utils.atomic_file import atomic_write_json,file_sha256
root=Path(__file__).resolve().parent;c=json.loads((root/'protocol.json').read_text(encoding='utf8'))
dates=np.load(root/'features/dates.npy');pool=np.load(root/'features/label_eligible.npy',mmap_mode='r')
labels={h:np.load(root/f'features/label_rank_{h}d.npy',mmap_mode='r') for h in c['model_horizons']}
rows=[]
for spec in fold_specs(c):
    for part in ['fit','stop']:
        if part not in spec:continue
        mask=label_split_mask(dates,*spec[part],60);eligible=pool[mask];total=int(eligible.sum())
        for h,y in labels.items():
            valid=eligible&np.isfinite(y[mask]);observed=int(valid.sum())
            rows.append({'fold':spec['name'],'partition':part,'horizon':h,'buyable_stock_days':total,'observed_label_stock_days':observed,'label_observation_fraction':observed/total,'dates':int(mask.sum())})
with (root/'label_coverage_training.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
atomic_write_json(root/'label_coverage_note.json',{'scope':'fit/stop periods only; no external labels inspected','rows':len(rows),'contract':'Existing economic open-return chain: missing intermediate quotes/close references can make a label unavailable; terminal zero-recovery handling is retained. No future buy/sell legality mask is applied.','interpretation':'Head-specific RankIC values use different observed-label subsets; equal task/day loss weighting does not make those subsets identical.','data_manifest_sha256':file_sha256(root/'features/data_manifest.json'),'csv_sha256':file_sha256(root/'label_coverage_training.csv')})
print({'training_coverage_rows':len(rows),'external_outcomes_read':False})
