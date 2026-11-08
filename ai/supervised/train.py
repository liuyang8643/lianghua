"""Fixed horizon models with a purged, held-out internal stopping year."""
from __future__ import annotations
import argparse
import gc
import json
import os
from pathlib import Path
import time

if __name__ == '__main__':
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
        os.environ[key]='1'
import numpy as np
from env.forward_returns import label_split_mask
from .model import LGBRankModel
from .ic import daily_ic,summary
from utils.atomic_file import atomic_write_json,file_sha256
from factor_db import require_allowed_factor_names


def predict_panel(model,columns,eligible,start=0,stop=None):
    stop=len(eligible) if stop is None else stop
    output=np.full((stop-start,eligible.shape[1]),np.nan,dtype=np.float64)
    for a in range(start,stop,20):
        b=min(a+20,stop);matrix=np.stack([x[a:b] for x in columns],axis=-1)
        valid=eligible[a:b]&np.isfinite(matrix).any(axis=2)
        if not valid.any():continue
        values=model.predict(matrix[valid])
        block=output[a-start:b-start];block[valid]=values
    return output


def run(root,threads=8):
    from scripts.run_research_guarded import require_guarded_job
    require_guarded_job()
    started=time.monotonic();config=json.loads((root/'protocol.json').read_text(encoding='utf-8'))
    source=root/'features';status=json.loads((source/'status.json').read_text())
    if status['state']!='complete':raise ValueError('selected feature preparation is incomplete')
    inventory=json.loads((source/'inventory.json').read_text(encoding='utf-8'))
    data_manifest=json.loads((source/'data_manifest.json').read_text())
    for name,sha in data_manifest.items():
        if file_sha256(source/name)!=sha:raise ValueError('training axes/labels changed: '+name)
    dates=np.load(source/'dates.npy');eligible=np.load(source/'eligible.npy',mmap_mode='r')
    label_eligible=np.load(source/'label_eligible.npy',mmap_mode='r')
    if label_eligible.shape!=eligible.shape or np.any(label_eligible&~eligible):raise ValueError('label universe must be a subset of the PIT signal universe')
    for h in config['model_horizons']:
        target=root/'models'/f'n{h}';target.mkdir(parents=True,exist_ok=True)
        selection=json.loads((root/'selection'/f'selected_{h}d.json').read_text(encoding='utf-8'))
        names=[r['name'] for r in selection['selected']]
        require_allowed_factor_names(names)
        identity={'features':{name:inventory[name]['sha256'] for name in names},'selection_sha256':file_sha256(root/'selection'/f'selected_{h}d.json'),
                  'data_manifest_sha256':file_sha256(source/'data_manifest.json'),
                  'protocol_sha256':file_sha256(root/'protocol.json'),'horizon':h,'model_source':file_sha256('ai/supervised/model.py'),
                  'factor_admission_sha256':file_sha256('factor_db/__init__.py'),
                  'driver_source':file_sha256(Path(__file__)),'internal_stopping':config['internal_validation'],
                  'fit':config['fit'],'final_refit':False,'retrain_schedule':'none','label_pool':'current_market_buyable'}
        if (target/'complete.json').exists():
            if json.loads((target/'identity.json').read_text())!=identity:raise ValueError('completed model identity differs')
            done=json.loads((target/'complete.json').read_text())
            for name,sha in done['artifact_sha256'].items():
                if file_sha256(target/name)!=sha:raise ValueError('completed model artifact changed: '+name)
            if np.load(target/'predictions.npy',mmap_mode='r').shape!=eligible.shape:raise ValueError('completed prediction axes differ')
            if np.load(target/'prediction_daily_ic.npy',mmap_mode='r').shape!=(len(dates),3):raise ValueError('completed IC axes differ')
            if json.loads((target/'model.json').read_text(encoding='utf-8'))['feature_names']!=names:raise ValueError('completed model feature order differs')
            continue
        if (target/'started.json').exists():raise RuntimeError('incomplete prior training needs explicit audited recovery, not blind restart')
        atomic_write_json(target/'identity.json',identity);atomic_write_json(target/'started.json',{'pid':os.getpid()})
        for name in names:
            if file_sha256(inventory[name]['path'])!=inventory[name]['sha256']:raise ValueError('input feature changed')
        columns=[np.load(inventory[name]['path'],mmap_mode='r') for name in names]
        raw_label=np.load(source/f'labels_{h}d.npy',mmap_mode='r')
        train_dates=label_split_mask(dates,*config['train'],h)
        rank_label=np.load(source/f'label_rank_{h}d.npy',mmap_mode='r')
        if rank_label.shape!=eligible.shape:raise ValueError('rank label shape differs')
        if np.any(np.isfinite(rank_label)&~label_eligible):raise ValueError('unbuyable stock has a ranking label')
        mask=label_eligible&np.isfinite(rank_label)&train_dates[:,None]
        observed=np.zeros(eligible.shape,dtype=bool)
        for column in columns:observed|=np.isfinite(column)
        excluded_no_input=int((mask&~observed).sum());mask&=observed;del observed
        flat=np.flatnonzero(mask.ravel());row=flat//eligible.shape[1]
        np.save(target/'sample_indices.npy',flat,allow_pickle=False)
        x=np.lib.format.open_memmap(target/'train_x.npy',mode='w+',dtype=np.float32,shape=(len(flat),len(names)))
        for a in range(0,len(flat),200000):
            indices=flat[a:a+200000]
            x[a:a+len(indices)]=np.stack([f.ravel()[indices] for f in columns],axis=1)
        x.flush();valid=np.isfinite(x).any(axis=1)
        if not valid.all():raise AssertionError('observed-input mask and matrix disagree')
        y=rank_label.ravel()[flat];del rank_label,mask
        day_count=np.bincount(row,minlength=len(dates))
        fit=label_split_mask(dates,*config['fit'],h)[row]
        stopping=label_split_mask(dates,*config['internal_validation'],h)[row]
        if not fit.any() or not stopping.any() or np.any(fit&stopping):raise ValueError('empty or overlapping fit/stopping samples')
        if dates[row[fit]+h].max()>=dates[row[stopping]].min():raise ValueError('label endpoints cross stopping boundary')
        weights=day_count[np.unique(row[fit])].mean()/day_count[row]
        model=LGBRankModel(names,seed=20261106+h,num_threads=threads)
        t=time.monotonic();model.fit(x[fit],y[fit],validation=(x[stopping],y[stopping]),weights=weights[fit])
        rounds=model.best_iteration;fit_seconds=time.monotonic()-t
        model.save(target/'model.txt',{'horizon':h,'fixed_training_interval':config['fit'],
            'internal_validation':config['internal_validation'],'refit_on_internal_validation':False,
            'training_rows':int(fit.sum()),'internal_validation_rows':int(stopping.sum()),
            'weight_normalization':'fit_dates_only_mean_one','fit_weight_mean':float(weights[fit].mean()),
            'actual_fit_decision_span':[str(dates[row[fit]].min()),str(dates[row[fit]].max())],
            'actual_fit_label_end_max':str(dates[row[fit]+h].max()),
            'actual_internal_decision_span':[str(dates[row[stopping]].min()),str(dates[row[stopping]].max())],
            'actual_internal_label_end_max':str(dates[row[stopping]+h].max()),
            'label':'cross-sectional percentile of economically adjusted open[T] to open[T+N] cumulative return in T-market-buyable pool','features':names})
        predictions=predict_panel(model,columns,eligible);np.save(target/'predictions.npy',predictions,allow_pickle=False)
        metrics=np.full((len(dates),3),np.nan)
        metrics[train_dates]=daily_ic(predictions[train_dates],raw_label[train_dates],label_eligible[train_dates])
        np.save(target/'prediction_daily_ic.npy',metrics,allow_pickle=False)
        # External holdout labels are not evaluated during training. Their
        # observations are produced only by final_evaluation after selection.
        diagnostics={split:summary(metrics,label_split_mask(dates,*config[split],h)) for split in ('fit','internal_validation','train')}
        receipt={'state':'complete','horizon':h,'features':len(names),'trees':rounds,'training_rows':int(fit.sum()),'internal_validation_rows':int(stopping.sum()),
                 'prediction_ic_scope':'train_only',
                 'excluded_no_observed_input_rows':excluded_no_input,
                 'fit_seconds':fit_seconds,'refit_on_internal_validation':False,
                 'model_sha256':file_sha256(target/'model.txt'),'predictions_sha256':file_sha256(target/'predictions.npy'),
                 'data_manifest_sha256':file_sha256(source/'data_manifest.json'),
                 'prediction_ic_sha256':file_sha256(target/'prediction_daily_ic.npy'),
                 'diagnostics':diagnostics}
        receipt['artifact_sha256']={name:file_sha256(target/name) for name in ['model.txt','model.json','predictions.npy','prediction_daily_ic.npy','identity.json','sample_indices.npy']}
        atomic_write_json(target/'complete.json',receipt,allow_nan=False);print(receipt,flush=True)
        del x,y,weights,model,predictions,metrics;gc.collect()
    atomic_write_json(root/'models/status.json',{'state':'complete','elapsed_seconds':time.monotonic()-started,'final_fixed_models':len(config['model_horizons'])})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--threads',type=int,default=8)
    a=p.parse_args();run(a.root,a.threads)
