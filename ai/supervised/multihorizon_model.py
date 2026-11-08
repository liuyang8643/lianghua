"""Memory-bounded categorical-horizon LightGBM using native Dataset Sequence."""
import argparse,gc,json,time
from pathlib import Path
import numpy as np
import lightgbm as lgb
from .model import DEFAULT_PARAMETERS
from .multihorizon_select import fold_specs
from .multihorizon_admission import verify_data
from .ic import daily_ic,summary
from env.forward_returns import label_split_mask
from factor_db import require_allowed_factor_names
from utils.atomic_file import atomic_write_json,file_sha256

class HorizonSequence(lgb.Sequence):
    batch_size=16384
    def __init__(self,columns,flat,task):self.columns=columns;self.flat=flat;self.task=task
    def __len__(self):return len(self.flat)
    def __getitem__(self,key):
        scalar=isinstance(key,(int,np.integer));index=np.atleast_1d(self.flat[key]);tasks=np.atleast_1d(self.task[key])
        out=np.empty((len(index),len(self.columns)+1),np.float64)
        for j,column in enumerate(self.columns):out[:,j]=column.ravel()[index]
        out[:,-1]=tasks
        return out[0] if scalar else out

def build_samples(columns,labels,eligible,mask):
    observed=np.zeros(eligible.shape,bool)
    for c in columns:observed|=np.isfinite(c)
    flats=[];tasks=[];ys=[];weights=[]
    for k,y in enumerate(labels):
        flat=np.flatnonzero((eligible&observed&np.isfinite(y)&mask[:,None]).ravel())
        if not len(flat):raise ValueError('empty horizon training sample')
        row=flat//eligible.shape[1];count=np.bincount(row,minlength=len(eligible));validdays=np.count_nonzero(count)
        flats.append(flat);tasks.append(np.full(len(flat),k,np.int8));ys.append(y.ravel()[flat])
        weights.append(1./(len(labels)*validdays*count[row]))
    weight=np.concatenate(weights);weight/=weight.mean()
    return np.concatenate(flats),np.concatenate(tasks),np.concatenate(ys),weight

def predict(model,columns,eligible,rows,horizons):
    result=np.full((len(rows),eligible.shape[1],len(horizons)),np.nan)
    for a in range(0,len(rows),8):
        rr=rows[a:a+8];matrix=np.stack([c[rr] for c in columns],axis=-1)
        valid=eligible[rr]&np.isfinite(matrix).any(axis=2)
        x=np.column_stack([matrix[valid],np.zeros(valid.sum())])
        for k in range(len(horizons)):
            x[:,-1]=k;out=result[a:a+len(rr),:,k];out[valid]=model.predict(x,num_threads=8)
    return result

def fit_one(root,spec,*,rounds=None):
    config=json.loads((root/'protocol.json').read_text(encoding='utf8'));source=root/'features';folder=root/'models'/spec['name'];folder.mkdir(parents=True,exist_ok=True)
    if (folder/'complete.json').exists():
        done=json.loads((folder/'complete.json').read_text());identity=json.loads((folder/'identity.json').read_text())
        assert identity['source']==file_sha256(__file__) and identity['protocol']==file_sha256(root/'protocol.json') and identity['spec']==spec
        assert identity['selection']==file_sha256(root/'fold_selection'/spec['name']/'complete.json')
        assert identity['data_manifest']==file_sha256(root/'features/data_manifest.json')
        for filename,key in [('model.txt','model_sha256'),('predictions.npy','predictions_sha256'),('rows.npy','rows_sha256')]:assert file_sha256(folder/filename)==done[key]
        return done
    if (folder/'started.json').exists():raise RuntimeError('incomplete model: inspect evidence before recovery')
    selection=root/'fold_selection'/spec['name']/'complete.json';names=json.loads(selection.read_text())['features'];require_allowed_factor_names(names)
    inventory=json.loads((source/'inventory.json').read_text(encoding='utf8'));columns=[np.load(inventory[n]['path'],mmap_mode='r') for n in names]
    dates=np.load(source/'dates.npy');eligible=np.load(source/'eligible.npy',mmap_mode='r');buyable=np.load(source/'label_eligible.npy',mmap_mode='r')
    labels=[np.load(source/f'label_rank_{h}d.npy',mmap_mode='r') for h in config['model_horizons']]
    fitmask=label_split_mask(dates,*spec['fit'],60);flat,task,y,w=build_samples(columns,labels,buyable,fitmask)
    parameters={**DEFAULT_PARAMETERS,'seed':20261106,'num_threads':8}
    identity={'spec':spec,'features':{n:inventory[n]['sha256'] for n in names},'selection':file_sha256(selection),
        'data_manifest':file_sha256(source/'data_manifest.json'),'protocol':file_sha256(root/'protocol.json'),
        'parameters':parameters,'task_values':config['model_horizons'],'source':file_sha256(__file__),
        'last_fit_label_end':str(dates[np.flatnonzero(fitmask)[-1]+60]),'fixed_rounds':rounds}
    atomic_write_json(folder/'identity.json',identity);atomic_write_json(folder/'started.json',{'state':'started'})
    np.savez_compressed(folder/'samples.npz',flat=flat,task=task,weight=w)
    features=[f'f{i:04d}' for i in range(len(names))]+['horizon']
    ds=lgb.Dataset(HorizonSequence(columns,flat,task),label=y,weight=w,feature_name=features,categorical_feature=['horizon'],params={'max_bin':127},free_raw_data=True)
    datasets=[ds];dsnames=['train'];callbacks=[lgb.log_evaluation(50)];evaluation={};callbacks.append(lgb.record_evaluation(evaluation))
    if rounds is None:
        vm=label_split_mask(dates,*spec['stop'],60);vf,vt,vy,vw=build_samples(columns,labels,buyable,vm)
        assert dates[np.flatnonzero(fitmask)[-1]+60]<dates[np.flatnonzero(vm)[0]]
        vd=lgb.Dataset(HorizonSequence(columns,vf,vt),label=vy,weight=vw,reference=ds,feature_name=features,categorical_feature=['horizon'],free_raw_data=True)
        datasets.append(vd);dsnames.append('valid');callbacks.append(lgb.early_stopping(60,verbose=False))
        np.savez_compressed(folder/'validation_samples.npz',flat=vf,task=vt,weight=vw)
    started=time.monotonic();model=lgb.train(parameters,ds,num_boost_round=rounds or 1000,valid_sets=datasets,valid_names=dsnames,callbacks=callbacks)
    best=int(model.best_iteration or model.current_iteration());model.save_model(str(folder/'model.txt'))
    atomic_write_json(folder/'model.json',{'feature_names':names,'horizons':config['model_horizons'],'best_iteration':best,'parameters':parameters,'evaluation':evaluation,'fit_seconds':time.monotonic()-started,'weighting':'each horizon1/5; each date equal within horizon'})
    # Free binned datasets before prediction. Model fitting never reads outer outcomes.
    model.free_dataset();del ds,datasets,flat,task,y,w
    if rounds is None:del vd,vf,vt,vy,vw
    gc.collect()
    diagnostics={}
    if rounds is None:
        vr=np.flatnonzero(vm);vp=predict(model,columns,eligible,vr,config['model_horizons'])
        for k,h in enumerate(config['model_horizons']):
            valid=buyable[vr]&np.isfinite(labels[k][vr])&np.isfinite(vp[:,:,k])
            count=valid.sum(axis=1);err=np.where(valid,(vp[:,:,k]-labels[k][vr])**2,0).sum(axis=1)
            mse=float(np.mean(err[count>0]/count[count>0]))
            raw=np.load(source/f'labels_{h}d.npy',mmap_mode='r')
            diagnostics[str(h)]={'weighted_date_mse':mse,**summary(daily_ic(vp[:,:,k],raw[vr],buyable[vr]))}
        atomic_write_json(folder/'validation_by_horizon.json',diagnostics);del vp
    rr=np.arange(len(dates)) if spec['name']=='final' else np.flatnonzero((dates>=np.datetime64(spec['evaluate'][0]))&(dates<=np.datetime64(spec['evaluate'][1])))
    predictions=predict(model,columns,eligible,rr,config['model_horizons']);np.save(folder/'predictions.npy',predictions);np.save(folder/'rows.npy',rr)
    # Actual reload parity on a fixed sampled set, including the categorical task column.
    reload=lgb.Booster(model_file=str(folder/'model.txt'));check=np.linspace(0,len(rr)-1,min(11,len(rr)),dtype=int)
    np.testing.assert_allclose(predict(reload,columns,eligible,rr[check],config['model_horizons']),predictions[check],rtol=1e-12,atol=1e-12,equal_nan=True)
    result={'state':'complete','spec':spec,'trees':best,'feature_count':len(names),'fit_and_prediction_seconds':time.monotonic()-started,'validation_by_horizon':diagnostics,
        'model_sha256':file_sha256(folder/'model.txt'),'predictions_sha256':file_sha256(folder/'predictions.npy'),
        'rows_sha256':file_sha256(folder/'rows.npy'),'identity_sha256':file_sha256(folder/'identity.json'),'reload_parity':True}
    atomic_write_json(folder/'complete.json',result);print(result,flush=True);return result

def run(root,final=False):
    from scripts.run_research_guarded import require_guarded_job
    require_guarded_job();c=json.loads((root/'protocol.json').read_text(encoding='utf8'))
    assert json.loads((root/'fold_selection/status.json').read_text())['state']=='complete'
    data_identity=verify_data(root)
    inventory=json.loads((root/'features/inventory.json').read_text(encoding='utf8'))
    for r in inventory.values():assert file_sha256(r['path'])==r['sha256']
    if final:
        assert (root/'search/winner.json').exists()
        best=[json.loads((root/f'models/fold{y}/complete.json').read_text())['trees'] for y in c['fold_years']]
        fit_one(root,fold_specs(c)[-1],rounds=int(np.median(best)))
    else:
        for spec in fold_specs(c)[:-1]:fit_one(root,spec)
    assert verify_data(root)==data_identity
    if not final:atomic_write_json(root/'models/oof_complete.json',{'state':'complete','folds':c['fold_years']})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--final',action='store_true');a=p.parse_args();run(a.root,a.final)
