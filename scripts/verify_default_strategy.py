"""Compare a portable release against saved full research cross sections; no fitting."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
from strategy_release import load_default

def main(root,output,model_name='final'):
 s=load_default();folder=root/'models'/model_name;meta=json.loads((folder/'model.json').read_text(encoding='utf8'));done=json.loads((folder/'complete.json').read_text(encoding='utf8'))
 assert s.names==meta['feature_names'] and hashlib.sha256((s.root/'model.txt').read_bytes()).hexdigest()==done['model_sha256']
 dates=np.load(root/'features/dates.npy');codes=np.load(root/'features/codes.npy').astype(str);eligible=np.load(root/'features/eligible.npy',mmap_mode='r');pred=np.load(folder/'predictions.npy',mmap_mode='r');rows=np.load(folder/'rows.npy') if (folder/'rows.npy').exists() else np.arange(len(dates))
 inv=json.loads((root/'features/inventory.json').read_text(encoding='utf8'));columns=[]
 for n in s.names:
  p=Path(inv[n]['path']);assert hashlib.sha256(p.read_bytes()).hexdigest()==inv[n]['sha256'];columns.append(np.load(p,mmap_mode='r'))
 records=[]
 for position in np.unique(np.linspace(0,len(rows)-1,5,dtype=int)):
  i=int(rows[position]);x=np.stack([col[i] for col in columns],axis=1);raw,score=s.predict(x,feature_names=s.names,codes=codes,eligible=eligible[i],complete_universe=True);expected=pred[position]
  np.testing.assert_allclose(raw,expected,rtol=1e-12,atol=1e-12,equal_nan=True)
  pool=eligible[i]&np.isfinite(expected).all(axis=1);ranks=np.full(expected.shape,np.nan,dtype=np.float32)
  for k in range(5):ranks[pool,k]=((rankdata(expected[pool,k])-.5)/pool.sum()).astype(np.float32)
  np.testing.assert_array_equal(score,np.sum(ranks*np.asarray(s.manifest['weights']),axis=1))
  records.append({'date':str(dates[i]),'stock_axis':len(codes),'eligible_predictions':int(pool.sum()),'raw_prediction_max_error':float(np.nanmax(abs(raw-expected))),'fused_scores_exact':True})
 result={'state':'pass','release':s.manifest['release_id'],'parameters':s.manifest['parameters'],'weights':s.manifest['weights'],'model_sha256':done['model_sha256'],'records':records,'scope':'portable normalized-input inference and fusion; upstream live factor supply and broker execution not certified'}
 output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8');print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);p.add_argument('--model-name',default='final');p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.study,a.output,a.model_name)
