"""Compare the portable release to full saved research cross sections; no training."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
from strategy_release import load_default

def main(root,output):
 s=load_default();meta=json.loads((root/'models/final/model.json').read_text(encoding='utf8'));assert s.names==meta['feature_names'];dates=np.load(root/'features/dates.npy');codes=np.load(root/'features/codes.npy').astype(str);eligible=np.load(root/'features/eligible.npy',mmap_mode='r');pred=np.load(root/'models/final/predictions.npy',mmap_mode='r');inv=json.loads((root/'features/inventory.json').read_text(encoding='utf8'));columns=[np.load(inv[n]['path'],mmap_mode='r') for n in s.names];records=[]
 for date in ['2011-01-04','2017-12-01','2021-12-31','2026-10-29']:
  i=int(np.searchsorted(dates,np.datetime64(date)));assert str(dates[i])==date;x=np.stack([col[i] for col in columns],axis=1);raw,score=s.predict(x,feature_names=s.names,codes=codes,eligible=eligible[i],complete_universe=True);expected=pred[i];np.testing.assert_allclose(raw,expected,rtol=1e-12,atol=1e-12,equal_nan=True)
  pool=eligible[i]&np.isfinite(expected).all(axis=1);ranks=np.full(expected.shape,np.nan,dtype=np.float32)
  for k in range(5):ranks[pool,k]=((rankdata(expected[pool,k])-.5)/pool.sum()).astype(np.float32)
  expected_score=np.sum(ranks*np.array([.2,.3,.1,.3,.1]),axis=1);np.testing.assert_array_equal(score,expected_score)
  records.append({'date':date,'stock_axis':len(codes),'eligible_predictions':int(pool.sum()),'raw_prediction_max_error':float(np.nanmax(abs(raw-expected))),'fused_scores_exact':True})
 result={'state':'pass','model_sha256':hashlib.sha256((s.root/'model.txt').read_bytes()).hexdigest(),'records':records,'scope':'portable normalized-input inference and fusion; raw live feature supply/broker execution not certified'};output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps(result))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();main(a.study,a.output)
