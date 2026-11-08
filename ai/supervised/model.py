"""Array adapter for Qlib's LightGBM MSE training and prediction workflow.

Derived from Microsoft Qlib qlib/contrib/model/gbdt.py (MIT). DatasetH and
MLflow orchestration are replaced by WBR's sealed arrays and local artifacts;
the LightGBM objective, callbacks, fit and predict operations are retained.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path

import numpy as np
from factor_db import require_allowed_factor_names


DEFAULT_PARAMETERS={
    'objective':'mse','metric':'l2','verbosity':-1,'learning_rate':.04,
    'num_leaves':31,'max_depth':6,'min_data_in_leaf':300,
    'feature_fraction':.9,'bagging_fraction':.85,'bagging_freq':1,
    'lambda_l1':.1,'lambda_l2':10.,'max_bin':127,
    'deterministic':True,'force_col_wise':True,
}


class LGBRankModel:
    def __init__(self,feature_names,*,seed=20261105,num_threads=8,parameters=None):
        self.feature_names=tuple(feature_names)
        require_allowed_factor_names(self.feature_names)
        if not self.feature_names or len(set(self.feature_names))!=len(self.feature_names):
            raise ValueError('model features must be nonempty and unique')
        self.parameters={**DEFAULT_PARAMETERS,'seed':int(seed),'num_threads':int(num_threads),**(parameters or {})}
        self.model=None
        self.evals_result={}

    def fit(self,x,y,*,rounds=1000,validation=None,early_stopping_rounds=60,weights=None):
        import lightgbm as lgb
        if self.model is not None:raise RuntimeError('a frozen model cannot be fitted again')
        if x.ndim!=2 or x.shape[1]!=len(self.feature_names) or np.shape(y)!=(len(x),):
            raise ValueError('model training axes differ')
        if not np.isfinite(y).all():raise ValueError('training labels must be observed finite outcomes')
        names=[f'f{index:04d}' for index in range(len(self.feature_names))]
        train=lgb.Dataset(x,label=y,weight=weights,feature_name=names,free_raw_data=True)
        datasets=[train];segments=['train']
        callbacks=[lgb.log_evaluation(period=50),lgb.record_evaluation(self.evals_result)]
        if validation is not None:
            vx,vy=validation
            if vx.shape[1]!=x.shape[1] or np.shape(vy)!=(len(vx),) or not np.isfinite(vy).all():
                raise ValueError('internal validation axes or labels invalid')
            datasets.append(lgb.Dataset(vx,label=vy,reference=train,feature_name=names,free_raw_data=True));segments.append('valid')
            callbacks.append(lgb.early_stopping(early_stopping_rounds,verbose=False))
        self.model=lgb.train(self.parameters,train,num_boost_round=int(rounds),valid_sets=datasets,valid_names=segments,callbacks=callbacks)
        return self

    def predict(self,x):
        if self.model is None:raise RuntimeError('model is not fitted')
        if x.ndim!=2 or x.shape[1]!=len(self.feature_names):raise ValueError('prediction feature order/shape differs')
        return np.asarray(self.model.predict(x,num_threads=self.parameters['num_threads']),dtype=np.float64)

    @property
    def best_iteration(self):
        if self.model is None:raise RuntimeError('model is not fitted')
        return int(self.model.best_iteration or self.model.current_iteration())

    def save(self,path,metadata):
        import lightgbm as lgb
        if self.model is None:raise RuntimeError('model is not fitted')
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        tmp=path.with_suffix('.tmp');self.model.save_model(str(tmp));os.replace(tmp,path)
        sidecar={'schema':'wbr-open-rank-lgb-v1','lightgbm_version':lgb.__version__,
                 'feature_names':self.feature_names,'parameters':self.parameters,
                 'best_iteration':self.best_iteration,'metadata':metadata,'evals_result':self.evals_result}
        path.with_suffix('.json').write_text(json.dumps(sidecar,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


@dataclass(frozen=True)
class RankReturnCalibration:
    """Train-only calibration: prediction-percentile bins to N-day excess return.

    Caller supplies chronologically out-of-fit predictions and equal-date
    sample weights. Bins are finer in the highest prediction percentiles so a
    top-20 portfolio is not collapsed into one coarse upper-tail bin. Means are
    isotonic-regularized; values outside support are clipped, not extrapolated.
    """
    percentile_knots: tuple[float,...]
    excess_return_knots: tuple[float,...]

    @classmethod
    def fit(cls,prediction_percentiles,excess_returns,weights=None,bins=20):
        from sklearn.isotonic import IsotonicRegression
        p=np.asarray(prediction_percentiles,dtype=np.float64);y=np.asarray(excess_returns,dtype=np.float64)
        w=np.ones_like(p) if weights is None else np.asarray(weights,dtype=np.float64)
        valid=np.isfinite(p)&np.isfinite(y)&np.isfinite(w)&(w>0)
        p,y,w=p[valid],y[valid],w[valid]
        if len(p)<bins or np.any(p<0) or np.any(p>1):raise ValueError('invalid or insufficient calibration samples')
        # Fixed before any financial model/search results: 20 broad, 20 upper,
        # and 40 extreme-tail bins by default. No return-dependent bin search.
        edges=np.r_[np.linspace(0.,.9,bins,endpoint=False),np.linspace(.9,.99,bins,endpoint=False),np.linspace(.99,1.,2*bins+1)]
        count=len(edges)-1;indices=np.clip(np.searchsorted(edges,p,side='right')-1,0,count-1)
        mass=np.bincount(indices,weights=w,minlength=count)
        sx=np.bincount(indices,weights=w*p,minlength=count);sy=np.bincount(indices,weights=w*y,minlength=count)
        good=mass>0
        x=sx[good]/mass[good];means=sy[good]/mass[good]
        fitted=IsotonicRegression(out_of_bounds='clip').fit_transform(x,means,sample_weight=mass[good])
        return cls(tuple(x.tolist()),tuple(fitted.tolist()))

    def predict(self,percentiles):
        return np.interp(percentiles,self.percentile_knots,self.excess_return_knots)
