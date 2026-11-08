"""The single missing-aware cross-sectional percentile transformation."""
import numpy as np
from scipy.stats import rankdata


def percentile(raw,eligible,*,binary=False):
    values=np.where(np.asarray(eligible,dtype=bool)&np.isfinite(raw),raw,np.nan)
    if binary:return values.astype(np.float32)
    ranks=rankdata(values,axis=-1,method='average',nan_policy='omit')
    counts=np.isfinite(values).sum(axis=-1,keepdims=True)
    return np.divide(ranks-.5,counts,out=np.full(values.shape,np.nan),where=counts>0).astype(np.float32)
