"""Single authority for pairwise cross-sectional IC and training-only selection."""
from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True, fastmath=False)
def average_ranks(values):
    order=np.argsort(values); result=np.empty(len(values),dtype=np.float64); a=0
    while a<len(values):
        b=a+1
        while b<len(values) and values[order[b]]==values[order[a]]:
            b+=1
        for i in range(a,b): result[order[i]]=(a+b-1)*.5
        a=b
    return result


@njit(cache=True, fastmath=False)
def correlation(a,b):
    # Decimal constants can have a rounded mean unequal to each element.
    # Test constancy before centering, otherwise a constant can get IC=1.
    if len(a)<2 or np.min(a)==np.max(a) or np.min(b)==np.max(b):
        return np.nan
    a=a-a.mean(); b=b-b.mean(); denominator=np.sqrt(np.dot(a,a)*np.dot(b,b))
    return np.dot(a,b)/denominator if denominator>0 else np.nan


@njit(cache=True, fastmath=False)
def daily_ic(values, labels, eligible, minimum_pairs=30):
    result=np.full((len(values),3),np.nan)
    for t in range(len(values)):
        good=eligible[t] & np.isfinite(values[t]) & np.isfinite(labels[t])
        a=values[t][good]; b=labels[t][good]; result[t,2]=len(a)
        if len(a)>=minimum_pairs:
            result[t,0]=correlation(a,b)
            result[t,1]=correlation(average_ranks(a),average_ranks(b))
    return result


def summary(daily, mask=None):
    data=daily if mask is None else daily[mask]
    out={}
    for column,name in [(0,'ic'),(1,'rank_ic')]:
        values=data[:,column]; values=values[np.isfinite(values)]
        mean=float(values.mean()) if len(values) else None
        sd=float(values.std(ddof=1)) if len(values)>1 else None
        out[name]=mean
        out[name+'_ir']=mean/sd if sd and sd>0 else None
        out[name+'_days']=len(values)
    out['mean_pairs']=float(np.nanmean(data[:,2])) if len(data) else None
    return out


@njit(cache=True, fastmath=False)
def pair_counts_and_constants(values,labels,eligible):
    """Linear-time compatibility check for reusing existing daily IC arrays."""
    counts=np.zeros(len(values),np.int64);constant=np.ones(len(values),np.bool_)
    for t in range(len(values)):
        first_a=0.;first_b=0.;var_a=False;var_b=False
        for j in range(values.shape[1]):
            if eligible[t,j] and np.isfinite(values[t,j]) and np.isfinite(labels[t,j]):
                a,b=values[t,j],labels[t,j]
                if counts[t]==0:first_a=a;first_b=b
                else:var_a|=a!=first_a;var_b|=b!=first_b
                counts[t]+=1
        constant[t]=not(var_a and var_b)
    return counts,constant
