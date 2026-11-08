"""Completed-bar low-attention research signals from BV16dmxBoE9Q.

The return signal is the negative 20-day maximum absolute difference from
the daily equal-weight PIT market simple return. The amount signal is the
negative 20-day maximum of amount divided by its prior-year mean. The video
does not specify year length or mean endpoint: 252 prior completed sessions,
excluding the ratio day, is an explicit research assumption. All windows
require complete finite observations; zero turnover is a valid observation,
negative amounts are not, and the prior-year mean must be positive. No
late-listed stock is backfilled.
"""
from __future__ import annotations
import hashlib
from pathlib import Path
import numpy as np
from factor.base import FactorDefinition, FactorMetadata


def _negative_completed_max(values, window):
    """Fixed-offset reduction, retaining NaN if any window value is missing."""
    out=np.full_like(values,np.nan,dtype=np.float64)
    count=len(values)-window
    if count<=0:return out
    # Doubling spans need O(log(window)) full-matrix reductions, not window
    # passes. np.maximum intentionally propagates any missing observation.
    target=values
    span=1
    while span*2<=window:
        target=np.maximum(target[:-span],target[span:])
        span*=2
    remainder=window-span
    if remainder:
        target=np.maximum(target[:-remainder],target[remainder:])
    np.negative(target[:-1], out=out[window:])
    return out


class BiliLowReturnAttention20:
    hist_days=20
    requires_full_history=False
    pre_ranked=False

    def calc_batch(self,panel):
        cl,pc=(np.asarray(panel[k],dtype=np.float64) for k in ('close','preClose'))
        age,delisted=(np.asarray(panel[k]) for k in ('listing_age','delisted_mask'))
        if cl.ndim!=2 or any(v.shape!=cl.shape for v in (pc,age,delisted)):
            raise ValueError('attention fields must share date-stock axes')
        # Public factor preparation converts required fields to float64,
        # including the boolean sidecar. Compare its explicit false value.
        member=(age>=0)&(delisted==0)
        valid=np.isfinite(cl)&(cl>0)&np.isfinite(pc)&(pc>0)&member
        returns=np.divide(cl,pc,out=np.full(cl.shape,np.nan),where=valid)
        returns-=1
        count=valid.sum(axis=1)
        mean=np.divide(np.sum(returns,where=valid,axis=1,initial=0.0),count,
                       out=np.full(len(cl),np.nan),where=count>0)
        returns-=mean[:,None]
        np.abs(returns,out=returns)
        return _negative_completed_max(returns,20)


class BiliLowAmountAttention20:
    hist_days=272
    requires_full_history=False
    pre_ranked=False

    def calc_batch(self,panel):
        amount=np.asarray(panel['amount'],dtype=np.float64)
        if amount.ndim!=2:raise ValueError('amount must be date-stock matrix')
        transposed=np.array(amount.T,order='C',copy=True)
        valid=np.isfinite(transposed)&(transposed>=0)
        # Contiguous time rows preserve each stock's accumulation order while
        # avoiding the large-stride traversal of a date-major cumsum.
        sums=np.empty((amount.shape[1],len(amount)+1),dtype=np.float64)
        sums[:,0]=0
        transposed[~valid]=0
        np.cumsum(transposed,axis=1,out=sums[:,1:])
        counts=np.empty((amount.shape[1],len(amount)+1),dtype=np.int32)
        counts[:,0]=0
        np.cumsum(valid,axis=1,dtype=np.int32,out=counts[:,1:])
        daily=np.full(transposed.shape,np.nan)
        if len(amount)>252:
            denominator=sums[:,252:-1]-sums[:,:-253]
            denominator/=252
            good=(counts[:,252:-1]-counts[:,:-253]==252)&valid[:,252:]&(denominator>0)
            np.divide(transposed[:,252:],denominator,out=daily[:,252:],where=good)
            del denominator,good
        del sums,counts,valid,transposed
        return _negative_completed_max(daily.T,20)


_SHA=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
DEFINITIONS=tuple(FactorDefinition(FactorMetadata(name=cls.__name__,version='bili-attention-v2-zero-amount-valid-252prior',hist_days=cls.hist_days,required_fields=fields,implementation_hash=_SHA),cls) for cls,fields in (
    (BiliLowReturnAttention20,('close','preClose','listing_age','delisted_mask')),
    (BiliLowAmountAttention20,('amount',)),
))
