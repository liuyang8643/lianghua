"""Forward labels under the existing open-to-open economic account convention.

These are future outcomes for supervised learning, never decision features.
Missing executable quotes remain missing. As in the local simulator, a known
delisting has zero recovery; it takes precedence over earlier quote gaps.
"""
from __future__ import annotations

import numpy as np


def open_forward_returns(panel, horizon: int) -> np.ndarray:
    if type(horizon) is not int or horizon < 1:
        raise ValueError('horizon must be a positive integer')
    opening, close, preclose = (np.asarray(panel[k],dtype=np.float64) for k in ('open','close','preClose'))
    if opening.ndim != 2 or close.shape != opening.shape or preclose.shape != opening.shape:
        raise ValueError('price arrays must have identical [date,stock] shapes')
    dead = np.asarray(panel['delisted_mask'],dtype=bool)
    member = (np.asarray(panel['listing_age']) >= 0) & ~dead
    if dead.shape != opening.shape or member.shape != opening.shape:
        raise ValueError('membership arrays differ from price shapes')
    result=np.full(opening.shape,np.nan)
    length=len(opening)-horizon
    if length <= 0:
        return result
    good=(np.isfinite(opening[:-1]) & (opening[:-1]>0)
          & np.isfinite(opening[1:]) & (opening[1:]>0)
          & np.isfinite(close[:-1]) & (close[:-1]>0)
          & np.isfinite(preclose[1:]) & (preclose[1:]>0))
    gross=np.full(good.shape,np.nan)
    with np.errstate(all='ignore'):
        gross[good]=(opening[1:][good]/opening[:-1][good])*(close[:-1][good]/preclose[1:][good])
    event=member[:-1] & dead[1:]
    gross[event]=0.; gross[dead[:-1]]=1.
    bad=~np.isfinite(gross) | (gross<0)
    zeros=gross==0
    logs=np.log(np.where(~bad & ~zeros,gross,1.))
    sums=np.zeros((length,opening.shape[1])); invalid=np.zeros_like(sums,dtype=bool); terminal=invalid.copy()
    for lag in range(horizon):
        sums+=logs[lag:lag+length]
        invalid|=bad[lag:lag+length]
        terminal|=zeros[lag:lag+length]
    result[:length]=np.where(terminal,-1.,np.where(invalid,np.nan,np.expm1(sums)))
    result[~member]=np.nan
    return result


def label_split_mask(dates, start, end, horizon):
    """Both the prediction date and final label open must be in the split."""
    days=np.asarray(dates,dtype='datetime64[D]')
    mask=(days>=np.datetime64(start)) & (days<=np.datetime64(end))
    if horizon < len(days):
        mask[:-horizon] &= days[horizon:] <= np.datetime64(end)
    mask[-horizon:]=False
    return mask
