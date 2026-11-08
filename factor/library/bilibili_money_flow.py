"""Completed-close video indicators over validated historical free-source inputs.

The public Sina desktop provider's r0 is the >= CNY 1m category. Its flow is
an estimate from directional trades, not an identification of investor accounts.
"""
from __future__ import annotations
import numpy as np

NAMES=('LowFloatTurnover20_10','MainDisagreement','MainStability','MainTurnover')

def _mean20(values):
    x=np.asarray(values,dtype=np.float64)
    out=np.full(x.shape,np.nan)
    if len(x)<20:return out
    n=len(x)-19
    valid=np.ones(n,bool);total=np.zeros(n)
    for k in range(20):
        a=x[k:k+n];valid &= np.isfinite(a);total += np.where(np.isfinite(a),a,0.)
    out[19:]=np.where(valid,total/20,np.nan)
    return out

def completed_close_stock_scores(volume,close,float_a_shares,share_pit_valid,main_net,main_gross):
    """Return four higher-is-preferred rows, including the current closed day.

    Source adapter owns alignment and rejection of provider placeholder zeros.
    Volume is canonical WBR lots (100 shares). Unknown shares/flows stay NaN.
    Caller uses these at next-month open, never at the same day's open.
    """
    arrays=[np.asarray(x) for x in (volume,close,float_a_shares,share_pit_valid,main_net,main_gross)]
    if any(x.ndim!=1 or x.shape!=arrays[0].shape for x in arrays):
        raise ValueError('Inputs must share one market-date axis')
    vol,price,shares,pit,net,gross=arrays
    if pit.dtype!=np.bool_:raise TypeError('share_pit_valid must be boolean')
    share_valid=pit&np.isfinite(shares)&(shares>0)
    valid=share_valid&np.isfinite(price)&(price>0)
    with np.errstate(over='ignore',invalid='ignore'):
        cap=price*shares
    cap=np.where(valid&np.isfinite(cap)&(cap>0),cap,np.nan)
    turnover=np.divide(vol*100.,shares,out=np.full(vol.shape,np.nan),where=share_valid&np.isfinite(vol)&(vol>=0))
    fraction=np.divide(net,cap,out=np.full(vol.shape,np.nan),where=valid&np.isfinite(net))
    gross_fraction=np.divide(gross,cap,out=np.full(vol.shape,np.nan),where=valid&np.isfinite(gross)&(gross>=0))
    mean=_mean20(fraction)
    # Explicit population std. All eligible stocks use exactly 20 observations.
    variance=np.zeros(max(0,len(vol)-19));finite=np.ones(len(variance),bool)
    for k in range(20):
        a=fraction[k:k+len(variance)];finite &= np.isfinite(a)
        variance += (np.where(np.isfinite(a),a,0)-np.nan_to_num(mean[19:]))**2/20
    sd=np.full(vol.shape,np.nan)
    if len(vol)>=20:sd[19:]=np.where(finite,np.sqrt(variance),np.nan)
    return np.vstack((-_mean20(turnover),-_mean20(np.abs(fraction)),-sd,-_mean20(gross_fraction)))
