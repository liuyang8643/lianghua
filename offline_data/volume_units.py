"""Resolve mixed historical share/lot volume from the observed daily bar.

The legacy store includes both share counts and 100-share lot counts. A
completed bar's amount/volume must imply a VWAP inside its low/high envelope.
Unknown or ambiguous units remain missing; they are never inferred from a
later day. Callers must still respect the normal one-day information lag.
"""
import numpy as np

VOLUME_UNIT_VERSION='completed-bar-volume-unit-resolution-v1'


def canonical_volume(panel, *, shares=True):
    volume,amount,low,high=(np.asarray(panel[k],dtype=np.float64) for k in ('volume','amount','low','high'))
    if any(x.shape!=volume.shape for x in (amount,low,high)):
        raise ValueError('volume-unit input shapes differ')
    good=np.isfinite(volume)&(volume>0)&np.isfinite(amount)&(amount>0)&np.isfinite(low)&(low>0)&np.isfinite(high)&(high>=low)
    ratio=np.divide(amount,volume,out=np.full(volume.shape,np.nan),where=good)
    # Two-percent slack tolerates vendor rounding, while the unit hypotheses
    # differ by 100 times. An overlap between them is explicitly ambiguous.
    lots=good&(ratio>=low*98)&(ratio<=high*102)
    units=good&(ratio>=low*.98)&(ratio<=high*1.02)
    ambiguous=lots&units;lots&=~ambiguous;units&=~ambiguous
    result=np.full(volume.shape,np.nan)
    if shares:
        np.multiply(volume,100.,out=result,where=lots)
        np.copyto(result,volume,where=units)
    else:
        np.copyto(result,volume,where=lots)
        np.divide(volume,100.,out=result,where=units)
    result[(volume==0)&((amount==0)|~np.isfinite(amount))]=0.
    return result
