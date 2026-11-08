"""Public MACD/KDJ video rules, completed-bar research adaptations.

Sources: BV1GaQJY5Evz (12/26/9 and 14/53/5 MACD), BV1MLf8YHEdE
(three bullish MACD/KDJ definitions), BV13T411c7yC (KDJ and D<20),
BV1M64y1Y7Pt (30-day close pivot/DIF divergence). BV1Ro4y1972t
duplicates basic MACD. These binary scores are NOT the original variable
position entry/exit portfolios: the current full-investment env owns trades.

Explicit conventions where the public video does not specify initialization:
causal adjusted OHLC from official close/preClose; EMA first valid price seed,
alpha=2/(span+1), reset on unavailable bar, 53 completed bars warmup for all
signals; RSV window9 includes its day, K/D start50, alpha1/3; flat9-day range
has RSV50. Video94 explicitly displays D using previous K; its two standalone
KDJ signals follow that formula. Video47 displays current K for D, followed
by its three resonance signals. All crosses are > now and <= previously. Pivot d uses d+1 close
for confirmation and may first signal at d+2 OPEN. The comparison pivot is
the preceding confirmed 30-day extreme, not the last successful divergence.
"""
from __future__ import annotations
import hashlib
from pathlib import Path
import numpy as np
from numba import njit
from factor.base import FactorDefinition,FactorMetadata

NAMES=('BiliMACDGoldenCross','BiliMACDPositiveAxisCross','BiliMACD14535FirstBar',
       'BiliKDJGoldenCross','BiliKDJOversoldCross','BiliMACDKDJHistogramResonance',
       'BiliMACDKDJLinesResonance','BiliMACDKDJAllPositiveResonance','BiliMACDBottomDivergence30')


@njit(cache=True)
def technical_event_cube(close,high,low,preclose,member):
    """Int8 [signal,date,stock]: -1 unavailable, 0 false, 1 true.

    Numba loops implement recursive indicator state; there are no Python
    per-stock/pandas loops, no centered rolling output and no future fills.
    """
    tmax,n=close.shape
    out=np.full((9,tmax,n),-1,dtype=np.int8)
    for stock in range(n):
        ac=np.empty(tmax);ah=np.empty(tmax);al=np.empty(tmax);difhist=np.empty(tmax);mh=np.empty(tmax)
        count=0;adj=1.;ef=0.;es=0.;dea=0.;efm=0.;esm=0.;deam=0.;k=50.;dd=50.;ddlag=50.
        oldhist=0.;oldmoddiff=0.;oldkd=0.;oldkdlag=0.;lastlow=np.nan;lastdif=np.nan
        for day in range(tmax):
            valid=member[day,stock] and np.isfinite(close[day,stock]) and close[day,stock]>0
            valid=valid and np.isfinite(preclose[day,stock]) and preclose[day,stock]>0
            valid=valid and np.isfinite(high[day,stock]) and np.isfinite(low[day,stock]) and low[day,stock]>0
            valid=valid and high[day,stock]>=close[day,stock] and low[day,stock]<=close[day,stock]
            if not valid:
                count=0;adj=1.;k=50.;dd=50.;ddlag=50.;lastlow=np.nan;lastdif=np.nan
                ac[day]=np.nan;ah[day]=np.nan;al[day]=np.nan;difhist[day]=np.nan;mh[day]=np.nan
                continue
            if count==0:
                adj=close[day,stock];ef=adj;es=adj;efm=adj;esm=adj;dea=0.;deam=0.
            else:
                adj*=close[day,stock]/preclose[day,stock]
                ef+=2./13*(adj-ef);es+=2./27*(adj-es)
                efm+=2./15*(adj-efm);esm+=2./54*(adj-esm)
            count+=1;ac[day]=adj;ah[day]=adj*high[day,stock]/close[day,stock];al[day]=adj*low[day,stock]/close[day,stock]
            dif=ef-es;dea+=2./10*(dif-dea);hist=2*(dif-dea);difhist[day]=dif
            dm=efm-esm;deam+=2./6*(dm-deam);mh[day]=2*(dm-deam)
            modd=0.
            if count>=5:
                mean=0.
                for lag in range(5):mean+=mh[day-lag]/5.
                modd=mh[day]-mean
            kd=0.;kdlag=0.
            if count>=9:
                hh=ah[day];ll=al[day]
                for lag in range(1,9):hh=max(hh,ah[day-lag]);ll=min(ll,al[day-lag])
                rsv=100*(adj-ll)/(hh-ll) if hh>ll else 50.
                ddlag=2./3*ddlag+k/3
                k=2./3*k+rsv/3;dd=2./3*dd+k/3;kd=k-dd;kdlag=k-ddlag
            divergence=False
            # day is the right-hand confirmation bar; pivot=day-1.
            if count>=32:
                pivot=day-1;extreme=ac[pivot]<ac[pivot-1] and ac[pivot]<ac[day]
                for lag in range(1,31):extreme=extreme and ac[pivot]<ac[pivot-lag]
                if extreme:
                    divergence=np.isfinite(lastlow) and ac[pivot]<lastlow and difhist[pivot]>lastdif
                    lastlow=ac[pivot];lastdif=difhist[pivot]
            if count>=53 and day+1<tmax:
                cross=hist>0 and oldhist<=0;kgold=kd>0 and oldkd<=0;klag=kdlag>0 and oldkdlag<=0
                out[0,day+1,stock]=cross
                out[1,day+1,stock]=cross and dif>0 and dea>0
                out[2,day+1,stock]=modd>0 and oldmoddiff<=0
                out[3,day+1,stock]=klag
                out[4,day+1,stock]=klag and ddlag<20
                out[5,day+1,stock]=kgold and hist>0
                out[6,day+1,stock]=kgold and dif>0 and dea>0
                out[7,day+1,stock]=kgold and hist>0 and dif>0 and dea>0
                out[8,day+1,stock]=divergence
            oldhist=hist;oldmoddiff=modd;oldkd=kd;oldkdlag=kdlag
    return out


_HASH=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
FIELDS=('close','high','low','preClose','listing_age','delisted_mask')
def prepare_technical_definitions(runtime):
    """Bind one causal calculation to the caller's immutable full history."""
    if runtime.manifest.actual_preload_rows>=runtime.manifest.requested_preload_rows:
        raise ValueError('recursive indicators require full-history preload')
    cube=technical_event_cube(*(runtime.field(k) for k in FIELDS[:4]),
                              (runtime.field('listing_age')>=0)&~runtime.field('delisted_mask'))
    cube.flags.writeable=False
    def definition(index,name):
        class BoundTechnical:
            def calc_batch(self,panel):
                if any(panel[k] is not runtime.field(k) or panel[k].flags.writeable for k in FIELDS):
                    raise ValueError('technical signal bound to another runtime')
                values=cube[index].astype(np.float64)
                values[values<0]=np.nan
                return values
        return FactorDefinition(FactorMetadata(name=name,version='bili-technical-v1-causal53-kdj9-Dlag94-Dcurrent47-pivot-confirmed',
            hist_days=53,required_fields=FIELDS,implementation_hash=_HASH,score_semantics='binary'),BoundTechnical,raw_runtime_view=True)
    return tuple(definition(i,name) for i,name in enumerate(NAMES))
