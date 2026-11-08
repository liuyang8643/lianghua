"""Additional independently usable components from public author videos.

BV1fQqGBdEFB: price exhaustion and RSI/DIF exhaustion are expressly tested
separately in the video. The volume component needs unavailable float turnover.
Price rule: completed5 official compounded returns < -8%, last return > -3%,
lower wick >1.5*abs(body). A positive last return is allowed by that inequality.
RSI rule: RSI14<25, strict RSI5 and DIF trough yesterday. Explicit convention:
Wilder RSI seeded by the first n changes' mean, zero total movement ->50;
EMA12/26 seeded by first adjusted close; 53 consecutive bars warmup.

BV1iwG16QEvJ: '声东击西' disclosed volume component. Explicit window/tie
interpretation: most recent closing-limit-up in the last7 bars INCLUDING
signal day; its next bar sets the volume base; a later bar <half that base.
Thus eligible lags2..6; source wording leaves overlap/boundary unspecified.
Exchange limits are supplied by the caller's canonical env, not recreated.
All outputs first appear at the next OPEN; current20 full-investment ranking
is an adaptation, not the source entry/exit portfolio.
BV1XT421v7iz: completed adjusted-close MA5 crosses above MA20.
"""
from pathlib import Path
import hashlib
import numpy as np
from numba import njit
from factor.base import FactorDefinition,FactorMetadata


@njit(cache=True)
def exhaustion_events(close,opening,high,low,preclose,member):
    tmax,n=close.shape;out=np.full((3,tmax,n),-1,dtype=np.int8)
    for j in range(n):
        count=0;adj=0.;ef=es=0.;up5=dn5=up14=dn14=0.;oldma=0.
        ac=np.empty(tmax);rsi=np.empty(tmax);dif=np.empty(tmax)
        for t in range(tmax):
            c=close[t,j];o=opening[t,j];h=high[t,j];l=low[t,j];p=preclose[t,j]
            good=member[t,j] and np.isfinite(c+o+h+l+p) and min(c,o,l,p)>0 and h>=max(c,o) and l<=min(c,o)
            if not good:count=0;up5=dn5=up14=dn14=0.;continue
            prev=adj
            if count==0:adj=c;ef=es=c
            else:adj*=c/p;ef+=2./13*(adj-ef);es+=2./27*(adj-es)
            count+=1;ac[t]=adj;dif[t]=ef-es
            if count>1:
                change=adj-prev;gain=max(change,0.);loss=max(-change,0.)
                if count<=6:up5+=gain/5;dn5+=loss/5
                else:up5=(up5*4+gain)/5;dn5=(dn5*4+loss)/5
                if count<=15:up14+=gain/14;dn14+=loss/14
                else:up14=(up14*13+gain)/14;dn14=(dn14*13+loss)/14
            rsi[t]=100*up5/(up5+dn5) if up5+dn5>0 else 50.
            if t+1<tmax and count>=6:
                event=ac[t]/ac[t-5]-1<-.08 and c/p-1>-.03
                event=event and min(c,o)-l>1.5*abs(c-o)
                out[0,t+1,j]=event
            if t+1<tmax and count>=53:
                rs14=100*up14/(up14+dn14) if up14+dn14>0 else 50.
                out[1,t+1,j]=rs14<25 and rsi[t-1]<rsi[t-2] and rsi[t]>rsi[t-1] and dif[t-1]<dif[t-2] and dif[t]>dif[t-1]
            if count>=20:
                madiff=0.
                for lag in range(20):
                    madiff-=ac[t-lag]/20
                    if lag<5:madiff+=ac[t-lag]/5
                if count>=21 and t+1<tmax:out[2,t+1,j]=madiff>0 and oldma<=0
                oldma=madiff
    return out


def shrink_after_limit(close,volume,limits,member):
    valid=member & np.isfinite(close) & (close>0) & np.isfinite(volume) & (volume>=0) & np.isfinite(limits)
    hit=valid & (np.rint(close*100)>=np.rint(limits*100))
    out=np.full(close.shape,-1,dtype=np.int8)
    if len(close)<8:return out
    n=len(close)-7
    available=np.ones((n,close.shape[1]),dtype=bool)
    for lag in range(7):available&=valid[6-lag:6-lag+n]
    signal=np.zeros_like(available)
    later_hit=hit[6:6+n]|hit[5:5+n]
    for lag in range(2,7):
        event=hit[6-lag:6-lag+n]&~later_hit
        base=volume[7-lag:7-lag+n]
        signal|=event & (base>0) & (volume[6:6+n]<base*.5)
        later_hit|=hit[6-lag:6-lag+n]
    out[7:]=np.where(available,signal,-1)
    return out


_HASH=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
FIELDS=('close','open','high','low','preClose','volume','listing_age','delisted_mask')
def prepare_supplemental_definitions(runtime,limit_up_prices):
    if runtime.manifest.actual_preload_rows>=runtime.manifest.requested_preload_rows:
        raise ValueError('recursive exhaustion requires full-history preload')
    member=(runtime.field('listing_age')>=0)&~runtime.field('delisted_mask')
    first=exhaustion_events(*(runtime.field(k) for k in FIELDS[:5]),member)
    second=shrink_after_limit(runtime.field('close'),runtime.field('volume'),limit_up_prices,member)
    arrays=(first[0],first[1],first[2],second)
    for x in arrays:x.flags.writeable=False
    def make(i,name,hist):
        class Bound:
            def calc_batch(self,panel):
                if any(panel[k] is not runtime.field(k) or panel[k].flags.writeable for k in FIELDS):
                    raise ValueError('supplemental signals bound to another runtime')
                a=arrays[i].astype(np.float64);a[a<0]=np.nan;return a
        return FactorDefinition(FactorMetadata(name=name,version='bili-supplement-v1-completed-wilder-firstseed-recent7inclusive',
            hist_days=hist,required_fields=FIELDS,implementation_hash=_HASH,score_semantics='binary'),Bound,raw_runtime_view=True)
    return tuple(make(i,name,hist) for i,(name,hist) in enumerate((('BiliPriceExhaustion5',6),('BiliRSIDIFExhaustion',53),('BiliMA5Cross20',21),('BiliPostLimitVolumeHalf7',7))))
