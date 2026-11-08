"""Frozen optimized factors; decision-open causal and explicit unit contracts."""
import numpy as np
from factor_db.factors.AmountBasedSmallCap import AmountBasedSmallCap
from factor_db.factors.TrueMarketCap import TrueMarketCap as _TrueMarketCap
from factor_db.factors.VolumeCV import VolumeCV as _VolumeCV
from factor.library.bilibili import BiliAdjustedIssueDiscount


def _live_gate(panel, score):
    opening = np.asarray(panel["open"])
    valid = np.isfinite(opening) & (opening >= 2.0) & (panel["st_mask"] == 0)
    return np.where(valid, score, np.nan)


class TrueMarketCap(_TrueMarketCap):
    """Negative legacy market-cap rank; caller lags WAN-share input by one day."""

    def calc_batch(self, panel):
        return _live_gate(panel, super().calc_batch(panel))


class VolumeCV(_VolumeCV):
    """Negative completed 20-day volume coefficient of variation."""

    def calc_batch(self, panel):
        return _live_gate(panel, super().calc_batch(panel))


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




class AmountBasedSmallCap5(AmountBasedSmallCap):
    hist_days=5

    def calc_batch(self, panel):
        return _live_gate(panel, super().calc_batch(panel))

class PB3ROE5Signal:
    """Shares are lagged WAN shares; financial profit/equity are CNY."""
    hist_days=1
    def calc_batch(self,panel):
        opening,shares,profit,equity=(np.asarray(panel[n],dtype=np.float64) for n in ('open','total_share','financial_profit_ttm','financial_equity'))
        with np.errstate(all='ignore'):
            cap=opening*(shares*10000.);pb=cap/equity;roe=profit/equity
        valid=np.isfinite(opening)&(opening>0)&np.isfinite(shares)&(shares>0)&np.isfinite(cap)&(cap>0)&np.isfinite(equity)&(equity>0)&np.isfinite(profit)
        return np.where(valid,((pb<3.)&(roe>.05)).astype(np.float64),np.nan)

class ProfitGrowthOriginalAvailability:
    """Lambda0 of the recorded profit-minus-cash-outflow-growth factor.

    Both original financial inputs must be observed; this is not a wider
    universe created by silently dropping the original availability contract.
    """
    hist_days=0
    def calc_batch(self,panel):
        a=np.asarray(panel['financial_profit_yoy'],dtype=np.float64)
        b=np.asarray(panel['financial_cash_outflow_yoy'],dtype=np.float64)
        with np.errstate(invalid='ignore'):
            return np.where(np.isfinite(a)&np.isfinite(b),a-0.*b,np.nan)

class BiliLowReturnAttention10:
    hist_days=10
    def calc_batch(self,panel):
        cl,pc=(np.asarray(panel[k],dtype=np.float64) for k in ('close','preClose'))
        member=(panel['listing_age']>=0)&(panel['delisted_mask']==0)
        valid=np.isfinite(cl)&(cl>0)&np.isfinite(pc)&(pc>0)&member
        r=np.divide(cl,pc,out=np.full(cl.shape,np.nan),where=valid);r-=1
        n=valid.sum(axis=1)
        market=np.divide(np.sum(r,where=valid,axis=1,initial=0.),n,out=np.full(len(r),np.nan),where=n>0)
        r-=market[:,None];np.abs(r,out=r)
        return _negative_completed_max(r,10)

class LowResidualVol20:
    hist_days=20
    def calc_batch(self,panel):
        cl,pc=(np.asarray(panel[k],dtype=np.float64) for k in ('close','preClose'))
        member=(panel['listing_age']>=0)&(panel['delisted_mask']==0)
        valid=member&np.isfinite(cl)&(cl>0)&np.isfinite(pc)&(pc>0)
        raw=np.divide(cl,pc,out=np.full(cl.shape,np.nan),where=valid)-1
        n=valid.sum(axis=1)
        market=np.divide(np.nansum(raw,axis=1),n,out=np.full(len(raw),np.nan),where=n>0)
        residual=raw-market[:,None]
        finite=np.isfinite(residual);x=np.where(finite,residual,0.)
        sums=np.vstack((np.zeros((1,x.shape[1])),np.cumsum(x,axis=0)))
        squares=np.vstack((np.zeros((1,x.shape[1])),np.cumsum(x*x,axis=0)))
        counts=np.vstack((np.zeros((1,x.shape[1]),dtype=np.int32),np.cumsum(finite,axis=0,dtype=np.int32)))
        w=20;out=np.full(x.shape,np.nan)
        if len(x)>w:
            mean=(sums[w:-1]-sums[:-w-1])/w
            var=(squares[w:-1]-squares[:-w-1])/w-mean*mean
            out[w:]=np.where(counts[w:-1]-counts[:-w-1]==w,np.sqrt(np.maximum(var,0)),np.nan)
        return -out.astype(np.float32)
