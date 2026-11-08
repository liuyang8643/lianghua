"""Local deep-value formulas using explicit strictly-announced inputs."""
import numpy as np
from scipy.stats import rankdata


def _pct_rank(values):
    ranks=rankdata(values,axis=1,method='average',nan_policy='omit')
    counts=np.isfinite(values).sum(axis=1)[:,None]
    return np.divide(ranks,counts,out=np.full_like(ranks,np.nan),where=counts>0).astype(np.float32)


NAMES=('BookToMarket','EarningsYield','CashFlowYield','ROEQuality','ProfitGrowth','DeepValueComposite','BookToMarketQuality')


def calculate(name,panel):
    opening=panel['open'];base=np.isfinite(opening)&(opening>=2)&~panel['st_mask']&(panel['listing_age']>=0)&~panel['delisted_mask']
    bps,eps,cf,roe,growth=(panel['pit_'+field] for field in ('bps','eps','operating_cf_ps','roe','profit_yoy'))
    with np.errstate(all='ignore'):
        if name=='BookToMarket':value=bps/opening;valid=base&np.isfinite(bps)&(bps>0)
        elif name=='EarningsYield':value=eps/opening;valid=base&np.isfinite(eps)
        elif name=='CashFlowYield':value=cf/opening;valid=base&np.isfinite(cf)
        elif name=='ROEQuality':value=roe;valid=base&np.isfinite(roe)
        elif name=='ProfitGrowth':value=growth;valid=base&np.isfinite(growth)
        elif name=='BookToMarketQuality':value=bps/opening;valid=base&np.isfinite(bps)&(bps>0)&np.isfinite(eps)&(eps>0)
        elif name=='DeepValueComposite':
            branches=[np.where(base&np.isfinite(x),x,np.nan) for x in (bps/opening,eps/opening,cf/opening)]
            value=np.nanmean(np.stack([_pct_rank(x) for x in branches]),axis=0);valid=base&np.isfinite(value)
        else:raise ValueError(name)
    return np.where(valid,value,np.nan)


for _name in NAMES:
    def _calc(self,panel,name=_name):return calculate(name,panel)
    globals()[_name]=type(_name,(),{'hist_days':0,'calc_batch':_calc,'__module__':__name__})
