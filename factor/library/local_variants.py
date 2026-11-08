"""All 175 previously researched parameter variants, independently selectable.

Formulae originate in the frozen 2026-11-04 study. PB uses corrected WAN-share
units. Selection belongs to the caller's training period, never this catalogue.
"""
import json,hashlib
from pathlib import Path
import numpy as np
from factor.base import FactorDefinition,FactorMetadata
from offline_data.volume_units import canonical_volume
from factor_db.factors.VolumeCV import VolumeCV
from factor_db.factors.AmountBasedSmallCap import AmountBasedSmallCap
from factor.library.styles import CompletedAmihudIlliquidity20
from factor.library.bilibili_attention import _negative_completed_max
from factor.library.attention_transition_research import completed_std,delay,percentile

BASE_PARAMS={
 'VolumeCV':{'window':20},'AmountBasedSmallCap':{'window':60},'CompletedAmihudIlliquidity20':{'window':20},
 'BiliLowReturnAttention20':{'window':20},'low_residual_vol20':{'window':20},
 'persistent_quiet20_60':{'recent':20,'prior':60},
 'BiliLowAmountAttention20':{'recent':20,'history':252},
 'attention_contraction_10_60':{'attention':20,'recent':10,'prior':60,'weight':.5},
 'PBBelowTwoROEAbove10Signal':{'pb':2.,'roe':.1},
 'LowCashOutflowProfitGrowthSpread':{'penalty':1.},'HighOperatingProfitRevenueGrowthSpread':{'penalty':1.}}

def candidates():
    rows=[]
    for name in list(BASE_PARAMS)[:5]:
        for w in (5,10,20,30,60,120):rows.append((name,{'window':w}))
    for recent in (10,20,30,60):
        for prior in (20,60,120):rows.append(('persistent_quiet20_60',dict(recent=recent,prior=prior)))
    for recent in (10,20,30,60):
        for history in (60,120,252):rows.append(('BiliLowAmountAttention20',dict(recent=recent,history=history)))
    for attention in (10,20,30):
        for recent in (5,10,20):
            for prior in (20,60,120):
                for weight in (.25,.5,.75):rows.append(('attention_contraction_10_60',dict(attention=attention,recent=recent,prior=prior,weight=weight)))
    for pb in (1.,1.5,2.,2.5,3.,4.):
        for roe in (.05,.1,.15,.2,.25):rows.append(('PBBelowTwoROEAbove10Signal',dict(pb=pb,roe=roe)))
    for name in ('LowCashOutflowProfitGrowthSpread','HighOperatingProfitRevenueGrowthSpread'):
        for penalty in (0.,.5,1.,1.5,2.):rows.append((name,dict(penalty=penalty)))
    assert len(rows)==175
    return rows

class Inputs:
    def __init__(self,panel,cache_dir=None):
        self.fields=panel;self.cache=Path(cache_dir) if cache_dir is not None else None
        self.memory={}
        if self.cache is not None:self.cache.mkdir(parents=True,exist_ok=True)
        self.member=(self.field('listing_age')>=0)&~self.field('delisted_mask')
    def field(self,k):
        if k=='volume':return self.cached('canonical_volume_lots',lambda:canonical_volume(self.fields,shares=False))
        return self.fields[k]
    def cached(self,key,calc):
        if self.cache is None:
            if key not in self.memory:self.memory[key]=calc()
            return self.memory[key]
        path=self.cache/(key+'.npy')
        if not path.exists():np.save(path,calc(),allow_pickle=False)
        return np.load(path,mmap_mode='r')
    def attention_abs(self):
        def calc():
            cl=np.asarray(self.field('close'),dtype=float);pc=np.asarray(self.field('preClose'),dtype=float)
            valid=np.isfinite(cl)&(cl>0)&np.isfinite(pc)&(pc>0)&self.member
            r=np.divide(cl,pc,out=np.full(cl.shape,np.nan),where=valid);r-=1
            n=valid.sum(axis=1);m=np.divide(np.sum(r,where=valid,axis=1,initial=0.),n,out=np.full(len(r),np.nan),where=n>0)
            r-=m[:,None];np.abs(r,out=r);return r
        return self.cached('attention_abs',calc)
    def residual(self):
        def calc():
            cl=np.asarray(self.field('close'),dtype=float);pc=np.asarray(self.field('preClose'),dtype=float)
            valid=self.member&np.isfinite(cl)&(cl>0)&np.isfinite(pc)&(pc>0)
            r=np.divide(cl,pc,out=np.full(cl.shape,np.nan),where=valid)-1
            n=valid.sum(axis=1);m=np.divide(np.nansum(r,axis=1),n,out=np.full(len(r),np.nan),where=n>0)
            return r-m[:,None]
        return self.cached('transition_residual',calc)
    def attention(self,w):return self.cached('attention_'+str(w),lambda:_negative_completed_max(self.attention_abs(),w))
    def vol(self,w):return self.cached('vol_'+str(w),lambda:completed_std(self.residual(),w).astype(np.float32))
    def attention_rank(self,w):return self.cached('att_rank_'+str(w),lambda:percentile(self.attention(w),self.member))
    def old_vol_rank(self,w,lag):return self.cached(f'old_vol_rank_{w}_{lag}',lambda:percentile(delay(self.vol(w),lag),self.member))
    def contraction_rank(self,recent,prior):
        def calc():
            old=delay(self.vol(prior),recent);current=self.vol(recent)
            ratio=np.divide(old,current,out=np.full(old.shape,np.nan),where=(old>0)&(current>0))
            return percentile(np.log(ratio),self.member)
        return self.cached(f'contraction_rank_{recent}_{prior}',calc)
    def amount_ratio(self,history):
        def calc():
            # Same transposed cumulative order and strict validity as the original.
            amount=np.asarray(self.field('amount'),dtype=float);x=np.array(amount.T,order='C',copy=True)
            valid=np.isfinite(x)&(x>=0);sums=np.empty((x.shape[0],x.shape[1]+1));sums[:,0]=0;x[~valid]=0
            np.cumsum(x,axis=1,out=sums[:,1:]);counts=np.empty(sums.shape,dtype=np.int32);counts[:,0]=0
            np.cumsum(valid,axis=1,dtype=np.int32,out=counts[:,1:]);daily=np.full(x.shape,np.nan)
            den=(sums[:,history:-1]-sums[:,:-history-1])/history
            good=(counts[:,history:-1]-counts[:,:-history-1]==history)&valid[:,history:]&(den>0)
            np.divide(x[:,history:],den,out=daily[:,history:],where=good)
            return daily.T
        return self.cached('amount_ratio_'+str(history),calc)
    def pb_roe(self):
        def calc():
            opening=np.asarray(self.field('open'),dtype=float);shares=delay(np.asarray(self.field('total_share'),dtype=float),1)
            profit=np.asarray(self.field('financial_profit_ttm'),dtype=float);equity=np.asarray(self.field('financial_equity'),dtype=float)
            with np.errstate(all='ignore'):cap=opening*(shares*10000.);pb=cap/equity;roe=profit/equity
            valid=np.isfinite(opening)&(opening>0)&np.isfinite(shares)&(shares>0)&np.isfinite(cap)&(cap>0)&np.isfinite(equity)&(equity>0)&np.isfinite(profit)
            return np.stack([np.where(valid,pb,np.nan),np.where(valid,roe,np.nan)])
        return self.cached('pb_roe',calc)
    def score(self,name,p):
        if name in ('VolumeCV','AmountBasedSmallCap','CompletedAmihudIlliquidity20'):
            cls={'VolumeCV':VolumeCV,'AmountBasedSmallCap':AmountBasedSmallCap,'CompletedAmihudIlliquidity20':CompletedAmihudIlliquidity20}[name]
            f=cls();f.hist_days=p['window']
            fields={'VolumeCV':('volume','open','st_mask'),'AmountBasedSmallCap':('amount','open','st_mask'),'CompletedAmihudIlliquidity20':('close','preClose','amount','open','st_mask')}[name]
            return f.calc_batch({k:np.asarray(self.field(k),dtype=float) for k in fields})
        if name=='BiliLowReturnAttention20':return self.attention(p['window'])
        if name=='low_residual_vol20':return -self.vol(p['window'])
        if name=='persistent_quiet20_60':return self.attention_rank(p['recent'])*(1-self.old_vol_rank(p['prior'],p['recent']))
        if name=='BiliLowAmountAttention20':return _negative_completed_max(self.amount_ratio(p['history']),p['recent'])
        if name=='attention_contraction_10_60':
            a=self.attention_rank(p['attention']);c=self.contraction_rank(p['recent'],p['prior'])
            return p['weight']*a+(1-p['weight'])*c
        if name=='PBBelowTwoROEAbove10Signal':
            x=self.pb_roe();return np.where(~np.isnan(x[0])&~np.isnan(x[1]),((x[0]<p['pb'])&(x[1]>p['roe'])).astype(float),np.nan)
        left,right={'LowCashOutflowProfitGrowthSpread':('financial_profit_yoy','financial_cash_outflow_yoy'),'HighOperatingProfitRevenueGrowthSpread':('financial_operating_profit_yoy','financial_revenue_yoy')}[name]
        a=np.asarray(self.field(left),dtype=float);b=np.asarray(self.field(right),dtype=float)
        with np.errstate(all='ignore'):score=a-p['penalty']*b
        return np.where(np.isfinite(a)&np.isfinite(b)&np.isfinite(score),score,np.nan)


def variant_name(name,parameters):
    return name+'__'+ '__'.join(f'{k}_{v:g}' for k,v in sorted(parameters.items()))


def definitions():
    source=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    common=('open','high','low','st_mask','listing_age','delisted_mask','close','preClose','volume','amount',
            'total_share','financial_profit_ttm','financial_equity','financial_profit_yoy',
            'financial_cash_outflow_yoy','financial_operating_profit_yoy','financial_revenue_yoy')
    result=[]
    for name,parameters in candidates():
        def calc(self,panel,name=name,parameters=parameters):return Inputs(panel).score(name,parameters)
        title=variant_name(name,parameters)
        cls=type(title,(),{'calc_batch':calc})
        identity=hashlib.sha256((source+json.dumps([name,parameters],sort_keys=True)).encode()).hexdigest()
        result.append(FactorDefinition(FactorMetadata(title,'local-parameter-catalog-v2-correct-shares',400,common,identity),cls))
    return tuple(result)
