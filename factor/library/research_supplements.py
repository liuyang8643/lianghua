"""Industry, implemented dividends, financial ratios and monthly source signals.

Source IO is supplied by the caller. Calculators have no hidden snapshot paths.
All event dates become usable strictly after their recorded availability date.
"""
from __future__ import annotations
import hashlib
from pathlib import Path
import numpy as np
from factor.base import FactorDefinition,FactorMetadata
from factor_db.factors.IndustryPeerReversalStrict import IndustryPeerReversal20Strict,IndustryPeerReversal60Strict,IndustryPeerReversal120Strict


def ratio(a,b):
    return np.divide(a,b,out=np.full(np.broadcast_shapes(a.shape,b.shape),np.nan),where=np.isfinite(a)&np.isfinite(b)&(b>0))


class HighInventoryOtherPayable:
    hist_days=0
    def calc_batch(self,p):return ratio(p['financial_inventory'],p['financial_other_payable'])


class HighReceivableOtherPayable:
    hist_days=0
    def calc_batch(self,p):return ratio(p['financial_receivable'],p['financial_other_payable'])


class HighOperatingCashSurplus:
    hist_days=0
    def calc_batch(self,p):
        surplus=p['financial_operating_cash_flow']-p['financial_cash_taxes_payable']-p['financial_cash_other_payable']-p['financial_cash_other_current_liability']
        return ratio(surplus,p['financial_sales_cash'])


def cash_dividend_scores(panel,events):
    """Preserve the local 252/504-row yields and three annual occupancy slots.

    Shares in event rows are shares; runtime total_share is lagged WAN shares.
    Unlike the historical restricted-pool implementation, all declared local
    SH/SZ stocks are covered. Missing share values remain missing yields.
    """
    dates=np.asarray(panel['trade_dates'],dtype='datetime64[D]');codes=tuple(panel['stock_codes']);lookup={c:i for i,c in enumerate(codes)}
    event=events[events['stock_code'].isin(lookup)].copy()
    col=event['stock_code'].map(lookup).to_numpy(dtype=np.intp)
    known=event['known_date'].to_numpy(dtype='datetime64[D]');ex=event['ex_date'].to_numpy(dtype='datetime64[D]')
    at=np.searchsorted(dates,known,side='right');valid=at<len(dates)
    cash=np.zeros((len(dates),len(codes)),dtype=np.float64)
    np.add.at(cash,(at[valid],col[valid]),event['total_cash_yuan'].to_numpy(dtype=float)[valid])
    cumulative=np.cumsum(cash,axis=0)
    shares=np.full(cash.shape,np.nan);shares[1:]=panel['total_share'][:-1]
    cap=panel['preClose']*shares*10000.
    result={}
    for window in (252,504):
        trailing=cumulative.copy();trailing[window:]-=cumulative[:-window]
        score=ratio(trailing,cap)
        score[dates-np.timedelta64(window*2,'D')<np.datetime64('2010-01-01')]=np.nan
        result[f'CompletedDividendYield{window}PIT']=score
    counts=np.zeros(cash.shape,dtype=np.int8)
    for slot in range(3):
        first=np.maximum(at,np.searchsorted(dates,ex+np.timedelta64(365*slot,'D'),side='right'))
        stop=np.searchsorted(dates,ex+np.timedelta64(365*(slot+1),'D'),side='right')
        keep=(first<stop)&(first<len(dates))
        delta=np.zeros((len(dates)+1,len(codes)),dtype=np.int16)
        np.add.at(delta,(first[keep],col[keep]),1);np.add.at(delta,(stop[keep],col[keep]),-1)
        counts+=(np.cumsum(delta[:-1],axis=0)>0).astype(np.int8)
    score=counts.astype(float)/3
    score[dates-np.timedelta64(1095,'D')<np.datetime64('2010-01-01')]=np.nan
    result['CompletedDividendConsistency3YPIT']=score
    return result


def monthly_pulse(dates,codes,payload,name):
    """A month-end observation is valid only on its first following open."""
    dates=np.asarray(dates,dtype='datetime64[D]');codes=tuple(codes)
    output=np.full((len(dates),len(codes)),np.nan)
    old_codes=tuple(payload['stock_codes'].astype(str));index={c:i for i,c in enumerate(codes)}
    columns=np.asarray([index.get(c,-1) for c in old_codes]);valid_cols=columns>=0
    names=list(payload['names'].astype(str));values=payload['scores'][names.index(name)]
    at=np.searchsorted(dates,payload['signal_dates'],side='right')
    for source_row,destination in enumerate(at):
        if destination<len(dates) and payload['signal_dates'][source_row]>=dates[0]:
            output[destination,columns[valid_cols]]=np.where(payload['eligible'][source_row,valid_cols],values[source_row,valid_cols],np.nan)
    return output


def definitions():
    digest=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    entries=[(IndustryPeerReversal20Strict,('close','preClose','industry_id')),
             (IndustryPeerReversal60Strict,('close','preClose','industry_id')),
             (IndustryPeerReversal120Strict,('close','preClose','industry_id')),
             (HighInventoryOtherPayable,('financial_inventory','financial_other_payable')),
             (HighReceivableOtherPayable,('financial_receivable','financial_other_payable')),
             (HighOperatingCashSurplus,('financial_operating_cash_flow','financial_cash_taxes_payable','financial_cash_other_payable','financial_cash_other_current_liability','financial_sales_cash'))]
    return tuple(FactorDefinition(FactorMetadata(cls.__name__,'strict-pit-supplement-v1',cls.hist_days,fields,digest),cls) for cls,fields in entries)
