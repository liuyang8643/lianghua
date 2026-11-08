"""Research-only completed-bar volatility/attention transitions (not registered)."""
from __future__ import annotations

import hashlib
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
from factor.base import FactorDefinition, FactorMetadata
from factor.library.bilibili_attention import BiliLowReturnAttention20


def completed_std(values, window):
    """Population std of [T-window,T); strict finite windows."""
    finite = np.isfinite(values)
    x = np.where(finite, values, 0.)
    sums = np.vstack((np.zeros((1, x.shape[1])), np.cumsum(x, axis=0)))
    squares = np.vstack((np.zeros((1, x.shape[1])), np.cumsum(x*x, axis=0)))
    counts = np.vstack((np.zeros((1, x.shape[1]), dtype=np.int32), np.cumsum(finite, axis=0, dtype=np.int32)))
    out = np.full(values.shape, np.nan)
    if len(x) > window:
        mean = (sums[window:-1]-sums[:-window-1])/window
        var = (squares[window:-1]-squares[:-window-1])/window-mean*mean
        out[window:] = np.where(counts[window:-1]-counts[:-window-1] == window, np.sqrt(np.maximum(var, 0)), np.nan)
    return out


def delay(values, days):
    out = np.full(values.shape, np.nan)
    out[days:] = values[:-days]
    return out


def percentile(values, member):
    ranks = rankdata(np.where(member, values, np.nan), axis=1, nan_policy='omit')
    counts = np.sum(np.isfinite(ranks), axis=1)[:, None]
    return np.divide(ranks-.5, counts, out=np.full(values.shape, np.nan), where=counts>0).astype(np.float32)


def candidate_specs():
    specs = [{'name': 'attention20', 'kind': 'attention'},
             {'name': 'low_raw_vol20', 'kind': 'raw_vol'},
             {'name': 'low_residual_vol20', 'kind': 'residual_vol'},
             {'name': 'prior_high_vol60', 'kind': 'prior_vol'}]
    for recent in (10, 20, 40):
        for prior in (20, 60):
            for kind in ('contraction', 'attention_contraction', 'high_to_quiet'):
                specs.append(dict(name=f'{kind}_{recent}_{prior}', kind=kind, recent=recent, prior=prior))
    specs += [dict(name='persistent_quiet20_60', kind='persistent'),
              dict(name='attention_decay20_60', kind='attention_decay'),
              dict(name='gradual_quiet20_60', kind='gradual')]
    return specs


class TransitionFeatures:
    def __init__(self, panel):
        close, pc = (np.asarray(panel[k], dtype=np.float64) for k in ('close', 'preClose'))
        self.member = (panel['listing_age'] >= 0) & (panel['delisted_mask'] == 0)
        valid = self.member & np.isfinite(close) & (close>0) & np.isfinite(pc) & (pc>0)
        raw = np.divide(close, pc, out=np.full(close.shape, np.nan), where=valid)-1
        count = valid.sum(axis=1)
        market = np.divide(np.nansum(raw, axis=1), count, out=np.full(len(raw), np.nan), where=count>0)
        self.raw_vol = completed_std(raw, 20).astype(np.float32)
        residual = raw-market[:, None]
        self.vol = {w: completed_std(residual, w).astype(np.float32) for w in (10, 20, 40, 60)}
        self.attention = BiliLowReturnAttention20().calc_batch(panel)
        self.a = percentile(self.attention, self.member)

    def calculate(self, spec):
        kind = spec['kind']
        if kind == 'attention': return self.attention
        if kind == 'raw_vol': return -self.raw_vol
        if kind == 'residual_vol': return -self.vol[20]
        old = delay(self.vol[spec.get('prior', 60)], spec.get('recent', 20))
        if kind == 'prior_vol': return old
        if kind == 'persistent': return self.a*(1-percentile(old, self.member))
        if kind == 'attention_decay':
            return self.a*percentile(-delay(self.attention, 60), self.member)
        if kind == 'gradual':
            middle, earliest = delay(self.vol[20],20), delay(self.vol[20],40)
            # Continuous reward for both consecutive declines, not merely endpoints.
            minimum_drop = np.minimum(earliest-middle, middle-self.vol[20])
            return self.a*percentile(minimum_drop, self.member)
        current = self.vol[spec['recent']]
        if kind == 'high_to_quiet':
            return self.a*percentile(old,self.member)*(1-percentile(current,self.member))
        ratio = np.divide(old, current, out=np.full(old.shape,np.nan), where=(old>0)&(current>0))
        contraction = np.log(ratio)
        if kind == 'contraction': return contraction
        if kind == 'attention_contraction': return .5*(self.a+percentile(contraction,self.member))
        raise ValueError(kind)


def array_definition(name, values):
    """Bind a calculated research array to the normal factor cache builder."""
    class BoundFactor:
        def calc_batch(self, panel):
            if panel['close'].shape != values.shape: raise ValueError('unaligned research array')
            return values
    sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return FactorDefinition(FactorMetadata(name, 'attention-transition-research-v1', 100,
                            ('close',), sha), BoundFactor)


def full_prefix_definition(path, dates, definition):
    """Preserve production cumulative arithmetic from the original date origin.

    Load only one factor's inputs at a time, then project its completed output;
    avoids retaining financial/actor panels before the episode warm-up.
    """
    with np.load(path, allow_pickle=False) as z:
        all_dates=z['trade_dates']
        lo=int(np.searchsorted(all_dates,dates[0]))
        hi=int(np.searchsorted(all_dates,dates[-1]))+1
        np.testing.assert_array_equal(all_dates[lo:hi],dates)
        panel={k:np.asarray(z[k][:hi],dtype=np.float64) for k in definition.metadata.required_fields}
    for k in definition.metadata.lagged_fields:
        panel[k]=delay(panel[k],1)
    values=np.array(definition.implementation().calc_batch(panel)[lo:hi],copy=True)
    del panel
    bound=array_definition(definition.metadata.name,values)
    # Keep original formula identity, and explicitly record full-origin projection.
    from dataclasses import replace
    metadata=replace(definition.metadata,version=definition.metadata.version+'-full-origin-projected',
                     implementation_hash=hashlib.sha256((definition.metadata.implementation_hash+bound.metadata.implementation_hash).encode()).hexdigest(),
                     required_fields=('close',),lagged_fields=())
    return FactorDefinition(metadata,bound.implementation)
