"""Exact rank-buffer definitions extracted from the verified research implementation."""
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class RankBufferConfig:
    buy_n: int
    sell_m: int
    hold_thresh: int
    max_replacements: int

    def __post_init__(self):
        if any(type(x) is not int for x in (self.buy_n,self.sell_m,self.hold_thresh,self.max_replacements)):
            raise TypeError('rank buffer parameters must be integers')
        if self.buy_n<1 or self.sell_m<self.buy_n or self.hold_thresh<1 or not 0<=self.max_replacements<=self.buy_n:
            raise ValueError('invalid rank buffer parameters')

def select_rank_buffer(scores,held,buyable,sellable,holding_days,config):
    order=np.argsort(-np.where(np.isfinite(scores),scores,-np.inf),kind='stable')
    observable_buyable=buyable&np.isfinite(scores)
    pool=held|observable_buyable;ranked=order[pool[order]]
    retained_rank=np.zeros(len(scores),bool);retained_rank[ranked[:config.sell_m]]=True
    worst=ranked[::-1]
    exit_mask=held&~retained_rank&sellable&(holding_days>=config.hold_thresh)
    exits=worst[exit_mask[worst]][:config.max_replacements]
    retain=held.copy();retain[exits]=False
    head=order[observable_buyable[order]][:config.buy_n]
    new=head[~held[head]][:max(0,config.sell_m-int(retain.sum()))]
    return np.flatnonzero(retain),new,exits


def allocate_rank_buffer(scores, held, current_weights, buyable, sellable, holding_days, config):
    keep,new,exits=select_rank_buffer(scores,held,buyable,sellable,holding_days,config)
    target=np.zeros(len(scores));target[keep]=current_weights[keep]
    available=max(0.,1-float(target.sum()))
    for i in new:
        budget=min(1/config.sell_m,available)
        if budget<=1e-12:break
        target[i]=budget;available-=budget
    return target,exits
