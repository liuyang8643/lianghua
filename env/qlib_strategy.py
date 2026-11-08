"""The frozen rank-buffer policy on the canonical account/market context.

Research origins include Microsoft Qlib (MIT); see factor/library/QLIB_LICENSE.txt.
Selection and allocation have one implementation in strategy_release.selection.
"""
import numpy as np
from env.planner import mark_open_positions
from env.legality import investment_universe_mask
from strategy_release.selection import RankBufferConfig,select_rank_buffer,allocate_rank_buffer


class _PortfolioState:
    def __init__(self,codes):
        self.codes=tuple(codes);self.lookup={c:i for i,c in enumerate(codes)};self.born={};self.last_day=None
        self.diagnostics=[]

    def context(self,market,account,day,scores):
        if self.last_day is not None and day<=self.last_day:self.born={};self.diagnostics=[]
        self.last_day=day
        held=np.zeros(len(self.codes),bool);sellable=np.zeros(len(self.codes),bool);ages=np.zeros(len(self.codes),int)
        legal=market.trade_legality(True)
        prices,marks,_,values,equity=mark_open_positions(market,account,self.lookup)
        weights=np.zeros(len(self.codes));locked=np.zeros(len(self.codes))
        self.born={code:value for code,value in self.born.items() if code in values}
        for code,value in values.items():
            i=self.lookup[code];held[i]=True;weights[i]=value/equity
            self.born.setdefault(code,day-1);ages[i]=day-self.born[code]
            quantity=account.positions[code];available=min(quantity,account.sellable_positions.get(code,0))
            sellable[i]=bool(legal.sell_allowed[i] and available>=quantity)
            locked[i]=(quantity-available)*marks[code]/equity if legal.sell_allowed[i] else weights[i]
        eligible=investment_universe_mask(market.open_prices,market.listing_age,market.st_mask,market.delisted_mask)&np.isfinite(scores)
        buyable=eligible&legal.buy_allowed
        return held,sellable,ages,weights,locked,buyable,equity


class RankBufferPortfolio(_PortfolioState):
    def __init__(self,codes,config):super().__init__(codes);self.config=config

    def targets(self,market,account,day,scores):
        if not np.isfinite(scores).any():raise ValueError('no observable model prediction at this decision date')
        held,sellable,ages,current,locked,buyable,equity=self.context(market,account,day,scores)
        weights,exits=allocate_rank_buffer(scores,held,current,buyable,sellable,ages,self.config)
        admitted=np.flatnonzero((weights>0)&~held)
        self.diagnostics.append({'day':int(day),'entries':len(admitted),'exits':len(exits),
            'target_positions':int(np.count_nonzero(weights)),'target_cash_weight':max(0.,1-float(weights.sum())),
            'entry_weight':1./self.config.sell_m,'max_replacements':self.config.max_replacements,'cash_sweep':False})
        return {self.codes[i]:float(weights[i]) for i in np.flatnonzero(weights>0)}
