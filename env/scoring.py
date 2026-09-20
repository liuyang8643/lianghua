"""Single signed-rank scoring law for selection and the T-1 prefilter."""

from typing import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

import numpy as np
from numba import njit, literal_unroll

from env.contracts import DayConfig


@njit(cache=True,fastmath=False,parallel=False)
def _score_rows(factors,size):
    result=np.zeros(size,np.float64)
    centered=np.zeros(size,np.float64)
    available=np.zeros(size,np.float64)
    complete=np.ones(size,np.bool_)
    observed=np.zeros(size,np.bool_)
    total=0.0;signed=0.0;invalid=False
    for row in literal_unroll(factors):
        weight=row[2];total+=abs(weight);signed+=weight
        rank,mask=row[0],row[1]
        for stock in range(size):
            if mask[stock]:
                value=np.float64(rank[stock])
                invalid |= not np.isfinite(value)
                result[stock]+=value*weight
                centered[stock]+=(value-0.5)*weight
                available[stock]+=abs(weight)
                observed[stock] |= weight!=0.0
            else:
                complete[stock]=False
    minimum=np.inf;any_observed=False
    for stock in range(size):
        if observed[stock]:
            if not complete[stock]:
                result[stock]=centered[stock]/available[stock]*total+0.5*signed
            any_observed=True
            minimum=np.nan if np.isnan(result[stock]) else min(minimum,result[stock])
    if invalid:raise ValueError('valid factor ranks must be finite')
    tail=minimum-max(total,1.0) if any_observed else 0.0
    for stock in range(size):
        if not observed[stock]:result[stock]=tail
    return result

@dataclass(frozen=True)
class FactorScoreRow:
    """Validated row layout reusable while a sealed market snapshot is alive.

    This caches input layout only, never a policy's scores. Valid rank values
    are still checked in the single numeric law on every evaluation.
    """
    factor_ranks: Mapping[str, np.ndarray]
    factor_validity: Mapping[str, np.ndarray]
    stock_count: int
    _positions: Mapping[str, int] = field(init=False, repr=False)
    _ranks: tuple[np.ndarray, ...] = field(init=False, repr=False)
    _masks: tuple[np.ndarray, ...] = field(init=False, repr=False)

    def __post_init__(self):
        if set(self.factor_ranks) != set(self.factor_validity):
            raise ValueError("factor inputs do not match DayConfig")
        names = tuple(self.factor_ranks)
        ranks = [np.asarray(self.factor_ranks[name]) for name in names]
        masks = [np.asarray(self.factor_validity[name], dtype=np.bool_) for name in names]
        if any(row.shape != (self.stock_count,) for row in (*ranks, *masks)):
            raise ValueError("factor rows must match the stock axis")
        if ranks and (any(row.dtype != ranks[0].dtype for row in ranks) or ranks[0].dtype not in (
            np.dtype('float32'), np.dtype('float64'),
        )):
            ranks = [np.asarray(row, dtype=np.float64) for row in ranks]
        def readonly_contiguous(row):
            value = np.ascontiguousarray(row)
            if value.flags.writeable:
                value = value.view()
                value.flags.writeable = False
            return value
        ranks = tuple(readonly_contiguous(row) for row in ranks)
        masks = tuple(readonly_contiguous(row) for row in masks)
        object.__setattr__(self, '_positions', MappingProxyType({name: index for index, name in enumerate(names)}))
        object.__setattr__(self, '_ranks', ranks)
        object.__setattr__(self, '_masks', masks)
        object.__setattr__(self, 'factor_ranks', MappingProxyType(dict(zip(names, ranks))))
        object.__setattr__(self, 'factor_validity', MappingProxyType(dict(zip(names, masks))))

    def score(self, config: DayConfig) -> np.ndarray:
        if any(name not in self._positions for name in config.factor_weights):
            raise ValueError("factor inputs do not match DayConfig")
        names = tuple(name for name, enabled in config.factor_enabled.items() if enabled)
        if not names:
            return np.zeros(self.stock_count, dtype=np.float64)
        positions = tuple(self._positions[name] for name in names)
        return _score_rows(
            tuple((self._ranks[index], self._masks[index], float(config.factor_weights[name]))
                  for name, index in zip(names, positions)),
            self.stock_count,
        )


def score_factor_ranks(
    factor_ranks: Mapping[str, np.ndarray],
    factor_validity: Mapping[str, np.ndarray],
    config: DayConfig,
    stock_count: int,
) -> np.ndarray:
    """The same score law for callers without a sealed reusable market row."""
    return FactorScoreRow(factor_ranks, factor_validity, stock_count).score(config)
