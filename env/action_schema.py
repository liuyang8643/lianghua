"""One DayConfig codec for static, GA, PPO and live decisions."""
from __future__ import annotations
from dataclasses import dataclass, field
from functools import cached_property
import hashlib
import json
import math
from typing import Literal, Mapping, Sequence
import numpy as np
from numpy.typing import NDArray
from env.contracts import DayConfig, PolicyHistory, PolicyMemory, decode_unit_action, encode_unit_action
from factor.registry import PRODUCTION_FACTOR_NAMES, PRODUCTION_FILTER_NAMES

CORE_FACTOR_NAMES = PRODUCTION_FACTOR_NAMES
CORE_FILTER_NAMES = PRODUCTION_FILTER_NAMES
FIXED_BUY_N = 20
FIXED_SELL_M = 25
GA_SELL_M_VALUES = tuple(range(20, 101, 5))
SERIALIZED_DAY_CONFIG_FIELDS = ('weights', 'factor_enabled', 'filter_factors', 'buy_n',
    'sell_m', 'turnover_rate', 'limit_up_protection', 'rebalance_band_pct', 'single_buy_pct')
STATIC_CONFIG_FIELDS = frozenset((*SERIALIZED_DAY_CONFIG_FIELDS, 'prefilter_n'))
STATIC_CONFIG_WRAPPER_FIELDS = frozenset(('ga_profile', 'individual_config', 'config_format', 'action_schema_hash', 'action_schema'))

@dataclass(frozen=True)
class ActionField:
    index: int
    name: str
    kind: Literal['continuous']
    minimum: float
    maximum: float
    transform: Literal['linear'] = 'linear'

@dataclass(frozen=True)
class ActionSchema:
    factor_names: tuple[str, ...] = CORE_FACTOR_NAMES
    filter_names: tuple[str, ...] = CORE_FILTER_NAMES
    fixed_buy_n: int = FIXED_BUY_N
    fixed_sell_m: int = FIXED_SELL_M
    fixed_turnover_rate: float = 1.0
    learn_turnover: bool = False
    # Fixed execution controls; M alone may have a GA search grid.
    sell_m_values: tuple[int, ...] = ()
    fixed_filter_flags: tuple[bool, ...] = (True, True)
    fixed_limit_up_protection: bool = True
    fixed_rebalance_band_pct: float = .01
    schema_version: str = 'day-config-v26-learned-replacement-cap'
    sub_strategy_names: tuple[str, ...] = ()
    sub_strategy_factor_weights: tuple[tuple[float, ...], ...] = ()
    ppo_action_bins: int = 0
    _layout: tuple[ActionField, ...] = field(init=False, repr=False)

    def __post_init__(self):
        for name in ('factor_names','filter_names','sell_m_values','fixed_filter_flags','sub_strategy_names','sub_strategy_factor_weights'):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        object.__setattr__(self, 'sub_strategy_factor_weights', tuple(tuple(row) for row in self.sub_strategy_factor_weights))
        if not self.factor_names or len(set(self.factor_names)) != len(self.factor_names):
            raise ValueError('factor_names must be non-empty and unique')
        if not self.filter_names or len(set(self.filter_names)) != len(self.filter_names):
            raise ValueError('filter_names must be non-empty and unique')
        if type(self.fixed_buy_n) is not int or self.fixed_buy_n <= 0:
            raise ValueError('fixed_buy_n must be a positive int')
        if type(self.fixed_sell_m) is not int or self.fixed_sell_m < self.fixed_buy_n:
            raise ValueError('fixed_sell_m must be an integer >= fixed_buy_n')
        if isinstance(self.fixed_turnover_rate, bool) or not math.isfinite(self.fixed_turnover_rate) or not 0 <= self.fixed_turnover_rate <= 1:
            raise ValueError('fixed_turnover_rate must be finite and in [0,1]')
        if self.sell_m_values and (any(type(x) is not int or x < self.fixed_buy_n for x in self.sell_m_values)
                or tuple(sorted(set(self.sell_m_values))) != self.sell_m_values):
            raise ValueError('sell_m_values must be sorted unique integers >= buy_n')
        if len(self.fixed_filter_flags) != len(self.filter_names) or any(type(x) is not bool for x in self.fixed_filter_flags):
            raise ValueError('fixed_filter_flags must match filter_names')
        if type(self.fixed_limit_up_protection) is not bool:
            raise TypeError('fixed_limit_up_protection must be bool')
        if not math.isfinite(self.fixed_rebalance_band_pct) or not 0 <= self.fixed_rebalance_band_pct < 1:
            raise ValueError('rebalance band must be in [0,1)')
        if self.sub_strategy_names:
            count = len(self.sub_strategy_names)
            rows = np.asarray(self.sub_strategy_factor_weights, dtype=np.float64)
            if count < 2 or len(set(self.sub_strategy_names)) != count or rows.shape != (count,len(self.factor_names)):
                raise ValueError('sub-strategy names and factor vectors must match')
            if not np.isfinite(rows).all() or np.any(rows < 0) or not np.allclose(rows.sum(axis=1),1,rtol=0,atol=1e-8):
                raise ValueError('sub-strategy vectors must be normalized and non-negative')
            if np.linalg.matrix_rank(rows[1:]-rows[0]) != count-1:
                raise ValueError('sub-strategy factor vectors must be affinely independent')
            if self.sell_m_values:
                raise ValueError('sub-strategy mixing keeps M fixed')
        elif self.sub_strategy_factor_weights:
            raise ValueError('sub-strategy vectors require names')
        if type(self.ppo_action_bins) is not int or self.ppo_action_bins not in (0, 11):
            raise ValueError('PPO action bins must be 0 or 11')
        if self.ppo_action_bins and len(self.sub_strategy_names) != 2:
            raise ValueError('categorical mixing requires exactly two sub-strategies')
        if type(self.learn_turnover) is not bool:
            raise TypeError('learn_turnover must be bool')
        if self.learn_turnover and (self.ppo_action_bins != 11 or self.fixed_buy_n != 20 or self.fixed_sell_m != 25 or self.sell_m_values):
            raise ValueError('learned cap requires categorical pair N20/M25')
        names = ([f'sub_strategy_share.{n}' for n in self.sub_strategy_names[1:]]
                 if self.sub_strategy_names else [f'factor_weight.{n}' for n in self.factor_names])
        layout = [ActionField(i,n,'continuous',0.,1.) for i,n in enumerate(names)]
        if self.sell_m_values:
            layout.append(ActionField(len(layout),'sell_m','continuous',float(self.sell_m_values[0]),float(self.sell_m_values[-1])))
        if self.learn_turnover:
            layout.append(ActionField(len(layout),'turnover_rate','continuous',0.,.2))
        object.__setattr__(self,'_layout',tuple(layout))

    @property
    def layout(self): return self._layout
    @property
    def action_names(self): return tuple(x.name for x in self.layout)
    @property
    def action_dim(self): return len(self.layout)
    @property
    def space_bounds(self): return np.full(self.action_dim,-1.,dtype=np.float32),np.full(self.action_dim,1.,dtype=np.float32)

    @property
    def policy_class_counts(self):
        return (self.ppo_action_bins, 5) if self.learn_turnover else (self.ppo_action_bins,)

    def policy_action_space(self):
        from gymnasium import spaces
        if self.learn_turnover:
            return spaces.MultiDiscrete(self.policy_class_counts)
        if self.ppo_action_bins:
            return spaces.Discrete(self.ppo_action_bins)
        low, high = self.space_bounds
        return spaces.Box(low=low, high=high, dtype=np.float32)

    def policy_action_coordinates(self, action):
        """Convert a sampled class ID into the one canonical DayConfig coordinate."""
        if not self.ppo_action_bins:
            return np.asarray(action, dtype=np.float32)
        value = np.asarray(action)
        if self.learn_turnover:
            counts = np.asarray(self.policy_class_counts)
            if value.shape != (2,) or not np.isfinite(value).all() or np.any(value != np.floor(value)) or np.any(value < 0) or np.any(value >= counts):
                raise ValueError('multi-categorical action must contain valid mixture/cap IDs')
            return np.asarray(2 * value / (counts - 1) - 1, dtype=np.float32)
        if value.shape not in ((), (1,)) or not np.isfinite(value).all():
            raise ValueError('categorical action must be one finite class ID')
        index = float(value.item())
        if not index.is_integer() or not 0 <= index < self.ppo_action_bins:
            raise ValueError('categorical action ID is outside its declared bins')
        return np.asarray([2 * index / (self.ppo_action_bins - 1) - 1], dtype=np.float32)

    def _schema_payload(self):
        return dict(schema_version=self.schema_version, ppo_action_bins=self.ppo_action_bins, factor_names=list(self.factor_names),
            filter_names=list(self.filter_names), fixed_buy_n=self.fixed_buy_n, fixed_sell_m=self.fixed_sell_m,
            sell_m_values=list(self.sell_m_values), fixed_turnover_rate=self.fixed_turnover_rate, learn_turnover=self.learn_turnover,
            replacement_rule='one ranked held-or-buyable pool; retain top M; voluntary exits capped at floor(N*rate); trim excess; fill vacancies to N',
            canonical_precision='field-relative IEEE-754 binary32 Box action; M ordinal bin centers',
            fixed_filter_flags=list(self.fixed_filter_flags), fixed_limit_up_protection=self.fixed_limit_up_protection,
            fixed_rebalance_band_pct=self.fixed_rebalance_band_pct, sub_strategy_names=list(self.sub_strategy_names),
            sub_strategy_factor_weights=[list(x) for x in self.sub_strategy_factor_weights],
            layout=[dict(index=x.index,name=x.name,kind=x.kind,minimum=x.minimum,maximum=x.maximum,transform=x.transform) for x in self.layout])

    @cached_property
    def schema_hash(self):
        return hashlib.sha256(json.dumps(self._schema_payload(),sort_keys=True,separators=(',',':')).encode()).hexdigest()
    def to_dict(self): return {**self._schema_payload(),'schema_hash':self.schema_hash}
    @classmethod
    def from_dict(cls,payload):
        schema = cls(**{k:payload[k] for k in ('schema_version','ppo_action_bins','factor_names','filter_names','fixed_buy_n','fixed_sell_m','fixed_turnover_rate','learn_turnover','sell_m_values','fixed_filter_flags','fixed_limit_up_protection','fixed_rebalance_band_pct','sub_strategy_names','sub_strategy_factor_weights')})
        if dict(payload) != schema.to_dict(): raise ValueError('action schema payload/hash mismatch')
        return schema

    @classmethod
    def from_config(cls,payload,*,factor_names=None,filter_names=None):
        from configs.strategy_config import resolve_strategy_config
        payload = resolve_strategy_config(payload)
        if 'action_schema' in payload:
            return cls.from_dict(payload['action_schema'])
        config=payload.get('individual_config',payload)
        names=tuple(factor_names or CORE_FACTOR_NAMES)
        filters=tuple(filter_names or CORE_FILTER_NAMES)
        return cls(factor_names=names,filter_names=filters,fixed_buy_n=config['buy_n'],
            fixed_sell_m=config['sell_m'],
            fixed_turnover_rate=config.get('turnover_rate',1.),
            fixed_filter_flags=tuple(config.get('filter_factors',{}).get(n,True) for n in filters),
            fixed_limit_up_protection=config.get('limit_up_protection',True),
            fixed_rebalance_band_pct=config.get('rebalance_band_pct',.01))

    def decode(self,action):
        values = np.asarray(action,dtype=np.float64)
        if values.shape != (self.action_dim,) or not np.isfinite(values).all() or np.any(np.abs(values)>1):
            raise ValueError(f'action must be finite Box coordinates with shape ({self.action_dim},)')
        units = decode_unit_action(encode_unit_action(decode_unit_action(values)))
        if self.sub_strategy_names:
            shares = np.zeros(len(self.sub_strategy_names)); remainder=1.
            for i,u in enumerate(units[:len(self.sub_strategy_names)-1],start=1):
                shares[i]=remainder*float(u); remainder-=shares[i]
            shares[0]=remainder
            weights=shares @ np.asarray(self.sub_strategy_factor_weights)
        else:
            weights=units[:len(self.factor_names)]
        m = (self.sell_m_values[min(int(units[-1]*len(self.sell_m_values)),len(self.sell_m_values)-1)]
             if self.sell_m_values else self.fixed_sell_m)
        return DayConfig(factor_weights=dict(zip(self.factor_names,weights.tolist())),
            factor_enabled=dict(zip(self.factor_names,(weights!=0).tolist())),
            filter_flags=dict(zip(self.filter_names,self.fixed_filter_flags)), buy_n=self.fixed_buy_n,sell_m=m,
            turnover_rate=(round(float(units[-1])*4)/20 if self.learn_turnover else self.fixed_turnover_rate),limit_up_protection=self.fixed_limit_up_protection,
            rebalance_band_pct=self.fixed_rebalance_band_pct,single_buy_pct=1./self.fixed_buy_n)

    def encode(self,config):
        self.validate_day_config(config)
        weights=np.array([config.factor_weights[n] for n in self.factor_names])
        if self.sub_strategy_names:
            rows=np.asarray(self.sub_strategy_factor_weights)
            coeff,*_=np.linalg.lstsq((rows[1:]-rows[0]).T,weights-rows[0],rcond=None)
            shares=np.concatenate(([1.-coeff.sum()],coeff))
            if np.any(shares < -1e-7) or np.any(shares > 1.+1e-7) or not np.allclose(shares@rows,weights,rtol=0,atol=2e-7):
                raise ValueError('DayConfig is outside the fixed sub-strategy mixture')
            units=[]; remainder=1.
            for share in shares[1:]:
                units.append(float(np.clip(share/remainder if remainder>1e-10 else 0.,0.,1.)));remainder-=share
        else: units=weights.tolist()
        if self.sell_m_values: units.append((self.sell_m_values.index(config.sell_m)+.5)/len(self.sell_m_values))
        if self.learn_turnover: units.append(config.turnover_rate / .2)
        return encode_unit_action(np.asarray(units))

    def canonicalize_day_config(self,config): return self.decode(self.encode(config))

    def validate_policy_memory(self, memory: PolicyMemory) -> None:
        """Require recorded history before an initialized state enters an actor."""
        if not isinstance(memory, PolicyMemory):
            raise TypeError("policy memory must be PolicyMemory")
        if not memory.initialized:
            return
        history = memory.history
        if history is None:
            raise ValueError("initialized policy memory is missing its actual dated history")
        if history.action_schema_hash != self.schema_hash or history.values.shape[1] != self.action_dim + 2:
            raise ValueError("policy history action schema mismatch")
        actions = history.values[:, :-2]
        canonical = np.ascontiguousarray(actions, dtype=np.float32)
        if not np.array_equal(actions, canonical):
            raise ValueError("every policy history action must use canonical action coordinates")
        if not np.array_equal(history.values[-1, :-2], self.encode(memory.previous_day_config)):
            raise ValueError("policy history must end with the canonical previous DayConfig")

    def advance_policy_memory(
        self,
        previous: PolicyMemory,
        settled: PolicyMemory,
        *,
        decision_date: str,
        history_length: int,
    ) -> PolicyMemory:
        """Append one actual Fill settlement, encoding its DayConfig exactly once."""
        if type(history_length) is not int or history_length <= 0:
            raise ValueError("policy history length must be a positive int")
        self.validate_policy_memory(previous)
        if not isinstance(settled, PolicyMemory) or not settled.initialized:
            raise ValueError("a recorded decision requires a completed config and Fill settlement")
        if settled.history is not None:
            raise ValueError("a completed decision must be appended exactly once")
        action = self.encode(settled.previous_day_config)
        row = np.asarray((
            *action, settled.previous_gross_turnover_ratio, settled.previous_total_cost_ratio,
        ), dtype=np.float64).reshape(1, -1)
        date = np.asarray([decision_date], dtype="datetime64[D]")
        if previous.history is not None:
            if date[0] <= previous.history.decision_dates[-1]:
                raise ValueError("a policy history decision date cannot repeat or go backwards")
            dates = np.concatenate((previous.history.decision_dates, date))[-history_length:]
            values = np.concatenate((previous.history.values, row), axis=0)[-history_length:]
        else:
            dates, values = date, row
        history = PolicyHistory(dates, values, self.schema_hash)
        return PolicyMemory(
            previous_day_config=settled.previous_day_config,
            previous_gross_turnover_ratio=settled.previous_gross_turnover_ratio,
            previous_total_cost_ratio=settled.previous_total_cost_ratio,
            history=history,
        )


    def validate_day_config(self,config):
        if tuple(config.factor_weights)!=self.factor_names or tuple(config.factor_enabled)!=self.factor_names:
            raise ValueError('DayConfig factor order does not match ActionSchema')
        if tuple(config.filter_flags)!=self.filter_names or tuple(config.filter_flags.values())!=self.fixed_filter_flags:
            raise ValueError('filter flags differ from fixed ActionSchema controls')
        if config.buy_n != self.fixed_buy_n: raise ValueError('buy_n differs from fixed ActionSchema control')
        if config.sell_m not in (self.sell_m_values or (self.fixed_sell_m,)): raise ValueError('sell_m outside declared values')
        if self.learn_turnover:
            if config.turnover_rate not in (0.,.05,.1,.15,.2): raise ValueError('turnover_rate outside learned replacement grid')
        elif config.turnover_rate != self.fixed_turnover_rate: raise ValueError('turnover_rate differs from fixed ActionSchema control')
        if config.limit_up_protection != self.fixed_limit_up_protection: raise ValueError('limit-up protection differs from fixed control')
        if config.rebalance_band_pct != self.fixed_rebalance_band_pct: raise ValueError('rebalance band differs from fixed control')
        if config.single_buy_pct != 1./config.buy_n: raise ValueError('single_buy_pct must equal 1 / buy_n')

    def from_static_config(self,payload):
        from configs.strategy_config import resolve_strategy_config
        payload = resolve_strategy_config(payload)
        if not isinstance(payload,Mapping): raise TypeError('static config must be a mapping')
        if 'individual_config' in payload:
            if set(payload)-STATIC_CONFIG_WRAPPER_FIELDS: raise ValueError('unexpected static wrapper fields')
            if 'action_schema_hash' in payload and payload['action_schema_hash']!=self.schema_hash:
                raise ValueError('static config action schema mismatch')
            payload=payload['individual_config']
        if not isinstance(payload,Mapping) or set(payload)-STATIC_CONFIG_FIELDS:
            raise ValueError('static config contains unexpected fields')
        if 'weights' not in payload:
            raise ValueError('static config requires weights')
        weights=payload['weights']
        if not isinstance(weights,Mapping) or set(weights)!=set(self.factor_names):
            raise ValueError('static factor vocabulary does not match ActionSchema')
        # Never renormalize trained or human coordinates during loading.
        ordered={n:weights[n] for n in self.factor_names}
        enabled={n:ordered[n]!=0 for n in self.factor_names}
        if 'factor_enabled' in payload and dict(payload['factor_enabled'])!=enabled:
            raise ValueError('factor_enabled must be derived from weights')
        raw_filters=payload.get('filter_factors',dict(zip(self.filter_names,self.fixed_filter_flags)))
        if set(raw_filters)!=set(self.filter_names): raise ValueError('static filter vocabulary does not match ActionSchema')
        day=DayConfig(factor_weights=ordered,factor_enabled=enabled,
            filter_flags={n:raw_filters[n] for n in self.filter_names},buy_n=payload['buy_n'],
            sell_m=payload['sell_m'],turnover_rate=payload.get('turnover_rate',1.),
            limit_up_protection=payload.get('limit_up_protection',self.fixed_limit_up_protection),
            rebalance_band_pct=payload.get('rebalance_band_pct',self.fixed_rebalance_band_pct),
            single_buy_pct=payload.get('single_buy_pct',1./payload['buy_n']))
        return self.canonicalize_day_config(day)

    def from_serialized_day_config(self,payload):
        if not isinstance(payload,Mapping) or set(payload)!=set(SERIALIZED_DAY_CONFIG_FIELDS):
            raise ValueError('serialized DayConfig fields must match exactly')
        return self.from_static_config(payload)

    def to_static_config(self,config):
        self.validate_day_config(config)
        return dict(weights=dict(config.factor_weights),factor_enabled=dict(config.factor_enabled),
            filter_factors=dict(config.filter_flags),buy_n=config.buy_n,sell_m=config.sell_m,
            turnover_rate=config.turnover_rate,limit_up_protection=config.limit_up_protection,
            rebalance_band_pct=config.rebalance_band_pct,single_buy_pct=config.single_buy_pct)
