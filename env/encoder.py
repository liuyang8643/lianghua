"""Lossless raw-observation transport and train-only field scaling.

The leading compact coordinate routes into an explicitly bound immutable
market store. It is transport metadata and must never enter a learned layer.
"""
from __future__ import annotations
from dataclasses import dataclass
from functools import cached_property
import json
from pathlib import Path
from typing import Mapping
import numpy as np
from numpy.typing import NDArray
from env.contracts import Observation
from env.observation import (AccountObservation, ObservationBuilder, ObservationSchema, _canonical_hash,
                             RAW_MISSING_VALUE, HISTORICAL_STOCK_FEATURE_NAMES, LATEST_STOCK_FEATURE_NAMES)

ENCODER_SCHEMA_VERSION = "wbr-raw-observation-transport-v22-source-state-12factors"
NORMALIZER_VERSION = "wbr-train-field-normalizer-v14-12factors"

@dataclass(frozen=True)
class EncodedObservationSchema:
    source_observation_schema: str
    lookback: int
    stock_count: int
    stock_feature_names: tuple[str, ...]
    position_feature_names: tuple[str, ...]
    portfolio_feature_names: tuple[str, ...]
    history_feature_names: tuple[str, ...]
    version: str = ENCODER_SCHEMA_VERSION

    def __post_init__(self):
        if self.version != ENCODER_SCHEMA_VERSION or self.lookback <= 0 or self.stock_count <= 0:
            raise ValueError("unsupported raw transport schema")
        for name in ("stock_feature_names", "position_feature_names", "portfolio_feature_names", "history_feature_names"):
            values = tuple(getattr(self, name))
            if not values or len(values) != len(set(values)):
                raise ValueError("raw field vocabulary must be nonempty and unique")
            object.__setattr__(self, name, values)

    @property
    def dimension(self):
        return 1 + self.stock_count*len(self.position_feature_names) + len(self.portfolio_feature_names) + self.lookback*len(self.history_feature_names)

    @property
    def position_slice(self):
        return slice(1, 1+self.stock_count*len(self.position_feature_names))

    @property
    def portfolio_slice(self):
        start = self.position_slice.stop
        return slice(start, start+len(self.portfolio_feature_names))

    @property
    def history_slice(self):
        return slice(self.portfolio_slice.stop, self.dimension)

    @cached_property
    def feature_names(self):
        return ("transport.raw_row_ref", *(f"position.{i}.{f}" for i in range(self.stock_count) for f in self.position_feature_names),
                *(f"portfolio.{f}" for f in self.portfolio_feature_names),
                *(f"policy_history.lag_{lag:04d}.{f}" for lag in range(self.lookback,0,-1) for f in self.history_feature_names))

    def _hash_payload(self):
        return {"version":self.version,"source_observation_schema":self.source_observation_schema,
                "lookback":self.lookback,"stock_count":self.stock_count,"dimension":self.dimension,
                "historical_stock_feature_names":list(HISTORICAL_STOCK_FEATURE_NAMES),
                "latest_stock_feature_names":list(LATEST_STOCK_FEATURE_NAMES),
                **{n:list(getattr(self,n)) for n in ("stock_feature_names","position_feature_names","portfolio_feature_names","history_feature_names")}}

    @property
    def schema_hash(self):
        return _canonical_hash(self._hash_payload())

    @property
    def identifier(self):
        return f"{self.version}:{self.schema_hash}"

    def to_dict(self):
        return {**self._hash_payload(),"schema_hash":self.schema_hash}

    @classmethod
    def from_dict(cls,payload):
        values=dict(payload)
        expected=values.pop("schema_hash")
        dimension=values.pop("dimension")
        if (values.pop("historical_stock_feature_names") != list(HISTORICAL_STOCK_FEATURE_NAMES)
                or values.pop("latest_stock_feature_names") != list(LATEST_STOCK_FEATURE_NAMES)):
            raise ValueError("raw history/latest partition differs")
        result=cls(**values)
        if result.schema_hash!=expected or result.dimension!=dimension:
            raise ValueError("raw transport schema hash/dimension mismatch")
        return result

class ObservationEncoder:
    """Serialize account state without pooling or transforming any field."""
    def __init__(self, observation_schema: ObservationSchema):
        self.observation_schema=observation_schema
        self.output_schema=EncodedObservationSchema(observation_schema.identifier,observation_schema.lookback,
            observation_schema.stock_count,observation_schema.stock_feature_names,observation_schema.position_feature_names,
            observation_schema.portfolio_feature_names,observation_schema.policy_history_feature_names)

    @property
    def output_dimension(self):
        return self.output_schema.dimension

    def encode_account(self, account: AccountObservation | Observation, store: "RawMarketStore"):
        if account.schema_version!=self.observation_schema.identifier or store.schema.identifier!=account.schema_version:
            raise ValueError("raw observation/store schema mismatch")
        schema=self.output_schema
        if account.position_panel.shape!=(schema.stock_count,len(schema.position_feature_names)) or account.portfolio.shape!=(len(schema.portfolio_feature_names),) or account.policy_history.shape!=(schema.lookback,len(schema.history_feature_names)):
            raise ValueError("raw account tensor shape mismatch")
        ref=store.row_reference(account.decision_date)
        encoded=np.concatenate((np.asarray([ref],dtype=np.float32),account.position_panel.reshape(-1),account.portfolio,account.policy_history.reshape(-1))).astype(np.float32,copy=False)
        if not np.isfinite(encoded).all():
            raise ValueError("raw compact observation must be finite")
        return encoded

    def encode(self, observation: Observation, *, store: "RawMarketStore"):
        return self.encode_account(observation,store)

@dataclass(frozen=True)
class RawMarketStore:
    """One raw row per date; overlapping windows are resolved only when used."""
    raw_rows: NDArray[np.float32]
    pit_universe_mask: NDArray[np.bool_]
    row_valid: NDArray[np.bool_]
    schema: ObservationSchema
    decision_start: int
    decision_stop: int
    decision_dates: tuple[str,...]
    row_start: int=0

    def __post_init__(self):
        raw=np.ascontiguousarray(self.raw_rows,dtype=np.float32)
        member=np.ascontiguousarray(self.pit_universe_mask,dtype=bool)
        valid=np.ascontiguousarray(self.row_valid,dtype=bool)
        if raw.ndim!=3 or raw.shape[1:]!=(self.schema.stock_count,self.schema.stock_feature_count) or member.shape!=raw.shape[:2] or valid.shape!=raw.shape[:1]:
            raise ValueError("raw store shape mismatch")
        if not 0<=self.decision_start<self.decision_stop<=len(raw) or len(self.decision_dates)!=self.decision_stop-self.decision_start:
            raise ValueError("raw store must declare exactly its legal decision references")
        if len(set(self.decision_dates))!=len(self.decision_dates) or not np.isfinite(raw).all():
            raise ValueError("raw rows must be finite with the reserved missing sentinel")
        if np.any(member[~valid]) or np.any(raw[~member]!=0):
            raise ValueError("nonmember and padded raw rows must remain zero")
        for name,values in (("raw_rows",raw),("pit_universe_mask",member),("row_valid",valid)):
            values.flags.writeable=False
            object.__setattr__(self,name,values)
        object.__setattr__(self,"decision_dates",tuple(self.decision_dates))

    @property
    def decision_indices(self):
        return tuple(range(self.row_start+self.decision_start,self.row_start+self.decision_stop))

    @cached_property
    def _reference_by_date(self):
        return {date:self.decision_start+i for i,date in enumerate(self.decision_dates)}

    def row_reference(self,decision_date:str)->int:
        return self._reference_by_date[decision_date]

    def window(self,row_reference:int):
        if type(row_reference) is not int or not self.decision_start<=row_reference<self.decision_stop:
            raise ValueError("reference outside bound sealed decision split")
        length=self.schema.lookback
        first=max(0,row_reference-length+1)
        rows=self.raw_rows[first:row_reference+1]
        raw=np.zeros((length,*rows.shape[1:]),dtype=np.float32)
        member=np.zeros(raw.shape[:2],dtype=bool)
        valid=np.zeros(length,dtype=bool)
        raw[-len(rows):]=rows
        member[-len(rows):]=self.pit_universe_mask[first:row_reference+1]
        valid[-len(rows):]=self.row_valid[first:row_reference+1]
        return raw,member,valid

    @classmethod
    def precompute(cls,builder:ObservationBuilder,decision_indices,*,chunk_rows:int=32):
        indices=tuple(int(i) for i in decision_indices)
        if not indices or indices!=tuple(range(indices[0],indices[-1]+1)) or chunk_rows<=0:
            raise ValueError("raw store decisions must be one nonempty continuous interval")
        first=max(0,indices[0]-builder.lookback+1)
        stop=indices[-1]+1
        raw=np.empty((stop-first,builder.schema.stock_count,builder.schema.stock_feature_count),dtype=np.float32)
        member=np.empty(raw.shape[:2],dtype=bool)
        for begin in range(first,stop,chunk_rows):
            end=min(begin+chunk_rows,stop)
            rows=builder.build_static_rows(begin,end)
            raw[begin-first:end-first]=rows.stock_panel
            member[begin-first:end-first]=rows.pit_universe_mask
        return cls(raw,member,np.ones(stop-first,dtype=bool),builder.schema,indices[0]-first,stop-first,
                   tuple(str(d) for d in builder.trade_dates[indices[0]:stop]),first)

    @classmethod
    def from_observation(cls,observation:Observation,encoder:ObservationEncoder):
        schema=encoder.observation_schema
        if observation.schema_version!=schema.identifier or observation.stock_panel.shape!=(schema.lookback,schema.stock_count,schema.stock_feature_count):
            raise ValueError("live observation does not match raw model schema")
        return cls(observation.stock_panel,observation.pit_universe_mask,observation.time_mask,schema,
                   schema.lookback-1,schema.lookback,(observation.decision_date,))

@dataclass(frozen=True)
class TrainOnlyNormalizer:
    encoder_schema:str
    stock_scale:NDArray[np.float32]
    position_scale:NDArray[np.float32]
    portfolio_scale:NDArray[np.float32]
    history_scale:NDArray[np.float32]
    sample_count:int
    version:str=NORMALIZER_VERSION

    def __post_init__(self):
        if self.version!=NORMALIZER_VERSION or self.sample_count<=0:
            raise ValueError("invalid field normalizer identity/sample count")
        for name in ("stock_scale","position_scale","portfolio_scale","history_scale"):
            values=np.ascontiguousarray(getattr(self,name),dtype=np.float32)
            if values.ndim!=1 or not len(values) or not np.isfinite(values).all() or np.any(values<=0):
                raise ValueError("field scales must be positive finite vectors")
            values.flags.writeable=False
            object.__setattr__(self,name,values)

    @classmethod
    def fit(cls,store:RawMarketStore,schema:EncodedObservationSchema,*,dataset_role:str,initial_cash:float):
        if dataset_role!="train" or schema.source_observation_schema!=store.schema.identifier:
            raise ValueError("normalizer may fit only the matched sealed train store")
        if not np.isfinite(initial_cash) or initial_cash<=0:
            raise ValueError("initial_cash must be positive")
        sums=np.zeros(len(schema.stock_feature_names),dtype=np.float64)
        counts=np.zeros_like(sums)
        for start in range(store.decision_start,store.decision_stop,32):
            end=min(start+32,store.decision_stop)
            values=store.raw_rows[start:end].astype(np.float64)
            valid=store.pit_universe_mask[start:end,:,None] & (values!=RAW_MISSING_VALUE)
            sums+=np.where(valid,np.square(values),0).sum(axis=(0,1))
            counts+=valid.sum(axis=(0,1))
        stock=np.sqrt(np.divide(sums,counts,out=np.ones_like(sums),where=counts>0))
        stock=np.where(stock>0,stock,1.0)
        for i,name in enumerate(schema.stock_feature_names):
            if name in ("st_mask","price_buy_allowed","price_sell_allowed"):
                stock[i]=1.0
        price=stock[schema.stock_feature_names.index("open")]
        position=np.asarray((initial_cash/price,price,initial_cash/price,price),dtype=np.float32)
        portfolio=np.asarray((initial_cash,initial_cash,initial_cash,1.0),dtype=np.float32)
        history=np.ones(len(schema.history_feature_names),dtype=np.float32)
        history[schema.history_feature_names.index("total_cost_ratio")]=0.01
        return cls(schema.identifier,stock.astype(np.float32),position,portfolio,history,store.decision_stop-store.decision_start)

    def _state_payload(self):
        return {"version":self.version,"encoder_schema":self.encoder_schema,"sample_count":self.sample_count,
                **{name:getattr(self,name).tolist() for name in ("stock_scale","position_scale","portfolio_scale","history_scale")}}

    @property
    def state_hash(self):
        return _canonical_hash(self._state_payload())

    def to_dict(self):
        return {**self._state_payload(),"state_hash":self.state_hash}

    @classmethod
    def from_dict(cls,payload:Mapping[str,object]):
        values=dict(payload)
        expected=values.pop("state_hash")
        result=cls(**values)
        if result.state_hash!=expected:
            raise ValueError("field normalizer hash mismatch")
        return result

    def save(self,path):
        Path(path).write_text(json.dumps(self.to_dict(),separators=(",",":")),encoding="utf-8")

    @classmethod
    def load(cls,path,*,expected_schema:EncodedObservationSchema):
        result=cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
        if result.encoder_schema!=expected_schema.identifier:
            raise ValueError("field normalizer schema mismatch")
        return result

__all__=["ENCODER_SCHEMA_VERSION","NORMALIZER_VERSION","EncodedObservationSchema","ObservationEncoder","RawMarketStore","TrainOnlyNormalizer"]
