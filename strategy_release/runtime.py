"""Canonical frozen strategy assembly for historical replay and file planning.

Factor formulae, normalization, prediction, allocation and order sizing are
shared. Callers supply dated inputs and confirmed account facts only.
"""
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np

from strategy_release import load_default
from ai.supervised.feature_data import materialize_factor_values
from ai.supervised.portfolio import PreparedModelPortfolio
from env.legality import investment_universe_mask


@dataclass(frozen=True)
class FeatureInputs:
    bound_names: tuple
    industry_history: object
    dividend_events: object
    monthly_payload: object
    identity: str = ''

    @classmethod
    def load(cls, directory):
        import pandas as pd
        from utils.atomic_file import file_sha256
        from offline_data.supplemental_fields import implemented_dividends
        root = Path(directory)
        manifest = json.loads((root/'manifest.json').read_text(encoding='utf8'))
        for name, digest in manifest['files'].items():
            path = (root/name).resolve()
            if not path.is_relative_to(root.resolve()) or file_sha256(path) != digest:
                raise ValueError('Strategy input checksum differs: '+name)
        with np.load(root/'monthly.npz', allow_pickle=False) as data:
            monthly = {key:data[key] for key in data.files}
        for values in monthly.values():values.flags.writeable=False
        return cls(tuple(manifest['bound_names']),
                   pd.read_parquet(root/'industry.parquet'),
                   implemented_dividends(pd.read_parquet(root/'dividends.parquet'))[0], monthly,
                   file_sha256(root/'manifest.json'))


@dataclass(frozen=True)
class OpeningCache:
    runtime: object
    index: int
    inputs_identity: str
    columns: dict


class FrozenStrategyRuntime:
    def __init__(self):
        self.strategy = load_default()

    def preopen_names(self):
        """Only formula families proven to exclude the decision day's quotes."""
        import ast,re
        from factor.library.qlib_open import CATALOG,_lookback
        from factor.library.bilibili import BILIBILI_FACTOR_NAMES
        from factor.library.bilibili_technical import NAMES as TECHNICAL_NAMES
        from factor.library.attention_transition_research import candidate_specs
        from factor.library.local_variants import candidates,variant_name
        known=set(BILIBILI_FACTOR_NAMES)|set(TECHNICAL_NAMES)|{s['name'] for s in candidate_specs()}
        for item in CATALOG['features']:
            expression=re.sub(r'\$([A-Za-z_]+)',r'FIELD_\1',item['expression'])
            if _lookback(ast.parse(expression,mode='eval').body)[0]>=1:known.add(item['name'])
        completed={'BiliLowReturnAttention20','low_residual_vol20','persistent_quiet20_60',
                   'BiliLowAmountAttention20','attention_contraction_10_60'}
        known.update(variant_name(n,p) for n,p in candidates() if n in completed)
        return [name for name in self.strategy.names if name in known]

    def prepare_open(self,runtime,index,inputs):
        columns={}
        materialize_factor_values(runtime,self.preopen_names(),lambda name,values:columns.update({name:np.asarray(values)}),
            bound_names=inputs.bound_names,industry_history=inputs.industry_history,
            dividend_events=inputs.dividend_events,monthly_payload=inputs.monthly_payload,
            output_indices=np.asarray([index]))
        for values in columns.values():values.flags.writeable=False
        return OpeningCache(runtime,index,inputs.identity,columns)

    def predict(self, runtime, indices, inputs, *, opening_cache=None, on_prediction=None):
        indices = np.asarray(indices, dtype=int)
        cached={}
        if opening_cache is not None:
            prior=opening_cache.runtime
            if (list(indices)!=[opening_cache.index] or opening_cache.inputs_identity!=inputs.identity
                    or runtime.stock_codes!=prior.stock_codes
                    or runtime.trade_dates is not prior.trade_dates):
                raise ValueError('Opening cache does not belong to this decision snapshot')
            for name,values in runtime.data.items():
                if name in ('open','preClose'):
                    if not np.array_equal(values[:-1],prior.field(name)[:-1],equal_nan=True):
                        raise ValueError('Opening cache historical prices changed')
                elif values is not prior.field(name):
                    raise ValueError('Opening cache non-quote inputs changed: '+name)
            cached=opening_cache.columns
        import tempfile
        with tempfile.TemporaryDirectory(prefix='wbr_features_') as temporary:
            columns = dict(cached)
            def collect(name, values):
                path = Path(temporary)/(name+'.npy')
                np.save(path,np.asarray(values),allow_pickle=False)
                columns[name] = np.load(path,mmap_mode='r')
            try:
                materialize_factor_values(runtime, [n for n in self.strategy.names if n not in cached], collect,
                    bound_names=inputs.bound_names, industry_history=inputs.industry_history,
                    dividend_events=inputs.dividend_events, monthly_payload=inputs.monthly_payload,
                    output_indices=indices)
                scores = np.full((runtime.n_dates,runtime.n_stocks),np.nan)
                for row,index in enumerate(indices):
                    raw = np.column_stack([columns[name][row] for name in self.strategy.names])
                    eligible = investment_universe_mask(runtime.field('open')[index],
                        runtime.field('listing_age')[index],runtime.field('st_mask')[index],
                        runtime.field('delisted_mask')[index])
                    normalized = self.strategy.normalize(raw, feature_names=self.strategy.names, eligible=eligible)
                    heads,scores[index] = self.strategy.predict(normalized, feature_names=self.strategy.names,
                        codes=runtime.stock_codes, eligible=eligible, complete_universe=True)
                    if on_prediction is not None:
                        on_prediction(index,raw,normalized,heads,scores[index],eligible)
            finally:
                for name,column in columns.items():
                    if name not in cached:column._mmap.close()
        return scores

    def prepare(self, runtime, scores, *, initial_cash, split):
        from utils.atomic_file import file_sha256
        return PreparedModelPortfolio.from_scores(runtime,scores,initial_cash=initial_cash,
            source_hash=file_sha256(self.strategy.root/'model.txt'),split=split)

    def plan(self, prepared, account, index, holding_days):
        return prepared.backtest_day(account,index,holding_days,self.strategy.manifest['parameters'])
