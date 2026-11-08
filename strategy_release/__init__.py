"""Portable default supervised strategy. This package never sends broker orders."""
from pathlib import Path
import hashlib
import json
import numpy as np
from .selection import RankBufferConfig,allocate_rank_buffer
from .normalization import percentile

REPOSITORY=Path(__file__).resolve().parent.parent
DEFAULT_CONFIG=REPOSITORY/'configs/default_supervised_strategy.json'

def _read(path):return json.loads(path.read_text(encoding='utf8'))
def _sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

class FrozenTopK:
    def __init__(self,config_path=DEFAULT_CONFIG):
        import lightgbm as lgb
        config=_read(Path(config_path));self.root=REPOSITORY/config['bundle']
        if config['strategy']!='new_model_topk' or config['mvo_enabled'] is not False:raise ValueError('Unsupported default strategy')
        if _sha(self.root/'manifest.json')!=config['manifest_sha256']:raise ValueError('Strategy manifest changed')
        self.manifest=_read(self.root/'manifest.json')
        for name,sha in self.manifest['files'].items():
            if Path(name).name!=name or _sha(self.root/name)!=sha:raise ValueError('Bundle file mismatch: '+name)
        self.features=_read(self.root/'features.json');self.names=self.features['feature_names']
        self.model=lgb.Booster(model_file=str(self.root/'model.txt'))
        if self.model.num_feature()!=len(self.names)+1 or self.model.num_trees()!=self.manifest['trees']:raise ValueError('Model schema mismatch')
        self.config=RankBufferConfig(**self.manifest['parameters'])

    def normalize(self,raw,*,feature_names,eligible):
        """One full daily cross section; compute causal raw factors upstream."""
        x=np.asarray(raw,dtype=np.float64);mask=np.asarray(eligible,dtype=bool)
        if list(feature_names)!=self.names or x.shape!=(len(mask),len(self.names)):raise ValueError('Feature schema differs')
        out=np.full(x.shape,np.nan,dtype=np.float32)
        for k,name in enumerate(self.names):
            out[:,k]=percentile(x[:,k],mask,binary=self.features['normalization'][name]=='binary')
        return out

    def predict(self,normalized,*,feature_names,codes,eligible,complete_universe=False):
        """Return raw five-head predictions and fused percentile scores.

        complete_universe is the provider's explicit completeness declaration,
        not an independently verified market-data completeness certificate.
        Stock order must match the provider's canonical axis (stable tie breaks).
        """
        if complete_universe is not True:raise ValueError('Full daily cross section required; never predict/rank only held or TopK stocks')
        x=np.asarray(normalized,dtype=np.float32);mask=np.asarray(eligible,dtype=bool)
        if list(feature_names)!=self.names or x.shape!=(len(codes),len(self.names)) or mask.shape!=(len(codes),):raise ValueError('Prediction axes differ')
        if len(set(map(str,codes)))!=len(codes) or np.isinf(x).any():raise ValueError('Duplicate codes or infinite features')
        valid=mask&np.isfinite(x).any(axis=1);raw=np.full((len(codes),5),np.nan)
        if not valid.any():raise ValueError('No eligible feature rows')
        model_x=np.column_stack([x[valid],np.zeros(valid.sum())])
        for k in range(5):
            model_x[:,-1]=k;raw[valid,k]=self.model.predict(model_x,num_threads=1)
        pool=mask&np.isfinite(raw).all(axis=1);ranks=np.full(raw.shape,np.nan,dtype=np.float32)
        for k in range(5):ranks[:,k]=percentile(raw[:,k],pool)
        scores=np.sum(ranks*np.asarray(self.manifest['weights']),axis=1)
        return raw,scores

    def targets(self,scores,*,held,current_weights,buyable,sellable,holding_days):
        """Target fractions of current open NAV, before lot sizing and fees.

        Caller supplies confirmed holdings, directional legality and trading-day
        ages, and must execute exits before entries with actual cash/T+1 checks.
        """
        s=np.asarray(scores,float);held=np.asarray(held,bool);w=np.asarray(current_weights,float)
        buy=np.asarray(buyable,bool);sell=np.asarray(sellable,bool);ages=np.asarray(holding_days,int)
        if any(a.shape!=s.shape for a in [held,w,buy,sell,ages]) or s.ndim!=1:raise ValueError('Account axes differ')
        if not np.isfinite(w).all() or np.any(w<0) or w.sum()>1+1e-8 or np.any((w>0)&~held) or np.any(ages<0):raise ValueError('Invalid opening account weights/ages')
        if held.sum()>self.config.sell_m:raise ValueError('Account exceeds strategy slots; explicit migration required')
        return allocate_rank_buffer(s,held,w,buy,sell,ages,self.config)

def load_default():return FrozenTopK()
