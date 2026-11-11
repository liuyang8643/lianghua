import numpy as np
import pytest
from strategy_release import load_default

@pytest.fixture(scope='module')
def strategy():return load_default()

def test_default_is_new_topk(strategy):
    assert strategy.manifest['default_method']=='rank_buffer'
    assert strategy.manifest['release_id']=='d_model_topk_20261111'
    assert strategy.manifest['weights']==[.2]*5
    assert strategy.manifest['early_stopping_interval']==['2026-01-01','2026-11-10']
    assert strategy.manifest['training_interval']==['2005-01-01','2025-12-31']
    assert strategy.config.buy_n==strategy.config.sell_m==40
    assert strategy.config.max_replacements==2 and strategy.config.hold_thresh==3
    assert strategy.model.num_trees()==583 and len(strategy.names)==278

def test_incomplete_universe_and_reordered_features_rejected(strategy):
    x=np.zeros((3,len(strategy.names)));mask=np.ones(3,bool);codes=['a','b','c']
    with pytest.raises(ValueError,match='Full daily'):strategy.predict(x,feature_names=strategy.names,codes=codes,eligible=mask)
    with pytest.raises(ValueError,match='axes'):strategy.predict(x,feature_names=strategy.names[::-1],codes=codes,eligible=mask,complete_universe=True)

def test_locked_and_young_positions_not_sold(strategy):
    scores=np.arange(100,dtype=float);held=np.arange(100)<40;w=held.astype(float)*.025
    target,exits=strategy.targets(scores,held=held,current_weights=w,buyable=np.ones(100,bool),sellable=np.zeros(100,bool),holding_days=np.full(100,10))
    np.testing.assert_array_equal(target,w);assert len(exits)==0
    target,exits=strategy.targets(scores,held=held,current_weights=w,buyable=np.ones(100,bool),sellable=held,holding_days=np.full(100,2))
    np.testing.assert_array_equal(target,w);assert len(exits)==0

def test_two_exit_slots_at_three_day_boundary_and_no_topup(strategy):
    held=np.arange(100)<40;w=held.astype(float)*.02
    target,exits=strategy.targets(np.arange(100),held=held,current_weights=w,buyable=np.ones(100,bool),sellable=held,holding_days=np.full(100,3))
    assert len(exits)==2 and np.count_nonzero(target)==40
    assert target[99]==target[98]==.025 and target.sum()<1
    keep=held.copy();keep[exits]=False;np.testing.assert_array_equal(target[keep],w[keep])

def test_raw_normalization_preserves_missing(strategy):
    x=np.tile(np.array([1.,1.,3.,np.nan])[:,None],(1,len(strategy.names)));eligible=np.array([True,True,True,False])
    y=strategy.normalize(x,feature_names=strategy.names,eligible=eligible)
    for k,n in enumerate(strategy.names):
        expected=x[:,k] if strategy.features['normalization'][n]=='binary' else np.array([1/3,1/3,5/6,np.nan],np.float32)
        np.testing.assert_allclose(y[:,k],expected,equal_nan=True)

def test_many_account_states_match_research(strategy):
    import ast,zipfile
    from dataclasses import dataclass
    # Use the bundled exact reference, not uncommitted modules in this workspace.
    with zipfile.ZipFile(strategy.root/'research_source.zip') as archive:
        source=archive.read('env/qlib_strategy.py').decode('utf8')
    class State:
        def __init__(self,codes):self.codes=codes;self.diagnostics=[]
    scope={'np':np,'dataclass':dataclass,'_PortfolioState':State}
    for node in ast.parse(source).body:
        if getattr(node,'name','') in ['RankBufferConfig','select_rank_buffer','RankBufferPortfolio']:
            text=ast.get_source_segment(source,node)
            if node.name=='RankBufferConfig':text='@dataclass(frozen=True)\n'+text
            exec(text,scope)
    RankBufferPortfolio=scope['RankBufferPortfolio'];RankBufferConfig=scope['RankBufferConfig']
    rng=np.random.default_rng(20261108)
    for _ in range(100):
        n=150;codes=[str(i) for i in range(n)];scores=rng.normal(size=n);scores[rng.random(n)<.05]=np.nan
        held=np.zeros(n,bool);held[rng.choice(n,int(rng.integers(0,41)),replace=False)]=True
        w=held*rng.uniform(.005,.025,n);buy=rng.random(n)>.2;sell=held&(rng.random(n)>.2);ages=rng.integers(0,20,n)
        reference=RankBufferPortfolio(codes,RankBufferConfig(40,40,3,2))
        reference.context=lambda *args:(held,sell,ages,w,np.where(sell,0,w),buy,1.)
        expected=reference.targets(None,None,1,scores)
        got,_=strategy.targets(scores,held=held,current_weights=w,buyable=buy,sellable=sell,holding_days=ages)
        np.testing.assert_array_equal(got,np.array([expected.get(c,0.) for c in codes]))
