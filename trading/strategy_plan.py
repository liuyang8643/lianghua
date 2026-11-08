"""Confirmed file account -> canonical research planner -> target document."""
from pathlib import Path
from types import SimpleNamespace
import numpy as np

from env.contracts import AccountState
from offline_data import load_runtime_slice
from strategy_release.runtime import FeatureInputs, FrozenStrategyRuntime


def calculate_plan(runtime_path, trade_date, positions, cash, ages, *, inputs_directory,
                   runtime=None,assembly=None,inputs=None,opening_cache=None,scores=None):
    assembly = assembly or FrozenStrategyRuntime()
    if runtime is None:
        runtime = load_runtime_slice(runtime_path, trade_date, trade_date, preload_rows=100001,
                                     cache_dir=Path('data/runtime_cache'))
    index = runtime.index_of(trade_date)
    missing = set(positions)-set(runtime.stock_codes)
    if missing:
        raise ValueError('Confirmed holdings missing from canonical stock axis: '+str(sorted(missing)))
    inputs = inputs or FeatureInputs.load(inputs_directory)
    if scores is None:
        scores = assembly.predict(runtime,[index],inputs,opening_cache=opening_cache)
    prepared = assembly.prepare(runtime,scores,initial_cash=cash,split='file_live')
    account = AccountState(cash=float(cash),positions={c:int(p.volume) for c,p in positions.items()},
        sellable_positions={c:int(p.can_use_volume) for c,p in positions.items()},
        last_prices={c:float(p.market_value)/int(p.volume) for c,p in positions.items()})
    plan,market,weights = assembly.plan(prepared,account,index,ages)
    managed = set(positions)|set(plan.buy_orders)
    lookup = {c:i for i,c in enumerate(runtime.stock_codes)}
    prices = {c:float(market.open_prices[lookup[c]]) for c in managed
              if np.isfinite(market.open_prices[lookup[c]]) and market.open_prices[lookup[c]]>0}
    traded = set(plan.buy_orders)|{c for c,_ in plan.sell_orders}
    if not traded.issubset(prices):
        raise ValueError('Planned order lacks executable opening price')
    legal=market.trade_legality(plan.day_config.limit_up_protection)
    freeze={c:max(price,float(legal.up_limit_prices[lookup[c]]))
            if np.isfinite(legal.up_limit_prices[lookup[c]]) and legal.up_limit_prices[lookup[c]]>0 else price
            for c,price in prices.items()}
    return SimpleNamespace(sell_orders=plan.sell_orders,buy_orders=dict(plan.buy_orders),
        prices=prices,limit_prices=freeze,buy_n_stocks=list(plan.buy_orders),
        strategy_id=assembly.strategy.manifest['release_id'],max_positions=assembly.strategy.config.sell_m,
        model_sha256=assembly.strategy.manifest['files']['model.txt'],target_weights=weights,
        strategy_open_nav=float(plan.diagnostics['total_equity']))
