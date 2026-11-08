"""Independent compiled equalization and actual shared-worker replay acceptance."""
from dataclasses import asdict, fields, replace
import multiprocessing
from multiprocessing.shared_memory import SharedMemory

import numpy as np
import pytest

from env.action_schema import ActionSchema, CORE_FACTOR_NAMES
from env.backtest import EpisodeSession, prepare_episode_from_runtime, run_day_config_episode
from env.fees import FeeSchedule
from env.planner import DayPlanner
from env.quantity import floor_buy_quantity, floor_partial_sell_quantity
from env.shared_episode import SharedPreparedEpisodeOwner
from env.simulator import DaySimulator
from env.contracts import AccountState
from test_backtest_lightweight import write_canonical_runtime
from test_rl_day_planner import _config, _market


pytestmark = pytest.mark.filterwarnings("error::numba.core.errors.NumbaIRAssumptionWarning")


def no_fees():
    return FeeSchedule(commission_rate=0.0, minimum_commission=0.0, stamp_tax_rate=0.0, transfer_fee_rate=0.0, slippage_rate=0.0)


def test_fee_parameters_cache_does_not_enter_serialized_dataclass_schema():
    fees = FeeSchedule(minimum_commission=5.0)
    original = asdict(fees)
    first = fees.parameters
    assert fees.parameters is first
    assert asdict(fees) == original
    assert tuple(item.name for item in fields(fees)) == (
        "commission_rate", "minimum_commission", "stamp_tax_rate",
        "transfer_fee_rate", "slippage_rate",
    )
    assert first == tuple(original[name] for name in original)
    with pytest.raises(TypeError):
        first[0] = 1.0


@pytest.mark.parametrize("code,minimum,step", [
    ("600001.SH", 100, 100), ("300001.SZ", 100, 100),
    ("688001.SH", 200, 1), ("689001.SH", 200, 1), ("430001.BJ", 100, 1),
])
def test_shared_buy_and_partial_sell_lot_formula_at_boundaries(code, minimum, step):
    cases = {0: 0, minimum - 1: 0, minimum: minimum, minimum + step: minimum + step}
    for raw, expected in cases.items():
        assert floor_buy_quantity(code, raw) == expected
        assert floor_partial_sell_quantity(code, raw) == expected
    assert floor_buy_quantity(code, minimum + step - 0.01) == minimum
    assert floor_partial_sell_quantity(code, minimum + step - 0.01) == minimum


def test_cash_sweep_ignores_rebalance_band_and_respects_frozen_affordability():
    codes = ("600001.SH", "600002.SH")
    market = _market(codes=codes)
    config = _config(buy_n=2, turnover_rate=1.0, band=0.3)
    account = AccountState(
        cash=2000.0, positions={codes[1]: 400, codes[0]: 400},
        sellable_positions={codes[0]: 400, codes[1]: 400},
        last_prices={code: 10.0 for code in codes}, nav=10000.0, peak_nav=10000.0,
    )
    plan = DayPlanner(fees=no_fees()).plan(market, account, config)
    assert plan.sell_orders == ()
    assert list(plan.buy_orders.items()) == [(codes[0], 100)]
    assert plan.diagnostics["planned_post_order_cash"] == 1000.0
    assert plan.diagnostics["skip_reasons"] == {codes[1]: "insufficient_frozen_cash"}
    assert plan.diagnostics["full_investment_contract_satisfied"]


def test_sell_first_funds_kcb_bj_purchases_with_one_share_steps():
    codes = ("688001.SH", "430001.BJ", "600003.SH")
    market = _market(codes=codes)
    config = _config(buy_n=2, turnover_rate=1.0)
    account = AccountState(
        cash=0.0, positions={codes[2]: 500}, sellable_positions={codes[2]: 500},
        last_prices={codes[2]: 10.0}, nav=5000.0, peak_nav=5000.0,
    )
    fees = no_fees()
    plan = DayPlanner(fees=fees).plan(market, account, config)
    assert plan.sell_orders == ((codes[2], 500),)
    assert list(plan.buy_orders.items()) == [(codes[0], 250), (codes[1], 192)]
    assert plan.diagnostics["planned_post_order_cash"] == 580.0
    assert plan.diagnostics["full_investment_contract_satisfied"]
    result = DaySimulator(fees).step(
        account, plan, {code: 10.0 for code in codes}, {code: 10.0 for code in codes},
        close_prices={code: 10.0 for code in codes},
        next_preclose_prices={code: 10.0 for code in codes},
        next_decision_date="2026-08-21",
    )
    assert [(fill.code, fill.side, fill.quantity) for fill in result.fills] == [
        (codes[2], "sell", 500), (codes[0], "buy", 250), (codes[1], "buy", 192),
    ]
    assert result.account_state.cash == 580.0
    assert result.account_state.nav == 5000.0


def test_full_liquidation_keeps_odd_lot_and_partial_liquidation_keeps_exchange_minimum():
    codes = ("688001.SH", "600002.SH")
    market = _market(codes=codes)
    config = _config(buy_n=1, turnover_rate=1.0)
    full = AccountState(
        cash=0.0, positions={codes[1]: 101}, sellable_positions={codes[1]: 101},
        last_prices={codes[1]: 10.0}, nav=1010.0, peak_nav=1010.0,
    )
    planner = DayPlanner(fees=no_fees())
    assert planner.plan(market, full, config).sell_orders == ((codes[1], 101),)
    partial = replace(full, sellable_positions={codes[1]: 99})
    assert planner.plan(market, partial, config).sell_orders == ()


def _trace_signature(trace):
    return {
        "nav": trace.nav, "cash": trace.cash, "returns": trace.portfolio_returns,
        "rewards": trace.rewards, "exposure": trace.exposure,
        "orders": [(list(plan["sell_orders"]), list(plan["buy_orders"].items())) for plan in trace.order_plans],
        "fills": [[asdict(fill) for fill in fills] for fills in trace.fills],
        "full_investment": trace.full_investment_contract,
    }


def _run_dynamic(episode):
    schema = ActionSchema()
    base = schema.decode(np.zeros(schema.action_dim))
    day = 0

    def policy(_observation):
        nonlocal day
        weights = {name: ((day + index) % 9 + 1) / 9 for index, name in enumerate(CORE_FACTOR_NAMES)}
        turnover_rate = (0.05, 0.2)[day % 2]
        day += 1
        return replace(base, factor_weights=weights, factor_enabled={name: True for name in weights}, turnover_rate=turnover_rate)

    return run_day_config_episode(EpisodeSession(episode), policy)


def _spawn_replay(descriptor):
    import warnings
    from numba.core.errors import NumbaIRAssumptionWarning

    warnings.simplefilter("error", NumbaIRAssumptionWarning)
    with descriptor.attach() as attached:
        episode = attached.episode
        market = episode.market_at(episode.decision_start)
        assert all(np.shares_memory(row, episode.factors.ranks) and not row.flags.writeable for row in market.factor_ranks.values())
        assert np.shares_memory(market.open_prices, episode.runtime.field("open"))
        assert np.shares_memory(market.preclose_prices, episode.runtime.field("preClose"))
        assert episode.listing_age is episode.runtime.field("listing_age")
        assert not episode.listing_age.flags.writeable
        signature = _trace_signature(_run_dynamic(episode))
        signature["actor_present"] = episode.observation_builder is not None
        signature["stock_count"] = len(episode.runtime.stock_codes)
        return signature


@pytest.mark.parametrize("encode", [False, True])
def test_actual_spawn_replay_preserves_shared_prices_ranks_and_dynamic_account(tmp_path, monkeypatch, encode):
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        monkeypatch.setenv(name, "1")
    path = tmp_path / "shared-equalization-runtime.npz"
    write_canonical_runtime(path, stocks=30)
    episode = prepare_episode_from_runtime(
        path, "2020-06-08", "2020-06-27", lookback=12,
        prefilter_n=21, encode_observations=encode,
    )
    expected = _trace_signature(_run_dynamic(episode))
    with SharedPreparedEpisodeOwner.create(episode) as owner:
        names = tuple(item.shared_memory_name for item in owner.descriptor.arrays)
        context = multiprocessing.get_context("spawn")
        with context.Pool(1) as pool:
            actual = pool.apply(_spawn_replay, (owner.descriptor,))
    assert actual.pop("actor_present") is encode
    assert actual.pop("stock_count") == 30
    assert actual.keys() == expected.keys()
    for name in ("orders", "fills"):
        assert actual[name] == expected[name]
    for name in ("nav", "cash", "returns", "rewards", "exposure", "full_investment"):
        np.testing.assert_array_equal(actual[name], expected[name])
    assert len(actual["rewards"]) == 19
    assert np.all(actual["full_investment"])
    for name in names:
        with pytest.raises(FileNotFoundError):
            SharedMemory(name=name, create=False)
