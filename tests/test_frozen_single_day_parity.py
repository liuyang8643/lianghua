"""Today's file plan is the same decision as an ordinary research account day."""
from dataclasses import replace
import json
from pathlib import Path
import socket
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from ai.supervised.portfolio import PreparedModelPortfolio
from env.contracts import AccountState
from offline_data import RuntimeManifest, RuntimeSlice
from test_rl_runtime_slice import _runtime_arrays


@pytest.fixture(autouse=True)
def no_external_connections(monkeypatch):
    before = set(sys.modules)

    def forbidden(*args, **kwargs):
        raise AssertionError('Single-day parity tests must stay offline')

    monkeypatch.setattr(socket, 'create_connection', forbidden)
    monkeypatch.setattr(socket.socket, 'connect', forbidden)
    yield
    assert not any(name == 'xtquant' or name.startswith('xtquant.')
                   for name in set(sys.modules) - before)


@pytest.fixture
def runtime():
    arrays = _runtime_arrays(rows=10, stocks=50)
    dates = arrays.pop('trade_dates')
    codes = tuple(arrays.pop('stock_codes'))
    arrays['listing_age'][:] = 100
    for field in ('eps', 'bps', 'roe', 'profit_yoy', 'revenue_yoy',
                  'operating_cf_ps', 'gross_margin'):
        arrays['pit_' + field] = arrays[field].copy()
    for array in (dates, *arrays.values()):
        array.flags.writeable = False
    manifest = RuntimeManifest(
        schema_version='single-day-parity-fixture', schema_hash='synthetic',
        source_path='synthetic', source_sha256='synthetic', source_size=0,
        stock_vocabulary_sha256='synthetic', requested_start=str(dates[3]),
        requested_end=str(dates[-1]), loaded_start=str(dates[0]),
        loaded_end=str(dates[-1]), requested_preload_rows=3,
        actual_preload_rows=3, fields=(),
    )
    return RuntimeSlice(codes, dates, arrays, 3, len(dates), manifest)


@pytest.fixture
def parameters():
    release = Path(__file__).resolve().parents[1] / 'strategy_release/new_model_topk_20261108/manifest.json'
    return json.loads(release.read_text(encoding='utf8'))['parameters']


def one_observation(runtime, index):
    dates = runtime.trade_dates[:index + 1]
    return replace(
        runtime, trade_dates=dates,
        data={name: values[:index + 1] if values.ndim == 2 else values
              for name, values in runtime.data.items()},
        decision_start=index, decision_stop=index + 1,
        manifest=replace(runtime.manifest, requested_start=str(dates[-1]),
                         requested_end=str(dates[-1]), loaded_end=str(dates[-1]),
                         actual_preload_rows=index),
    )


def prepare(runtime, scores):
    return PreparedModelPortfolio.from_scores(
        runtime, scores, initial_cash=1_000_000,
        source_hash='synthetic-known-scores', split='single-day-parity',
    )


def test_each_single_day_matches_settled_session_including_held_ages(runtime, parameters):
    scores = np.stack([np.roll(np.arange(runtime.n_stocks, dtype=float), day * 8)
                       for day in range(runtime.n_dates)])
    account_days = []
    prepare(runtime, scores).evaluate(
        parameters, details=True,
        on_transition=lambda account, transition: account_days.append((account, transition)),
    )
    born = {}
    for index, (account, transition) in enumerate(account_days, runtime.decision_start):
        born = {code: born[code] if code in born else index - 1 for code in account.positions}
        ages = {code: index - acquired for code, acquired in born.items()}
        today = prepare(one_observation(runtime, index), scores[:index + 1])
        plan, market, weights = today.backtest_day(account, index, ages, parameters)
        expected = transition.order_plan
        assert today.episode.transition_count == 0
        assert plan.decision_date == expected.decision_date
        assert plan.sell_orders == expected.sell_orders
        assert dict(plan.buy_orders) == dict(expected.buy_orders)
        assert plan.day_config == expected.day_config
        assert market.decision_date == expected.decision_date
        assert len(weights) <= parameters['sell_m']
        with pytest.raises(ValueError, match='next-open'):
            today.evaluate(parameters)
    assert len(account_days[0][1].order_plan.buy_orders) == 40
    assert any(transition.order_plan.sell_orders for _, transition in account_days[1:])


@pytest.mark.parametrize('verified_through, accepted', [
    ('2020-01-05', True),   # Sunday evidence covers the Monday decision.
    ('2020-01-03', False),  # Friday alone does not cover weekend disclosures.
])
def test_t_open_overlay_is_readonly_and_can_prepare_todays_plan(
        runtime, parameters, monkeypatch, tmp_path, verified_through, accepted):
    from data.strategy_inputs import with_open_overlay
    import data.db.stock_name as stock_name
    import data.build_runtime as runtime_builder
    import offline_data.financial_versions as financial
    import offline_data.financial_research as pershare
    import offline_data.financial_store as financial_store
    import offline_data.capital_history as capital

    history = one_observation(runtime, 2)
    today = np.datetime64('2020-01-06')
    manifest = {'verified_through': verified_through}
    frames = {table: None for table in (*financial.FINANCIAL_FIELDS, 'PershareIndex', 'Capital')}
    def read_tables(directory):
        assert Path(directory) == tmp_path
        return frames, manifest
    monkeypatch.setattr(financial_store, 'read_tables', read_tables)
    monkeypatch.setattr(financial, 'normalize_financial_events', lambda *args: None)
    monkeypatch.setattr(financial, 'iter_financial_fields',
                        lambda dates, count, core: iter([(today, {})]))
    per_share = np.column_stack([runtime.field('pit_' + name)[3] for name in pershare.PERSHARE_FIELDS])
    monkeypatch.setattr(pershare, 'replay_latest_fields', lambda *args: iter([per_share]))
    monkeypatch.setattr(capital, 'apply_capital_events', lambda *args: None)
    monkeypatch.setattr(stock_name, 'build_st_mask',
                        lambda codes, dates: np.zeros((len(dates), len(codes)), dtype=bool))
    monkeypatch.setattr(runtime_builder, 'build_delisted_mask',
                        lambda codes, dates: np.zeros((len(dates), len(codes)), dtype=bool))
    overlay = {code: {name: np.array([runtime.field(name)[3, column]])
                      for name in ('open', 'preClose')}
               for column, code in enumerate(runtime.stock_codes)}

    if not accepted:
        with pytest.raises(ValueError, match='refreshed through T-1'):
            with_open_overlay(history, today, overlay, tmp_path)
        return
    opened = with_open_overlay(history, today, overlay, tmp_path)
    assert not opened.trade_dates.flags.writeable
    assert all(not values.flags.writeable for values in opened.data.values())
    for name in ('high', 'low', 'close', 'volume', 'amount'):
        assert np.isnan(opened.field(name)[-1]).all()
    np.testing.assert_array_equal(opened.field('open')[-1], runtime.field('open')[3])
    for name in history.data:
        original = history.field(name)
        np.testing.assert_array_equal(opened.field(name)[:-1] if original.ndim == 2
                                      else opened.field(name), original)

    scores = np.tile(np.arange(opened.n_stocks, dtype=float), (opened.n_dates, 1))
    prepared = prepare(opened, scores)
    plan, _, _ = prepared.backtest_day(AccountState(cash=1_000_000), 3, {}, parameters)
    assert len(plan.buy_orders) == 40
    assert not plan.sell_orders


def test_preopen_materialized_families_ignore_current_open_and_preclose():
    from factor.library.bilibili import BILIBILI_FACTOR_NAMES
    from factor.library.bilibili_technical import NAMES as TECHNICAL_NAMES
    from factor.library.attention_transition_research import candidate_specs
    from strategy_release.runtime import FeatureInputs, FrozenStrategyRuntime

    arrays = _runtime_arrays(rows=340, stocks=8)
    dates = arrays.pop('trade_dates')
    codes = tuple(arrays.pop('stock_codes'))
    for name in ('eps', 'bps', 'roe', 'profit_yoy', 'revenue_yoy',
                 'operating_cf_ps', 'gross_margin'):
        arrays['pit_' + name] = arrays[name].copy()
    for array in (dates, *arrays.values()):
        array.flags.writeable = False
    manifest = RuntimeManifest(
        schema_version='preopen-cache-fixture', schema_hash='synthetic',
        source_path='synthetic', source_sha256='synthetic', source_size=0,
        stock_vocabulary_sha256='synthetic', requested_start=str(dates[-1]),
        requested_end=str(dates[-1]), loaded_start=str(dates[0]),
        loaded_end=str(dates[-1]), requested_preload_rows=100001,
        actual_preload_rows=len(dates) - 1, fields=(),
    )
    runtime = RuntimeSlice(codes, dates, arrays, len(dates) - 1, len(dates), manifest)
    features = Path(__file__).resolve().parents[1] / 'strategy_release/new_model_topk_20261108/features.json'
    assembly = FrozenStrategyRuntime.__new__(FrozenStrategyRuntime)
    assembly.strategy = SimpleNamespace(names=json.loads(features.read_text(encoding='utf8'))['feature_names'])
    bound = set(BILIBILI_FACTOR_NAMES) | set(TECHNICAL_NAMES) | {s['name'] for s in candidate_specs()}
    inputs = FeatureInputs(tuple(sorted(bound)), None, None, None, 'synthetic-inputs')
    original = assembly.prepare_open(runtime, runtime.decision_start, inputs)

    changed = dict(runtime.data)
    for name in ('open', 'preClose'):
        values = runtime.field(name).copy()
        values[-1] = [np.nan, 0., .01, 1e6, 2., 500., 80., .5]
        values.flags.writeable = False
        changed[name] = values
    perturbed = assembly.prepare_open(replace(runtime, data=changed), runtime.decision_start, inputs)

    assert set(original.columns) == set(assembly.preopen_names()) == set(perturbed.columns)
    for name, values in original.columns.items():
        np.testing.assert_array_equal(values, perturbed.columns[name], err_msg=name)
        assert not values.flags.writeable
    # Ensure the assertion exercises meaningful values from every cached
    # family, not only all-NaN warm-up rows.
    representatives = (
        'Alpha158_MA60', 'BiliAdjustedIssueDiscount', 'BiliLowLifetimeRangeRatio',
        'BiliKDJOversoldCross', 'BiliMACDBottomDivergence30', 'contraction_20_60',
        'attention_decay20_60', 'BiliLowAmountAttention20__history_252__recent_60',
        'low_residual_vol20__window_120',
    )
    for name in representatives:
        assert np.isfinite(original.columns[name]).any(), name
