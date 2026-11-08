"""Independent equivalence and invalidation checks for sealed market rows."""
from dataclasses import fields, replace

import numpy as np
import pytest

from env.action_schema import ActionSchema, CORE_FACTOR_NAMES
from env.backtest import EpisodeSession, build_day_market, prepare_episode_from_runtime
from env.legality import evaluate_trade_legality
from env.planner import DayPlanner
from test_backtest_lightweight import write_canonical_runtime
from test_rl_day_planner import _cash_account, _config, _market


def assert_legality_equal(left, right):
    for field in fields(left):
        np.testing.assert_array_equal(getattr(left, field.name), getattr(right, field.name))


@pytest.mark.parametrize("field_name", [
    "open_prices", "preclose_prices", "issue_prices", "listing_age",
    "st_mask", "delisted_mask", "candidate_mask",
])
def test_seal_owns_rows_even_if_supplied_as_readonly_views(field_name):
    market = _market()
    backing = np.array(getattr(market, field_name), copy=True)
    readonly_view = backing.view()
    readonly_view.flags.writeable = False
    market = replace(market, **{field_name: readonly_view})
    sealed = market.seal()
    expected = getattr(sealed, field_name).copy()
    backing[0] = not bool(backing[0]) if backing.dtype == np.bool_ else backing[0] + 7
    np.testing.assert_array_equal(getattr(sealed, field_name), expected)


@pytest.mark.parametrize("field_name", ["factor_ranks", "factor_validity", "filter_masks"])
def test_seal_owns_factor_and_filter_rows_behind_readonly_views(field_name):
    market = _market()
    mapping = dict(getattr(market, field_name))
    name = next(iter(mapping))
    backing = mapping[name].copy()
    readonly_view = backing.view()
    readonly_view.flags.writeable = False
    mapping[name] = readonly_view
    sealed = replace(market, **{field_name: mapping}).seal()
    expected = getattr(sealed, field_name)[name].copy()
    backing[:] = False if backing.dtype == np.bool_ else 0.123
    np.testing.assert_array_equal(getattr(sealed, field_name)[name], expected)
    with pytest.raises(TypeError):
        getattr(sealed, field_name)[name] = backing


def test_external_price_owner_cannot_make_a_cached_legality_result_stale():
    backing = np.full(4, 10.0)
    readonly_view = backing.view()
    readonly_view.flags.writeable = False
    sealed = _market(opens=readonly_view).seal()
    first = sealed.trade_legality(True)
    backing[0] = 11.0
    fresh = evaluate_trade_legality(
        decision_date=sealed.decision_date, stock_codes=sealed.stock_codes,
        listing_age=sealed.listing_age, open_prices=sealed.open_prices,
        preclose_prices=sealed.preclose_prices, issue_prices=sealed.issue_prices,
        st_mask=sealed.st_mask, delisted_mask=sealed.delisted_mask,
        limit_up_protection=True,
    )
    assert_legality_equal(first, fresh)
    assert sealed.open_prices[0] == 10.0


@pytest.mark.parametrize("readonly_buffer_view", [False, True])
def test_seal_owns_readonly_numpy_view_of_mutable_python_buffer(readonly_buffer_view):
    backing = bytearray(np.full(4, 10.0).tobytes())
    buffer = memoryview(backing).toreadonly() if readonly_buffer_view else backing
    readonly_view = np.frombuffer(buffer, dtype=np.float64)
    readonly_view.flags.writeable = False
    sealed = _market(opens=readonly_view).seal()
    original = sealed.trade_legality(True)
    np.frombuffer(backing, dtype=np.float64)[0] = 11.0
    assert sealed.open_prices[0] == 10.0
    assert sealed.trade_legality(True).buy_allowed[0] == original.buy_allowed[0]


def test_unsealed_live_market_recomputes_legality_after_price_change():
    market = _market()
    before = market.trade_legality(True)
    market.open_prices[0] = 11.0
    after = market.trade_legality(True)
    assert before.buy_allowed[0]
    assert not after.buy_allowed[0]
    assert not after.sell_allowed[0]
    assert after.buy_reasons[0] == "limit_up"
    assert market._legality_cache == {}


def test_candidate_clones_share_only_market_legality_and_respect_protection(monkeypatch):
    import env.planner as planner_module

    market = _market(opens=np.array([11.0, 10.0, 10.0, 10.0])).seal()
    real_evaluate = planner_module.evaluate_trade_legality
    calls = []

    def counted_evaluate(**kwargs):
        calls.append(kwargs["limit_up_protection"])
        return real_evaluate(**kwargs)

    monkeypatch.setattr(planner_module, "evaluate_trade_legality", counted_evaluate)
    without_protection = market.trade_legality(False)
    with_protection = market.trade_legality(True)
    assert without_protection.sell_allowed[0]
    assert not with_protection.sell_allowed[0]
    for mask in (np.array([False, True, False, False]), np.array([False, False, True, True])):
        clone = market.with_candidate_mask(mask)
        expected_mask = mask.copy()
        mask[:] = False
        np.testing.assert_array_equal(clone.candidate_mask, expected_mask)
        assert clone.trade_legality(False) is without_protection
        assert clone.trade_legality(True) is with_protection
        cached_plan = DayPlanner().plan(clone, _cash_account(), _config())
        fresh_plan = DayPlanner().plan(replace(clone), _cash_account(), _config())
        assert cached_plan == fresh_plan
        assert set(cached_plan.buy_orders) <= {
            code for code, allowed in zip(market.stock_codes, expected_mask) if allowed
        }
    assert calls == [False, True, False, False]
    np.testing.assert_array_equal(market.candidate_mask, np.ones(4, dtype=bool))
    for field in fields(with_protection):
        values = getattr(with_protection, field.name)
        with pytest.raises(ValueError, match="read-only"):
            values[0] = values[0]


def test_different_dates_listing_delisting_and_st_rules_do_not_share_cache():
    common = _market(opens=np.full(4, 10.5), st=np.ones(4, dtype=bool))
    before_reform = replace(common, decision_date="2026-07-03").seal()
    after_reform = replace(common, decision_date="2026-07-06").seal()
    assert not before_reform.trade_legality(True).buy_allowed.any()
    assert after_reform.trade_legality(True).buy_allowed.all()
    missing_member = replace(after_reform, listing_age=np.array([-1, 10, 10, 10], dtype=np.int32)).seal()
    delisted = replace(after_reform, delisted_mask=np.array([False, True, False, False])).seal()
    assert missing_member.trade_legality(True).buy_reasons[0] == "not_listed"
    assert delisted.trade_legality(True).buy_reasons[1] == "delisted"
    assert after_reform.trade_legality(True).buy_allowed.all()


class FreshMarketSession(EpisodeSession):
    """Use the same domain kernel with pre-optimization market construction."""

    def _market_day(self, index):
        return build_day_market(self.episode.runtime, self.episode.factors, self.episode.listing_age, index)


@pytest.fixture
def episode(tmp_path):
    runtime_path = tmp_path / "cache-runtime.npz"
    write_canonical_runtime(runtime_path, stocks=30)
    with np.load(runtime_path, allow_pickle=False) as source:
        arrays = {name: source[name].copy() for name in source.files}
    start = int(np.searchsorted(arrays["trade_dates"], np.datetime64("2020-06-08")))
    arrays["listing_age"][: start + 4, -1] = -1
    arrays["listing_age"][start + 4 :, -1] = np.arange(len(arrays["trade_dates"]) - start - 4)
    arrays["delisted_mask"][start + 9 :, -2] = True
    arrays["st_mask"][start + 3 : start + 7, -3] = True
    arrays["open"][start + 5, -4] = np.nan
    arrays["open"][start + 8, -5] = arrays["preClose"][start + 8, -5] * 1.1
    np.savez_compressed(runtime_path, **arrays)
    return prepare_episode_from_runtime(runtime_path, "2020-06-08", "2020-06-27", lookback=12, prefilter_n=25)


def test_prepared_rows_stay_in_split_and_match_fresh_market(episode):
    for index in range(episode.decision_start, episode.decision_stop):
        cached = episode.market_at(index)
        assert cached is episode.market_at(index)
        fresh = build_day_market(episode.runtime, episode.factors, episode.listing_age, index)
        assert cached.decision_date == fresh.decision_date
        assert cached.stock_codes == fresh.stock_codes
        for name in ("open_prices", "preclose_prices", "issue_prices", "st_mask", "delisted_mask", "listing_age", "candidate_mask"):
            np.testing.assert_array_equal(getattr(cached, name), getattr(fresh, name))
        for name in ("factor_ranks", "factor_validity", "filter_masks"):
            for key, values in getattr(cached, name).items():
                np.testing.assert_array_equal(values, getattr(fresh, name)[key])
    for index in (episode.decision_start - 1, episode.decision_stop):
        with pytest.raises(IndexError, match="sealed episode"):
            episode.market_at(index)


@pytest.mark.parametrize("dynamic", [False, True])
def test_complete_30_stock_account_timeline_exactly_matches_uncached_path(episode, dynamic):
    schema = ActionSchema()
    fixed = schema.decode(np.zeros(schema.action_dim))
    cached = EpisodeSession(episode, action_schema=schema)
    uncached = FreshMarketSession(episode, action_schema=schema)
    first_cached, info_cached = cached.reset()
    first_uncached, info_uncached = uncached.reset()
    np.testing.assert_array_equal(first_cached, first_uncached)
    assert info_cached == info_uncached
    transitions = 0
    while not cached.terminated:
        weights = {name: (1 + (index + transitions) % 11) / 11 for index, name in enumerate(CORE_FACTOR_NAMES)}
        turnover_rate = (0.05, 0.2)[transitions % 2]
        config = replace(fixed, factor_weights=weights, factor_enabled={name: True for name in weights}, turnover_rate=turnover_rate) if dynamic else fixed
        actual = cached.step(config)
        expected = uncached.step(config)
        assert actual.order_plan == expected.order_plan
        assert actual.step_result == expected.step_result
        assert actual.info == expected.info
        np.testing.assert_array_equal(actual.observation, expected.observation)
        assert cached.current_account == uncached.current_account
        assert cached.current_policy_memory == uncached.current_policy_memory
        assert cached.reward_state == uncached.reward_state
        assert actual.info["full_investment_contract_satisfied"]
        transitions += 1
    assert transitions == 19
    assert uncached.terminated
