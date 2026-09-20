"""Summary recording must preserve the complete canonical account trajectory."""
from dataclasses import fields, replace

import numpy as np
import pytest

from ai.ga.config import canonicalize_ga_genes
from ai.ga.train import _evaluate_individual
from env.action_schema import ActionSchema, CORE_FACTOR_NAMES
from env.backtest import EpisodeSession, RolloutSeries, RolloutSummary, RolloutTrace, prepare_episode_from_runtime, run_day_config_episode
from env.contracts import AccountState
from test_backtest_lightweight import write_canonical_runtime


pytestmark = pytest.mark.filterwarnings("error::numba.core.errors.NumbaIRAssumptionWarning")


@pytest.fixture
def summary_episode(tmp_path):
    path = tmp_path / "summary-runtime.npz"
    write_canonical_runtime(path, stocks=30)
    with np.load(path, allow_pickle=False) as payload:
        arrays = {name: np.array(payload[name], copy=True) for name in payload.files}
    split = int(np.searchsorted(arrays["trade_dates"], np.datetime64("2020-06-15")))
    arrays["preClose"][split] /= 1.5
    delist = int(np.searchsorted(arrays["trade_dates"], np.datetime64("2020-06-20")))
    arrays["delisted_mask"][delist:, :15] = True
    np.savez_compressed(path, **arrays)
    return prepare_episode_from_runtime(path, "2020-06-08", "2020-06-27", lookback=12, prefilter_n=21, encode_observations=False)


class RecordingSession(EpisodeSession):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.recorded = []

    def step(self, config):
        transition = super().step(config)
        self.recorded.append(transition)
        return transition


def dynamic_provider():
    schema = ActionSchema()
    base = schema.decode(np.zeros(schema.action_dim))
    cursor = 0

    def provide(_observation):
        nonlocal cursor
        weights = {name: ((cursor + index) % 9 + 1) / 9 for index, name in enumerate(CORE_FACTOR_NAMES)}
        buy_n = (20, 25)[cursor % 2]
        cursor += 1
        return replace(base, factor_weights=weights, factor_enabled={name: True for name in weights}, turnover_rate=(.05 if buy_n == 20 else .2))

    return provide


def assert_common_exact(summary, trace):
    for item in fields(RolloutSeries):
        actual, expected = getattr(summary, item.name), getattr(trace, item.name)
        if isinstance(actual, np.ndarray):
            assert actual.shape == expected.shape
            assert actual.dtype == expected.dtype
            assert actual.tobytes() == expected.tobytes(), item.name
        else:
            assert actual == expected, item.name
    assert summary.metrics == trace.metrics
    assert summary.average_exposure == trace.average_exposure
    assert summary.full_investment_contract_satisfied == trace.full_investment_contract_satisfied
    assert summary.executed_sell_count == trace.executed_sell_count


def test_summary_and_full_trace_execute_identical_every_day_fills_and_events(summary_episode):
    full_session, summary_session = RecordingSession(summary_episode), RecordingSession(summary_episode)
    full = run_day_config_episode(full_session, dynamic_provider())
    summary = run_day_config_episode(summary_session, dynamic_provider(), record_details=False)
    assert isinstance(full, RolloutTrace)
    assert type(summary) is RolloutSummary
    assert_common_exact(summary, full)
    assert len(summary_session.recorded) == len(full_session.recorded) == 19
    for actual, expected in zip(summary_session.recorded, full_session.recorded, strict=True):
        assert actual.order_plan == expected.order_plan
        assert actual.step_result == expected.step_result
        assert actual.info == expected.info
        assert actual.observation.tobytes() == expected.observation.tobytes()
    assert summary_session.current_policy_memory == full_session.current_policy_memory
    assert summary.total_fees == sum(item.info['total_fees'] for item in full_session.recorded)
    assert summary.sum_gross_turnover_ratio == sum(item.info['gross_turnover_ratio'] for item in full_session.recorded)
    assert summary.sum_total_cost_ratio == pytest.approx(sum(item.info['total_cost_ratio'] for item in full_session.recorded), rel=1e-15)
    assert full.executed_sell_count > 0
    assert sum(len(day) for day in full.fills) > 0
    assert sum(item["total"] for day in full.fee_breakdowns for item in day) > 0
    assert full.account_events
    assert any(transition.step_result.diagnostics["corporate_action_adjustments"] for transition in full_session.recorded)
    assert "fills" not in vars(summary)
    assert "order_plans" not in vars(summary)
    assert "day_configs" not in vars(summary)


def test_ga_consumes_the_same_metrics_as_a_full_fixed_trace(summary_episode):
    schema = ActionSchema()
    payload = schema.to_static_config(schema.decode(np.full(schema.action_dim, 0.25)))
    canonical, config = canonicalize_ga_genes(payload, action_schema=schema)
    assert tuple(config.factor_weights.values()) == (0.625,) * len(schema.factor_names)
    session = RecordingSession(summary_episode)
    full = run_day_config_episode(session, lambda _: config)
    evaluated = _evaluate_individual(summary_episode, payload)
    assert evaluated.pop('evaluation_elapsed_seconds') > 0
    assert evaluated.pop('execution_diagnostics') == {
        'executed_sell_count': full.executed_sell_count,
        'total_fees': sum(item.info['total_fees'] for item in session.recorded),
        'mean_gross_turnover_ratio': sum(item.info['gross_turnover_ratio'] for item in session.recorded) / len(full.rewards),
        'mean_total_cost_ratio': sum(item.info['total_cost_ratio'] for item in session.recorded) / len(full.rewards),
        'reward_sum': float(full.rewards.sum()),
    }
    assert evaluated == {
        "individual_config": canonical,
        "metrics": full.metrics.as_dict(),
        "total_return": float((full.nav[-1] / full.nav[0] - 1.0) * 100.0),
        "calmar": float(full.metrics.calmar),
        "average_exposure": float(np.mean(full.exposure)),
        "full_investment_contract_satisfied": True,
        "sharpe": float(full.metrics.sharpe),
        "annualized": float(full.metrics.annualized_return * 100.0),
        "max_drawdown": float(-full.metrics.max_drawdown * 100.0),
    }


@pytest.mark.parametrize("record_details", [True, False])
@pytest.mark.parametrize("corruption", ["events_sequence", "event_mapping", "fees_length", "fee_mapping"])
def test_summary_keeps_diagnostic_boundary_rejections(summary_episode, record_details, corruption):
    session = EpisodeSession(summary_episode)
    real_step = session.step
    corrupted = False

    def corrupted_step(config):
        nonlocal corrupted
        result = real_step(config)
        if corrupted:
            return result
        corrupted = True
        diagnostics = dict(result.step_result.diagnostics)
        if corruption == "events_sequence":
            diagnostics["account_events"] = "invalid-events"
        elif corruption == "event_mapping":
            diagnostics["account_events"] = ["invalid-event"]
        elif corruption == "fees_length":
            assert result.step_result.fills
            diagnostics["fee_breakdown"] = ()
        else:
            assert result.step_result.fills
            diagnostics["fee_breakdown"] = ["invalid-fee" for _ in result.step_result.fills]
        return replace(result, step_result=replace(result.step_result, diagnostics=diagnostics))

    session.step = corrupted_step
    expected_error = (TypeError, ValueError) if corruption == "fee_mapping" else (ValueError if corruption == "fees_length" else TypeError)
    with pytest.raises(expected_error):
        run_day_config_episode(session, dynamic_provider(), record_details=record_details)


@pytest.mark.parametrize("record_details", [True, False])
def test_summary_does_not_skip_config_validation(summary_episode, record_details):
    schema = ActionSchema()
    base = schema.decode(np.zeros(schema.action_dim))
    config = replace(base, buy_n=21, single_buy_pct=1 / 21)
    with pytest.raises(ValueError):
        run_day_config_episode(EpisodeSession(summary_episode), lambda _: config, record_details=record_details)


@pytest.mark.parametrize("record_details", [None, 0, 1, "false"])
def test_summary_flag_has_no_truthiness_alias(summary_episode, record_details):
    with pytest.raises(TypeError, match="record_details must be bool"):
        run_day_config_episode(EpisodeSession(summary_episode), dynamic_provider(), record_details=record_details)


def test_settlement_gathers_only_known_relevant_codes_after_plan(summary_episode):
    session = EpisodeSession(summary_episode)
    session.reset()
    index = int(np.searchsorted(summary_episode.runtime.trade_dates, np.datetime64("2020-06-19")))
    known = summary_episode.runtime.stock_codes[0]
    account = AccountState(cash=1000.0, positions={"OUTSIDE": 100, known: 100},
        sellable_positions={known: 100, "OUTSIDE": 100}, last_prices={known: 10.0, "OUTSIDE": 10.0},
        nav=3000.0, peak_nav=3000.0)

    class Captured(Exception):
        pass

    class CaptureSimulator:
        def step(self, actual_account, plan, opens, following, *, close_prices, next_preclose_prices,
                 next_delisted_codes, next_decision_date, terminated):
            assert actual_account is account
            relevant = set(account.positions) | set(plan.buy_orders) | {code for code, _ in plan.sell_orders}
            known_relevant = relevant & set(summary_episode.runtime.stock_codes)
            for actual, field_name, day in (
                (opens, "open", index), (following, "open", index + 1),
                (close_prices, "close", index), (next_preclose_prices, "preClose", index + 1),
            ):
                expected = {code: float(summary_episode.runtime.field(field_name)[day, summary_episode.code_to_index[code]])
                            for code in known_relevant}
                assert actual.keys() == expected.keys()
                for code in expected:
                    assert np.float64(actual[code]).tobytes() == np.float64(expected[code]).tobytes()
            expected_delisted = {code for code in known_relevant if summary_episode.runtime.field("delisted_mask")[index + 1, summary_episode.code_to_index[code]]}
            assert set(next_delisted_codes) == expected_delisted
            assert known in next_delisted_codes
            assert "OUTSIDE" not in opens
            assert next_decision_date == "2020-06-20"
            raise Captured

    session._simulator = CaptureSimulator()
    with pytest.raises(Captured):
        session._settle_account(summary_episode.market_at(index), account, dynamic_provider()(np.empty(0)),
                                index=index, next_index=index + 1, terminated=False)
