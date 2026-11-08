from pathlib import Path
from datetime import date

import numpy as np
import pytest

from env.action_schema import ActionSchema
from env.contracts import (
    AccountState,
    Fill,
    Observation,
    OrderPlan,
    PolicyMemory,
)
from trade.journal import DecisionJournal
from trade.runtime import LiveDecision
from trade.post_close import reconcile_pending_execution, summarize_replay
from trade.replay import replay_reports
from trade.executor import AcceptedBrokerOrder, BrokerExecutionError
from trade.main import execute_recorded_decision


def _decision(schema: ActionSchema, decision_date: str) -> LiveDecision:
    config = schema.decode(np.zeros(schema.action_dim, dtype=np.float32))
    observation = Observation(
        stock_panel=np.zeros((2, 3, 4), dtype=np.float32),
        position_panel=np.zeros((3, 6), dtype=np.float32),
        portfolio=np.zeros(7, dtype=np.float32),
        policy_history=np.zeros((2, schema.action_dim + 3), dtype=np.float32),
        time_mask=np.ones(2, dtype=np.bool_),
        pit_universe_mask=np.ones((2, 3), dtype=np.bool_),
        schema_version="observation:test",
        decision_date=decision_date,
    )
    account = AccountState(
        cash=99_000.0,
        positions={"600001.SH": 100},
        sellable_positions={"600001.SH": 100},
        average_costs={"600001.SH": 9.0},
        last_prices={"600001.SH": 10.0},
        mark_provenance={"600001.SH": "broker.last_price"},
        nav=100_000.0,
        peak_nav=100_000.0,
    )
    plan = OrderPlan(
        decision_date=decision_date,
        buy_orders={"600001.SH": 1_000},
        day_config=config,
        diagnostics={
            "full_investment_contract_satisfied": True,
            "prices": {"600001.SH": 10.0},
        },
    )
    return LiveDecision(
        observation=observation,
        action=schema.encode(config),
        day_config=config,
        order_plan=plan,
        account_before=account,
        policy_memory_before=PolicyMemory(),
        snapshot_identity={
            "decision_date": decision_date,
            "runtime_path": "D:/sealed/runtime.npz",
            "runtime_source_sha256": "1" * 64,
            "runtime_schema_hash": "2" * 64,
            "stock_vocabulary_sha256": "3" * 64,
            "stock_count": 3,
            "factor_schema_hash": "4" * 64,
            "observation_schema": observation.schema_version,
            "prefilter_n": 300,
        },
        policy_identity={
            "bundle_version": "test",
            "created_at": "2026-08-28T00:00:00",
            "algorithm": "stable_baselines3.PPO",
            "manifest_sha256": "9" * 64,
            "model_sha256": "5" * 64,
            "normalizer_sha256": "6" * 64,
            "config_sha256": "7" * 64,
            "source_sha256": "8" * 64,
            "action_schema_hash": schema.schema_hash,
            "observation_schema": observation.schema_version,
            "environment_schema_hash": "a" * 64,
        },
    )


def test_journal_roundtrip_and_policy_memory_use_actual_fills(tmp_path: Path):
    schema = ActionSchema()
    journal = DecisionJournal(tmp_path, schema)
    decision = _decision(schema, "2026-08-28")
    journal.record_decision(decision)
    fills = (
        Fill(
            code="600001.SH",
            side="buy",
            quantity=1_000,
            price=10.0,
            fee=12.5,
            timestamp="2026-08-28T09:30:01",
        ),
    )
    memory = journal.record_fills("2026-08-28", fills)

    replay = journal.replay("2026-08-28")
    np.testing.assert_array_equal(replay.action, decision.action)
    np.testing.assert_array_equal(
        replay.observation.stock_panel,
        decision.observation.stock_panel,
    )
    np.testing.assert_array_equal(
        replay.observation.pit_universe_mask,
        decision.observation.pit_universe_mask,
    )
    assert replay.order_plan == decision.order_plan
    assert replay.fills == fills
    assert replay.snapshot_identity == dict(decision.snapshot_identity)
    assert replay.policy_identity == dict(decision.policy_identity)
    assert replay.account_before.mark_provenance == {
        "600001.SH": "broker.last_price"
    }
    assert memory.previous_gross_turnover_ratio == pytest.approx(0.1)
    assert memory.previous_total_cost_ratio == pytest.approx(0.000125)


def test_journal_cold_start_is_explicit_and_only_for_empty_chain(tmp_path: Path):
    schema = ActionSchema()
    journal = DecisionJournal(tmp_path, schema)
    with pytest.raises(FileNotFoundError, match="cold-start"):
        journal.load_prior_state("2026-08-28")
    assert journal.load_prior_state(
        "2026-08-28", initialize_new_chain=True
    ) == (None, PolicyMemory())

    journal.record_decision(_decision(schema, "2026-08-28"))
    journal.record_fills("2026-08-28", ())
    previous_account, memory = journal.load_prior_state("2026-08-29")
    assert previous_account is not None
    assert memory.initialized


def test_journal_refuses_to_overwrite_a_decision(tmp_path: Path):
    schema = ActionSchema()
    journal = DecisionJournal(tmp_path, schema)
    decision = _decision(schema, "2026-08-28")
    journal.record_decision(decision)
    with pytest.raises(FileExistsError):
        journal.record_decision(decision)


def test_post_close_and_multi_day_replay_only_read_journal(tmp_path: Path):
    schema = ActionSchema()
    journal = DecisionJournal(tmp_path, schema)
    decision = _decision(schema, "2026-08-28")
    journal.record_decision(decision)
    journal.record_fills(
        "2026-08-28",
        (
            Fill(
                code="600001.SH",
                side="buy",
                quantity=1_000,
                price=10.0,
                fee=10.0,
            ),
        ),
    )

    summary = summarize_replay(journal.replay("2026-08-28"))
    reports = replay_reports(
        date(2026, 8, 27),
        date(2026, 8, 29),
        journal=journal,
    )

    assert reports == (summary,)
    assert summary.model_sha256 == "5" * 64
    assert summary.runtime_source_sha256 == "1" * 64


def test_unconfirmed_execution_stays_pending_until_terminal_reconciliation(
    tmp_path: Path,
):
    schema = ActionSchema()
    journal = DecisionJournal(tmp_path, schema)
    decision = _decision(schema, "2026-08-28")

    class UnsettledExecutor:
        def execute(self, order_plan):
            raise BrokerExecutionError(
                "terminal state unknown",
                (),
                terminal_confirmed=False,
                accepted_orders=(
                    AcceptedBrokerOrder(
                        order_id=17,
                        side="buy",
                        code="600001.SH",
                        planned_quantity=1_000,
                    ),
                ),
            )

    with pytest.raises(BrokerExecutionError, match="terminal state unknown"):
        execute_recorded_decision(
            decision=decision,
            journal=journal,
            broker_executor=UnsettledExecutor(),
        )

    pending = journal.load_pending_execution("2026-08-28")
    assert pending["state"] == "broker_terminal_unconfirmed"
    assert pending["accepted_orders"] == (
        AcceptedBrokerOrder(17, "buy", "600001.SH", 1_000),
    )
    with pytest.raises(RuntimeError, match="pending terminal reconciliation"):
        journal.replay("2026-08-28")
    with pytest.raises(FileNotFoundError, match="no complete Fill"):
        journal.load_prior_state("2026-08-29")

    final_fill = Fill(
        code="600001.SH",
        side="buy",
        quantity=1_000,
        price=10.0,
        fee=10.0,
    )
    class LaterFillExecutor:
        def reconcile(self, order_plan, accepted_orders):
            assert order_plan == decision.order_plan
            assert accepted_orders == pending["accepted_orders"]
            return (final_fill,)

    reconcile_pending_execution(
        trade_date=date(2026, 8, 28),
        journal=journal,
        broker_executor=LaterFillExecutor(),
    )
    assert journal.replay("2026-08-28").fills == (final_fill,)
    assert journal.load_prior_state("2026-08-29")[1].initialized
