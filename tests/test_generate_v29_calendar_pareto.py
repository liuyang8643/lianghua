import research_generate_v29_calendar_pareto as generator
import json
from pathlib import Path


PLAN = (
    Path(__file__).resolve().parents[1]
    / "results/strategy_opt_20260721/v29_calendar_pareto_preregistered_plan.json"
)


def test_v29_pareto_set_is_fixed_and_contains_only_training_rows():
    payload = generator.build_payload()

    assert payload["selection_scope"] == "training_only_2010_2018"
    assert payload["count"] == 8
    assert [item["candidate_id"] for item in payload["configs"]] == [
        f"v29_pareto_{index:02d}" for index in range(8)
    ]
    assert len(
        {
            (
                item["individual_config"]["buy_n"],
                tuple(item["individual_config"]["weights"].items()),
            )
            for item in payload["configs"]
        }
    ) == 8


def test_v29_pareto_frontier_has_no_internal_dominance():
    rows = generator.build_pareto_rows()

    assert all(
        not generator._dominates(other, row)
        for row in rows
        for other in rows
        if other is not row
    )
    assert all(row["calmar"] >= 2.5 for row in rows)
    assert all(row["average_exposure"] >= 0.45 for row in rows)


def test_v29_first_candidate_is_the_already_audited_v28_weight_winner():
    first = generator.build_payload()["configs"][0]

    assert first["individual_config"]["buy_n"] == 30
    assert first["individual_config"]["weights"] == {
        "PreCloseMarketCap": 1.0,
        "AmihudIlliquidityStrict": 0.5,
        "TrendReversalPreCloseStrict": 0.1,
        "VolumeCVStrict": 0.3,
    }


def test_v29_plan_freezes_exact_frontier_and_natural_calendar_gates():
    plan = json.loads(PLAN.read_text(encoding="utf-8"))

    assert plan["status"] == "preregistered_before_remaining_seven_full_audits"
    assert plan["frontier_definition"]["frontier_count"] == 8
    assert len(plan["candidates"]) == 8
    assert plan["full_audit_contract"]["calendar_blocks"] == [
        "2010-2012",
        "2013-2015",
        "2016-2018",
    ]
    assert plan["full_audit_contract"]["block_initial_nav"] == 1.0
    assert "Reject the whole V29 set" in plan["selection_rule"]


def test_v29_decision_rejects_closed_set_without_holdout():
    decision = json.loads(
        PLAN.with_name("v29_calendar_pareto_training_decision.json").read_text(
            encoding="utf-8"
        )
    )

    assert decision["status"] == "all_eight_candidates_rejected"
    assert decision["strategy_holdout_evaluated"] is False
    assert decision["audit_summary"]["candidate_count"] == 8
    assert decision["audit_summary"]["trade_open_audit_all_pass"] is True
    assert decision["audit_summary"]["candidates_passing_every_gate"] == 0
    assert all(not item["passes_all"] for item in decision["candidates"])
