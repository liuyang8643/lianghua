from __future__ import annotations

import copy
import inspect
import json
from datetime import date
from pathlib import Path

import numpy as np
import pytest

import research_v35_prior_month_turnover_ablation as v35


def _fixed_date_strings() -> list[str]:
    all_dates = np.arange(
        np.datetime64(v35.TRAIN_FIRST_DATE, "D"),
        np.datetime64(v35.TRAIN_LAST_DATE, "D") + np.timedelta64(1, "D"),
    )
    positions = np.linspace(
        0,
        len(all_dates) - 1,
        v35.EXPECTED_TRAIN_DAYS,
        dtype=np.int64,
    )
    selected = all_dates[positions]
    assert len(np.unique(selected)) == v35.EXPECTED_TRAIN_DAYS
    return selected.astype(str).tolist()


def _gate_row(*, role: str, label: str) -> dict:
    return {
        "candidate_index": 0 if role == "control" else 1,
        "role": role,
        "label": label,
        "full_calmar": 3.0,
        "fold_calmars": {
            "2010-2012": 1.6,
            "2013-2015": 4.0,
            "2016-2018": 1.7,
        },
        "worst_fold_calmar": 1.6,
        "robust_score": 2.3,
        "average_exposure": 0.50,
        "annualized": 30.0,
        "max_drawdown": -10.0,
        "sharpe": 1.8,
        "terminal_nav": 10.0,
        "total_return": 900.0,
        "minimum_daily_selectable_candidates": 30,
    }


def _trade_row(
    *,
    price: float = 10.0,
    field: str = "open",
    amount: float | None = None,
) -> dict:
    volume = 100
    return {
        "code": "000001.SZ",
        "action": "buy",
        "date": "2018-12-28",
        "trade_date": date(2018, 12, 28),
        "price": np.float64(price),
        "price_field": field,
        "volume": np.int64(volume),
        "amount": np.float64(price * volume if amount is None else amount),
    }


def test_plan_amendment_hashes_and_decision_fields_are_pinned(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    assert v35._sha256(v35.PLAN_PATH) == v35.EXPECTED_PLAN_SHA256
    assert v35._sha256(v35.AMENDMENT_PATH) == v35.EXPECTED_AMENDMENT_SHA256
    assert (
        v35._sha256(v35.CANDIDATE_SOURCE_PATH)
        == v35.EXPECTED_CANDIDATE_SOURCE_SHA256
    )
    plan = v35.load_and_validate_plan()
    amendment = v35.load_and_validate_amendment()
    assert plan["fixed_candidate"]["weights"] == v35.CANDIDATE_WEIGHTS
    assert plan["primary_gates"]["minimum_daily_selectable_candidates"] == 30
    assert amendment["cost_stress_contract"] == v35.AMENDED_COST_STRESS_CONTRACT
    assert amendment["secondary_order"] == v35.AMENDED_SECONDARY_ORDER

    changed_plan = json.loads(v35.PLAN_PATH.read_text(encoding="utf-8"))
    changed_plan["fixed_candidate"]["weights"][v35.NEW_FACTOR_NAME] = 0.2
    changed_plan_path = tmp_path / "changed_plan.json"
    changed_plan_path.write_text(json.dumps(changed_plan), encoding="utf-8")
    with pytest.raises(ValueError, match="plan SHA256 changed"):
        v35.load_and_validate_plan(changed_plan_path)
    monkeypatch.setattr(v35, "EXPECTED_PLAN_SHA256", v35._sha256(changed_plan_path))
    with pytest.raises(ValueError, match="fixed candidate semantics changed"):
        v35.load_and_validate_plan(changed_plan_path)
    monkeypatch.undo()

    changed_amendment = json.loads(v35.AMENDMENT_PATH.read_text(encoding="utf-8"))
    changed_amendment["cost_stress_contract"]["full_calmar_min"] = 2.49
    changed_amendment_path = tmp_path / "changed_amendment.json"
    changed_amendment_path.write_text(
        json.dumps(changed_amendment), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="amendment SHA256 changed"):
        v35.load_and_validate_amendment(changed_amendment_path)
    monkeypatch.setattr(
        v35, "EXPECTED_AMENDMENT_SHA256", v35._sha256(changed_amendment_path)
    )
    with pytest.raises(ValueError, match="cost-stress contract changed"):
        v35.load_and_validate_amendment(changed_amendment_path)


def test_factor_contract_is_the_fixed_prior_complete_month_turnover():
    factor_class = v35._new_factor_class()
    factor = factor_class()
    assert factor_class.__name__ == v35.NEW_FACTOR_NAME
    assert factor.hist_days == 60
    assert factor.pre_ranked is False
    assert factor.requires_full_history is False

    dates = np.asarray(
        [
            "2018-01-02",
            "2018-01-03",
            "2018-02-01",
            "2018-02-02",
            "2018-03-01",
            "2018-03-02",
        ],
        dtype="datetime64[D]",
    )
    total_share = np.full((6, 1), 2.0)
    volume = np.asarray([[100.0], [300.0], [200.0], [600.0], [1.0], [1.0]])
    result = factor.calc_batch(
        {
            "trade_dates": dates,
            "volume": volume,
            "total_share": total_share,
        }
    )
    assert np.isnan(result[:4]).all()
    # February rates are 1.0 and 3.0, so March is frozen at negative mean -2.
    np.testing.assert_array_equal(result[4:, 0], np.float32(-2.0))


def test_exact_two_arm_config_diff_is_only_the_fixed_weight():
    plan = v35.load_and_validate_plan()
    source = v35.load_control_source()
    candidates = v35.build_fixed_candidates(plan, source)
    assert [(row["role"], row["label"]) for row in candidates] == [
        ("control", v35.CONTROL_LABEL),
        ("candidate", v35.CANDIDATE_LABEL),
    ]
    assert candidates[0]["individual_config"] == source["individual_config"]
    control = copy.deepcopy(candidates[0]["individual_config"])
    candidate = copy.deepcopy(candidates[1]["individual_config"])
    assert control["weights"] == v35.CONTROL_WEIGHTS
    assert candidate["weights"] == v35.CANDIDATE_WEIGHTS
    assert candidates[1]["change_from_control"] == {v35.NEW_FACTOR_NAME: 0.1}
    control.pop("weights")
    candidate.pop("weights")
    assert candidate == control


def test_pool_local_ranking_preserves_dates_and_ignores_outside_pool():
    class DateAwareFactor:
        hist_days = 1

        def calc_batch(self, panel):
            assert np.issubdtype(panel["trade_dates"].dtype, np.datetime64)
            return np.asarray([[1.0, 2.0, 100.0], [3.0, 1.0, -100.0]])

    class AllFilter:
        hist_days = 1

        def calc_batch(self, panel):
            return np.ones((2, 3), dtype=np.float32)

    data = {
        "stock_codes": np.asarray(["600001", "000001", "688001"]),
        "trade_dates": np.asarray(
            ["2018-01-02", "2018-01-03"], dtype="datetime64[D]"
        ),
    }
    arrays, score_keys = v35.compute_research_arrays(
        data, [DateAwareFactor], [AllFilter]
    )
    assert score_keys == {"DateAwareFactor"}
    np.testing.assert_allclose(
        arrays["DateAwareFactor"],
        np.asarray([[0.5, 1.0, 0.0], [1.0, 0.5, 0.0]], dtype=np.float32),
    )


def test_runtime_is_physically_copied_and_loader_deleted_before_factors(
    monkeypatch: pytest.MonkeyPatch,
):
    dates = np.asarray(
        ["2018-12-27", "2018-12-28", "2019-01-02"], dtype="datetime64[D]"
    )
    opens = np.arange(6, dtype=np.float64).reshape(3, 2)
    source = {
        "trade_dates": dates,
        "open": opens,
        "close": opens + 1.0,
        "stock_codes": np.asarray(["600001", "000001"]),
        "issue_price": np.asarray([1.0, 2.0]),
    }
    truncated, discarded = v35.physically_truncate_runtime(source)
    assert discarded == 1
    assert str(truncated["trade_dates"][-1]) == v35.TRAIN_LAST_DATE
    assert not np.shares_memory(truncated["trade_dates"], dates)
    assert not np.shares_memory(truncated["open"], opens)

    loaded = {"loader_owned": object()}
    copied = {
        "trade_dates": np.asarray(
            ["2018-12-27", "2018-12-28"], dtype="datetime64[D]"
        )
    }
    monkeypatch.setattr(v35, "load_runtime_npz", lambda *_args, **_kwargs: loaded)
    monkeypatch.setattr(
        v35, "physically_truncate_runtime", lambda value: (copied, 7)
    )
    result, count = v35._load_physically_truncated_training_panel(60)
    assert result is copied
    assert count == 7

    helper_source = inspect.getsource(v35._load_physically_truncated_training_panel)
    assert helper_source.index("physically_truncate_runtime(loaded_data)") < (
        helper_source.index("del loaded_data")
    ) < helper_source.index("return data")
    run_source = inspect.getsource(v35.run)
    assert run_source.index("_load_physically_truncated_training_panel") < (
        run_source.index("compute_research_arrays(")
    )


def test_daily_exposures_read_canonical_detailed_snapshots():
    detailed_result = {
        "daily_snapshots": [
            {
                "trade_date": "2018-12-27",
                "exposure": np.float64(0.25),
                "rebalance_funds_ratio": 0.10,
                "raw_buy_n_list": ["000001.SZ"],
                "executed_buy_list": ["000001.SZ"],
                "executed_sell_list": [],
            },
            {
                "trade_date": "2018-12-28",
                "exposure": np.float32(0.75),
                "rebalance_funds_ratio": 0.20,
                "raw_buy_n_list": ["600001.SH"],
                "executed_buy_list": ["600001.SH"],
                "executed_sell_list": ["000001.SZ"],
            },
        ]
    }
    exposures = v35._daily_exposures(detailed_result, 2)
    assert exposures.dtype == np.float64
    np.testing.assert_allclose(exposures, [0.25, 0.75], rtol=0.0, atol=0.0)


@pytest.mark.parametrize(
    ("result", "expected_length", "message"),
    [
        ({}, 1, "omitted daily_snapshots"),
        ({"daily_snapshots": ({"exposure": 0.5},)}, 1, "must be a list"),
        ({"daily_snapshots": [{"exposure": 0.5}]}, 2, "wrong length"),
        ({"daily_snapshots": [{"trade_date": "2018-12-28"}]}, 1, "missing numeric exposure"),
        ({"daily_snapshots": [{"exposure": np.nan}]}, 1, "non-finite"),
        ({"daily_snapshots": [{"exposure": np.inf}]}, 1, "non-finite"),
        ({"daily_snapshots": [{"exposure": -0.0001}]}, 1, "outside"),
        ({"daily_snapshots": [{"exposure": 1.0001}]}, 1, "outside"),
    ],
)
def test_daily_exposures_reject_missing_wrong_length_nonfinite_and_out_of_range(
    result: dict,
    expected_length: int,
    message: str,
):
    with pytest.raises(ValueError, match=message):
        v35._daily_exposures(result, expected_length)


def test_control_validity_is_not_affected_by_new_factor_missingness():
    candidates = v35.build_fixed_candidates(
        v35.load_and_validate_plan(), v35.load_control_source()
    )
    shape = (2, 3)
    arrays = {
        f"_factor_valid_{name}": np.ones(shape, dtype=bool)
        for name in v35.BASE_FACTOR_NAMES
    }
    new_valid = np.asarray(
        [[True, False, True], [False, True, True]], dtype=bool
    )
    arrays[f"_factor_valid_{v35.NEW_FACTOR_NAME}"] = new_valid
    for name in v35.EXPECTED_FILTER_NAMES:
        arrays[name] = np.ones(shape, dtype=bool)
    control_masks = v35.candidate_filter_masks(
        arrays, v35.ALL_FACTOR_NAMES, candidates[0]["individual_config"]
    )
    candidate_masks = v35.candidate_filter_masks(
        arrays, v35.ALL_FACTOR_NAMES, candidates[1]["individual_config"]
    )
    np.testing.assert_array_equal(
        control_masks["_active_factor_intersection"], np.ones(shape, dtype=bool)
    )
    np.testing.assert_array_equal(
        candidate_masks["_active_factor_intersection"], new_valid
    )
    v35._verify_candidate_validity_semantics(
        arrays, set(v35.ALL_FACTOR_NAMES), candidates
    )


def test_daily_selectable_count_is_pool_validity_and_enabled_filters():
    arrays = {
        "_factor_valid_active": np.asarray(
            [[True, True, True, True], [True, False, True, True]], dtype=bool
        ),
        "_factor_valid_zero": np.zeros((2, 4), dtype=bool),
        "FilterST": np.asarray(
            [[True, True, False, True], [True, True, True, True]], dtype=bool
        ),
        "FilterLowPrice": np.asarray(
            [[True, False, True, True], [True, True, True, True]], dtype=bool
        ),
    }
    config = {
        "weights": {"active": 1.0, "zero": 0.0},
        "filter_factors": {"FilterST": True, "FilterLowPrice": True},
    }
    counts = v35.daily_selectable_counts(
        arrays, {"active", "zero"}, config, [0, 1], [0, 1, 2]
    )
    np.testing.assert_array_equal(counts, [1, 2])


def test_control_canary_and_primary_gates_are_strict():
    canary = {
        "role": "control",
        "label": v35.CONTROL_LABEL,
        **copy.deepcopy(v35.CONTROL_CANARY),
    }
    v35.validate_control_canary(canary)
    changed = copy.deepcopy(canary)
    changed["fold_calmars"]["2016-2018"] += 2e-12
    with pytest.raises(ValueError, match="control canary mismatch"):
        v35.validate_control_canary(changed)

    control = _gate_row(role="control", label=v35.CONTROL_LABEL)
    control["robust_score"] = 2.0
    control["fold_calmars"]["2010-2012"] = 1.4
    control["fold_calmars"]["2016-2018"] = 1.3
    candidate = _gate_row(role="candidate", label=v35.CANDIDATE_LABEL)
    assert all(v35.primary_gate_checks(candidate, control).values())
    candidate["minimum_daily_selectable_candidates"] = 29
    annotated, decision = v35.annotate_primary_decision([control, candidate])
    assert annotated[1]["passes_all_primary_gates"] is False
    assert decision["status"] == "fixed_candidate_rejected_at_primary_gates"
    assert decision["secondary_audit_allowed"] is False
    assert decision["configuration_frozen"] is False
    assert decision["holdout_open_allowed"] is False


def _fake_natural_metrics(
    *, full_calmar: float = 2.5, worst_fold: float = 1.5
) -> dict:
    full = {
        "annualized": 25.0,
        "max_drawdown": -10.0,
        "sharpe": 1.0,
        "calmar": full_calmar,
        "terminal_nav": 2.0,
    }
    folds = {
        "2010-2012": {**full, "calmar": worst_fold},
        "2013-2015": {**full, "calmar": 3.0},
        "2016-2018": {**full, "calmar": 1.6},
    }
    return {
        "full": full,
        "folds": folds,
        "worst_fold_calmar": worst_fold,
        "robust_score": 0.5 * full_calmar + 0.5 * worst_fold,
    }


def test_secondary_runs_fixed_5bps_and_only_all_pass_can_freeze(
    monkeypatch: pytest.MonkeyPatch,
):
    dates = _fixed_date_strings()
    daily = np.zeros(v35.EXPECTED_TRAIN_DAYS, dtype=np.float64)
    calls = []
    monkeypatch.setattr(
        v35,
        "_rolling_blocks",
        lambda *_args: {"p10_calmar": 1.5814941448598532},
    )
    monkeypatch.setattr(
        v35,
        "_return_concentration",
        lambda *_args: {"largest_year_log_wealth_share": 0.4664240680214493},
    )
    monkeypatch.setattr(
        v35,
        "_report_charts",
        lambda _path: {
            "equity": {
                "trade_dates": dates,
                "rebalance_funds_pct": [1.0] * len(dates),
            }
        },
    )

    def fake_stress(values, rebalanced, bps):
        calls.append((len(values), float(rebalanced[0]), bps))
        return np.asarray(values, dtype=np.float64) - 0.001

    monkeypatch.setattr(v35, "stress_daily_returns", fake_stress)
    monkeypatch.setattr(
        v35, "natural_calendar_metrics", lambda *_args: _fake_natural_metrics()
    )
    audit, rebalance, stressed = v35._secondary_audits(
        dates=dates,
        candidate_daily_returns=daily,
        candidate_report_path=Path("candidate/single_report.html"),
        amendment=v35.load_and_validate_amendment(),
    )
    assert calls == [(v35.EXPECTED_TRAIN_DAYS, 1.0, 5.0)]
    assert audit["passes_all_secondary_gates"] is True
    assert rebalance.shape == stressed.shape == daily.shape

    control = _gate_row(role="control", label=v35.CONTROL_LABEL)
    control["robust_score"] = 2.0
    control["fold_calmars"]["2010-2012"] = 1.4
    control["fold_calmars"]["2016-2018"] = 1.3
    candidate = _gate_row(role="candidate", label=v35.CANDIDATE_LABEL)
    primary_rows, primary = v35.annotate_primary_decision([control, candidate])
    final_rows, final = v35.finalize_training_decision(
        primary_rows, primary, audit
    )
    assert final["configuration_frozen"] is True
    assert final["holdout_open_allowed"] is False
    assert final["holdout_opened"] is False
    assert final_rows[1]["configuration_frozen"] is True

    failed_audit = copy.deepcopy(audit)
    failed_audit["passes_all_secondary_gates"] = False
    failed_audit["status"] = "rejected_at_secondary_gates"
    failed_audit["checks"]["stressed_full_calmar_at_least_2p5"] = False
    failed_rows, failed = v35.finalize_training_decision(
        primary_rows, primary, failed_audit
    )
    assert failed["status"] == "fixed_candidate_rejected_at_secondary_gates"
    assert failed["configuration_frozen"] is False
    assert failed["holdout_open_allowed"] is False
    assert failed_rows[1]["configuration_frozen"] is False


def test_primary_failure_forbids_secondary_and_freeze():
    control = _gate_row(role="control", label=v35.CONTROL_LABEL)
    candidate = _gate_row(role="candidate", label=v35.CANDIDATE_LABEL)
    candidate["minimum_daily_selectable_candidates"] = 29
    rows, primary = v35.annotate_primary_decision([control, candidate])
    final_rows, final = v35.finalize_training_decision(rows, primary, None)
    assert final == primary
    assert final_rows[1]["configuration_frozen"] is False
    with pytest.raises(ValueError, match="secondary audit ran after a primary failure"):
        v35.finalize_training_decision(
            rows,
            primary,
            {"passes_all_secondary_gates": True},
        )


def test_exact_open_audit_rejects_one_ulp_field_and_amount_drift():
    exact = v35._audit_trade_log_exact_open(
        role="control",
        label=v35.CONTROL_LABEL,
        trade_log=[_trade_row()],
        open_price=np.asarray([[10.0]], dtype=np.float64),
        date_indices={"2018-12-28": 0},
        stock_indices={"000001.SZ": 0},
    )
    assert exact["passes"] is True
    assert exact["comparison"].endswith("zero tolerance")
    assert exact["per_trade_checks"][0]["exact_amount_match"] is True

    one_ulp = float(np.nextafter(np.float64(10.0), np.inf))
    amount_ulp = float(np.nextafter(np.float64(1000.0), np.inf))
    failed = v35._audit_trade_log_exact_open(
        role="candidate",
        label=v35.CANDIDATE_LABEL,
        trade_log=[
            _trade_row(price=one_ulp),
            _trade_row(field="close"),
            _trade_row(amount=amount_ulp),
        ],
        open_price=np.asarray([[10.0]], dtype=np.float64),
        date_indices={"2018-12-28": 0},
        stock_indices={"000001.SZ": 0},
    )
    assert failed["passes"] is False
    assert failed["issues"]["open_price_mismatches"] == 1
    assert failed["issues"]["non_open_price_field"] == 1
    assert failed["issues"]["amount_mismatches"] == 1


def test_real_html_decodes_only_truncated_training_klines_for_both_arms(
    tmp_path: Path,
):
    trade_log = [
        {
            "code": "000001.SZ",
            "action": "buy",
            "trade_date": "2010-01-04",
            "price": 10.0,
            "price_field": "open",
            "volume": 100,
            "amount": 1000.0,
        },
        {
            "code": "000001.SZ",
            "action": "sell",
            "trade_date": "2018-12-28",
            "price": 12.0,
            "price_field": "open",
            "volume": 100,
            "amount": 1200.0,
        },
    ]
    runtime_dates = np.asarray(
        ["2009-12-31", "2010-01-04", "2014-06-30", "2018-12-28"],
        dtype="datetime64[D]",
    )
    data = {
        "trade_dates": runtime_dates,
        "open": np.asarray([[9.0], [10.0], [11.0], [12.0]]),
        "high": np.asarray([[9.1], [10.1], [11.1], [12.1]]),
        "low": np.asarray([[8.9], [9.9], [10.9], [11.9]]),
        "close": np.asarray([[9.0], [10.0], [11.0], [12.0]]),
        "amount": np.asarray([[900.0], [1000.0], [1100.0], [1200.0]]),
    }
    report_payload = {
        "individual_config": {"weights": {"factor": 1.0}},
        "total_return": 20.0,
        "daily_returns": [0.0, 20.0],
        "cumulative_returns": [0.0, 20.0],
        "trade_dates": ["2010-01-04", "2018-12-28"],
        "trade_log": trade_log,
        "daily_snapshots": [],
        "positions": [],
        "cleared_positions": [],
        "delist_events": [],
        "stock_name_map": {"000001.SZ": "样本"},
        "holding_stats": {},
        "metrics": {},
        "hs300_returns": [],
        "init_cash": 1_000_000.0,
        "final_asset": 1_200_000.0,
        "period": {
            "start": v35.TRAIN_FIRST_DATE,
            "end": v35.TRAIN_LAST_DATE,
            "trade_start": v35.TRAIN_FIRST_DATE,
            "trade_end": v35.TRAIN_LAST_DATE,
        },
    }
    original = v35.single_report_module._collect_kline_payload
    for label in (v35.CONTROL_LABEL, v35.CANDIDATE_LABEL):
        report_path = v35._generate_train_capped_single_report(
            copy.deepcopy(report_payload),
            tmp_path / label,
            data,
            {"000001.SZ": 0},
        )
        assert v35.single_report_module._collect_kline_payload is original
        decoded = v35._decode_html_kline_payload(report_path)
        assert decoded["000001.SZ"]["d"] == [
            "2010-01-04",
            "2014-06-30",
            "2018-12-28",
        ]
        assert decoded["000001.SZ"]["events"][0]["date"] == "2010-01-04"
        assert decoded["000001.SZ"]["episodes"][0]["window_start"] == (
            v35.TRAIN_FIRST_DATE
        )
        assert decoded["000001.SZ"]["episodes"][0]["window_end"] == (
            v35.TRAIN_LAST_DATE
        )
        audit = v35._audit_html_kline_training_bounds(
            report_path, {"000001.SZ"}
        )
        assert audit["passes"] is True
        rendered = v35._audit_rendered_trade_table_exact_open(
            report_path=report_path,
            open_price=np.asarray(data["open"], dtype=np.float64),
            date_indices={str(value): i for i, value in enumerate(runtime_dates)},
            stock_indices={"000001.SZ": 0},
        )
        assert rendered["passes"] is True


def test_report_payload_is_full_and_train_isolated(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(v35, "compute_per_year_metrics", lambda *_: [])
    monkeypatch.setattr(v35, "compute_hs300_cumulative_returns", lambda *_: [])
    result = {
        "total_return": 1.0,
        "daily_returns": [1.0],
        "cumulative_returns": [1.0],
        "trade_log": [_trade_row()],
        "daily_snapshots": [],
        "positions": [],
        "cleared_positions": [],
        "delist_events": [],
        "stock_name_map": {},
        "holding_stats": {},
        "executed_buy_count": 1,
        "executed_sell_count": 0,
        "delist_count": 0,
        "round_trip_count": 0,
        "final_asset": 1_010_000.0,
        "cleared_positions_count": 0,
        "current_positions_count": 1,
    }
    payload = v35._report_payload(
        config={"weights": {"factor": 1.0}},
        result=result,
        dates=["2018-12-28"],
        legacy_metrics={},
        identity_hashes={"runtime": {"sha256": "abc"}},
        stock_pool_size=100,
    )
    assert payload["trade_log"] is result["trade_log"]
    assert payload["rebalance_rule"]["price_field"] == "open"
    assert payload["report_metadata"]["kline_source"] == (
        "physically_truncated_training_runtime"
    )
    assert payload["report_metadata"]["kline_first_date"] == v35.TRAIN_FIRST_DATE
    assert payload["report_metadata"]["kline_last_date"] == v35.TRAIN_LAST_DATE
    assert payload["report_metadata"][
        "strategy_holdout_available_to_factor_or_worker"
    ] is False


def test_both_arm_report_obligations_are_enforced(tmp_path: Path):
    summaries = []
    for role, label in (
        ("control", v35.CONTROL_LABEL),
        ("candidate", v35.CANDIDATE_LABEL),
    ):
        run_dir = tmp_path / label
        run_dir.mkdir()
        for filename in v35.ARM_REQUIRED_FILENAMES:
            (run_dir / filename).write_text(filename, encoding="utf-8")
        summaries.append(
            {
                "role": role,
                "label": label,
                "full_report_obligation_completed": True,
                "trade_open_audit": {"passes": True},
                "html_kline_training_bounds": {"passes": True},
            }
        )
    v35._validate_report_obligations(tmp_path, summaries)
    (tmp_path / v35.CANDIDATE_LABEL / "record.json").unlink()
    with pytest.raises(FileNotFoundError, match="mandatory artifacts"):
        v35._validate_report_obligations(tmp_path, summaries)


def test_daily_returns_exposures_and_selectable_counts_cover_both_arms(
    tmp_path: Path,
):
    output = tmp_path / "daily_returns.npz"
    v35._write_daily_series(
        output,
        ["2018-12-27", "2018-12-28"],
        {
            "control": np.asarray([0.1, 0.2]),
            "candidate": np.asarray([0.3, 0.4]),
        },
        {
            "control": np.asarray([0.5, 0.6]),
            "candidate": np.asarray([0.7, 0.8]),
        },
        {
            "control": np.asarray([100, 101]),
            "candidate": np.asarray([80, 81]),
        },
    )
    with np.load(output, allow_pickle=False) as saved:
        assert set(saved.files) == {
            "dates",
            "control_daily_returns",
            "candidate_daily_returns",
            "control_daily_exposures",
            "candidate_daily_exposures",
            "control_daily_selectable_candidates",
            "candidate_daily_selectable_candidates",
        }
        np.testing.assert_array_equal(
            saved["candidate_daily_selectable_candidates"], [80, 81]
        )
    with pytest.raises(ValueError, match="both arms"):
        v35._write_daily_series(
            tmp_path / "incomplete.npz",
            ["2018-12-28"],
            {"control": np.asarray([0.0])},
            {
                "control": np.asarray([0.5]),
                "candidate": np.asarray([0.5]),
            },
            {
                "control": np.asarray([30]),
                "candidate": np.asarray([30]),
            },
        )


def test_manifest_hashes_all_generated_files_except_itself(tmp_path: Path):
    output = tmp_path / "output"
    nested = output / v35.CONTROL_LABEL
    nested.mkdir(parents=True)
    (output / "summary.json").write_text("summary", encoding="utf-8")
    (nested / "report.json").write_text("report", encoding="utf-8")
    (output / "manifest.json").write_text("excluded", encoding="utf-8")
    hashes = v35._generated_file_hashes(output)
    assert set(hashes) == {
        "summary.json",
        f"{v35.CONTROL_LABEL}/report.json",
    }
    assert hashes["summary.json"] == v35._sha256(output / "summary.json")


def test_input_hash_recheck_catches_plan_amendment_runtime_or_source_change(
    tmp_path: Path,
):
    identity = {}
    for name in (
        "preregistered_plan",
        "execution_amendment",
        "runtime",
        "candidate_source",
        "authoritative_script",
    ):
        path = tmp_path / f"{name}.bin"
        path.write_bytes(name.encode("ascii"))
        identity[name] = {"path": str(path), "sha256": v35._sha256(path)}
    factor = tmp_path / "factor.py"
    factor.write_text("factor", encoding="utf-8")
    identity["factor_sources"] = {
        "Factor": {"path": str(factor), "sha256": v35._sha256(factor)}
    }
    v35._assert_input_hashes_unchanged(identity)
    identity["runtime"]["path"] = str(tmp_path / "runtime.bin")
    Path(identity["runtime"]["path"]).write_text("changed", encoding="utf-8")
    with pytest.raises(RuntimeError, match="input changed during run"):
        v35._assert_input_hashes_unchanged(identity)


def test_existing_output_fails_before_any_input_read(tmp_path: Path):
    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        v35.run(
            output_dir=output,
            plan_path=tmp_path / "missing_plan.json",
            amendment_path=tmp_path / "missing_amendment.json",
            candidate_source_path=tmp_path / "missing_candidates.json",
        )


def test_authoritative_source_has_exactly_two_full_runs_and_no_forbidden_path():
    run_source = inspect.getsource(v35.run)
    module_source = Path(v35.__file__).read_text(encoding="utf-8")
    assert "lightweight=False" in run_source
    assert "lightweight=True" not in run_source
    assert run_source.count("_backtest_direct(") == 1
    assert "for identity in candidates" in run_source
    assert run_source.index("_validate_report_obligations") < run_source.index(
        "annotate_primary_decision"
    )
    primary_branch = run_source.index(
        'if primary_decision["passes_all_primary_gates"]:'
    )
    assert primary_branch < run_source.index("_secondary_audits(")
    assert "_run_ga(" not in module_source
    assert "warm_start" not in module_source
    assert "holdout_diagnostics" not in module_source
    assert "load_validation" not in module_source
    assert "load_test" not in module_source
    assert "multiprocessing" not in module_source
    assert "subprocess" not in module_source
    assert v35.load_and_validate_plan()["ga_authorized"] is False
    assert v35.load_and_validate_plan()["framework_changes"] is False
