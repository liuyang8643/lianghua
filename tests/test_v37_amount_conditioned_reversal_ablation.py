from __future__ import annotations

import ast
import copy
import gzip
import base64
import inspect
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

import research_v37_amount_conditioned_reversal_ablation as v37


def _fixed_date_strings() -> list[str]:
    all_dates = np.arange(
        np.datetime64(v37.TRAIN_FIRST_DATE, "D"),
        np.datetime64(v37.TRAIN_LAST_DATE, "D") + np.timedelta64(1, "D"),
    )
    positions = np.linspace(
        0,
        len(all_dates) - 1,
        v37.EXPECTED_TRAIN_DAYS,
        dtype=np.int64,
    )
    selected = all_dates[positions]
    assert len(np.unique(selected)) == v37.EXPECTED_TRAIN_DAYS
    return selected.astype(str).tolist()


def _gate_row(*, role: str, label: str) -> dict:
    return {
        "candidate_index": 0 if role == "control" else 1,
        "role": role,
        "label": label,
        "full_calmar": 2.5,
        "fold_calmars": {
            "2010-2012": 1.4,
            "2013-2015": 3.0,
            "2016-2018": 1.3,
        },
        "worst_fold_calmar": 1.5,
        "robust_score": 2.0,
        "average_exposure": 0.45,
        "annualized": 25.0,
        "max_drawdown": -10.0,
        "sharpe": 1.5,
        "terminal_nav": 2.0,
        "total_return": 100.0,
        "minimum_daily_selectable_candidates": 30,
    }


def _trade_row(**updates) -> dict:
    row = {
        "signal_date": "2018-12-28",
        "trade_date": "2018-12-28",
        "date": "2018-12-28",
        "code": "000001.SZ",
        "action": "buy",
        "price_field": "open",
        "price": 10.0,
        "volume": 100,
        "amount": 1000.0,
        "commission": 0.0,
        "income": None,
        "reason": "fixed synthetic trade",
    }
    row.update(updates)
    return row


def _write_report_data(path: Path, report_data: dict) -> None:
    path.write_text(
        '<script id="report-data" type="application/json">'
        + json.dumps(report_data, ensure_ascii=False, allow_nan=False)
        + "</script>",
        encoding="utf-8",
    )


def _encode_kline_payload(payload: dict) -> str:
    raw = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
    return base64.b64encode(gzip.compress(raw)).decode("ascii")


def _valid_kline_payload() -> dict:
    return {
        "000001.SZ": {
            "n": "平安银行",
            "d": [v37.TRAIN_FIRST_DATE, v37.TRAIN_LAST_DATE],
            "o": [9.0, 10.0],
            "h": [9.1, 10.1],
            "l": [8.9, 9.9],
            "c": [9.0, 10.0],
            "a": [900.0, 1000.0],
            "events": [{"date": v37.TRAIN_LAST_DATE, "action": "buy"}],
            "episodes": [
                {
                    "start": v37.TRAIN_LAST_DATE,
                    "end": v37.TRAIN_LAST_DATE,
                    "window_start": v37.TRAIN_FIRST_DATE,
                    "window_end": v37.TRAIN_LAST_DATE,
                }
            ],
        }
    }


def _write_kline_report(path: Path, payload: dict) -> None:
    _write_report_data(
        path,
        {
            "tables": {"trades": {"rows": []}},
            "charts": {"equity": {"trade_dates": []}},
            "kline_b64": _encode_kline_payload(payload),
        },
    )


def _structural_fixture() -> tuple[dict, set[str], list[dict], list[int], np.ndarray]:
    rows = v37.EXPECTED_TRAIN_DAYS + 1
    columns = 30
    shape = (rows, columns)
    arrays = {
        f"_factor_valid_{name}": np.ones(shape, dtype=bool)
        for name in v37.ALL_FACTOR_NAMES
    }
    arrays.update(
        {name: np.ones(shape, dtype=bool) for name in v37.EXPECTED_FILTER_NAMES}
    )
    control_config = {
        "weights": copy.deepcopy(v37.CONTROL_WEIGHTS),
        "filter_factors": {name: True for name in v37.EXPECTED_FILTER_NAMES},
    }
    candidate_config = copy.deepcopy(control_config)
    candidate_config["weights"] = copy.deepcopy(v37.CANDIDATE_WEIGHTS)
    candidates = [
        {
            "role": "control",
            "label": v37.CONTROL_LABEL,
            "individual_config": control_config,
        },
        {
            "role": "candidate",
            "label": v37.CANDIDATE_LABEL,
            "individual_config": candidate_config,
        },
    ]
    return (
        arrays,
        set(v37.ALL_FACTOR_NAMES),
        candidates,
        list(range(v37.EXPECTED_TRAIN_DAYS)),
        np.arange(columns, dtype=np.intp),
    )


def _valid_source_lock(digest: str = "a" * 64) -> dict:
    return {
        "experiment": v37.EXPERIMENT,
        "status": v37.SOURCE_LOCK_STATUS,
        "sources": {
            name: {
                "path": str(path).replace("\\", "/"),
                "sha256": (
                    v37.EXPECTED_NEW_FACTOR_SHA256
                    if name == "factor"
                    else v37.EXPECTED_NEW_FACTOR_TEST_SHA256
                    if name == "factor_tests"
                    else digest
                ),
            }
            for name, path in v37.NEW_V37_SOURCE_PATHS.items()
        },
    }


def _v36_npz_fields() -> set[str]:
    return {
        "dates",
        "control_daily_returns",
        "candidate_daily_returns",
        "control_daily_exposures",
        "candidate_daily_exposures",
        "control_daily_selectable_candidates",
        "candidate_daily_selectable_candidates",
    }


def _v36_control_fields() -> set[str]:
    return {
        "dates",
        "control_daily_returns",
        "control_daily_exposures",
        "control_daily_selectable_candidates",
    }


def _synthetic_v36_npz_arrays() -> dict[str, np.ndarray]:
    dates = np.asarray(_fixed_date_strings(), dtype="U10")
    days = v37.EXPECTED_TRAIN_DAYS
    return {
        "dates": dates,
        "control_daily_returns": np.zeros(days, dtype=np.float64),
        "candidate_daily_returns": np.full(days, 0.01, dtype=np.float64),
        "control_daily_exposures": np.full(days, 0.50, dtype=np.float64),
        "candidate_daily_exposures": np.full(days, 0.51, dtype=np.float64),
        "control_daily_selectable_candidates": np.full(
            days, 138, dtype=np.int64
        ),
        "candidate_daily_selectable_candidates": np.full(
            days, 137, dtype=np.int64
        ),
    }


def _write_synthetic_v36_npz(
    path: Path, arrays: dict[str, np.ndarray]
) -> None:
    assert set(arrays) == _v36_npz_fields()
    np.savez_compressed(path, **arrays)


def test_plan_sha_and_execution_semantics_are_strictly_pinned(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    assert v37._sha256(v37.PLAN_PATH) == (
        "4e5a3458813faa8495b46c01242dbf4e91a261eefd55bd9f8d9cd44b50581cf6"
    )
    plan = v37.load_and_validate_plan()
    assert plan["factor"]["class_name"] == v37.NEW_FACTOR_NAME
    assert plan["factor"]["hist_days"] == 20
    assert plan["control"]["fixed_training_metrics"] == (
        v37.CONTROL_FIXED_TRAINING_METRICS
    )
    assert plan["fixed_candidate"]["weights"] == v37.CANDIDATE_WEIGHTS
    assert plan["decision_order"] == v37.EXPECTED_DECISION_ORDER
    assert plan["ga_authorized"] is False
    assert plan["parameter_search_authorized"] is False
    assert plan["framework_changes_authorized"] is False
    assert plan["retry_authorized"] is False
    assert plan["amendment_authorized"] is False
    assert plan["training_holdout_path_authorized"] is False
    assert (
        plan["configuration_freeze_allowed_before_all_training_gates"] is False
    )
    secondary = plan["secondary_training_gates_only_after_every_primary_gate_passes"]
    assert secondary["rolling_3y_window_days"] == 756
    assert secondary["rolling_step_days"] == 63
    assert "appending n-756 exactly once" in secondary["rolling_start_rule"]
    assert "default linear method" in secondary["rolling_p10_algorithm"]
    assert secondary["additional_one_way_cost_bps_each_buy_and_sell"] == 5.0
    assert plan["factor"]["formula"] == v37.FACTOR_FORMULA
    assert plan["factor"]["range"] == [-2.0, 2.0]
    assert plan["new_v37_external_source_lock"][
        "exact_source_roles_and_paths"
    ] == {
        name: str(path).replace("\\", "/")
        for name, path in v37.NEW_V37_SOURCE_PATHS.items()
    }

    mutations = [
        ("factor", "hist_days", 19),
        ("control", "buy_n", 29),
        ("fixed_candidate", "buy_n", 29),
        ("sealed_holdout", "may_be_loaded_by_factor_or_training_worker", True),
        ("ga_authorized", None, True),
    ]
    original = json.loads(v37.PLAN_PATH.read_text(encoding="utf-8"))
    for index, (section, field, value) in enumerate(mutations):
        changed = copy.deepcopy(original)
        if field is None:
            changed[section] = value
        else:
            changed[section][field] = value
        path = tmp_path / f"changed_plan_{index}.json"
        path.write_text(json.dumps(changed), encoding="utf-8")
        with pytest.raises(ValueError, match="plan SHA256 changed"):
            v37.load_and_validate_plan(path)
        monkeypatch.setattr(v37, "EXPECTED_PLAN_SHA256", v37._sha256(path))
        with pytest.raises(ValueError):
            v37.load_and_validate_plan(path)
        monkeypatch.undo()


def test_every_execution_bearing_plan_section_rejects_a_rehashed_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    original = json.loads(v37.PLAN_PATH.read_text(encoding="utf-8"))

    def mutations():
        changed = copy.deepcopy(original)
        changed["decision_order"][0] += " changed"
        yield "decision_order", changed
        changed = copy.deepcopy(original)
        changed["structural_isolation"]["failure_policy"] += " changed"
        yield "structural", changed
        changed = copy.deepcopy(original)
        changed["primary_training_gates"]["control_canary"][
            "metrics_compared"
        ].pop()
        yield "control_canary", changed
        for field, value in (
            ("rolling_3y_window_days", 755),
            ("rolling_step_days", 62),
            ("rolling_start_rule", "changed"),
            ("rolling_p10_algorithm", "changed"),
            ("concentration_algorithm", "changed"),
        ):
            changed = copy.deepcopy(original)
            changed[
                "secondary_training_gates_only_after_every_primary_gate_passes"
            ][field] = value
            yield f"secondary_{field}", changed
        changed = copy.deepcopy(original)
        changed[
            "secondary_training_gates_only_after_every_primary_gate_passes"
        ]["cost_function"]["formula"] = "changed"
        yield "cost_formula", changed
        changed = copy.deepcopy(original)
        changed[
            "training_integrity_gates_required_before_any_metric_decision_or_freeze"
        ]["machine_trade_audit"] = "changed"
        yield "integrity", changed
        changed = copy.deepcopy(original)
        changed["sealed_holdout"]["test_actual_trade_dates"][1] = "2026-07-21"
        yield "sealed_holdout", changed
        changed = copy.deepcopy(original)
        changed["frozen_holdout_sequence"]["validation"][
            "required_trade_day_count"
        ] = 971
        yield "frozen_sequence", changed

    for name, changed in mutations():
        path = tmp_path / f"semantic_{name}.json"
        path.write_text(json.dumps(changed), encoding="utf-8")
        monkeypatch.setattr(v37, "EXPECTED_PLAN_SHA256", v37._sha256(path))
        with pytest.raises(ValueError):
            v37.load_and_validate_plan(path)
        monkeypatch.undo()


def test_preverified_factor_and_v36_control_reference_hashes_are_exact():
    pinned = {
        v37.NEW_V37_SOURCE_PATHS["factor"]: v37.EXPECTED_NEW_FACTOR_SHA256,
        v37.NEW_V37_SOURCE_PATHS[
            "factor_tests"
        ]: v37.EXPECTED_NEW_FACTOR_TEST_SHA256,
        v37.CONTROL_DAILY_REFERENCE_PATH: (
            v37.EXPECTED_CONTROL_DAILY_REFERENCE_SHA256
        ),
        v37.V36_FORMAL_MANIFEST_PATH: v37.EXPECTED_V36_FORMAL_MANIFEST_SHA256,
        v37.V36_INDEPENDENT_SOURCE_LOCK_PATH: (
            v37.EXPECTED_V36_INDEPENDENT_SOURCE_LOCK_SHA256
        ),
    }
    assert {path: v37._sha256(path) for path in pinned} == pinned
    for mapping in (
        v37.EXPECTED_BASE_SOURCE_SHA256,
        v37.EXPECTED_EXECUTION_SOURCE_SHA256,
    ):
        assert {
            path: v37._sha256(Path(path)) for path in mapping
        } == mapping


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda payload: payload.update(extra=True), "schema changed"),
        (
            lambda payload: payload["sources"].pop("factor_tests"),
            "source set changed",
        ),
        (
            lambda payload: payload["sources"].update(
                extra={"path": "extra.py", "sha256": "a" * 64}
            ),
            "source set changed",
        ),
        (
            lambda payload: payload["sources"]["factor"].update(
                path="factor_db/factors/Wrong.py"
            ),
            "path changed",
        ),
        (
            lambda payload: payload["sources"]["adapter"].update(
                sha256="A" * 64
            ),
            "SHA256 is invalid",
        ),
        (
            lambda payload: payload["sources"]["factor"].update(
                sha256="f" * 64
            ),
            "factor SHA256 changed",
        ),
        (
            lambda payload: payload["sources"]["factor_tests"].update(
                sha256="f" * 64
            ),
            "factor-tests SHA256 changed",
        ),
        (
            lambda payload: payload.update(status="not_independently_verified"),
            "identity/status changed",
        ),
    ],
)
def test_source_lock_rejects_missing_extra_and_wrong_identity(
    tmp_path: Path,
    mutation,
    message: str,
):
    payload = _valid_source_lock()
    mutation(payload)
    path = tmp_path / "source_lock.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        v37.load_and_validate_source_lock(path)


def test_source_lock_missing_file_fails_loudly(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        v37.load_and_validate_source_lock(tmp_path / "missing_source_lock.json")


def test_capture_identity_rejects_validly_formed_but_wrong_source_digest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    repo_root = Path(v37.__file__).resolve().parent
    factor_paths = {
        name: repo_root / "factor_db" / "factors" / f"{name}.py"
        for name in v37.BASE_FACTOR_NAMES
    }
    factor_paths[v37.NEW_FACTOR_NAME] = repo_root / v37.NEW_V37_SOURCE_PATHS[
        "factor"
    ]
    filter_path = repo_root / "factor_db" / "factors" / "filter.py"
    factor_classes = [type(name, (), {}) for name in v37.ALL_FACTOR_NAMES]
    filter_classes = [type(name, (), {}) for name in v37.EXPECTED_FILTER_NAMES]
    source_paths = {
        **{
            factor_class: factor_paths[factor_class.__name__]
            for factor_class in factor_classes
        },
        **{filter_class: filter_path for filter_class in filter_classes},
    }
    monkeypatch.setattr(v37, "_source_path", lambda value: source_paths[value])
    source_digests = {
        "factor": v37.EXPECTED_NEW_FACTOR_SHA256,
        "factor_tests": v37.EXPECTED_NEW_FACTOR_TEST_SHA256,
        "adapter": f"{3:064x}",
        "adapter_tests": f"{4:064x}",
    }
    lock = _valid_source_lock()
    for name, digest in source_digests.items():
        lock["sources"][name]["sha256"] = digest
    lock_path = tmp_path / "source_lock.json"
    lock_path.write_text(json.dumps(lock), encoding="utf-8")

    def fake_identity_entry(path: Path) -> dict[str, str]:
        resolved = Path(path).resolve()
        try:
            relative = str(resolved.relative_to(repo_root)).replace("\\", "/")
        except ValueError:
            relative = ""
        if relative in v37.EXPECTED_BASE_SOURCE_SHA256:
            digest = v37.EXPECTED_BASE_SOURCE_SHA256[relative]
        elif relative in v37.EXPECTED_EXECUTION_SOURCE_SHA256:
            digest = v37.EXPECTED_EXECUTION_SOURCE_SHA256[relative]
        else:
            new_name = next(
                (
                    name
                    for name, source_path in v37.NEW_V37_SOURCE_PATHS.items()
                    if relative == str(source_path).replace("\\", "/")
                ),
                None,
            )
            if new_name is not None:
                digest = source_digests[new_name]
            elif resolved == v37.PLAN_PATH.resolve():
                digest = v37.EXPECTED_PLAN_SHA256
            elif resolved == v37.CANDIDATE_SOURCE_PATH.resolve():
                digest = v37.EXPECTED_CANDIDATE_SOURCE_SHA256
            elif resolved == v37.CONTROL_DAILY_REFERENCE_PATH.resolve():
                digest = v37.EXPECTED_CONTROL_DAILY_REFERENCE_SHA256
            elif resolved == v37.V36_FORMAL_MANIFEST_PATH.resolve():
                digest = v37.EXPECTED_V36_FORMAL_MANIFEST_SHA256
            elif resolved == v37.V36_INDEPENDENT_SOURCE_LOCK_PATH.resolve():
                digest = v37.EXPECTED_V36_INDEPENDENT_SOURCE_LOCK_SHA256
            elif resolved == v37.EXPECTED_RUNTIME_PATH.resolve():
                digest = v37.EXPECTED_RUNTIME_SHA256
            else:
                digest = "e" * 64
        return {"path": str(resolved).replace("\\", "/"), "sha256": digest}

    monkeypatch.setattr(v37, "_identity_entry", fake_identity_entry)
    identity = v37._capture_input_identity(
        plan_path=v37.PLAN_PATH,
        source_lock_path=lock_path,
        candidate_source_path=v37.CANDIDATE_SOURCE_PATH,
        control_daily_reference_path=v37.CONTROL_DAILY_REFERENCE_PATH,
        runtime_path=v37.EXPECTED_RUNTIME_PATH,
        script_path=Path(v37.__file__),
        factor_classes=factor_classes,
        filter_classes=filter_classes,
    )
    assert set(identity["new_v37_sources"]) == set(v37.NEW_V37_SOURCE_PATHS)
    assert "independent_new_source_lock" in identity
    assert identity["v36_formal_manifest"]["sha256"] == (
        v37.EXPECTED_V36_FORMAL_MANIFEST_SHA256
    )
    assert identity["v36_independent_source_lock"]["sha256"] == (
        v37.EXPECTED_V36_INDEPENDENT_SOURCE_LOCK_SHA256
    )

    lock["sources"]["adapter"]["sha256"] = "f" * 64
    lock_path.write_text(json.dumps(lock), encoding="utf-8")
    with pytest.raises(ValueError, match="current source differs"):
        v37._capture_input_identity(
            plan_path=v37.PLAN_PATH,
            source_lock_path=lock_path,
            candidate_source_path=v37.CANDIDATE_SOURCE_PATH,
            control_daily_reference_path=v37.CONTROL_DAILY_REFERENCE_PATH,
            runtime_path=v37.EXPECTED_RUNTIME_PATH,
            script_path=Path(v37.__file__),
            factor_classes=factor_classes,
            filter_classes=filter_classes,
        )


@pytest.mark.parametrize(
    "stage", ["structure", "control", "candidate", "manifest"]
)
def test_any_locked_source_change_is_detected_at_every_recheck_stage(
    tmp_path: Path,
    stage: str,
):
    source = tmp_path / f"{stage}.py"
    source.write_text("verified source\n", encoding="utf-8")
    identity = {
        "new_v37_sources": {
            "adapter": {
                "path": str(source),
                "sha256": v37._sha256(source),
            }
        }
    }
    v37._assert_input_hashes_unchanged(identity)
    source.write_text("changed by one byte!\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="input changed during run"):
        v37._assert_input_hashes_unchanged(identity)


def test_source_rechecks_are_positioned_before_structure_each_arm_and_manifest():
    run_source = inspect.getsource(v37.run)
    arm_source = inspect.getsource(v37._run_authoritative_arm)
    manifest_source = inspect.getsource(v37._write_and_verify_manifest)
    assert run_source.index("_assert_input_hashes_unchanged(identity_hashes)") < (
        run_source.index("_structural_isolation_audit(")
    )
    assert arm_source.index("_assert_input_hashes_unchanged(identity_hashes)") < (
        arm_source.index("portfolio_call_sequence.append(role)")
    ) < arm_source.index("_backtest_direct(")
    assert manifest_source.count("_assert_input_hashes_unchanged(identity_hashes)") == 2
    assert manifest_source.index("_assert_input_hashes_unchanged(identity_hashes)") < (
        manifest_source.index("_generated_file_hashes(output_dir)")
    ) < manifest_source.index("_write_json(manifest_path")


@pytest.mark.parametrize(
    ("role", "initial_sequence"),
    [("control", []), ("candidate", ["control"])],
)
def test_each_real_authoritative_arm_calls_once_lightweight_false_without_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    role: str,
    initial_sequence: list[str],
):
    class StopAfterCall(RuntimeError):
        pass

    calls = []
    lock_checks = []
    identity = {
        "role": role,
        "label": v37.CONTROL_LABEL if role == "control" else v37.CANDIDATE_LABEL,
        "individual_config": {
            "weights": copy.deepcopy(
                v37.CONTROL_WEIGHTS if role == "control" else v37.CANDIDATE_WEIGHTS
            ),
            "filter_factors": {
                name: True for name in v37.EXPECTED_FILTER_NAMES
            },
            "stock_pool": list(v37.RANK_PREFIXES),
            "timing_enabled": False,
            "trend_risk_overlay": {"enabled": True, "mode": "dual_completed"},
            "buy_n": 30,
            "sell_m": 30,
            "holding_period": 1,
            "limit_up_protection": True,
            "rebalance": True,
        },
    }
    arrays = {
        **{name: np.ones((1, 1), dtype=np.float32) for name in v37.ALL_FACTOR_NAMES},
        **{
            f"_factor_valid_{name}": np.ones((1, 1), dtype=bool)
            for name in v37.ALL_FACTOR_NAMES
        },
        **{
            name: np.ones((1, 1), dtype=bool)
            for name in v37.EXPECTED_FILTER_NAMES
        },
    }
    monkeypatch.setattr(
        v37,
        "compute_configured_timing_multipliers",
        lambda **_kwargs: np.ones(1, dtype=np.float64),
    )
    monkeypatch.setattr(
        v37,
        "_assert_input_hashes_unchanged",
        lambda value: lock_checks.append(value),
    )

    def stop_backtest(*_args, **kwargs):
        calls.append(kwargs)
        raise StopAfterCall

    monkeypatch.setattr(v37, "_backtest_direct", stop_backtest)
    sequence = list(initial_sequence)
    with pytest.raises(StopAfterCall):
        v37._run_authoritative_arm(
            identity=identity,
            output_dir=tmp_path,
            data={"stock_codes": np.asarray(["000001.SZ"])},
            arrays=arrays,
            score_keys=set(v37.ALL_FACTOR_NAMES),
            valid_dates=[datetime(2018, 12, 28)],
            date_indices=[0],
            dates=["2018-12-28"],
            stock_codes=["000001.SZ"],
            stock_indices={"000001.SZ": 0},
            pool_columns=np.asarray([0], dtype=np.intp),
            list_dates_map={},
            runtime_open=np.asarray([[10.0]]),
            runtime_date_indices={"2018-12-28": 0},
            post_train_rows=0,
            identity_hashes={"stage": role},
            expected_selectable=np.asarray([1]),
            portfolio_call_sequence=sequence,
        )
    assert sequence == [*initial_sequence, role]
    assert len(lock_checks) == 1
    assert len(calls) == 1
    assert calls[0]["lightweight"] is False


def test_training_entrypoint_has_no_amendment_old_factor_search_or_stage_path():
    module_source = Path(v37.__file__).read_text(encoding="utf-8")
    tree = ast.parse(module_source)
    imported_modules = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    imported_from_modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    called_names = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "CompletedSignedAmountImbalance20Strict" not in imported_modules
    assert (
        "factor_db.factors.CompletedSignedAmountImbalance20Strict"
        not in imported_from_modules
    )
    assert "research_v36_signed_amount_imbalance_ablation" not in (
        imported_from_modules | imported_modules
    )
    assert "_run_ga" not in called_names
    assert "subprocess" not in imported_modules
    assert "multiprocessing" not in imported_modules
    assert list(inspect.signature(v37.run).parameters) == ["output_dir"]
    main_source = inspect.getsource(v37.main)
    assert main_source.count("add_argument(") == 1
    assert '"--output"' in main_source
    for forbidden in (
        "--amendment",
        "--warm-start",
        "--resume",
        "--stage",
        "--start-date",
        "--end-date",
        "load_validation",
        "load_test",
        "holdout_diagnostics",
    ):
        assert forbidden not in main_source
    arm_source = inspect.getsource(v37._run_authoritative_arm)
    assert arm_source.count("_backtest_direct(") == 1
    assert "lightweight=False" in arm_source
    assert "lightweight=True" not in arm_source
    assert "retry" not in arm_source.lower()


def test_module_has_no_hidden_old_factor_ga_search_resume_or_holdout_executor():
    module_source = Path(v37.__file__).read_text(encoding="utf-8")
    tree = ast.parse(module_source)
    names = {
        node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
    }
    attributes = {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    }
    definitions = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }
    for forbidden in (
        "AMENDMENT_PATH",
        "CompletedSignedAmountImbalance20Strict",
        "_run_ga",
        "run_ga",
        "warm_start",
        "resume",
        "load_validation",
        "load_test",
        "run_validation",
        "run_test",
        "holdout_diagnostics",
    ):
        assert forbidden not in names | attributes | definitions
    assert str(v37.CONTROL_DAILY_REFERENCE_PATH).replace("\\", "/") == (
        "results/strategy_opt_20260721/"
        "v36_signed_amount_imbalance_fixed_ablation/daily_returns.npz"
    )
    run_source = inspect.getsource(v37.run)
    for holdout_date in (
        "2019-01-01",
        "2019-01-02",
        "2022-12-30",
        "2023-01-01",
        "2023-01-03",
        "2026-07-22",
    ):
        assert holdout_date not in run_source


def test_new_factor_loader_is_exactly_hist20_and_not_a_profile_mutation():
    factor_class = v37._new_factor_class()
    assert factor_class.__name__ == v37.NEW_FACTOR_NAME
    assert factor_class.hist_days == 20
    assert factor_class.pre_ranked is False
    assert factor_class.requires_full_history is False
    source = inspect.getsource(v37._new_factor_class)
    assert "CompletedAmountConditionedReversal20Strict" in source
    assert "get_profile" not in source


def test_ranking_is_confined_to_the_pinned_4737_strategy_pool(
    monkeypatch: pytest.MonkeyPatch,
):
    assert v37.EXPECTED_RANK_POOL_SIZE == 4737
    run_source = inspect.getsource(v37.run)
    assert "len(pool_columns) != EXPECTED_RANK_POOL_SIZE" in run_source
    calls: list[np.ndarray] = []

    class RankProbe:
        def calc_batch(self, data):
            return data["probe"]

    def fake_ranks(values: np.ndarray) -> np.ndarray:
        calls.append(values.copy())
        return np.full(values.shape, 0.25, dtype=np.float32)

    monkeypatch.setattr(v37, "scores_to_ranks", fake_ranks)
    probe = np.asarray(
        [[1.0, 2.0, 3.0, 1.0e30, -1.0e30]], dtype=np.float64
    )
    arrays, keys = v37.compute_research_arrays(
        {
            "stock_codes": np.asarray(
                ["600000.SH", "000001.SZ", "300001.SZ", "688001.SH", "430001.BJ"]
            ),
            "trade_dates": np.asarray(["2018-12-28"], dtype="datetime64[D]"),
            "probe": probe,
        },
        [RankProbe],
        [],
    )
    assert keys == {"RankProbe"}
    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0], probe[:, :3].astype(np.float32))
    np.testing.assert_array_equal(arrays["RankProbe"][:, :3], 0.25)
    np.testing.assert_array_equal(arrays["RankProbe"][:, 3:], 0.0)


def test_runtime_is_loaded_with_60_rows_then_physically_capped_before_factors(
    monkeypatch: pytest.MonkeyPatch,
):
    calls: list[tuple[str, object]] = []
    loaded = {"untruncated": True}
    truncated = {
        "trade_dates": np.asarray([v37.TRAIN_LAST_DATE], dtype="datetime64[D]")
    }
    monkeypatch.setattr(v37, "_requested_training_dates", lambda: ["train-only"])

    def fake_load(dates, *, max_lookback):
        calls.append(("load", (dates, max_lookback)))
        return loaded

    def fake_truncate(panel):
        calls.append(("truncate", panel))
        return truncated, 1831

    monkeypatch.setattr(v37, "load_runtime_npz", fake_load)
    monkeypatch.setattr(v37, "physically_truncate_runtime", fake_truncate)
    result, post_rows = v37._load_physically_truncated_training_panel()
    assert calls == [
        ("load", (["train-only"], 60)),
        ("truncate", loaded),
    ]
    assert result is truncated
    assert post_rows == 1831
    loader_source = inspect.getsource(v37._load_physically_truncated_training_panel)
    assert loader_source.index("max_lookback=60") < loader_source.index(
        "physically_truncate_runtime(loaded_data)"
    ) < loader_source.index("del loaded_data")
    run_source = inspect.getsource(v37.run)
    assert run_source.index("_load_physically_truncated_training_panel()") < (
        run_source.index("compute_research_arrays(")
    )


def test_full_panel_v37_amihud_one_bit_mismatch_aborts_before_any_arm():
    arrays, score_keys, candidates, date_indices, pool_columns = (
        _structural_fixture()
    )
    arrays[f"_factor_valid_{v37.NEW_FACTOR_NAME}"][-1, -1] = False
    arm_calls = []
    with pytest.raises(ValueError, match="finite mask differs from Amihud"):
        v37._structural_isolation_audit(
            arrays=arrays,
            score_keys=score_keys,
            candidates=candidates,
            date_indices=date_indices,
            pool_columns=pool_columns,
        )
        arm_calls.append("unreachable")
    assert arm_calls == []


@pytest.mark.parametrize(
    "mask_name",
    ["_active_factor_intersection", *v37.EXPECTED_FILTER_NAMES],
)
def test_each_training_pool_mask_one_bit_mismatch_aborts_at_zero_calls(
    mask_name: str,
    monkeypatch: pytest.MonkeyPatch,
):
    arrays, score_keys, candidates, date_indices, pool_columns = (
        _structural_fixture()
    )
    shape = arrays[f"_factor_valid_{v37.NEW_FACTOR_NAME}"].shape

    def fake_masks(_arrays, _score_keys, config):
        masks = {
            "_active_factor_intersection": np.ones(shape, dtype=bool),
            **{
                name: np.ones(shape, dtype=bool)
                for name in v37.EXPECTED_FILTER_NAMES
            },
        }
        if v37.NEW_FACTOR_NAME in config["weights"]:
            masks[mask_name][0, 0] = False
        return masks

    monkeypatch.setattr(v37, "candidate_filter_masks", fake_masks)
    arm_calls = []
    with pytest.raises(ValueError, match="control/candidate mask differs"):
        v37._structural_isolation_audit(
            arrays=arrays,
            score_keys=score_keys,
            candidates=candidates,
            date_indices=date_indices,
            pool_columns=pool_columns,
        )
        arm_calls.append("unreachable")
    assert arm_calls == []


@pytest.mark.parametrize(
    ("comparison_call", "message"),
    [(6, "composite pool masks differ"), (7, "daily selectable counts differ")],
)
def test_defensive_composite_and_count_mismatch_branches_abort_at_zero_calls(
    comparison_call: int,
    message: str,
    monkeypatch: pytest.MonkeyPatch,
):
    arrays, score_keys, candidates, date_indices, pool_columns = (
        _structural_fixture()
    )
    original = np.array_equal
    calls = 0

    def targeted_array_equal(left, right, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == comparison_call:
            return False
        return original(left, right, *args, **kwargs)

    monkeypatch.setattr(v37.np, "array_equal", targeted_array_equal)
    arm_calls = []
    with pytest.raises(ValueError, match=message):
        v37._structural_isolation_audit(
            arrays=arrays,
            score_keys=score_keys,
            candidates=candidates,
            date_indices=date_indices,
            pool_columns=pool_columns,
        )
        arm_calls.append("unreachable")
    assert arm_calls == []


def _set_metric(row: dict, path: str, value: float | int) -> None:
    if path.startswith("fold_calmars."):
        row["fold_calmars"][path.split(".", 1)[1]] = value
    else:
        row[path] = value


@pytest.mark.parametrize(
    "metric_path",
    [
        "full_calmar",
        "fold_calmars.2010-2012",
        "fold_calmars.2013-2015",
        "fold_calmars.2016-2018",
        "worst_fold_calmar",
        "robust_score",
        "average_exposure",
        "annualized",
        "max_drawdown",
        "sharpe",
        "terminal_nav",
        "total_return",
        "minimum_daily_selectable_candidates",
    ],
)
def test_every_control_canary_metric_obeys_the_absolute_tolerance_boundary(
    metric_path: str,
):
    row = {
        "role": "control",
        "label": v37.CONTROL_LABEL,
        **copy.deepcopy(v37.CONTROL_CANARY),
    }
    if metric_path.startswith("fold_calmars."):
        expected = row["fold_calmars"][metric_path.split(".", 1)[1]]
    else:
        expected = row[metric_path]
    if metric_path == "minimum_daily_selectable_candidates":
        v37.validate_control_canary(row)
        _set_metric(row, metric_path, int(expected) + 1)
    else:
        edge = float(expected) + v37.CANARY_ABSOLUTE_TOLERANCE
        inside = np.nextafter(edge, float(expected))
        outside = np.nextafter(edge, np.inf)
        _set_metric(row, metric_path, inside)
        v37.validate_control_canary(row)
        _set_metric(row, metric_path, outside)
    with pytest.raises(ValueError, match=f"control canary mismatch for {metric_path.split('.')[-1]}"):
        v37.validate_control_canary(row)


def test_control_canary_selectable_count_is_exact_integer_not_float_tolerance():
    row = {
        "role": "control",
        "label": v37.CONTROL_LABEL,
        **copy.deepcopy(v37.CONTROL_CANARY),
    }
    row["minimum_daily_selectable_candidates"] = np.int64(138)
    v37.validate_control_canary(row)
    for invalid in (
        138.0,
        np.float64(138.0),
        np.nextafter(np.float64(138.0), np.inf),
        True,
        139,
    ):
        row["minimum_daily_selectable_candidates"] = invalid
        with pytest.raises(ValueError, match="minimum selectable|mismatch"):
            v37.validate_control_canary(row)


@pytest.mark.parametrize(
    ("field", "mutate"),
    [
        ("dates", lambda value: np.asarray([value[0], "2018-12-27"], dtype="U10")),
        (
            "control_daily_returns",
            lambda value: np.asarray(
                [value[0], np.nextafter(value[1], np.inf)], dtype=np.float64
            ),
        ),
        (
            "control_daily_exposures",
            lambda value: np.asarray(
                [value[0], np.nextafter(value[1], np.inf)], dtype=np.float64
            ),
        ),
        (
            "control_daily_selectable_candidates",
            lambda value: np.asarray([value[0], value[1] + 1], dtype=np.int64),
        ),
    ],
)
def test_control_reference_uses_np_array_equal_for_all_four_arrays(
    field: str,
    mutate,
):
    dates = [v37.TRAIN_FIRST_DATE, v37.TRAIN_LAST_DATE]
    returns = np.asarray([0.0, 0.1], dtype=np.float64)
    exposures = np.asarray([0.5, 0.6], dtype=np.float64)
    counts = np.asarray([138, 139], dtype=np.int64)
    reference = {
        "dates": np.asarray(dates, dtype="U10"),
        "control_daily_returns": returns.copy(),
        "control_daily_exposures": exposures.copy(),
        "control_daily_selectable_candidates": counts.copy(),
    }
    audit = v37._validate_control_daily_reference(
        reference=reference,
        dates=dates,
        daily_returns=returns,
        daily_exposures=exposures,
        daily_selectable=counts,
    )
    assert audit["comparison_tolerance"] == 0
    assert all(
        item["np_array_equal"] is True for item in audit["arrays"].values()
    )
    reference[field] = mutate(reference[field])
    with pytest.raises(ValueError, match=f"pinned V36 {field}"):
        v37._validate_control_daily_reference(
            reference=reference,
            dates=dates,
            daily_returns=returns,
            daily_exposures=exposures,
            daily_selectable=counts,
        )


def test_default_pinned_v36_npz_has_seven_fields_but_loader_returns_control_only():
    assert v37._sha256(v37.CONTROL_DAILY_REFERENCE_PATH) == (
        v37.EXPECTED_CONTROL_DAILY_REFERENCE_SHA256
    )
    with np.load(v37.CONTROL_DAILY_REFERENCE_PATH, allow_pickle=False) as saved:
        assert set(saved.files) == _v36_npz_fields()
    reference = v37._load_control_daily_reference()
    assert set(reference) == _v36_control_fields()
    assert all(
        value.shape == (v37.EXPECTED_TRAIN_DAYS,)
        for value in reference.values()
    )
    assert reference["dates"].dtype.kind == "U"
    assert str(reference["dates"][0]) == v37.TRAIN_FIRST_DATE
    assert str(reference["dates"][-1]) == v37.TRAIN_LAST_DATE
    assert np.all(np.isfinite(reference["control_daily_returns"]))
    assert np.all(np.isfinite(reference["control_daily_exposures"]))
    counts = reference["control_daily_selectable_candidates"]
    assert np.issubdtype(counts.dtype, np.integer)
    assert np.all(counts >= 0)


def test_synthetic_exact_seven_field_v36_npz_returns_only_control_arrays(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    path = tmp_path / "seven_fields.npz"
    arrays = _synthetic_v36_npz_arrays()
    _write_synthetic_v36_npz(path, arrays)
    monkeypatch.setattr(
        v37, "EXPECTED_CONTROL_DAILY_REFERENCE_SHA256", v37._sha256(path)
    )
    reference = v37._load_control_daily_reference(path)
    assert set(reference) == _v36_control_fields()
    for name in _v36_control_fields():
        np.testing.assert_array_equal(reference[name], arrays[name])


@pytest.mark.parametrize("field_change", ["missing", "extra"])
def test_v36_npz_missing_or_extra_field_fails_loudly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field_change: str,
):
    arrays = _synthetic_v36_npz_arrays()
    if field_change == "missing":
        arrays.pop("candidate_daily_returns")
    else:
        arrays["unexpected_candidate_metric"] = np.zeros(
            v37.EXPECTED_TRAIN_DAYS, dtype=np.float64
        )
    path = tmp_path / f"{field_change}_field.npz"
    np.savez_compressed(path, **arrays)
    monkeypatch.setattr(
        v37, "EXPECTED_CONTROL_DAILY_REFERENCE_SHA256", v37._sha256(path)
    )
    with pytest.raises(ValueError, match="reference fields changed"):
        v37._load_control_daily_reference(path)


@pytest.mark.parametrize(
    "field",
    [
        "candidate_daily_returns",
        "candidate_daily_exposures",
        "candidate_daily_selectable_candidates",
    ],
)
def test_each_candidate_reference_array_wrong_shape_fails_loudly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
):
    arrays = _synthetic_v36_npz_arrays()
    arrays[field] = arrays[field][:-1]
    path = tmp_path / f"wrong_shape_{field}.npz"
    _write_synthetic_v36_npz(path, arrays)
    monkeypatch.setattr(
        v37, "EXPECTED_CONTROL_DAILY_REFERENCE_SHA256", v37._sha256(path)
    )
    with pytest.raises(ValueError, match=f"shape changed for {field}"):
        v37._load_control_daily_reference(path)


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("candidate_daily_returns", np.nan),
        ("candidate_daily_exposures", np.inf),
        ("candidate_daily_selectable_candidates", np.nan),
    ],
)
def test_each_candidate_numeric_family_nonfinite_fails_loudly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    bad_value: float,
):
    arrays = _synthetic_v36_npz_arrays()
    arrays[field] = arrays[field].astype(np.float64)
    arrays[field][0] = bad_value
    path = tmp_path / f"nonfinite_{field}.npz"
    _write_synthetic_v36_npz(path, arrays)
    monkeypatch.setattr(
        v37, "EXPECTED_CONTROL_DAILY_REFERENCE_SHA256", v37._sha256(path)
    )
    with pytest.raises(ValueError, match=f"non-finite/non-numeric {field}"):
        v37._load_control_daily_reference(path)


def test_candidate_reference_negative_or_nonintegral_count_fails_loudly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    for label, values, message in (
        (
            "negative",
            np.full(v37.EXPECTED_TRAIN_DAYS, -1, dtype=np.int64),
            "contains a negative candidate_daily_selectable_candidates",
        ),
        (
            "float",
            np.full(v37.EXPECTED_TRAIN_DAYS, 137.0, dtype=np.float64),
            "count is not integral: candidate_daily_selectable_candidates",
        ),
    ):
        arrays = _synthetic_v36_npz_arrays()
        arrays["candidate_daily_selectable_candidates"] = values
        path = tmp_path / f"{label}_candidate_count.npz"
        _write_synthetic_v36_npz(path, arrays)
        monkeypatch.setattr(
            v37, "EXPECTED_CONTROL_DAILY_REFERENCE_SHA256", v37._sha256(path)
        )
        with pytest.raises(ValueError, match=message):
            v37._load_control_daily_reference(path)


def test_primary_gate_boundaries_distinguish_at_least_from_strictly_exceeds():
    control = _gate_row(role="control", label=v37.CONTROL_LABEL)
    candidate = _gate_row(role="candidate", label=v37.CANDIDATE_LABEL)
    candidate["robust_score"] = control["robust_score"]
    candidate["fold_calmars"]["2010-2012"] = control["fold_calmars"]["2010-2012"]
    candidate["fold_calmars"]["2016-2018"] = control["fold_calmars"]["2016-2018"]
    checks = v37.primary_gate_checks(candidate, control)
    assert checks["full_calmar_at_least_2p5"] is True
    assert checks["worst_fold_calmar_at_least_1p5"] is True
    assert checks["average_exposure_at_least_0p45"] is True
    assert checks["minimum_daily_selectable_candidates_at_least_30"] is True
    assert checks["robust_score_strictly_exceeds_control"] is False
    assert checks["fold_2010_2012_strictly_exceeds_control"] is False
    assert checks["fold_2016_2018_strictly_exceeds_control"] is False

    candidate["robust_score"] = np.nextafter(control["robust_score"], np.inf)
    for label in ("2010-2012", "2016-2018"):
        candidate["fold_calmars"][label] = np.nextafter(
            control["fold_calmars"][label], np.inf
        )
    assert all(v37.primary_gate_checks(candidate, control).values())


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_strict_json_rejects_every_nonfinite_constant(constant: str):
    with pytest.raises(ValueError, match="forbidden constant"):
        v37._loads_strict_json(f'{{"nested": [1, {constant}]}}')


def test_strict_json_rejects_duplicate_keys_at_every_nested_level():
    with pytest.raises(ValueError, match="duplicate key 'same'"):
        v37._loads_strict_json('{"outer": {"same": 1, "same": 2}}')


@pytest.mark.parametrize(
    ("updates", "issue"),
    [
        (
            {
                "price": float(np.nextafter(np.float64(10.0), np.inf)),
                "amount": float(np.nextafter(np.float64(10.0), np.inf)) * 100,
            },
            "open_price_mismatches",
        ),
        (
            {"amount": float(np.nextafter(np.float64(1000.0), np.inf))},
            "amount_mismatches",
        ),
        ({"action": "hold"}, "invalid_action"),
        ({"price_field": "close"}, "non_open_price_field"),
        (
            {"trade_date": "2019-01-02", "date": "2019-01-02"},
            "outside_training_period",
        ),
        ({"code": "000002.SZ"}, "runtime_missing"),
        ({"volume": 0}, "nonfinite_or_nonpositive_volume"),
    ],
)
def test_machine_trade_audit_rejects_every_exactness_and_identity_drift(
    updates: dict,
    issue: str,
):
    baseline = v37._audit_trade_log_exact_open(
        role="control",
        label=v37.CONTROL_LABEL,
        trade_log=[_trade_row()],
        open_price=np.asarray([[10.0]], dtype=np.float64),
        date_indices={"2018-12-28": 0},
        stock_indices={"000001.SZ": 0},
    )
    assert baseline["passes"] is True
    failed = v37._audit_trade_log_exact_open(
        role="candidate",
        label=v37.CANDIDATE_LABEL,
        trade_log=[_trade_row(**updates)],
        open_price=np.asarray([[10.0]], dtype=np.float64),
        date_indices={"2018-12-28": 0},
        stock_indices={"000001.SZ": 0},
    )
    assert failed["passes"] is False
    assert failed["issues"][issue] == 1


@pytest.mark.parametrize("cell_index", range(11))
def test_rendered_trade_audit_compares_all_sort_cells_in_original_order(
    tmp_path: Path,
    cell_index: int,
):
    trades = [
        _trade_row(),
        _trade_row(
            action="sell",
            signal_date="2018-12-27",
            commission=1.0,
            income=5.0,
            reason="synthetic sell",
        ),
    ]
    names = {"000001.SZ": "平安银行"}
    table = v37.single_report_module._make_trade_table(trades, names)
    report_path = tmp_path / "single_report.html"
    _write_report_data(report_path, {"tables": {"trades": table}})
    passed = v37._audit_rendered_trade_table_exact_open(
        report_path=report_path,
        trade_log=trades,
        stock_name_map=names,
        open_price=np.asarray([[10.0]], dtype=np.float64),
        date_indices={"2018-12-28": 0},
        stock_indices={"000001.SZ": 0},
    )
    assert passed["passes"] is True
    assert passed["machine_action_to_rendered_action"] == {
        "buy": "买入",
        "sell": "卖出",
    }
    assert [row[3]["sort"] for row in table["rows"]] == ["买入", "卖出"]
    assert all(
        row["all_11_sort_cells_exact_in_original_order"] is True
        for row in passed["row_comparisons"]
    )

    changed = copy.deepcopy(table)
    changed["rows"][0][cell_index]["sort"] += "_drift"
    _write_report_data(report_path, {"tables": {"trades": changed}})
    failed = v37._audit_rendered_trade_table_exact_open(
        report_path=report_path,
        trade_log=trades,
        stock_name_map=names,
        open_price=np.asarray([[10.0]], dtype=np.float64),
        date_indices={"2018-12-28": 0},
        stock_indices={"000001.SZ": 0},
    )
    assert failed["passes"] is False
    assert failed["issues"]["sort_cell_mismatches"] == 1


def test_rendered_trade_audit_rejects_row_reordering(tmp_path: Path):
    trades = [
        _trade_row(reason="first"),
        _trade_row(action="sell", reason="second"),
    ]
    table = v37.single_report_module._make_trade_table(trades, {})
    table["rows"].reverse()
    report_path = tmp_path / "single_report.html"
    _write_report_data(report_path, {"tables": {"trades": table}})
    failed = v37._audit_rendered_trade_table_exact_open(
        report_path=report_path,
        trade_log=trades,
        stock_name_map={},
        open_price=np.asarray([[10.0]], dtype=np.float64),
        date_indices={"2018-12-28": 0},
        stock_indices={"000001.SZ": 0},
    )
    assert failed["passes"] is False
    assert failed["issues"]["sort_cell_mismatches"] == 2


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda payload: payload.update(
                {"000002.SZ": payload.pop("000001.SZ")}
            ),
            "code set differs",
        ),
        (
            lambda payload: payload["000001.SZ"]["d"].__setitem__(
                1, "2019-01-02"
            ),
            "date leaks outside training",
        ),
        (
            lambda payload: payload["000001.SZ"]["events"][0].update(
                date="2018-12-27"
            ),
            "omits an event date",
        ),
        (
            lambda payload: payload["000001.SZ"]["episodes"][0].update(
                window_end="2019-01-02"
            ),
            "episode window_end leaks outside training",
        ),
        (
            lambda payload: payload["000001.SZ"]["d"].reverse(),
            "dates are not strictly ordered",
        ),
        (
            lambda payload: payload["000001.SZ"]["a"].pop(),
            "columns do not align",
        ),
    ],
)
def test_kline_audit_rejects_code_date_event_episode_order_and_column_drift(
    tmp_path: Path,
    mutation,
    message: str,
):
    payload = _valid_kline_payload()
    mutation(payload)
    path = tmp_path / "report.html"
    _write_kline_report(path, payload)
    with pytest.raises(ValueError, match=message):
        v37._audit_html_kline_training_bounds(path, {"000001.SZ"})


def test_kline_report_decodes_and_audits_exact_training_bounds(tmp_path: Path):
    path = tmp_path / "report.html"
    payload = _valid_kline_payload()
    _write_kline_report(path, payload)
    assert v37._decode_html_kline_payload(path) == payload
    audit = v37._audit_html_kline_training_bounds(path, {"000001.SZ"})
    assert audit["passes"] is True
    assert audit["actual_first_date"] == v37.TRAIN_FIRST_DATE
    assert audit["actual_last_date"] == v37.TRAIN_LAST_DATE
    assert audit["all_executed_trade_codes_present"] is True


def test_html_and_kline_decoders_reject_duplicate_nonfinite_and_invalid_base64(
    tmp_path: Path,
):
    path = tmp_path / "report.html"
    marker = '<script id="report-data" type="application/json">'
    path.write_text(marker + "{} </script>" + marker + "{} </script>", encoding="utf-8")
    with pytest.raises(ValueError, match="payload count is invalid"):
        v37._decode_html_report_data(path)

    path.write_text(marker + '{"value": NaN}</script>', encoding="utf-8")
    with pytest.raises(ValueError, match="forbidden constant"):
        v37._decode_html_report_data(path)

    _write_report_data(path, {"kline_b64": "not base64!"})
    with pytest.raises(ValueError):
        v37._decode_html_kline_payload(path)


def _create_arm_artifacts(output_dir: Path, role: str, label: str) -> dict:
    run_dir = output_dir / label
    run_dir.mkdir(parents=True)
    dates = _fixed_date_strings()
    daily = [0.0] * len(dates)
    trade = _trade_row()
    names = {"000001.SZ": "平安银行"}
    table = v37.single_report_module._make_trade_table([trade], names)
    report_data = {
        "tables": {"trades": table},
        "charts": {"equity": {"trade_dates": dates}},
        "kline_b64": _encode_kline_payload(_valid_kline_payload()),
    }
    stable = run_dir / "single_report.html"
    timestamped = run_dir / "single_report_20181228_120000.html"
    _write_report_data(stable, report_data)
    timestamped.write_bytes(stable.read_bytes())
    report = {
        "signal_dates": dates,
        "trade_dates": dates,
        "daily_returns": daily,
        "trade_log": [trade],
        "stock_name_map": names,
    }
    record = {"dates": dates, "daily_returns": daily}
    trades = {
        "role": role,
        "label": label,
        "price_field": "open",
        "trade_rows": 1,
        "trades": [trade],
    }
    rendered = {
        "trade_rows": 1,
        "passes": True,
        "all_11_sort_cells_compared_in_original_order": True,
        "row_comparisons": [{"passes": True}],
    }
    audit = {
        "trade_rows": 1,
        "passes": True,
        "per_trade_checks": [{"passes": True}],
        "rendered_html_cross_check": rendered,
    }
    audit_summary = {"role": role, "label": label}
    for name, payload in (
        ("report.json", report),
        ("record.json", record),
        ("trades.json", trades),
        ("trade_open_audit.json", audit),
        ("audit_summary.json", audit_summary),
    ):
        v37._write_json(run_dir / name, payload)
    (run_dir / "single.log").write_text(
        "synthetic V37 arm completed successfully\n", encoding="utf-8"
    )
    return {
        "role": role,
        "label": label,
        "full_report_obligation_completed": True,
        "trade_open_audit": {"passes": True},
        "html_kline_training_bounds": {"passes": True},
    }


def _create_both_arm_artifacts(output_dir: Path) -> list[dict]:
    return [
        _create_arm_artifacts(output_dir, "control", v37.CONTROL_LABEL),
        _create_arm_artifacts(output_dir, "candidate", v37.CANDIDATE_LABEL),
    ]


def test_both_arm_report_kline_json_log_and_date_obligations_parse(tmp_path: Path):
    summaries = _create_both_arm_artifacts(tmp_path)
    v37._validate_report_obligations(tmp_path, summaries)


@pytest.mark.parametrize(
    ("corruption", "error_type", "message"),
    [
        ("missing", FileNotFoundError, "missing mandatory artifacts"),
        ("nonfinite_json", ValueError, "forbidden constant"),
        ("date_disagreement", ValueError, "dates do not exactly agree"),
        ("empty_log", ValueError, "log is empty"),
        ("traceback_log", ValueError, "contains a traceback"),
        ("nonfinite_log", ValueError, "non-finite numeric token"),
    ],
)
def test_report_obligations_fail_closed_on_missing_unparseable_or_bad_dates(
    tmp_path: Path,
    corruption: str,
    error_type: type[Exception],
    message: str,
):
    summaries = _create_both_arm_artifacts(tmp_path)
    run_dir = tmp_path / v37.CANDIDATE_LABEL
    if corruption == "missing":
        (run_dir / "record.json").unlink()
    elif corruption == "nonfinite_json":
        (run_dir / "report.json").write_text('{"value": NaN}', encoding="utf-8")
    elif corruption == "date_disagreement":
        record = v37._load_strict_json(run_dir / "record.json")
        record["dates"][-1] = "2018-12-27"
        v37._write_json(run_dir / "record.json", record)
    elif corruption == "empty_log":
        (run_dir / "single.log").write_text("", encoding="utf-8")
    elif corruption == "traceback_log":
        (run_dir / "single.log").write_text(
            "Traceback (most recent call last):\n", encoding="utf-8"
        )
    else:
        (run_dir / "single.log").write_text("metric = NaN\n", encoding="utf-8")
    with pytest.raises(error_type, match=message):
        v37._validate_report_obligations(tmp_path, summaries)


def test_manifest_rehash_detects_any_artifact_write_during_commit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    output = tmp_path / "output"
    output.mkdir()
    artifact = output / "summary.json"
    artifact.write_text("before", encoding="utf-8")
    source = tmp_path / "source.py"
    source.write_text("locked", encoding="utf-8")
    identity = {
        "source": {"path": str(source), "sha256": v37._sha256(source)}
    }
    original_write = v37._write_json

    def write_then_mutate(path, payload):
        original_write(path, payload)
        if Path(path).name == "manifest.json":
            artifact.write_text("changed during manifest commit", encoding="utf-8")

    monkeypatch.setattr(v37, "_write_json", write_then_mutate)
    with pytest.raises(RuntimeError, match="artifact hashes failed revalidation"):
        v37._write_and_verify_manifest(
            output_dir=output,
            status="synthetic",
            identity_hashes=identity,
        )


def test_rolling_contract_uses_756_63_final_start_and_linear_p10():
    rows = 1000
    dates = [f"synthetic-{index:04d}" for index in range(rows)]
    x = np.arange(rows, dtype=np.float64)
    daily = 0.08 + 0.35 * np.sin(x / 17.0) - 0.20 * np.cos(x / 29.0)
    rolling = v37._rolling_blocks(dates, daily)
    expected_starts = list(range(0, rows - 756 + 1, 63))
    final_start = rows - 756
    if expected_starts[-1] != final_start:
        expected_starts.append(final_start)
    assert rolling["window_days"] == 756
    assert rolling["step_days"] == 63
    assert rolling["count"] == len(expected_starts)
    assert [window["start"] for window in rolling["windows"]] == [
        dates[index] for index in expected_starts
    ]
    assert expected_starts.count(final_start) == 1
    calmars = np.asarray([window["calmar"] for window in rolling["windows"]])
    assert rolling["p10_calmar"] == float(
        np.quantile(calmars, 0.10, method="linear")
    )


def test_concentration_uses_calendar_log_wealth_and_nonpositive_default():
    dates = ["2010-01-04", "2010-01-05", "2011-01-04", "2011-01-05"]
    daily = np.asarray([10.0, -2.0, 5.0, 1.0], dtype=np.float64)
    audit = v37._return_concentration(dates, daily)
    log_returns = np.log1p(daily / 100.0)
    expected_yearly = {
        "2010": float(np.sum(log_returns[:2])),
        "2011": float(np.sum(log_returns[2:])),
    }
    assert audit["yearly_log_wealth"] == expected_yearly
    assert audit["largest_year_log_wealth_share"] == (
        max(expected_yearly.values()) / float(np.sum(log_returns))
    )
    losing = v37._return_concentration(dates, np.full(4, -1.0))
    assert losing["largest_year_log_wealth_share"] == 1.0


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


def test_secondary_is_fixed_5bps_and_freeze_requires_every_secondary_gate(
    monkeypatch: pytest.MonkeyPatch,
):
    dates = _fixed_date_strings()
    daily = np.zeros(v37.EXPECTED_TRAIN_DAYS, dtype=np.float64)
    calls = []
    monkeypatch.setattr(
        v37,
        "_rolling_blocks",
        lambda *_args: {
            "p10_calmar": v37.SECONDARY_AUDITS["rolling_3y_p10_calmar_min"]
        },
    )
    monkeypatch.setattr(
        v37,
        "_return_concentration",
        lambda *_args: {
            "largest_year_log_wealth_share": v37.SECONDARY_AUDITS[
                "largest_year_log_wealth_share_max"
            ]
        },
    )
    monkeypatch.setattr(
        v37,
        "_report_charts",
        lambda _path: {
            "equity": {
                "trade_dates": dates,
                "rebalance_funds_pct": [1.0] * len(dates),
            }
        },
    )

    def fake_stress(values, rebalance, bps):
        calls.append((len(values), float(rebalance[0]), bps))
        return np.asarray(values, dtype=np.float64)

    monkeypatch.setattr(v37, "stress_daily_returns", fake_stress)
    monkeypatch.setattr(
        v37, "natural_calendar_metrics", lambda *_args: _fake_natural_metrics()
    )
    audit, rebalance, stressed = v37._secondary_audits(
        dates=dates,
        candidate_daily_returns=daily,
        candidate_report_path=Path("synthetic_report.html"),
    )
    assert calls == [(v37.EXPECTED_TRAIN_DAYS, 1.0, 5.0)]
    assert audit["passes_all_secondary_gates"] is True
    assert rebalance.shape == stressed.shape == daily.shape

    control = _gate_row(role="control", label=v37.CONTROL_LABEL)
    candidate = _gate_row(role="candidate", label=v37.CANDIDATE_LABEL)
    candidate["robust_score"] = np.nextafter(control["robust_score"], np.inf)
    for label in ("2010-2012", "2016-2018"):
        candidate["fold_calmars"][label] = np.nextafter(
            control["fold_calmars"][label], np.inf
        )
    primary_rows, primary = v37.annotate_primary_decision([control, candidate])
    frozen_rows, frozen = v37.finalize_training_decision(
        primary_rows, primary, audit
    )
    assert frozen["configuration_frozen"] is True
    assert frozen_rows[1]["configuration_frozen"] is True
    assert frozen["holdout_opened"] is False

    rejected_audit = copy.deepcopy(audit)
    rejected_audit["passes_all_secondary_gates"] = False
    rejected_audit["status"] = "rejected_at_secondary_gates"
    rejected_audit["checks"]["stressed_full_calmar_at_least_2p5"] = False
    rejected_rows, rejected = v37.finalize_training_decision(
        primary_rows, primary, rejected_audit
    )
    assert rejected["configuration_frozen"] is False
    assert rejected_rows[1]["configuration_frozen"] is False


def test_secondary_is_forbidden_after_any_primary_failure():
    control = _gate_row(role="control", label=v37.CONTROL_LABEL)
    candidate = _gate_row(role="candidate", label=v37.CANDIDATE_LABEL)
    candidate["minimum_daily_selectable_candidates"] = 29
    rows, primary = v37.annotate_primary_decision([control, candidate])
    assert primary["secondary_audit_allowed"] is False
    unchanged_rows, unchanged = v37.finalize_training_decision(
        rows, primary, None
    )
    assert unchanged == primary
    assert unchanged_rows[1]["configuration_frozen"] is False
    with pytest.raises(ValueError, match="secondary audit ran after a primary failure"):
        v37.finalize_training_decision(
            rows, primary, {"passes_all_secondary_gates": True}
        )


def _install_synthetic_run(
    monkeypatch: pytest.MonkeyPatch,
    *,
    primary_passes: bool,
) -> dict:
    dates = _fixed_date_strings()
    valid_dates = [datetime.strptime(value, "%Y-%m-%d") for value in dates]
    days = v37.EXPECTED_TRAIN_DAYS
    returns = np.zeros(days, dtype=np.float64)
    exposures = np.full(days, 0.5, dtype=np.float64)
    counts = np.full(days, 138, dtype=np.int64)
    data = {
        "trade_dates": np.asarray(dates, dtype="datetime64[D]"),
        "stock_codes": np.asarray(["000001.SZ"]),
        "open": np.full((days, 1), 10.0, dtype=np.float64),
    }
    control_config = {
        "weights": copy.deepcopy(v37.CONTROL_WEIGHTS),
        "filter_factors": {name: True for name in v37.EXPECTED_FILTER_NAMES},
    }
    candidate_config = copy.deepcopy(control_config)
    candidate_config["weights"] = copy.deepcopy(v37.CANDIDATE_WEIGHTS)
    identities = [
        {
            "candidate_index": 0,
            "role": "control",
            "label": v37.CONTROL_LABEL,
            "source_candidate_index": v37.CONTROL_SOURCE_INDEX,
            "change_from_control": {},
            "individual_config": control_config,
        },
        {
            "candidate_index": 1,
            "role": "candidate",
            "label": v37.CANDIDATE_LABEL,
            "source_candidate_index": v37.CONTROL_SOURCE_INDEX,
            "change_from_control": {v37.NEW_FACTOR_NAME: 0.1},
            "individual_config": candidate_config,
        },
    ]
    control_row = {
        **copy.deepcopy(identities[0]),
        **copy.deepcopy(v37.CONTROL_CANARY),
    }
    candidate_row = {
        **copy.deepcopy(identities[1]),
        "full_calmar": 3.2,
        "fold_calmars": {
            "2010-2012": 1.6,
            "2013-2015": 9.4,
            "2016-2018": 1.7,
        },
        "worst_fold_calmar": 1.6,
        "robust_score": 2.4,
        "average_exposure": 0.5,
        "annualized": 30.0,
        "max_drawdown": -10.0,
        "sharpe": 2.0,
        "terminal_nav": 10.0,
        "total_return": 900.0,
        "minimum_daily_selectable_candidates": 30 if primary_passes else 29,
    }
    structural_audit = {
        "passes": True,
        "new_factor_finite_mask_equals_amihud_full_panel": True,
        "active_factor_masks_exactly_equal": True,
        "shared_filter_masks_exactly_equal": {
            name: True for name in v37.EXPECTED_FILTER_NAMES
        },
        "composite_pool_masks_exactly_equal": True,
        "daily_selectable_counts_exactly_equal": True,
    }
    state = {"calls": [], "secondary_calls": 0}

    class DummyFactor:
        hist_days = 1

    script_hash = v37._sha256(Path(v37.__file__))
    identity_hashes = {
        "authoritative_script": {
            "path": str(Path(v37.__file__).resolve()),
            "sha256": script_hash,
        },
        "synthetic_test_identity": "no_real_data_loaded",
    }
    monkeypatch.setattr(v37, "load_and_validate_plan", lambda _path: {})
    monkeypatch.setattr(
        v37, "_factor_classes", lambda: ([DummyFactor], [DummyFactor])
    )
    monkeypatch.setattr(
        v37, "_unambiguous_runtime_path", lambda: v37.EXPECTED_RUNTIME_PATH
    )
    monkeypatch.setattr(
        v37, "_capture_input_identity", lambda **_kwargs: identity_hashes
    )
    monkeypatch.setattr(v37, "load_control_source", lambda _path: {})
    monkeypatch.setattr(
        v37,
        "build_fixed_candidates",
        lambda _plan, _source: copy.deepcopy(identities),
    )
    monkeypatch.setattr(
        v37,
        "_load_physically_truncated_training_panel",
        lambda: (data, 100),
    )
    monkeypatch.setattr(v37, "EXPECTED_RANK_POOL_SIZE", 1)
    monkeypatch.setattr(
        v37,
        "_build_dates_and_indices",
        lambda _data: (valid_dates, list(range(days))),
    )
    monkeypatch.setattr(
        v37,
        "compute_research_arrays",
        lambda *_args, **_kwargs: ({}, set(v37.ALL_FACTOR_NAMES)),
    )
    monkeypatch.setattr(v37, "_verify_candidate_validity_semantics", lambda *_: None)
    monkeypatch.setattr(v37, "_compute_list_dates", lambda *_: {})
    monkeypatch.setattr(v37, "_assert_input_hashes_unchanged", lambda _identity: None)
    monkeypatch.setattr(
        v37,
        "_structural_isolation_audit",
        lambda **_kwargs: (
            copy.deepcopy(structural_audit),
            {"control": counts.copy(), "candidate": counts.copy()},
        ),
    )
    reference = {
        "dates": np.asarray(dates, dtype="U10"),
        "control_daily_returns": returns.copy(),
        "control_daily_exposures": exposures.copy(),
        "control_daily_selectable_candidates": counts.copy(),
    }
    monkeypatch.setattr(
        v37, "_load_control_daily_reference", lambda: copy.deepcopy(reference)
    )

    def fake_arm(*, identity, output_dir, portfolio_call_sequence, **_kwargs):
        role = identity["role"]
        portfolio_call_sequence.append(role)
        state["calls"].append(role)
        run_dir = output_dir / identity["label"]
        run_dir.mkdir()
        row = control_row if role == "control" else candidate_row
        return {
            "row": copy.deepcopy(row),
            "daily_returns": returns.copy(),
            "daily_exposures": exposures.copy(),
            "daily_selectable": counts.copy(),
            "arm_summary": {
                "role": role,
                "label": identity["label"],
                "trade_open_audit": {"passes": True},
                "html_kline_training_bounds": {"passes": True},
                "full_report_obligation_completed": True,
            },
            "trade_audit_summary": {"trade_rows": 1, "passes": True},
            "report_path": run_dir / "single_report.html",
        }

    monkeypatch.setattr(v37, "_run_authoritative_arm", fake_arm)
    monkeypatch.setattr(v37, "_validate_report_obligations", lambda *_args, **_kwargs: None)

    def fake_secondary(**_kwargs):
        state["secondary_calls"] += 1
        audit = {
            "status": "passed_all_secondary_gates",
            "passes_all_secondary_gates": True,
            "checks": {"all_fixed_secondary_checks": True},
        }
        return audit, np.ones(days), returns.copy()

    monkeypatch.setattr(v37, "_secondary_audits", fake_secondary)
    monkeypatch.setattr(
        v37,
        "_build_summary",
        lambda **kwargs: {
            "experiment": v37.EXPERIMENT,
            "status": kwargs["decision"]["status"],
            "portfolio_call_sequence": kwargs["portfolio_call_sequence"],
        },
    )
    monkeypatch.setattr(
        v37,
        "_build_run_metadata",
        lambda **kwargs: {
            "experiment": v37.EXPERIMENT,
            "portfolio_call_sequence": kwargs["portfolio_call_sequence"],
        },
    )
    monkeypatch.setattr(
        v37,
        "_write_frozen_config",
        lambda path, *_args: v37._write_json(path, {"configuration_frozen": True}),
    )
    return state


@pytest.mark.parametrize(
    ("failure_stage", "expected_calls"),
    [
        ("structural", []),
        ("canary", ["control"]),
        ("reference", ["control"]),
        ("primary", ["control", "candidate"]),
        ("success", ["control", "candidate"]),
    ],
)
def test_synthetic_run_has_exact_fail_closed_call_order_and_manifests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_stage: str,
    expected_calls: list[str],
):
    state = _install_synthetic_run(
        monkeypatch,
        primary_passes=failure_stage not in {"primary"},
    )
    if failure_stage == "structural":
        monkeypatch.setattr(
            v37,
            "_structural_isolation_audit",
            lambda **_kwargs: (_ for _ in ()).throw(
                ValueError("synthetic structural failure")
            ),
        )
    elif failure_stage == "canary":
        monkeypatch.setattr(
            v37,
            "validate_control_canary",
            lambda _row: (_ for _ in ()).throw(
                ValueError("synthetic canary failure")
            ),
        )
    elif failure_stage == "reference":
        monkeypatch.setattr(
            v37,
            "_validate_control_daily_reference",
            lambda **_kwargs: (_ for _ in ()).throw(
                ValueError("synthetic reference failure")
            ),
        )

    output = tmp_path / failure_stage
    if failure_stage in {"structural", "canary", "reference"}:
        with pytest.raises(ValueError, match=f"synthetic {failure_stage}"):
            v37.run(output)
    else:
        v37.run(output)
    assert state["calls"] == expected_calls
    assert state["secondary_calls"] == int(failure_stage == "success")

    if failure_stage == "structural":
        assert not output.exists()
        return
    manifest = v37._load_strict_json(output / "manifest.json")
    assert manifest["manifest_written_after_every_other_artifact"] is True
    assert "manifest.json" not in manifest["generated_artifact_sha256"]
    assert manifest["generated_artifact_sha256"] == v37._generated_file_hashes(
        output
    )
    assert not (output / v37.CANDIDATE_LABEL).exists() == (
        failure_stage in {"canary", "reference"}
    )
    if failure_stage in {"canary", "reference"}:
        assert (output / "failure.json").is_file()
        assert not (output / "frozen_config.json").exists()
        with np.load(output / "daily_returns.npz", allow_pickle=False) as saved:
            assert set(saved.files) == {
                "dates",
                "control_daily_returns",
                "control_daily_exposures",
                "control_daily_selectable_candidates",
            }
    elif failure_stage == "primary":
        assert (output / "failure.json").is_file()
        assert not (output / "secondary_audits.json").exists()
        assert not (output / "frozen_config.json").exists()
    else:
        assert not (output / "failure.json").exists()
        assert (output / "secondary_audits.json").is_file()
        assert (output / "secondary_cost_stress.npz").is_file()
        assert (output / "frozen_config.json").is_file()


def test_existing_output_is_rejected_before_any_input_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    output = tmp_path / "already_exists"
    output.mkdir()
    reads = []
    monkeypatch.setattr(
        v37, "load_and_validate_plan", lambda *_args: reads.append("plan")
    )
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        v37.run(output)
    assert reads == []
