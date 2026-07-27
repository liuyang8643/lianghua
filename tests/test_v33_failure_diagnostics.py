from __future__ import annotations

import json
from pathlib import Path
import re

import numpy as np
import pytest

import research_v33_failure_diagnostics as diagnostics


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _three_role_series(values: list[float]) -> dict[str, np.ndarray]:
    base = np.asarray(values, dtype=np.float64)
    return {
        "control": base,
        "standalone": base * 2.0,
        "blend": base * 1.25,
    }


def test_frozen_sha256_constants_are_complete_lowercase_digests():
    frozen_digests = [
        *diagnostics.EXPECTED_AUTHORITATIVE_SHA256.values(),
        diagnostics.EXPECTED_PLAN_SHA256,
        diagnostics.EXPECTED_RUNTIME_SHA256,
        diagnostics.EXPECTED_TRAIN_REPORT_MANIFEST_SHA256,
    ]

    assert all(re.fullmatch(r"[0-9a-f]{64}", value) for value in frozen_digests)


def test_default_training_report_manifest_matches_frozen_identity():
    manifest = diagnostics.TRAIN_REPORT_DIR / "manifest.json"

    assert manifest.is_file()
    assert diagnostics._sha256(manifest) == (
        diagnostics.EXPECTED_TRAIN_REPORT_MANIFEST_SHA256
    )


def test_authoritative_loader_checks_hash_before_parsing(tmp_path: Path):
    expected = {}
    for name in diagnostics.AUTHORITATIVE_FILENAMES:
        path = tmp_path / name
        path.write_bytes(name.encode("ascii"))
        expected[name] = diagnostics._sha256(path)
    expected["summary.json"] = "0" * 64

    with pytest.raises(ValueError, match="authoritative summary.json SHA256"):
        diagnostics.load_authoritative_result(tmp_path, expected)


def test_training_report_manifest_cross_checks_and_hashes_every_artifact(
    tmp_path: Path,
):
    report_dir = tmp_path / "reports"
    report_dir.mkdir()
    summary = {
        "selection_scope": "training_only_2010_2018",
        "reports_are_diagnostic_and_cannot_change_selection": True,
        "runtime_physically_truncated_before_factor_calculation": True,
        "strategy_holdout_available_to_factor_or_worker": False,
        "strategy_holdout_evaluated": False,
    }
    _write_json(report_dir / "summary.json", summary)
    nested = report_dir / "control"
    nested.mkdir()
    (nested / "trades.json").write_text("training only\n", encoding="utf-8")

    authoritative = {}
    for name in diagnostics.AUTHORITATIVE_FILENAMES:
        path = tmp_path / name
        path.write_text(name, encoding="utf-8")
        authoritative[name] = {
            "path": str(path),
            "sha256": diagnostics._sha256(path),
        }
    plan = tmp_path / "plan.json"
    runtime = tmp_path / "runtime.npz"
    plan.write_text("plan", encoding="utf-8")
    runtime.write_text("runtime", encoding="utf-8")
    plan_identity = {"path": str(plan), "sha256": diagnostics._sha256(plan)}
    runtime_identity = {
        "path": str(runtime),
        "sha256": diagnostics._sha256(runtime),
    }
    manifest = {
        "experiment": "v33_completed_52week_high_sleeve_train_reports",
        "source_runtime_plan_and_authoritative_result_sha256": {
            "preregistered_plan": plan_identity,
            "runtime": runtime_identity,
            "candidate_source": {"sha256": "recorded-only"},
            "authoritative_result": authoritative,
        },
        "generated_artifact_sha256": {
            "summary.json": diagnostics._sha256(report_dir / "summary.json"),
            "control/trades.json": diagnostics._sha256(
                nested / "trades.json"
            ),
        },
    }
    _write_json(report_dir / "manifest.json", manifest)

    identities, loaded_summary = diagnostics.validate_training_report_manifest(
        report_dir,
        authoritative_identities=authoritative,
        plan_identity=plan_identity,
        runtime_identity=runtime_identity,
        expected_manifest_sha256=diagnostics._sha256(
            report_dir / "manifest.json"
        ),
    )

    assert loaded_summary == summary
    assert set(identities["generated_artifacts"]) == {
        "summary.json",
        "control/trades.json",
    }
    assert identities["candidate_source_recorded_not_loaded"] == {
        "sha256": "recorded-only",
        "loaded_by_this_diagnostic": False,
    }

    (nested / "trades.json").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="artifact control/trades.json SHA256"):
        diagnostics.validate_training_report_manifest(
            report_dir,
            authoritative_identities=authoritative,
            plan_identity=plan_identity,
            runtime_identity=runtime_identity,
            expected_manifest_sha256=diagnostics._sha256(
                report_dir / "manifest.json"
            ),
        )


def test_raw_factor_is_called_only_after_physical_train_truncation():
    seen = {}

    class InspectingFactor:
        def calc_batch(self, panel):
            dates = np.asarray(panel["trade_dates"], dtype="datetime64[D]")
            seen["last_date"] = str(dates[-1])
            seen["rows"] = len(dates)
            return np.ones(
                (len(dates), len(panel["stock_codes"])), dtype=np.float32
            )

    loaded = {
        "trade_dates": np.asarray(
            ["2018-12-27", "2018-12-28", "2019-01-02"],
            dtype="datetime64[D]",
        ),
        "stock_codes": np.asarray(["000001", "600001"]),
        "open": np.ones((3, 2), dtype=np.float64),
    }
    original_open = loaded["open"]

    truncated, raw, discarded = (
        diagnostics.compute_raw_factor_after_physical_truncation(
            loaded, InspectingFactor
        )
    )

    assert discarded == 1
    assert seen == {"last_date": diagnostics.TRAIN_LAST_DATE, "rows": 2}
    assert raw.shape == (2, 2)
    assert not np.shares_memory(truncated["open"], original_open)
    assert np.max(truncated["trade_dates"]) == np.datetime64(
        diagnostics.TRAIN_LAST_DATE
    )


def test_return_summaries_use_compounding_and_fixed_rolling_windows():
    dates = [f"2010-01-{day:02d}" for day in range(1, 9)]
    series = _three_role_series([1.0, -0.5, 2.0, -1.0, 0.5, 1.0, -0.5, 2.0])

    monthly = diagnostics.monthly_return_diagnostics(dates, series)
    expected = (np.prod(1.0 + series["control"] / 100.0) - 1.0) * 100.0
    assert monthly["months"][0]["returns_pct"]["control"] == pytest.approx(
        expected
    )
    assert monthly["summary_by_role"]["control"]["month_count"] == 1

    rolling = diagnostics.rolling_return_diagnostics(
        dates, series, window_days=3
    )
    assert rolling["summary_by_role"]["control"]["window_count"] == 6
    assert 0.0 <= rolling["summary_by_role"]["control"][
        "positive_window_fraction"
    ] <= 1.0

    rolling_3y = diagnostics.fixed_rolling_3y_diagnostics(
        dates, series, window_days=4, step_days=3
    )
    assert rolling_3y["summary_by_role"]["control"]["window_count"] == 3
    windows = rolling_3y["summary_by_role"]["control"]["windows"]
    assert windows[-1]["start"] == dates[4]
    assert windows[-1]["end"] == dates[-1]


def test_correlations_cover_full_control_negative_and_three_folds():
    dates = [
        "2010-01-04",
        "2010-01-05",
        "2013-01-04",
        "2013-01-05",
        "2016-01-04",
        "2016-01-05",
    ]
    control = np.asarray([-2.0, 1.0, -1.0, 2.0, -3.0, 3.0])
    series = {
        "control": control,
        "standalone": -control,
        "blend": 0.5 * control,
    }
    all_mask = np.ones(len(dates), dtype=bool)
    full = diagnostics._pairwise_correlations(series, all_mask)
    negative = diagnostics._pairwise_correlations(series, control < 0.0)

    assert full["control__standalone"]["pearson"] == pytest.approx(-1.0)
    assert full["control__blend"]["pearson"] == pytest.approx(1.0)
    assert negative["control__standalone"]["observations"] == 3

    date_array = np.asarray(dates)
    for label in diagnostics.FOLD_LABELS:
        first, last = label.split("-")
        mask = (date_array >= f"{first}-01-01") & (
            date_array <= f"{last}-12-31"
        )
        fold = diagnostics._pairwise_correlations(series, mask)
        assert fold["control__standalone"]["observations"] == 2


def test_raw_month_diagnostics_report_coverage_saturation_and_overlap():
    dates = np.asarray(
        ["2010-01-04", "2010-02-01", "2013-01-04", "2016-01-04"],
        dtype="datetime64[D]",
    )
    codes = np.asarray([f"000{index:03d}" for index in range(1, 36)])
    raw = np.full((4, 35), 0.5, dtype=np.float32)
    raw[0, :30] = 1.0
    raw[1, 4:34] = 1.0
    raw[2] = np.linspace(0.1, 0.9, 35, dtype=np.float32)
    raw[3] = np.linspace(0.9, 0.1, 35, dtype=np.float32)
    data = {
        "trade_dates": dates,
        "stock_codes": codes,
        "open": np.full((4, 35), 10.0),
        "st_mask": np.zeros((4, 35), dtype=bool),
    }
    training_dates = dates.astype(str).tolist()

    result = diagnostics.raw_factor_diagnostics(data, raw, training_dates)

    first = result["months"][0]
    assert first["raw_valid_count"] == 35
    assert first["raw_valid_fraction_of_open_st_legal_pool"] == 1.0
    assert first["score_exactly_one_count"] == 30
    assert first["top30_saturated_by_score_one"] is True
    assert result["score_one_and_top30_saturation"][
        "top30_saturated_months"
    ] == ["2010-01", "2010-02"]
    assert set(result["fold_month_first_coverage"]) == set(
        diagnostics.FOLD_LABELS
    )
    adjacent = result["adjacent_month_raw_top30_intersection"]["pairs"]
    assert adjacent[0]["intersection_count"] == 26
    assert adjacent[0]["intersection_codes"] == [
        f"000{index:03d}" for index in range(5, 31)
    ]


def test_existing_output_is_rejected_before_any_input_is_opened(tmp_path: Path):
    output = tmp_path / "diagnostic.json"
    output.write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        diagnostics.run(
            output,
            result_dir=tmp_path / "missing-result",
            report_dir=tmp_path / "missing-report",
            plan_path=tmp_path / "missing-plan",
            runtime_path=tmp_path / "missing-runtime",
        )

    assert output.read_text(encoding="utf-8") == "keep"


def test_output_metadata_states_training_only_non_selection_contract():
    payload = diagnostics.build_output_payload(
        validated={
            "authoritative": {
                "summary": {"status": "fixed_blend_rejected_at_primary_gates"}
            }
        },
        identities={"runtime": {"path": "runtime", "sha256": "abc"}},
        return_analysis={"yearly": {}},
        raw_analysis={"months": []},
        post_train_rows_discarded=7,
        script_identity={"path": "script", "sha256": "def"},
    )

    contract = payload["diagnostic_contract"]
    assert contract["read_only"] is True
    assert contract["reports_are_diagnostic_and_cannot_change_selection"] is True
    assert contract["authoritative_rejection_and_selection_remain_unchanged"] is True
    assert contract["configuration_freeze_allowed"] is False
    assert contract["validation_test_holdout_loaded_or_evaluated"] is False
    assert contract["candidate_source_loaded"] is False
    assert contract["runtime_post_train_rows_discarded_before_raw_factor"] == 7
    assert payload["input_identity_sha256"]["runtime"]["sha256"] == "abc"
