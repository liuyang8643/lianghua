import json

import pytest

from research_candidate_coverage import audit_run


def test_candidate_coverage_summarizes_targets_by_year(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "record.json").write_text(
        json.dumps(
            {
                "dates": ["2010-12-31", "2011-01-03", "2011-01-04"],
                "topn": [["A", "B"], [], ["C"]],
                "buy_n": 2,
            }
        ),
        encoding="utf-8",
    )

    result = audit_run(run_dir)

    assert result["overall"]["days_below_buy_n"] == 2
    assert result["overall"]["days_zero"] == 1
    assert result["overall"]["median"] == 1
    assert result["calendar_years"]["2010"]["minimum"] == 2
    assert result["calendar_years"]["2011"]["mean"] == 0.5
    assert result["zero_target_dates"] == ["2011-01-03"]


def test_candidate_coverage_rejects_misaligned_record(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "record.json").write_text(
        json.dumps({"dates": ["2010-01-04"], "topn": [], "buy_n": 40}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="invalid dates/topn lengths"):
        audit_run(run_dir)
