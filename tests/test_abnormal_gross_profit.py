from dataclasses import replace
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from factor.library.abnormal_gross_profit import abnormal_gross_profit
from offline_data.financial_versions import (
    ABNORMAL_GROSS_PROFIT_FIELD_SET, ABNORMAL_GROSS_PROFIT_FIELDS,
    FINANCIAL_FIELDS, FINANCIAL_PANEL_FIELDS, iter_financial_fields,
    load_financial_events, normalize_financial_events,
)


def research_events():
    # Single Q1 / Q2: 2020 GP 40 / 60, cash 80 / 120;
    # 2021 GP 60 / 100, cash 100 / 180; assets 1000 / 2000.
    rows = [
        ("20200331", "20200420", 100., 60., 80., 800.),
        ("20200630", "20200820", 250., 150., 200., 900.),
        ("20210331", "20210420", 160., 100., 100., 1000.),
        ("20210630", "20210820", 400., 240., 280., 2000.),
    ]
    frames = {table: [] for table in ABNORMAL_GROSS_PROFIT_FIELDS}
    for period, announcement, revenue, cost, cash, assets in rows:
        common = dict(stock_code="000001.SZ", m_timetag=period, m_anntime=announcement)
        frames["Income"].append(dict(**common, revenue=revenue, total_expense=cost))
        frames["CashFlow"].append(dict(**common, goods_sale_and_service_render_cash=cash))
        frames["Balance"].append(dict(**common, tot_assets=assets))
    return {table: normalize_financial_events(pd.DataFrame(frames[table]), ("000001.SZ",), fields)
            for table, fields in ABNORMAL_GROSS_PROFIT_FIELDS.items()}


def replay(events, days):
    return list(iter_financial_fields(np.array(days, dtype="datetime64[D]"), 1, events,
                                     field_set=ABNORMAL_GROSS_PROFIT_FIELD_SET))


def test_hand_computed_single_quarters_cash_scaling_and_period_assets():
    rows = replay(research_events(), ["2021-04-21", "2021-08-21"])
    np.testing.assert_allclose(abnormal_gross_profit(rows[0][1]), [.01])
    np.testing.assert_allclose(abnormal_gross_profit(rows[1][1]), [.005])
    fields = rows[1][1]
    assert fields["abnormal_revenue_quarter"][0] == 240.
    assert fields["abnormal_cost_quarter"][0] == 140.
    assert fields["abnormal_sales_cash_quarter"][0] == 180.
    assert fields["abnormal_revenue_prior_year_quarter"][0] == 150.
    assert fields["abnormal_total_assets"][0] == 2000.
    assert all(not value.flags.writeable for value in fields.values())
    assert not abnormal_gross_profit(fields).flags.writeable


def test_same_day_announcement_excluded_and_three_tables_aligned():
    events = research_events()
    cash = events["CashFlow"]
    announcements = cash.announcement_dates.copy()
    announcements[-1] = np.datetime64("2021-08-25")
    events["CashFlow"] = replace(cash, announcement_dates=announcements)
    rows = replay(events, ["2021-04-20", "2021-04-21", "2021-08-25", "2021-08-26"])
    assert np.isnan(abnormal_gross_profit(rows[0][1])).all()
    np.testing.assert_allclose([abnormal_gross_profit(fields)[0] for _, fields in rows[1:]], [.01, .01, .005])
    # New Income / Balance Q2 cannot be mixed with old CashFlow Q1.
    assert rows[2][1]["abnormal_total_assets"][0] == 1000.


def test_missing_selected_operand_and_previous_quarter_do_not_fallback():
    events = research_events()
    income = events["Income"]
    values = income.values.copy()
    values[-1, 1] = np.nan
    bad = {**events, "Income": replace(income, values=values)}
    assert np.isnan(abnormal_gross_profit(replay(bad, ["2021-08-21"])[0][1])).all()
    # Remove Q1 YTD needed to derive previous year's Q2 single quarter.
    keep = income.quarters != 2020 * 4
    bad["Income"] = replace(income, announcement_dates=income.announcement_dates[keep],
                            quarters=income.quarters[keep], stock_columns=income.stock_columns[keep],
                            values=income.values[keep])
    assert np.isnan(abnormal_gross_profit(replay(bad, ["2021-08-21"])[0][1])).all()


def test_revisions_apply_only_after_actual_disclosure_without_past_backfill():
    events = research_events()
    income = events["Income"]
    revised = replace(income,
                      announcement_dates=np.append(income.announcement_dates, np.datetime64("2021-09-01")),
                      quarters=np.append(income.quarters, 2020 * 4 + 1),
                      stock_columns=np.append(income.stock_columns, 0),
                      values=np.vstack([income.values, [250., 170.]]))
    days = ["2021-04-21", "2021-08-21", "2021-09-01", "2021-09-02"]
    original = replay(events, days)
    after = replay({**events, "Income": revised}, days)
    for (_, before), (_, new) in zip(original[:3], after[:3], strict=True):
        for field in before:
            np.testing.assert_array_equal(before[field], new[field])
    np.testing.assert_allclose(abnormal_gross_profit(after[-1][1]), [.02])


@pytest.mark.parametrize("field,value", [("abnormal_total_assets", 0.),
    ("abnormal_total_assets", -1.), ("abnormal_sales_cash_prior_year_quarter", 0.),
    ("abnormal_sales_cash_prior_year_quarter", -1.), ("abnormal_sales_cash_quarter", -1.),
    ("abnormal_cost_quarter", np.nan), ("abnormal_revenue_quarter", np.inf)])
def test_invalid_operands_are_nan(field, value):
    fields = dict(replay(research_events(), ["2021-04-21"])[0][1])
    fields[field] = np.array([value])
    assert np.isnan(abnormal_gross_profit(fields)).all()


def test_current_zero_cash_and_signed_profit_are_preserved():
    fields = dict(replay(research_events(), ["2021-04-21"])[0][1])
    fields["abnormal_sales_cash_quarter"] = np.array([0.])
    np.testing.assert_allclose(abnormal_gross_profit(fields), [.06])
    fields["abnormal_cost_quarter"] = np.array([200.])
    np.testing.assert_allclose(abnormal_gross_profit(fields), [-.04])


def test_default_production_schema_versions_all_fields_and_research_subset(tmp_path):
    from test_video_financial_versions import events as production_events
    events = production_events()
    directory = tmp_path / "sealed"
    batch = directory / "batch_00000"
    batch.mkdir(parents=True)
    request = {"codes": ["000001.SZ"]}
    (directory / "request.json").write_text(json.dumps(request))
    manifest = {"sha256": {}}
    for table, event in events.items():
        frame = pd.DataFrame(event.values, columns=event.fields)
        frame["stock_code"] = "000001.SZ"
        frame["m_timetag"] = [str(pd.Period(year=int(q // 4), quarter=int(q % 4 + 1), freq="Q").end_time.date()) for q in event.quarters]
        frame["m_anntime"] = event.announcement_dates.astype(str)
        for field in ABNORMAL_GROSS_PROFIT_FIELDS.get(table, ()):
            if field not in frame:
                frame[field] = 100.
        path = batch / f"{table}.parquet"
        frame.to_parquet(path)
        manifest["sha256"][table] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest_bytes = json.dumps(manifest).encode()
    (batch / "manifest.json").write_bytes(manifest_bytes)
    loaded, identity = load_financial_events(directory, ("000001.SZ",))
    expected = {"schema": "financial-announcement-events-v3-abnormal-gross-profit",
                "request": request, "batch_manifest_hashes": {"0": hashlib.sha256(manifest_bytes).hexdigest()},
                "audit": {table: event.audit for table, event in loaded.items()},
                "field_schema": {table: list(fields) for table, fields in FINANCIAL_FIELDS.items()}}
    expected["sha256"] = hashlib.sha256(json.dumps(expected, sort_keys=True).encode()).hexdigest()
    assert identity == expected
    assert {table: event.fields for table, event in loaded.items()} == FINANCIAL_FIELDS
    days = np.array(["2021-04-21", "2021-08-21"], dtype="datetime64[D]")
    for (_, original), (_, actual) in zip(iter_financial_fields(days, 1, events),
                                         iter_financial_fields(days, 1, loaded), strict=True):
        assert len(actual) == 21 and set(FINANCIAL_PANEL_FIELDS) <= set(actual)
        for name in original:
            np.testing.assert_array_equal(original[name], actual[name])
    research, research_identity = load_financial_events(directory, ("000001.SZ",),
                                                        field_set=ABNORMAL_GROSS_PROFIT_FIELD_SET)
    assert research_identity["field_schema"] == {table: list(fields) for table, fields in ABNORMAL_GROSS_PROFIT_FIELDS.items()}
    assert research_identity["sha256"] != identity["sha256"]
    assert research["Income"].fields == ("revenue", "total_expense")
    load_financial_events(directory, ("000001.SZ",), field_set=ABNORMAL_GROSS_PROFIT_FIELD_SET,
                          expected_identity=research_identity)
    with pytest.raises(ValueError, match="identity differs"):
        load_financial_events(directory, ("000001.SZ",), field_set=ABNORMAL_GROSS_PROFIT_FIELD_SET,
                              expected_identity=identity)
    with pytest.raises(ValueError, match="field order differs"):
        list(iter_financial_fields(days, 1, research))


def test_production_and_research_share_exact_quarter_operands_and_causality():
    from offline_data.financial_versions import ABNORMAL_GROSS_PROFIT_PANEL_FIELDS as data_fields
    from factor.library.abnormal_gross_profit import ABNORMAL_GROSS_PROFIT_PANEL_FIELDS as factor_fields
    assert data_fields == factor_fields
    research = research_events()
    production = {}
    for table, event in research.items():
        values = np.full((len(event.values), len(FINANCIAL_FIELDS[table])), 123.)
        for index, field in enumerate(event.fields):
            values[:, FINANCIAL_FIELDS[table].index(field)] = event.values[:, index]
        production[table] = replace(event, fields=FINANCIAL_FIELDS[table], values=values)
    days = np.array(["2021-04-20", "2021-04-21", "2021-08-20", "2021-08-21"], dtype="datetime64[D]")
    rows = list(iter_financial_fields(days, 1, production))
    for (_, actual), (_, expected) in zip(rows, replay(research, days), strict=True):
        for name in data_fields:
            np.testing.assert_array_equal(actual[name], expected[name])
    assert np.isnan(abnormal_gross_profit(rows[0][1])).all()
    np.testing.assert_allclose(abnormal_gross_profit(rows[1][1]), [.01])
    np.testing.assert_allclose(abnormal_gross_profit(rows[2][1]), [.01])
    np.testing.assert_allclose(abnormal_gross_profit(rows[3][1]), [.005])
