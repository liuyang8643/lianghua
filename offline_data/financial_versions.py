"""Read sealed QMT announcement vintages without touching the network."""
from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Iterator, Mapping
import hashlib
from io import BytesIO
import json
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pandas as pd


FINANCIAL_FIELDS = {
    "Income": ("net_profit_excl_min_int_inc", "oper_profit", "revenue_inc", "revenue", "total_expense"),
    "Balance": ("tot_shrhldr_eqy_excl_min_int", "inventories", "other_payable",
                "account_receivable", "taxes_surcharges_payable", "other_current_liability", "tot_assets"),
    "CashFlow": ("stot_cash_outflows_oper_act", "net_cash_flows_oper_act", "goods_sale_and_service_render_cash"),
    "PershareIndex": ("equity_roe",),
}

FINANCIAL_REPLAY_VERSION = "financial-causal-raw-fields-v3-source-state"
LEGACY_FINANCIAL_PANEL_FIELDS = (
    "financial_profit_ttm", "financial_equity", "financial_cash_outflow_yoy",
    "financial_profit_yoy", "financial_operating_profit_yoy", "financial_revenue_yoy",
)

# The research subset retains its original identity; production explicitly
# versions the superset used by the eleven-factor snapshot.
ABNORMAL_GROSS_PROFIT_FIELD_SET = "abnormal-gross-profit-v1"
ABNORMAL_GROSS_PROFIT_FIELDS = {
    "Income": ("revenue", "total_expense"),
    "Balance": ("tot_assets",),
    "CashFlow": ("goods_sale_and_service_render_cash",),
}
ABNORMAL_GROSS_PROFIT_PANEL_FIELDS = (
    "abnormal_revenue_quarter", "abnormal_cost_quarter", "abnormal_sales_cash_quarter",
    "abnormal_revenue_prior_year_quarter", "abnormal_cost_prior_year_quarter",
    "abnormal_sales_cash_prior_year_quarter", "abnormal_total_assets",
)

# Source values, without TTM, growth, factor thresholds or cross-table alignment.
RAW_FINANCIAL_FIELDS = {
    "Income": FINANCIAL_FIELDS["Income"],
    "Balance": ("tot_shrhldr_eqy_excl_min_int", "tot_assets"),
    "CashFlow": ("stot_cash_outflows_oper_act", "goods_sale_and_service_render_cash"),
}
RAW_FINANCIAL_VALUE_NAMES = tuple(
    f"financial_raw.{table}.{field}" for table, fields in RAW_FINANCIAL_FIELDS.items() for field in fields)
RAW_FINANCIAL_PERIOD_NAMES = tuple(f"financial_raw.{table}.report_quarter" for table in RAW_FINANCIAL_FIELDS)
RAW_FINANCIAL_TIME_NAMES = tuple(
    f"financial_raw.{table}.{field}" for table in RAW_FINANCIAL_FIELDS
    for field in ("report_age_days", "announcement_age_days"))
RAW_FINANCIAL_STATE_VERSION = "financial-source-state-v1-announced-ytd-and-balance"
FINANCIAL_PANEL_FIELDS = (*LEGACY_FINANCIAL_PANEL_FIELDS, *ABNORMAL_GROSS_PROFIT_PANEL_FIELDS,
                          *RAW_FINANCIAL_VALUE_NAMES, *RAW_FINANCIAL_PERIOD_NAMES, *RAW_FINANCIAL_TIME_NAMES)


def _field_schema(field_set: str | None) -> Mapping[str, tuple[str, ...]]:
    if field_set is None:
        return FINANCIAL_FIELDS
    if field_set == ABNORMAL_GROSS_PROFIT_FIELD_SET:
        return ABNORMAL_GROSS_PROFIT_FIELDS
    raise ValueError(f"unknown financial research field set: {field_set}")


@dataclass(frozen=True)
class FinancialEvents:
    fields: tuple[str, ...]
    announcement_dates: np.ndarray
    quarters: np.ndarray
    stock_columns: np.ndarray
    values: np.ndarray
    audit: dict


def load_financial_events(directory: Path, stock_codes: tuple[str, ...], *,
                          expected_identity: Mapping | None = None,
                          field_set: str | None = None) -> tuple[dict[str, FinancialEvents], dict]:
    """Load authenticated raw fields; opt-in research gets a distinct identity.

    ``field_set=None`` loads the explicit production field schema.
    Research fields are selected from the same sealed source files, without
    changing those files or the production panel vocabulary.
    """
    field_schema = _field_schema(field_set)
    directory = Path(directory)
    request = json.loads((directory / "request.json").read_text(encoding="utf-8"))
    if expected_identity is not None:
        payload = {k: v for k, v in expected_identity.items() if k != "sha256"}
        if hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest() != expected_identity["sha256"]:
            raise ValueError("financial identity hash mismatch")
        if request != expected_identity["request"]:
            raise ValueError("financial request differs from sealed identity")
    if request["codes"] != list(stock_codes):
        raise ValueError("financial snapshot must retain the complete runtime stock axis")
    expected = list(range(0, len(stock_codes), 20))
    frames = {table: [] for table in field_schema}
    source_hashes = {}
    for offset in expected:
        batch = directory / f"batch_{offset:05d}"
        manifest_path = batch / "manifest.json"
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes)
        source_hashes[str(offset)] = hashlib.sha256(manifest_bytes).hexdigest()
        if expected_identity is not None and source_hashes[str(offset)] != expected_identity["batch_manifest_hashes"][str(offset)]:
            raise ValueError(f"financial batch manifest differs from sealed identity: {manifest_path}")
        for table in frames:
            path = batch / f"{table}.parquet"
            parquet_bytes = path.read_bytes()
            if hashlib.sha256(parquet_bytes).hexdigest() != manifest["sha256"][table]:
                raise ValueError(f"corrupted financial snapshot: {path}")
            frame = pd.read_parquet(BytesIO(parquet_bytes))
            if not frame.empty:
                frames[table].append(frame[["stock_code", "m_timetag", "m_anntime", *field_schema[table]]])
    events = {table: normalize_financial_events(
                  pd.concat(parts, ignore_index=True) if parts else
                  pd.DataFrame(columns=["stock_code", "m_timetag", "m_anntime", *fields]), stock_codes, fields)
              for (table, fields), parts in zip(field_schema.items(), frames.values(), strict=True)}
    identity = {"schema": "financial-announcement-events-v3-abnormal-gross-profit", "request": request,
                "batch_manifest_hashes": source_hashes, "audit": {k: v.audit for k, v in events.items()},
                "field_schema": {table: list(fields) for table, fields in field_schema.items()}}
    if field_set is not None:
        identity.update(schema="financial-announcement-research-events-v1", field_set=field_set,
                        field_schema={table: list(fields) for table, fields in field_schema.items()})
    identity["sha256"] = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    if expected_identity is not None and identity != dict(expected_identity):
        raise ValueError("normalized financial identity differs from sealed identity")
    return events, identity


def normalize_financial_events(frame: pd.DataFrame, stock_codes: tuple[str, ...], fields: tuple[str, ...]) -> FinancialEvents:
    """Keep every disclosure version; fail closed on same-key value conflicts.

    A conflicting field becomes unavailable at that announcement, and is
    counted. There is no arbitrary last-row preference or future backfill.
    """
    frame = frame.copy()
    input_rows = len(frame)
    for column in ("m_timetag", "m_anntime"):
        text = frame[column].astype(str).str.replace("-", "", regex=False).str[:8]
        frame[column] = pd.to_datetime(text, format="%Y%m%d", errors="raise")
    if frame[["m_timetag", "m_anntime"]].isna().any().any():
        raise ValueError("financial dates contain NaT")
    if (frame.m_anntime < frame.m_timetag).any():
        raise ValueError("financial announcement predates reporting period")
    if not frame.m_timetag.dt.is_quarter_end.all():
        raise ValueError("financial record is not a calendar-quarter report")
    # Legacy provider rows sometimes repeat the period end as publication date.
    # Without original-disclosure evidence that timestamp cannot establish PIT.
    # Preserve the raw archive, but do not activate such a report in research.
    unproven_date = frame.m_anntime == frame.m_timetag
    quarantined_rows = int(unproven_date.sum())
    frame = frame.loc[~unproven_date].copy()
    for field in fields:
        frame[field] = pd.to_numeric(frame[field], errors="raise")
        if np.isinf(frame[field]).any():
            raise ValueError(f"infinite {field}")
    keys = ["stock_code", "m_timetag", "m_anntime"]
    frame = frame.drop_duplicates(keys + list(fields))
    conflicts = {}
    if frame.duplicated(keys, keep=False).any():
        grouped = frame.groupby(keys, sort=False, dropna=False)
        counts = grouped[list(fields)].nunique(dropna=False)
        clean = grouped[list(fields)].first()
        for field in fields:
            conflict = counts[field] > 1
            clean.loc[conflict, field] = np.nan
            conflicts[field] = int(conflict.sum())
        frame = clean.reset_index()
    frame = frame.sort_values(["m_anntime", "m_timetag", "stock_code"], kind="stable")
    code_index = {code: index for index, code in enumerate(stock_codes)}
    columns = frame.stock_code.map(code_index)
    if columns.isna().any():
        raise ValueError("financial record outside immutable stock universe")
    announcements = frame.m_anntime.to_numpy(dtype="datetime64[D]")
    quarters = (frame.m_timetag.dt.year * 4 + frame.m_timetag.dt.quarter - 1).to_numpy(dtype=np.int32)
    values = frame[list(fields)].to_numpy(dtype=np.float64)
    columns = columns.to_numpy(dtype=np.int32)
    for value in (announcements, quarters, columns, values):
        value.setflags(write=False)
    audit = {"input_rows": input_rows, "version_rows": len(frame), "stocks": int(frame.stock_code.nunique()),
             "quarantined_period_end_announcement_rows": quarantined_rows,
             "multiple_version_periods": int((frame.groupby(["stock_code", "m_timetag"]).size() > 1).sum()),
             "conflicting_fields_unavailable": conflicts,
             "first_report": str(frame.m_timetag.min().date()) if len(frame) else None,
             "last_report": str(frame.m_timetag.max().date()) if len(frame) else None,
             "first_announcement": str(announcements.min()) if len(frame) else None,
             "last_announcement": str(announcements.max()) if len(frame) else None,
             "effective_rule": "strictly after announcement date; first following runtime trading day"}
    return FinancialEvents(fields, announcements, quarters, columns, values, audit)


def iter_financial_fields(dates: np.ndarray, n_stocks: int,
                          events: Mapping[str, FinancialEvents], *,
                          field_set: str | None = None) -> Iterator[tuple[np.datetime64, Mapping[str, np.ndarray]]]:
    """Yield independent read-only float64 fields using only announcements < T.

    TTM profit/equity use min(latest Income, latest Balance); cash outflow and
    profit YoY use min(latest Income, latest CashFlow); operating profit and
    revenue YoY both use latest Income. A missing selected period stays NaN:
    there is no older-period fallback. TTM is current YTD + prior annual -
    prior same YTD (annual reports use their own value). YoY uses single
    quarters, (current - prior) / abs(prior), with zero prior unavailable.

    Additional balance/cash fields preserve the existing research formulae.
    No market prices, ranks, stock selection, or revised annual fallback enter
    this data-layer replay. Published dates are date-only, so same-day
    announcements are deliberately excluded, including from prior operands.

    ``abnormal-gross-profit-v1`` instead emits seven raw quarterly operands
    aligned to min(latest Income, latest CashFlow, latest Balance). Revenue,
    cost and sales cash are converted from YTD to single-quarter values;
    assets remain the selected period's balance. The factor layer owns the
    resulting score and operand validity rules.
    """
    field_schema = _field_schema(field_set)
    dates = np.asarray(dates, dtype="datetime64[D]")
    if dates.ndim != 1 or np.isnat(dates).any() or np.any(dates[1:] <= dates[:-1]):
        raise ValueError("dates must be finite, sorted and unique")
    if n_stocks <= 0:
        raise ValueError("financial replay requires the full non-empty stock axis")
    tables = ("Income", "Balance", "CashFlow")
    for table in tables:
        event = events[table]
        length = len(event.announcement_dates)
        if event.fields != field_schema[table]:
            raise ValueError(f"financial field order differs: {table}")
        if (event.announcement_dates.shape != (length,) or event.quarters.shape != (length,)
                or event.stock_columns.shape != (length,)
                or event.values.shape != (length, len(event.fields))):
            raise ValueError(f"financial event axes differ: {table}")
        if (np.isnat(event.announcement_dates).any()
                or np.any(event.announcement_dates[1:] < event.announcement_dates[:-1])
                or np.any(event.stock_columns < 0) or np.any(event.stock_columns >= n_stocks)
                or np.isinf(event.values).any()):
            raise ValueError(f"invalid normalized financial events: {table}")
    nonempty = [events[t].quarters for t in tables if len(events[t].quarters)]
    first_q = min(int(q.min()) for q in nonempty) - 5 if nonempty else 0
    last_q = max(int(q.max()) for q in nonempty) if nonempty else 0
    state = {t: np.full((last_q - first_q + 1, n_stocks, len(events[t].fields)), np.nan) for t in tables}
    disclosed = {t: np.full((last_q - first_q + 1, n_stocks), np.datetime64('NaT', 'D')) for t in tables}
    latest = {t: np.full(n_stocks, -1, dtype=np.int32) for t in tables}
    pointers = {t: 0 for t in tables}
    columns = np.arange(n_stocks)

    def at(table, q, field):
        valid = (q >= first_q) & (q <= last_q)
        return np.where(valid, state[table][np.clip(q - first_q, 0, last_q - first_q), columns, field], np.nan)

    def ttm(table, q, field):
        value = at(table, q, field)
        return np.where(q % 4 == 3, value, value + at(table, q // 4 * 4 - 1, field) - at(table, q - 4, field))

    def quarter(table, q, field):
        return at(table, q, field) - np.where(q % 4 == 0, 0.0, at(table, q - 1, field))

    def yoy(table, q, field):
        previous = quarter(table, q - 4, field)
        numerator, denominator = quarter(table, q, field) - previous, np.abs(previous)
        result = np.full(n_stocks, np.nan)
        np.divide(numerator, denominator, out=result,
                  where=np.isfinite(numerator) & np.isfinite(denominator) & (denominator > 0))
        return result

    for day in dates:
        for table in tables:
            event = events[table]
            stop = int(np.searchsorted(event.announcement_dates, day, side="left"))
            start = pointers[table]
            # Revisions remain serial by announcement; stock updates are vectorised.
            for announcement in np.unique(event.announcement_dates[start:stop]):
                a = max(start, int(np.searchsorted(event.announcement_dates, announcement, side="left")))
                b = min(stop, int(np.searchsorted(event.announcement_dates, announcement, side="right")))
                c, q = event.stock_columns[a:b], event.quarters[a:b]
                state[table][q - first_q, c] = event.values[a:b]
                disclosed[table][q - first_q, c] = announcement
                np.maximum.at(latest[table], c, q)
            pointers[table] = stop
        # The research subset and production superset share this exact path.
        # All operands use one selected quarter, without an older-period fallback.
        q_abnormal = np.minimum(np.minimum(latest["Income"], latest["CashFlow"]), latest["Balance"])
        revenue_i = field_schema["Income"].index("revenue")
        cost_i = field_schema["Income"].index("total_expense")
        cash_i = field_schema["CashFlow"].index("goods_sale_and_service_render_cash")
        assets_i = field_schema["Balance"].index("tot_assets")
        abnormal_fields = {
            "abnormal_revenue_quarter": quarter("Income", q_abnormal, revenue_i),
            "abnormal_cost_quarter": quarter("Income", q_abnormal, cost_i),
            "abnormal_sales_cash_quarter": quarter("CashFlow", q_abnormal, cash_i),
            "abnormal_revenue_prior_year_quarter": quarter("Income", q_abnormal - 4, revenue_i),
            "abnormal_cost_prior_year_quarter": quarter("Income", q_abnormal - 4, cost_i),
            "abnormal_sales_cash_prior_year_quarter": quarter("CashFlow", q_abnormal - 4, cash_i),
            "abnormal_total_assets": at("Balance", q_abnormal, assets_i),
        }
        if field_set == ABNORMAL_GROSS_PROFIT_FIELD_SET:
            fields = abnormal_fields
            for value in fields.values():
                value.flags.writeable = False
            yield day, MappingProxyType(fields)
            continue
        q = np.minimum(latest["Income"], latest["Balance"])
        q_flow = np.minimum(latest["Income"], latest["CashFlow"])
        q_balance = latest["Balance"]
        q_cash = np.minimum(latest["Balance"], latest["CashFlow"])
        fields = {
            "financial_profit_ttm": ttm("Income", q, 0),
            "financial_equity": at("Balance", q, 0),
            "financial_cash_outflow_yoy": yoy("CashFlow", q_flow, 0),
            "financial_profit_yoy": yoy("Income", q_flow, 0),
            "financial_operating_profit_yoy": yoy("Income", latest["Income"], 1),
            "financial_revenue_yoy": yoy("Income", latest["Income"], 2),
            "financial_inventory": at("Balance", q_balance, 1),
            "financial_other_payable": at("Balance", q_balance, 2),
            "financial_receivable": at("Balance", q_balance, 3),
            "financial_operating_cash_flow": at("CashFlow", q_cash, 1),
            "financial_cash_taxes_payable": at("Balance", q_cash, 4),
            "financial_cash_other_payable": at("Balance", q_cash, 2),
            "financial_cash_other_current_liability": at("Balance", q_cash, 5),
            "financial_sales_cash": at("CashFlow", q_cash, 2),
            **abnormal_fields,
        }
        for table, raw_names in RAW_FINANCIAL_FIELDS.items():
            q = latest[table]
            known = q >= first_q
            fields[f'financial_raw.{table}.report_quarter'] = np.where(known, q % 4 + 1, np.nan)
            period_end = (np.asarray(q // 4 - 1970, dtype='datetime64[Y]').astype('datetime64[M]')
                          + (q % 4 + 1) * 3).astype('datetime64[D]') - np.timedelta64(1, 'D')
            announcement = disclosed[table][np.clip(q - first_q, 0, last_q - first_q), columns]
            fields[f'financial_raw.{table}.report_age_days'] = np.where(
                known, (day - period_end).astype('timedelta64[D]').astype(float), np.nan)
            fields[f'financial_raw.{table}.announcement_age_days'] = np.where(
                known, (day - announcement).astype('timedelta64[D]').astype(float), np.nan)
            for name in raw_names:
                fields[f'financial_raw.{table}.{name}'] = at(table, q, field_schema[table].index(name))
        for value in fields.values():
            value.flags.writeable = False
        yield day, MappingProxyType(fields)
