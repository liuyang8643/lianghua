"""构建 runtime npz 文件：将所有 parquet 中间数据合并为 stocks × dates 维度的 numpy 数组。

输出格式（按 CLAUDE.md 规范）：
  np.savez_compressed(
    'runtime_{start}_{end}.npz',
    stock_codes=np.array(..., dtype='U12'),
    trade_dates=np.array(..., dtype='datetime64[D]'),
    open=ndarray(n_dates, n_stocks),       # 不复权真实价
    high=ndarray(n_dates, n_stocks),
    low=ndarray(n_dates, n_stocks),
    close=ndarray(n_dates, n_stocks),
    volume=ndarray(n_dates, n_stocks),
    amount=ndarray(n_dates, n_stocks),
    preClose=ndarray(n_dates, n_stocks),   # 官方前收盘价(除权除息参考价)，涨跌停/收益基准
    issue_price=ndarray(n_stocks,),        # 每股发行价（元），从新浪 IPO 快照获取
    issue_date=ndarray(n_stocks,),         # 发行价对应上市日，datetime64[D]
    st_mask=ndarray(bool, n_dates, n_stocks),
    listing_age=ndarray(int32, n_dates, n_stocks), # -1 未上市，0 首个有效 open
    delisted_mask=ndarray(bool, n_dates, n_stocks),  # 退市日后首个交易日起持续为 True
    total_share=ndarray(n_dates, n_stocks),
    eps=ndarray(n_dates, n_stocks),
    roe=ndarray(n_dates, n_stocks),
    profit_yoy=ndarray(n_dates, n_stocks),
    revenue_yoy=ndarray(n_dates, n_stocks),
    operating_cf_ps=ndarray(n_dates, n_stocks),
    gross_margin=ndarray(n_dates, n_stocks),
  )

用法:
  uv run python data/build_runtime.py
"""
import os
import time
from datetime import date
from pathlib import Path
from typing import Collection, NamedTuple

import numpy as np
import pandas as pd

from data.financial_pit import (
    build_pit_source_indices,
    materialize_pit_field,
)
from data.kline_mootdx import (
    BAOSTOCK_EVIDENCE_PATH,
    load_baostock_daily_evidence,
)
from data.kline_secondary_evidence import (
    DEFAULT_SNAPSHOT_PATH as SECONDARY_EVIDENCE_PATH,
    load_verified_daily_evidence as load_secondary_daily_evidence,
)
from data.db.issue_price import resolve_terminal_active_codes


def save_runtime_npz_atomic(output_path: Path, **arrays) -> None:
    temp_path = output_path.with_name(f'.{output_path.name}.{os.getpid()}.tmp.npz')
    try:
        np.savez_compressed(temp_path, **arrays)
        temp_path.replace(output_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()

DATA_DIR = Path(__file__).resolve().parent
OUT_DIR = DATA_DIR / "runtime"
KLINE_DIR = DATA_DIR / "k-line"             # mootdx 不复权（唯一原始价源 / 交易日历 / 股票全集）

# 原始字段（含官方 preClose）
RAW_FIELDS = ['open', 'high', 'low', 'close', 'volume', 'amount', 'preClose']


class ListingDateAlignmentError(RuntimeError):
    """First-bar/listing-evidence conflict with serializable diagnostics."""

    def __init__(self, diagnostics: list[dict[str, object]]) -> None:
        self.diagnostics = tuple(diagnostics)
        shown = ", ".join(
            f"{item['code']}(first_kline={item['first_kline_date']}, "
            f"first_executable={item['first_executable_date']}, expected "
            f"{'/'.join(item['expected_first_executable_dates'])})"
            for item in diagnostics[:20]
        )
        suffix = f" ...(+{len(diagnostics) - 20})" if len(diagnostics) > 20 else ""
        super().__init__(
            "股票首条 K 线与上市日期证据不一致，拒绝生成 runtime: "
            f"{len(diagnostics)} 只: {shown}{suffix}"
        )


class KlineDailyEvidenceError(RuntimeError):
    """Local K-line rows conflict with sealed independent daily evidence."""

    def __init__(self, diagnostics: list[dict[str, object]]) -> None:
        self.diagnostics = tuple(diagnostics)
        shown = ", ".join(
            f"{item['code']}({item['date']}:{item['kind']})"
            for item in diagnostics[:20]
        )
        suffix = f" ...(+{len(diagnostics) - 20})" if len(diagnostics) > 20 else ""
        super().__init__(
            "K 线逐日可交易证据缺失或冲突，拒绝生成 runtime: "
            f"{len(diagnostics)} 行: {shown}{suffix}"
        )


class DailyEvidenceContext(NamedTuple):
    """One validated evidence snapshot reused across one runtime build."""

    primary: pd.DataFrame
    secondary: pd.DataFrame
    roles: pd.DataFrame
    lifecycle: dict[str, dict[str, object]]
    role_indices_by_code: dict[str, np.ndarray]


def _align_into(arr_cols: list[np.ndarray], df: pd.DataFrame, src_fields: list[str],
                trade_date_list: np.ndarray, j: int):
    """把单只 parquet（时间倒序）的 src_fields 列按交易日对齐写入 arr_cols 的第 j 列。"""
    kline_dates = df['time'].to_numpy(np.int64).astype('datetime64[ms]').astype('datetime64[D]')
    sort_idx = np.argsort(kline_dates)
    kline_dates = kline_dates[sort_idx]
    indices = np.searchsorted(kline_dates, trade_date_list, side='left')
    indices = np.clip(indices, 0, len(kline_dates) - 1)
    match = kline_dates[indices] == trade_date_list
    idx = indices[match]
    for arr, f in zip(arr_cols, src_fields):
        arr[match, j] = df[f].to_numpy(float)[sort_idx][idx]


def load_kline_panel(
    stock_codes: np.ndarray,
    trade_dates: np.ndarray,
    *,
    evidence_context: DailyEvidenceContext | None = None,
) -> dict[str, np.ndarray]:
    """构建 OHLCV 面板：原始价来自 k-line/。

    零容错：stock_codes 来自 k-line/ 全集；缺文件/缺列直接报错。
    """
    n_dates = len(trade_dates)
    n_stocks = len(stock_codes)
    arrays = {f: np.full((n_dates, n_stocks), np.nan, dtype=np.float64)
              for f in RAW_FIELDS}
    trade_date_list = trade_dates.astype('datetime64[D]')

    t0 = time.time()
    last_log = t0
    for j, code in enumerate(stock_codes):
        raw = pd.read_parquet(KLINE_DIR / f"{code}.parquet").sort_values('time').reset_index(drop=True)
        _align_into([arrays[f] for f in RAW_FIELDS], raw, RAW_FIELDS, trade_date_list, j)

        now = time.time()
        if now - last_log >= 5 or j == n_stocks - 1:
            elapsed = now - t0
            speed = (j + 1) / elapsed if elapsed > 0 else 0
            eta = (n_stocks - j - 1) / speed if speed > 0 else 0
            print(f"[{time.strftime('%H:%M:%S')}] K线面板: {j+1}/{n_stocks} "
                  f"(耗时 {elapsed:.0f}s, 速度 {speed:.0f}只/s, 预计剩余 {eta:.0f}s)")
            last_log = now

    apply_baostock_daily_evidence(
        arrays,
        stock_codes,
        trade_dates,
        evidence_context=evidence_context,
    )
    print(f"K线面板完成: {n_stocks} 只（不复权 k-line/）")
    return arrays


def _daily_executable_mask(rows: pd.DataFrame) -> np.ndarray:
    """Return independent daily execution semantics without volume/amount.

    Explicit status is authoritative when present.  Old source rows can leave
    ``tradestatus`` empty, in which case only that source's positive
    ``reference_open`` proves an executable price bar.
    """
    status = rows["tradestatus"].fillna("").astype(str).str.strip()
    if not status.isin({"", "0", "1"}).all():
        raise ValueError("daily evidence tradestatus 只能是空串/0/1")
    reference_open = pd.to_numeric(rows["reference_open"], errors="coerce").to_numpy(
        dtype=np.float64
    )
    return (
        status.eq("1").to_numpy()
        | (
            status.eq("").to_numpy()
            & np.isfinite(reference_open)
            & (reference_open > 0.0)
        )
    )


def _secondary_executable_mask(rows: pd.DataFrame) -> np.ndarray:
    """Validate exact secondary execution evidence without copying its price.

    Adjusted historical sources can report finite negative reference prices
    across entity replacement/corporate-action boundaries.  The sealed row's
    role is only to prove that an exact date traded: status must be ``1`` and
    the reference value must be finite.  It is never a canonical K-line price.
    """

    status = rows["tradestatus"].fillna("").astype(str).str.strip()
    reference_open = pd.to_numeric(
        rows["reference_open"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    if not status.eq("1").all() or not np.isfinite(reference_open).all():
        raise ValueError("次级 T-open evidence 必须是明确可成交证据日")
    return np.ones(len(rows), dtype=bool)


def _load_secondary_daily_evidence(path: Path | None = None) -> pd.DataFrame:
    """Load the optional hash-sealed T-open evidence without any network I/O."""
    target = SECONDARY_EVIDENCE_PATH if path is None else Path(path)
    if path is None and not target.exists():
        return pd.DataFrame(
            columns=("stock_code", "date", "reference_open", "tradestatus")
        )
    frame = load_secondary_daily_evidence(path=target)
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("load_verified_daily_evidence 必须返回 DataFrame")
    required = {"stock_code", "date", "reference_open", "tradestatus"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"次级 T-open evidence 缺少列: {sorted(missing)}")
    result = frame.copy()
    result["stock_code"] = (
        result["stock_code"].astype(str).str.strip().str.upper()
    )
    result["date"] = pd.to_datetime(result["date"], errors="raise").values.astype(
        "datetime64[D]"
    )
    if result.duplicated(["stock_code", "date"]).any():
        raise ValueError("次级 T-open evidence stock_code/date 重复")
    # Evaluate once here as a contract check.  The returned frame retains only
    # the source fields allowed to affect executable-date semantics.
    _secondary_executable_mask(result)
    return result


def _daily_evidence_roles(
    baostock_evidence: pd.DataFrame,
    secondary_evidence: pd.DataFrame,
) -> pd.DataFrame:
    """Preserve primary validation and secondary missing-bar roles.

    Primary status can validate or mask only a date already present in local
    K.  A row previously marked ``normalization_applied`` is an audited
    placeholder candidate and primary status alone cannot retain it.  Only a
    hash-sealed exact secondary row can require/retain a missing candidate.
    """

    parts: list[pd.DataFrame] = []
    if not baostock_evidence.empty:
        primary = baostock_evidence.loc[:, ["stock_code", "date"]].copy()
        audited_candidate = (
            baostock_evidence["normalization_applied"].fillna(False).astype(bool)
        )
        primary["primary_present"] = True
        primary["primary_executable"] = (
            _daily_executable_mask(baostock_evidence) & ~audited_candidate.to_numpy()
        )
        primary["secondary_executable"] = False
        parts.append(primary)
    if not secondary_evidence.empty:
        secondary = secondary_evidence.loc[:, ["stock_code", "date"]].copy()
        secondary["primary_present"] = False
        secondary["primary_executable"] = False
        secondary["secondary_executable"] = _secondary_executable_mask(
            secondary_evidence
        )
        parts.append(secondary)
    if not parts:
        return pd.DataFrame(
            columns=(
                "stock_code",
                "date",
                "primary_present",
                "primary_executable",
                "secondary_executable",
            )
        )
    combined = pd.concat(parts, ignore_index=True)
    return (
        combined.groupby(["stock_code", "date"], as_index=False, sort=True)[
            [
                "primary_present",
                "primary_executable",
                "secondary_executable",
            ]
        ]
        .any()
        .reset_index(drop=True)
    )


def _baostock_lifecycle_by_code(
    evidence: pd.DataFrame,
    secondary_evidence: pd.DataFrame | None = None,
    *,
    roles: pd.DataFrame | None = None,
) -> dict[str, dict[str, object]]:
    """Derive lifecycle metadata without promoting primary-only missing bars."""
    secondary = (
        secondary_evidence
        if secondary_evidence is not None
        else pd.DataFrame(
            columns=("stock_code", "date", "reference_open", "tradestatus")
        )
    )
    resolved_roles = (
        roles if roles is not None else _daily_evidence_roles(evidence, secondary)
    )
    role_indices_by_code = {
        str(code): indices
        for code, indices in resolved_roles.groupby(
            "stock_code", sort=False
        ).indices.items()
    }
    result: dict[str, dict[str, object]] = {}
    for code, rows in evidence.groupby("stock_code", sort=False):
        code = str(code)
        first = rows.iloc[0]
        primary_dates = rows.loc[
            _daily_executable_mask(rows), "date"
        ].to_numpy(dtype="datetime64[D]")
        role_indices = role_indices_by_code.get(code)
        code_roles = (
            resolved_roles.iloc[role_indices]
            if role_indices is not None
            else None
        )
        secondary_dates = (
            code_roles.loc[
                code_roles["secondary_executable"], "date"
            ].to_numpy(dtype="datetime64[D]")
            if code_roles is not None
            else np.array([], dtype="datetime64[D]")
        )
        result[code] = {
            "listing_date": np.datetime64(first["listing_date"], "D"),
            "primary_first_executable_date": (
                primary_dates[0] if primary_dates.size else None
            ),
            "secondary_first_executable_date": (
                secondary_dates[0] if secondary_dates.size else None
            ),
            "secondary_executable_dates": secondary_dates,
        }
    primary_codes = set(result)
    for code, role_indices in role_indices_by_code.items():
        if code in primary_codes:
            continue
        rows = resolved_roles.iloc[role_indices]
        rows = rows.loc[rows["secondary_executable"]]
        if rows.empty:
            continue
        executable_dates = rows["date"].to_numpy(dtype="datetime64[D]")
        result[code] = {
            "listing_date": None,
            "primary_first_executable_date": None,
            "secondary_first_executable_date": executable_dates[0],
            "secondary_executable_dates": executable_dates,
        }
    return result


def load_daily_evidence_context(
    *,
    evidence_path: Path = BAOSTOCK_EVIDENCE_PATH,
    secondary_evidence_path: Path | None = None,
) -> DailyEvidenceContext:
    """Validate primary/secondary evidence exactly once per runtime build."""

    primary = load_baostock_daily_evidence(evidence_path)
    secondary = _load_secondary_daily_evidence(secondary_evidence_path)
    roles = _daily_evidence_roles(primary, secondary)
    lifecycle = _baostock_lifecycle_by_code(
        primary,
        secondary,
        roles=roles,
    )
    role_indices_by_code = {
        str(code): indices
        for code, indices in roles.groupby(
            "stock_code", sort=False
        ).indices.items()
    }
    return DailyEvidenceContext(
        primary=primary,
        secondary=secondary,
        roles=roles,
        lifecycle=lifecycle,
        role_indices_by_code=role_indices_by_code,
    )


def apply_baostock_daily_evidence(
    arrays: dict[str, np.ndarray],
    stock_codes: np.ndarray,
    trade_dates: np.ndarray,
    *,
    evidence_path: Path = BAOSTOCK_EVIDENCE_PATH,
    secondary_evidence_path: Path | None = None,
    evidence_context: DailyEvidenceContext | None = None,
) -> None:
    """Mask non-executable rows using the strict primary/secondary union.

    Each source uses only ``tradestatus`` and ``reference_open``.  T-day
    runtime volume and amount never participate in this normalization.
    """
    required = {"open", "high", "low", "close", "volume", "amount", "preClose"}
    missing_fields = required.difference(arrays)
    if missing_fields:
        raise ValueError(f"kline_arrays 缺少字段: {sorted(missing_fields)}")
    codes = np.asarray(stock_codes).astype(str)
    dates = np.asarray(trade_dates, dtype="datetime64[D]")
    expected_shape = (dates.size, codes.size)
    for field in required:
        if np.asarray(arrays[field]).shape != expected_shape:
            raise ValueError(f"kline_arrays[{field!r}] shape 不一致")

    opens = np.asarray(arrays["open"], dtype=np.float64)
    context = evidence_context or load_daily_evidence_context(
        evidence_path=evidence_path,
        secondary_evidence_path=secondary_evidence_path,
    )
    roles = context.roles
    diagnostics: list[dict[str, object]] = []
    mask_plan: list[tuple[int, np.ndarray]] = []
    code_to_stock = {value: index for index, value in enumerate(codes)}
    for raw_code, rows in roles.groupby("stock_code", sort=False):
        code = str(raw_code)
        stock = code_to_stock.get(code)
        if stock is None:
            continue
        reference_dates = rows["date"].to_numpy(dtype="datetime64[D]")
        positive_rows = np.flatnonzero(
            np.isfinite(opens[:, stock]) & (opens[:, stock] > 0.0)
        )
        positive_dates = dates[positive_rows]
        has_exact_evidence = np.isin(
            positive_dates,
            reference_dates,
            assume_unique=True,
        )
        unsupported = ~has_exact_evidence
        for value in positive_dates[unsupported]:
            diagnostics.append(
                {
                    "code": code,
                    "date": str(value),
                    "kind": "missing_daily_evidence",
                }
            )

        reference_rows = np.searchsorted(dates, reference_dates, side="left")
        in_runtime = reference_rows < dates.size
        if in_runtime.any():
            in_runtime[in_runtime] &= (
                dates[reference_rows[in_runtime]] == reference_dates[in_runtime]
            )
        primary_executable = rows["primary_executable"].to_numpy(dtype=np.bool_)
        secondary_executable = rows["secondary_executable"].to_numpy(
            dtype=np.bool_
        )
        effective_executable = primary_executable | secondary_executable
        within_runtime_range = (
            (reference_dates >= dates[0]) & (reference_dates <= dates[-1])
        )
        for value in reference_dates[
            secondary_executable & within_runtime_range & ~in_runtime
        ]:
            diagnostics.append(
                {
                    "code": code,
                    "date": str(value),
                    "kind": "missing_trade_date_axis",
                }
            )
        for source_row in np.flatnonzero(secondary_executable & in_runtime):
            local_row = int(reference_rows[source_row])
            if not (
                np.isfinite(opens[local_row, stock])
                and opens[local_row, stock] > 0.0
            ):
                diagnostics.append(
                    {
                        "code": code,
                        "date": str(reference_dates[source_row]),
                        "kind": "missing_local_executable_history",
                    }
                )
        # Primary-only absent dates are intentionally ignored.  For an exact
        # primary row already present on the runtime axis, a non-executable
        # status (including an audited placeholder candidate) is a mask; an
        # exact secondary executable row is the only override.
        non_executable = rows["primary_present"].to_numpy(dtype=np.bool_) & ~effective_executable
        rows_to_mask = reference_rows[in_runtime & non_executable]
        mask_plan.append((stock, rows_to_mask))
    if diagnostics:
        raise KlineDailyEvidenceError(diagnostics)
    for stock, rows_to_mask in mask_plan:
        for field in ("open", "high", "low", "close", "preClose"):
            arrays[field][rows_to_mask, stock] = np.nan
        arrays["volume"][rows_to_mask, stock] = 0.0
        arrays["amount"][rows_to_mask, stock] = 0.0


def build_st_mask(
    stock_codes: np.ndarray,
    trade_dates: np.ndarray,
) -> np.ndarray:
    """从 st_changes.parquet 构建 ST 掩码。

    Returns:
        bool ndarray (n_dates, n_stocks), True = ST/*ST/退市状态
    """
    n_dates = len(trade_dates)
    n_stocks = len(stock_codes)
    st_path = DATA_DIR / "stock_name" / "st_changes.parquet"

    if not st_path.exists():
        raise FileNotFoundError(f"ST 历史快照不存在，拒绝生成 runtime: {st_path}")

    df_st = pd.read_parquet(st_path)
    from data.db.stock_name import validate_st_changes

    validate_st_changes(df_st)

    trade_date_arr = trade_dates.astype('datetime64[ns]')
    result = np.zeros((n_dates, n_stocks), dtype=bool)

    _KEEP_ST_KEYWORDS = ("披*", "退市整理", "戴帽", "暂停上市", "终止上市")
    _CLEAR_KEYWORDS = ("摘帽", "恢复上市", "新股上市", "重新上市", "转板上市", "摘*摘帽", "发行失败", "拟上市")

    for j, code in enumerate(stock_codes):
        bare = str(code).split('.')[0]
        sc_records = df_st[df_st['bare_code'] == bare]
        if sc_records.empty:
            continue

        dates = sc_records['date'].values.astype('datetime64[ns]')
        events = sc_records['event'].values

        # _KEEP_ST_KEYWORDS 优先：命中则保持 ST
        is_keep = np.array([any(kw in str(e) for kw in _KEEP_ST_KEYWORDS) for e in events])
        is_clear = np.array([any(kw in str(e) for kw in _CLEAR_KEYWORDS) for e in events])
        # keep 优先级最高, clear 次之, 其余保持 ST
        status_changes = np.where(is_keep, True, ~is_clear)

        indices = np.searchsorted(dates, trade_date_arr, side='right') - 1
        valid = indices >= 0
        if valid.any():
            result[valid, j] = status_changes[indices[valid]]

    print(f"ST掩码: {result.sum()} 个 True / {result.size} ({result.sum()/result.size*100:.1f}%)")
    return result


def build_financial_arrays(
    stock_codes: np.ndarray,
    trade_dates: np.ndarray,
) -> dict[str, np.ndarray]:
    """从 deep_indicators.parquet（同花顺深历史，唯一财务源）构建财务指标数组。

    PIT：使用法定最晚披露日后的首个交易日生效，持续至下一份报告披露。
    单一数据源、无兜底/无合并；所有字段绑定同一报告期，源报告缺失即为 NaN。
    """
    n_dates = len(trade_dates)
    n_stocks = len(stock_codes)

    # 输出字段 -> deep_indicators 列名
    colmap = {
        'bps': 'bps',
        'eps': 'eps',
        'roe': 'roe',
        'operating_cf_ps': 'ocfps',
        'profit_yoy': 'profit_yoy',
        'revenue_yoy': 'revenue_yoy',
        'gross_margin': 'gross_margin',
    }
    results = {k: np.full((n_dates, n_stocks), np.nan, dtype=np.float64) for k in colmap}

    deep_path = DATA_DIR / "financial" / "deep_indicators.parquet"
    if not deep_path.exists():
        print("警告: deep_indicators.parquet 不存在，财务指标全部为 NaN")
        return results

    df = pd.read_parquet(deep_path)
    td = trade_dates.astype('datetime64[D]')

    source_indices = build_pit_source_indices(df, stock_codes, td)

    for out_name, src_col in colmap.items():
        if src_col not in df.columns:
            continue
        results[out_name] = materialize_pit_field(
            df,
            source_indices,
            src_col,
        )

    cov = {k: int(np.isfinite(v).any(axis=0).sum()) for k, v in results.items()}
    print(f"财务面板完成（deep_indicators, 法定披露日后 PIT, 单一源）: 覆盖股票 {cov}")
    return results


def build_total_share(
    stock_codes: np.ndarray,
    trade_dates: np.ndarray,
) -> np.ndarray:
    """从 balance.parquet 的 cap_stk 构建历史总股本数组 (n_dates, n_stocks)。

    使用 m_anntime（披露日期）对齐交易日期，避免未来信息泄露。
    """
    n_dates = len(trade_dates)
    n_stocks = len(stock_codes)
    result = np.full((n_dates, n_stocks), np.nan, dtype=np.float64)

    balance_path = DATA_DIR / "financial" / "balance.parquet"
    if not balance_path.exists():
        raise FileNotFoundError(f"缺少股本数据: {balance_path}")

    trade_date_ts = trade_dates.astype('datetime64[ns]')
    df_all = pd.read_parquet(balance_path)
    required = {"stock_code", "m_anntime", "cap_stk"}
    missing_columns = required.difference(df_all.columns)
    if missing_columns:
        raise ValueError(f"官方股本数据缺少列: {sorted(missing_columns)}")
    if df_all.duplicated(["stock_code", "m_anntime"]).any():
        raise ValueError("官方股本数据存在重复 (stock_code, m_anntime)")
    df_all = df_all.sort_values(['stock_code', 'm_anntime'])

    t0 = time.time()
    last_log = t0

    for j, code in enumerate(stock_codes):
        df_code = df_all[df_all['stock_code'] == code]
        if df_code.empty:
            continue

        anntimes = df_code['m_anntime'].values.astype('datetime64[ns]')
        if len(anntimes) == 0:
            continue

        indices = np.searchsorted(anntimes, trade_date_ts, side='right') - 1
        valid = indices >= 0
        if not valid.any():
            continue

        cap_stk = df_code['cap_stk'].to_numpy(dtype=np.float64)
        result[valid, j] = cap_stk[indices[valid]]

        now = time.time()
        if now - last_log >= 5 or j == n_stocks - 1:
            elapsed = now - t0
            speed = (j + 1) / elapsed if elapsed > 0 else 0
            eta = (n_stocks - j - 1) / speed if speed > 0 else 0
            print(f"[{time.strftime('%H:%M:%S')}] 总股本: {j+1}/{n_stocks} "
                  f"(耗时 {elapsed:.0f}s, 速度 {speed:.0f}只/s, 预计剩余 {eta:.0f}s)")
            last_log = now

    covered = np.isfinite(result).sum()
    print(f"总股本: {covered}/{result.size} 有效 ({covered/result.size*100:.1f}%)")
    return result


def build_delisted_mask(
    stock_codes: np.ndarray,
    trade_dates: np.ndarray,
) -> np.ndarray:
    """Build the PIT write-off state used by the unified env settlement.

    The mask turns true on the first runtime trading day strictly after the
    locally downloaded delist date and remains true. Future delist metadata
    therefore cannot affect an earlier decision row.
    """

    from data.db.delist import get_delist_stock_info

    result = np.zeros((len(trade_dates), len(stock_codes)), dtype=np.bool_)
    delist_info = get_delist_stock_info()
    date_axis = trade_dates.astype("datetime64[D]")
    for stock_index, raw_code in enumerate(stock_codes):
        info = delist_info.get(str(raw_code))
        if info is None:
            continue
        first_written_off = int(
            np.searchsorted(
                date_axis,
                np.datetime64(info.delist_date, "D"),
                side="right",
            )
        )
        result[first_written_off:, stock_index] = True
    return result


def build_listing_age(open_prices: np.ndarray) -> np.ndarray:
    """Build split-independent trading-row ages from the full runtime history."""

    opens = np.asarray(open_prices)
    if opens.ndim != 2 or not np.issubdtype(opens.dtype, np.number):
        raise ValueError("open_prices must be a numeric [date, stock] matrix")
    valid = np.isfinite(opens) & (opens > 0.0)
    has_open = valid.any(axis=0)
    first = np.where(has_open, valid.argmax(axis=0), -1).astype(np.int32)
    rows = np.arange(opens.shape[0], dtype=np.int32)[:, None]
    ages = rows - first[None, :]
    ages[(first[None, :] < 0) | (ages < 0)] = -1
    return np.ascontiguousarray(ages, dtype=np.int32)


def sanitize_first_bar_preclose(
    open_prices: np.ndarray,
    preclose_prices: np.ndarray,
    issue_prices: np.ndarray,
    trade_dates: np.ndarray,
    issue_dates: np.ndarray,
) -> np.ndarray:
    """Use issue price only when its listing date is the first tradable row."""
    opens = np.asarray(open_prices, dtype=np.float64)
    precloses = np.array(preclose_prices, dtype=np.float64, copy=True, order="C")
    issues = np.asarray(issue_prices, dtype=np.float64)
    dates = np.asarray(trade_dates, dtype="datetime64[D]")
    listed = np.asarray(issue_dates, dtype="datetime64[D]")
    if opens.shape != precloses.shape or opens.ndim != 2:
        raise ValueError("open_prices and preclose_prices must share shape [date, stock]")
    if issues.shape != (opens.shape[1],):
        raise ValueError("issue_prices must have shape [stock]")
    if dates.shape != (opens.shape[0],):
        raise ValueError("trade_dates must have shape [date]")
    if listed.shape != (opens.shape[1],):
        raise ValueError("issue_dates must have shape [stock]")
    valid_open = np.isfinite(opens) & (opens > 0.0)
    has_open = valid_open.any(axis=0)
    stock_indices = np.flatnonzero(has_open)
    first_rows = valid_open[:, has_open].argmax(axis=0)
    references = issues[has_open]
    exact_listing_day = dates[first_rows] == listed[has_open]
    references = np.where(
        exact_listing_day & np.isfinite(references) & (references > 0.0),
        references,
        np.nan,
    )
    precloses[first_rows, stock_indices] = references
    return precloses


def validate_listing_date_alignment(
    stock_codes: np.ndarray,
    trade_dates: np.ndarray,
    open_prices: np.ndarray,
    issue_dates: np.ndarray,
    *,
    evidence_path: Path = BAOSTOCK_EVIDENCE_PATH,
    secondary_evidence_path: Path | None = None,
    evidence_context: DailyEvidenceContext | None = None,
) -> None:
    """Reject first-executable conflicts without equating IPO and execution dates."""

    codes = tuple(str(code) for code in np.asarray(stock_codes))
    dates = np.asarray(trade_dates, dtype="datetime64[D]")
    opens = np.asarray(open_prices, dtype=np.float64)
    issues = np.asarray(issue_dates, dtype="datetime64[D]")
    if opens.shape != (len(dates), len(codes)):
        raise ValueError("open_prices must match trade_dates × stock_codes")
    if issues.shape != (len(codes),):
        raise ValueError("issue_dates must have shape [stock]")

    listing_events = _load_listing_event_dates()
    context = evidence_context or load_daily_evidence_context(
        evidence_path=evidence_path,
        secondary_evidence_path=secondary_evidence_path,
    )
    roles = context.roles
    baostock = context.lifecycle
    role_indices_by_code = context.role_indices_by_code
    valid_open = np.isfinite(opens) & (opens > 0.0)
    has_open = valid_open.any(axis=0)
    first_open_rows = valid_open.argmax(axis=0)
    date_to_row = {value: index for index, value in enumerate(dates)}
    failures: list[dict[str, object]] = []
    for stock_index, code in enumerate(codes):
        if not has_open[stock_index]:
            continue
        all_event_evidence = {
            np.datetime64(value, "D")
            for value in listing_events.get(code[:6], ())
        }
        event_evidence = {
            value
            for value in all_event_evidence
            if np.datetime64(value, "D") >= dates[0]
        }
        listing_evidence = set(event_evidence)
        issue_source: np.datetime64 | None = None
        issue_evidence: np.datetime64 | None = None
        if not np.isnat(issues[stock_index]):
            issue_source = issues[stock_index]
            if issue_source >= dates[0]:
                issue_evidence = issue_source
                listing_evidence.add(issue_evidence)
        first_date = dates[int(first_open_rows[stock_index])]
        independent = baostock.get(code)
        baostock_listing = (
            independent["listing_date"] if independent is not None else None
        )
        baostock_first = (
            independent["primary_first_executable_date"]
            if independent is not None
            else None
        )
        secondary_first = (
            independent["secondary_first_executable_date"]
            if independent is not None
            else None
        )
        if baostock_listing is not None and baostock_listing >= dates[0]:
            listing_evidence.add(baostock_listing)
        role_indices = role_indices_by_code.get(code)
        code_roles = roles.iloc[role_indices] if role_indices is not None else None
        if code_roles is None:
            supported_source_dates: set[np.datetime64] = set()
            secondary_required_dates: set[np.datetime64] = set()
        else:
            supported_source_dates = set(
                code_roles.loc[
                    code_roles["primary_executable"]
                    | code_roles["secondary_executable"],
                    "date",
                ].to_numpy(dtype="datetime64[D]")
            )
            secondary_required_dates = {
                value
                for value in code_roles.loc[
                    code_roles["secondary_executable"], "date"
                ].to_numpy(dtype="datetime64[D]")
                if dates[0] <= value <= dates[-1]
            }
        local_supported_dates = {
            value
            for value in supported_source_dates
            if value in date_to_row
            and valid_open[date_to_row[value], stock_index]
        }
        exact_candidates = local_supported_dates | secondary_required_dates
        if exact_candidates:
            diagnostic_first = min(exact_candidates)
            expected_executable = {diagnostic_first}
        else:
            expected_executable = set(listing_evidence)
            diagnostic_first = first_date
        has_listing_evidence = (
            bool(all_event_evidence)
            or issue_source is not None
            or baostock_listing is not None
        )
        if has_listing_evidence and (
            not expected_executable or first_date in expected_executable
        ):
            continue
        diagnostic: dict[str, object] = {
            "code": code,
            "first_kline_date": str(first_date),
            "first_executable_date": str(diagnostic_first),
            "expected_listing_dates": [
                str(value) for value in sorted(listing_evidence)
            ],
            "expected_first_executable_dates": [
                str(value) for value in sorted(expected_executable)
            ],
            "evidence": {
                "stock_name_events": [
                    str(value) for value in sorted(event_evidence)
                ],
                "issue_date": (
                    str(issue_evidence) if issue_evidence is not None else None
                ),
                "baostock_listing_date": (
                    str(baostock_listing)
                    if baostock_listing is not None
                    else None
                ),
                "baostock_first_executable_date": (
                    str(baostock_first) if baostock_first is not None else None
                ),
            },
        }
        if not has_listing_evidence:
            diagnostic["reason"] = "missing_listing_evidence"
        if secondary_first is not None:
            diagnostic["evidence"]["secondary_first_executable_date"] = str(
                secondary_first
            )
        failures.append(diagnostic)
    if failures:
        raise ListingDateAlignmentError(failures)


def _relevant_delisted_stocks(trade_dates: np.ndarray) -> dict:
    from data.db.delist import get_delist_stock_info
    from utils.stock.info import is_b_stock

    if len(trade_dates) == 0:
        raise ValueError("runtime trade date axis is empty")
    range_start = pd.Timestamp(trade_dates[0]).date()
    range_end = pd.Timestamp(trade_dates[-1]).date()
    return {
        code: info
        for code, info in get_delist_stock_info().items()
        if not is_b_stock(code)
        and info.list_date <= range_end
        and info.delist_date >= range_start
    }


def validate_delisted_kline_coverage(
    kline_stocks: set[str],
    trade_dates: np.ndarray,
) -> None:
    """Reject a runtime whose requested history omits an active delisted A share."""
    relevant = _relevant_delisted_stocks(trade_dates)
    missing = sorted(
        code
        for code in relevant
        if code not in kline_stocks
    )
    if missing:
        shown = ", ".join(missing[:20])
        suffix = f" ...(+{len(missing) - 20})" if len(missing) > 20 else ""
        raise RuntimeError(
            "退市股票 K 线覆盖不完整，拒绝生成存在幸存者偏差的 runtime: "
            f"{len(missing)} 只: {shown}{suffix}"
        )


def _resolve_expected_latest_codes(
    partial_live_candidates: Collection[str] | None,
) -> set[str]:
    """Return the current-market codes that must have the terminal T open.

    ``None`` is the offline/default path and always means the complete current
    stock list.  A non-empty explicit subset is reserved for the live 09:25
    prefilter snapshot; callers cannot silently request an empty bypass.
    """
    from data.db.stock_list import load_current_stock_codes

    current = set(load_current_stock_codes())
    if not current:
        raise RuntimeError("当前在市股票列表为空，拒绝生成 runtime")
    if partial_live_candidates is None:
        return current
    requested = {str(code).strip().upper() for code in partial_live_candidates}
    if not requested:
        raise ValueError("partial_live_candidates 不得为空")
    unknown = sorted(requested.difference(current))
    if unknown:
        shown = ", ".join(unknown[:20])
        suffix = f" ...(+{len(unknown) - 20})" if len(unknown) > 20 else ""
        raise ValueError(
            "partial_live_candidates 必须是当前在市股票子集: "
            f"{shown}{suffix}"
        )
    return requested


def validate_current_kline_terminal_coverage(
    stock_codes: np.ndarray,
    trade_dates: np.ndarray,
    preclose_prices: np.ndarray,
    expected_latest_codes: Collection[str],
) -> None:
    """Require terminal source evidence for every terminal-active stock.

    A positive terminal preClose proves the source represented the stock even
    when a same-day suspension legitimately leaves T-open missing.  The caller
    must first remove only lifecycle-proven ``list_date > T`` securities via
    :func:`resolve_terminal_active_codes`.
    """
    codes = np.asarray(stock_codes).astype(str)
    dates = np.asarray(trade_dates, dtype="datetime64[D]")
    precloses = np.asarray(preclose_prices, dtype=np.float64)
    if dates.ndim != 1 or dates.size == 0:
        raise ValueError("runtime trade date axis is empty")
    if precloses.shape != (dates.size, codes.size):
        raise ValueError("preclose_prices shape must match trade_dates × stock_codes")
    code_to_index = {code: index for index, code in enumerate(codes)}
    required = sorted(set(expected_latest_codes))
    if not required:
        raise ValueError("terminal-active 股票集合不得为空")
    missing_axis = [code for code in required if code not in code_to_index]
    stale = []
    for code in required:
        if code not in code_to_index:
            continue
        stock_index = code_to_index[code]
        covered = (
            np.isfinite(precloses[-1, stock_index])
            and precloses[-1, stock_index] > 0.0
        )
        if not covered:
            stale.append(code)
    failures = [*(f"{code}(missing file)" for code in missing_axis), *stale]
    if failures:
        shown = ", ".join(failures[:20])
        suffix = f" ...(+{len(failures) - 20})" if len(failures) > 20 else ""
        raise RuntimeError(
            f"当前在市股票末端 {dates[-1]} 行情覆盖不完整，拒绝生成 runtime: "
            f"{len(failures)} 只: {shown}{suffix}"
        )


def apply_live_open_overlay(
    kline_arrays: dict[str, np.ndarray],
    stock_codes: np.ndarray,
    trade_dates: np.ndarray,
    overlay_path: Path,
    current_stock_codes: Collection[str],
) -> None:
    """Fill only missing terminal open/preClose from a sealed full-axis quote.

    This is exclusively for the partial 09:25 live fetch. Candidate values
    already sourced from mootdx remain untouched, and no other terminal field
    is filled, so the full runtime stock axis is preserved without fabricating
    OHLCVA data.
    """
    required_columns = {"trade_date", "stock_code", "open", "preClose"}
    if not overlay_path.exists():
        raise FileNotFoundError(f"实盘 T-open overlay 不存在: {overlay_path}")
    frame = pd.read_parquet(overlay_path)
    if set(frame.columns) != required_columns:
        raise ValueError(
            "实盘 T-open overlay 必须且只能包含列: "
            f"{sorted(required_columns)}"
        )
    if frame.empty:
        raise ValueError("实盘 T-open overlay 不得为空")

    frame = frame.copy()
    frame["stock_code"] = (
        frame["stock_code"].astype(str).str.strip().str.upper()
    )
    if frame["stock_code"].duplicated().any():
        raise ValueError("实盘 T-open overlay stock_code 重复")
    expected = set(current_stock_codes)
    actual = set(frame["stock_code"])
    if actual != expected:
        missing = sorted(expected.difference(actual))
        extra = sorted(actual.difference(expected))
        raise ValueError(
            "实盘 T-open overlay 未严格覆盖 current stock_list: "
            f"missing={missing[:20]}, extra={extra[:20]}"
        )

    terminal_date = np.asarray(trade_dates, dtype="datetime64[D]")[-1]
    overlay_dates = pd.to_datetime(frame["trade_date"], errors="raise").values.astype(
        "datetime64[D]"
    )
    if not np.all(overlay_dates == terminal_date):
        raise ValueError(
            f"实盘 T-open overlay 日期必须全部等于 runtime 末日 {terminal_date}"
        )
    frame["preClose"] = pd.to_numeric(frame["preClose"], errors="raise")
    preclose_values = frame["preClose"].to_numpy(dtype=np.float64)
    if not (np.isfinite(preclose_values) & (preclose_values > 0.0)).all():
        raise ValueError("实盘 T-open overlay preClose 必须全部为有限正值")
    frame["open"] = pd.to_numeric(frame["open"], errors="raise")
    open_values = frame["open"].to_numpy(dtype=np.float64)
    if np.isinf(open_values).any() or np.any(open_values < 0.0):
        raise ValueError("实盘 T-open overlay open 只能是正值、0 或 NaN")
    frame.loc[frame["open"] == 0.0, "open"] = np.nan

    code_to_index = {
        str(code): index for index, code in enumerate(np.asarray(stock_codes))
    }
    missing_axis = sorted(expected.difference(code_to_index))
    if missing_axis:
        raise RuntimeError(
            "实盘 T-open overlay 中的当前股票缺少历史 K 文件: "
            + ", ".join(missing_axis[:20])
        )
    rows = frame.set_index("stock_code").loc[sorted(expected)]
    indices = np.array([code_to_index[code] for code in rows.index], dtype=np.int64)
    for field in ("open", "preClose"):
        terminal = kline_arrays[field][-1, indices]
        missing = ~(np.isfinite(terminal) & (terminal > 0.0))
        kline_arrays[field][-1, indices[missing]] = rows[field].to_numpy(
            dtype=np.float64
        )[missing]


def validate_delisted_kline_panel(
    stock_codes: np.ndarray,
    trade_dates: np.ndarray,
    kline_arrays: dict[str, np.ndarray],
    *,
    evidence_path: Path = BAOSTOCK_EVIDENCE_PATH,
    secondary_evidence_path: Path | None = None,
    evidence_context: DailyEvidenceContext | None = None,
) -> None:
    """Require every independently executable delisted-stock row and preClose."""
    relevant = _relevant_delisted_stocks(trade_dates)
    code_to_index = {str(code): index for index, code in enumerate(stock_codes)}
    opens = np.asarray(kline_arrays["open"], dtype=np.float64)
    precloses = np.asarray(kline_arrays["preClose"], dtype=np.float64)
    dates = np.asarray(trade_dates, dtype="datetime64[D]")
    expected_shape = (dates.size, len(stock_codes))
    if opens.shape != expected_shape or precloses.shape != expected_shape:
        raise ValueError("open/preClose shape must match trade_dates × stock_codes")
    failures: list[str] = []
    context = evidence_context or load_daily_evidence_context(
        evidence_path=evidence_path,
        secondary_evidence_path=secondary_evidence_path,
    )
    roles = context.roles
    baostock = context.lifecycle
    role_indices_by_code = context.role_indices_by_code
    date_to_row = {value: index for index, value in enumerate(dates)}
    for code, info in relevant.items():
        if code not in code_to_index:
            failures.append(f"{code}(missing K-line axis)")
            continue
        stock_index = code_to_index[code]
        independent = baostock.get(code)
        if independent is None:
            failures.append(f"{code}(missing independent daily evidence)")
            continue

        role_indices = role_indices_by_code.get(code)
        code_roles = roles.iloc[role_indices] if role_indices is not None else None
        if code_roles is None:
            supported_dates: set[np.datetime64] = set()
            secondary_dates: set[np.datetime64] = set()
        else:
            supported_dates = set(
                code_roles.loc[
                    code_roles["primary_executable"]
                    | code_roles["secondary_executable"],
                    "date",
                ].to_numpy(dtype="datetime64[D]")
            )
            secondary_dates = set(
                code_roles.loc[
                    code_roles["secondary_executable"], "date"
                ].to_numpy(dtype="datetime64[D]")
            )
        required_secondary = {
            value
            for value in secondary_dates
            if dates[0] <= value <= dates[-1]
        }
        missing_axis_dates = sorted(
            value for value in required_secondary if value not in date_to_row
        )
        if missing_axis_dates:
            shown = "/".join(str(value) for value in missing_axis_dates[:5])
            failures.append(f"{code}(trade-date axis missing {shown})")

        required_rows = {
            date_to_row[value]: value
            for value in required_secondary
            if value in date_to_row
        }
        missing_open_dates = sorted(
            value
            for row, value in required_rows.items()
            if not (
                np.isfinite(opens[row, stock_index])
                and opens[row, stock_index] > 0.0
            )
        )
        if missing_open_dates:
            shown = "/".join(str(value) for value in missing_open_dates[:5])
            failures.append(f"{code}(open missing {shown})")

        administrative_lifecycle = (
            (dates >= np.datetime64(info.list_date, "D"))
            & (dates <= np.datetime64(info.delist_date, "D"))
        )
        local_open_dates = set(
            dates[
                administrative_lifecycle
                & np.isfinite(opens[:, stock_index])
                & (opens[:, stock_index] > 0.0)
            ]
        )
        unexpected_open_dates = sorted(local_open_dates.difference(supported_dates))
        if unexpected_open_dates:
            shown = "/".join(str(value) for value in unexpected_open_dates[:5])
            failures.append(f"{code}(open lacks executable evidence {shown})")

        expected_dates = local_open_dates | required_secondary
        global_first = min(expected_dates) if expected_dates else None
        missing_preclose_dates = sorted(
            value
            for value in local_open_dates
            if value != global_first
            and not (
                np.isfinite(precloses[date_to_row[value], stock_index])
                and precloses[date_to_row[value], stock_index] > 0.0
            )
        )
        if missing_preclose_dates:
            shown = "/".join(str(value) for value in missing_preclose_dates[:5])
            failures.append(f"{code}(preClose missing {shown})")

        if not expected_dates:
            # A requested slice can overlap the administrative lifecycle after
            # the final trade.  Independent evidence makes this empty period
            # legitimate rather than an implicit success.
            continue
        if required_secondary and not local_open_dates:
            failures.append(f"{code}(no executable lifecycle rows)")
    if failures:
        shown = ", ".join(failures[:20])
        suffix = f" ...(+{len(failures) - 20})" if len(failures) > 20 else ""
        raise RuntimeError(
            "退市股票生命周期 K 线不完整，拒绝生成 runtime: "
            f"{len(failures)} 只: {shown}{suffix}"
        )


def _load_listing_event_dates() -> dict[str, tuple[date, ...]]:
    """Return local A-share listing events used to disambiguate source rows."""
    path = DATA_DIR / "stock_name" / "st_changes.parquet"
    if not path.exists():
        return {}
    frame = pd.read_parquet(path, columns=["bare_code", "date", "event"])
    events = frame[
        frame["event"].astype(str).str.contains("新股上市", regex=False, na=False)
    ].copy()
    if events.empty:
        return {}
    events["bare_code"] = events["bare_code"].astype(str).str.strip().str.zfill(6)
    events["date"] = pd.to_datetime(events["date"], errors="raise").dt.date
    return {
        str(code): tuple(sorted(set(rows["date"])))
        for code, rows in events.groupby("bare_code", sort=False)
    }


def get_trade_dates_from_kline() -> np.ndarray:
    """从所有 k-line parquet 文件中提取并集交易日列表。

    Returns:
        sorted unique numpy array of datetime64[D] dates
    """
    parquet_files = sorted(KLINE_DIR.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"未找到 k-line parquet 文件: {KLINE_DIR}")

    all_dates = set()
    t0 = time.time()

    for i, path in enumerate(parquet_files):
        df = pd.read_parquet(path, columns=['time'])
        if df.empty:
            continue
        # time 是 ms 时间戳 → date
        dates = pd.to_datetime(df['time'], unit='ms').dt.date
        all_dates.update(dates)

        if (i + 1) % 500 == 0:
            print(f"[{time.strftime('%H:%M:%S')}] 交易日收集: {i+1}/{len(parquet_files)}")

    dates_arr = np.array(sorted(all_dates), dtype='datetime64[D]')
    print(f"交易日收集完成: {len(all_dates)} 个, 耗时 {time.time()-t0:.1f}s")
    return dates_arr


def build_issue_data(
    stock_codes: np.ndarray,
    *,
    issue_reference: pd.DataFrame | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """构建发行价及其上市日期；只有二者配对才可作为首日因果基准。

    未匹配到的股票发行价为 NaN。
    """
    n_stocks = len(stock_codes)
    prices = np.full(n_stocks, np.nan, dtype=np.float64)
    dates = np.full(n_stocks, np.datetime64("NaT", "D"), dtype="datetime64[D]")

    from data.db.issue_price import load_issue_reference

    df = load_issue_reference() if issue_reference is None else issue_reference
    ip_map = {}
    for _, row in df.iterrows():
        ip_map[str(row['stock_code']).zfill(6)] = (
            float(row['issue_price']),
            np.datetime64(pd.Timestamp(row['list_date']).date(), "D"),
        )

    matched = 0
    for j, code in enumerate(stock_codes):
        bare = code[:6]
        issue = ip_map.get(bare)
        if issue is not None and issue[0] > 0:
            prices[j], dates[j] = issue
            matched += 1

    print(f"发行价: {matched}/{n_stocks} 匹配 ({matched/n_stocks*100:.1f}%)")
    return prices, dates


def build_runtime(
    *,
    partial_live_candidates: Collection[str] | None = None,
    live_open_overlay: Path | None = None,
):
    """构建唯一的全历史、全股票生产 runtime。"""
    overall_t0 = time.time()
    if partial_live_candidates is None and live_open_overlay is not None:
        raise ValueError("普通离线 runtime 禁止读取 live_open_overlay")
    if partial_live_candidates is not None and live_open_overlay is None:
        raise ValueError("partial live runtime 必须显式提供 live_open_overlay")
    requested_latest_codes = (
        _resolve_expected_latest_codes(partial_live_candidates)
        if partial_live_candidates is not None
        else None
    )

    print(f"[{time.strftime('%H:%M:%S')}] ===== 1/9 收集交易日 =====")
    all_trade_dates = get_trade_dates_from_kline()

    # 剔除1970脏数据（最早A股交易在1990-12-19）
    all_trade_dates = all_trade_dates[all_trade_dates >= np.datetime64('1990-12-01')]

    print(f"交易日范围: {all_trade_dates[0]} ~ {all_trade_dates[-1]}, 共 {len(all_trade_dates)} 天")

    # 确定 stock_codes: 有 K 线 parquet 的历史全集，不能用当前可买池过滤历史退市股。
    kline_stocks = sorted([f.stem for f in KLINE_DIR.glob("*.parquet")])
    validate_delisted_kline_coverage(set(kline_stocks), all_trade_dates)
    stock_codes = np.array(kline_stocks, dtype='U12')
    from data.db.stock_list import load_current_stock_codes

    from data.db.issue_price import load_issue_reference

    issue_reference = load_issue_reference()
    current_stock_codes = load_current_stock_codes()
    terminal_active_codes = resolve_terminal_active_codes(
        current_stock_codes,
        all_trade_dates[-1],
        kline_stocks,
        issue_reference=issue_reference,
    )
    active_missing_axis = sorted(
        set(terminal_active_codes).difference(kline_stocks)
    )
    if active_missing_axis:
        shown = ", ".join(active_missing_axis[:20])
        suffix = (
            f" ...(+{len(active_missing_axis) - 20})"
            if len(active_missing_axis) > 20
            else ""
        )
        raise RuntimeError(
            "runtime 末日已上市股票缺少 K 线文件，拒绝加载完整面板: "
            f"{shown}{suffix}"
        )
    if requested_latest_codes is not None:
        not_yet_listed = sorted(
            set(requested_latest_codes).difference(terminal_active_codes)
        )
        if not_yet_listed:
            raise ValueError(
                "partial_live_candidates 包含 runtime 末日尚未上市股票: "
                + ", ".join(not_yet_listed[:20])
            )

    print(f"有K线: {len(kline_stocks)} 只, 使用: {len(stock_codes)} 只")

    print(f"[{time.strftime('%H:%M:%S')}] ===== 2/9 校验证据并构建K线面板 =====")
    evidence_context = load_daily_evidence_context()
    kline_arrays = load_kline_panel(
        stock_codes,
        all_trade_dates,
        evidence_context=evidence_context,
    )
    if partial_live_candidates is None:
        validate_current_kline_terminal_coverage(
            stock_codes,
            all_trade_dates,
            kline_arrays["preClose"],
            terminal_active_codes,
        )
    else:
        apply_live_open_overlay(
            kline_arrays,
            stock_codes,
            all_trade_dates,
            Path(live_open_overlay),
            terminal_active_codes,
        )
        validate_current_kline_terminal_coverage(
            stock_codes,
            all_trade_dates,
            kline_arrays["preClose"],
            terminal_active_codes,
        )

    # issue_price: 每股发行价，用于 IPO 首日涨跌停基准
    issue_price, issue_dates = build_issue_data(
        stock_codes,
        issue_reference=issue_reference,
    )
    kline_arrays['preClose'] = sanitize_first_bar_preclose(
        kline_arrays['open'],
        kline_arrays['preClose'],
        issue_price,
        all_trade_dates,
        issue_dates,
    )
    validate_listing_date_alignment(
        stock_codes,
        all_trade_dates,
        kline_arrays['open'],
        issue_dates,
        evidence_context=evidence_context,
    )
    validate_delisted_kline_panel(
        stock_codes,
        all_trade_dates,
        kline_arrays,
        evidence_context=evidence_context,
    )

    print(f"[{time.strftime('%H:%M:%S')}] ===== 3/9 构建ST掩码 =====")
    st_mask = build_st_mask(stock_codes, all_trade_dates)

    print(f"[{time.strftime('%H:%M:%S')}] ===== 4/9 构建总股本 =====")
    total_share = build_total_share(stock_codes, all_trade_dates)

    print(f"[{time.strftime('%H:%M:%S')}] ===== 5/9 构建财务面板 =====")
    fin_arrays = build_financial_arrays(stock_codes, all_trade_dates)

    print(f"[{time.strftime('%H:%M:%S')}] ===== 6/9 构建上市年龄 =====")
    listing_age = build_listing_age(kline_arrays['open'])

    print(f"[{time.strftime('%H:%M:%S')}] ===== 7/9 构建退市状态 =====")
    delisted_mask = build_delisted_mask(stock_codes, all_trade_dates)

    print(f"[{time.strftime('%H:%M:%S')}] ===== 8/8 保存 npz =====")
    output_name = f"runtime_{all_trade_dates[0]}_{all_trade_dates[-1]}.npz"
    output_path = OUT_DIR / output_name
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    save_runtime_npz_atomic(
        output_path,
        stock_codes=stock_codes,
        trade_dates=all_trade_dates,
        open=kline_arrays['open'],
        high=kline_arrays['high'],
        low=kline_arrays['low'],
        close=kline_arrays['close'],
        volume=kline_arrays['volume'],
        amount=kline_arrays['amount'],
        preClose=kline_arrays['preClose'],
        issue_price=issue_price,
        issue_date=issue_dates,
        st_mask=st_mask,
        listing_age=listing_age,
        delisted_mask=delisted_mask,
        total_share=total_share,
        bps=fin_arrays['bps'],
        eps=fin_arrays['eps'],
        roe=fin_arrays['roe'],
        profit_yoy=fin_arrays['profit_yoy'],
        revenue_yoy=fin_arrays['revenue_yoy'],
        operating_cf_ps=fin_arrays['operating_cf_ps'],
        gross_margin=fin_arrays['gross_margin'],
    )

    # 新 runtime 原子落盘成功后才清理旧文件。构建失败或被中断时，
    # 盘前进程仍可继续读取上一份完整 runtime。
    for old in OUT_DIR.glob("runtime_*.npz"):
        if old.name != output_name:
            old.unlink()
            print(f"  删除旧文件: {old.name}")

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    elapsed = time.time() - overall_t0
    print(f"构建完成: {output_path} ({file_size_mb:.1f} MB, 耗时 {elapsed:.0f}s)")
    return output_path


def main():
    build_runtime()


if __name__ == "__main__":
    main()
