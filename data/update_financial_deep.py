"""深历史财务指标预下载（akshare 同花顺 stock_financial_abstract_ths，回溯至 1990s）。

现有 data/financial/pershare_index.parquet 仅回溯到 2020；本脚本补齐深历史，
用于 1993-2018 区间的价值/质量/成长财务因子回测。

数据源选择：东财 stock_financial_abstract 约 50 次请求后 IP 限流；同花顺
stock_financial_abstract_ths 限流宽松（实测 >4 req/s 稳定），且深度到 1996，
含每股净资产/每股收益/每股经营现金流/ROE/增长率等。

产物：data/financial/deep_indicators.parquet
  列：stock_code(带后缀), report_period(int YYYYMMDD), bps/eps/ocfps/roe/
      net_profit/revenue/profit_yoy/revenue_yoy/net_margin/debt_ratio
  每行 = (股票, 报告期)。可断点续传，按"首个交易日"升序抓取。

用法：
  uv run python data/update_financial_deep.py

红线：本脚本属预下载入口，允许联网（akshare）。
"""
import argparse
import time
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger('update_financial_deep')

DATA_DIR = Path(__file__).resolve().parent
OUT_PATH = DATA_DIR / 'financial' / 'deep_indicators.parquet'

# THS 列名 -> 输出列名
_COLMAP = {
    '每股净资产': 'bps',
    '基本每股收益': 'eps',
    '每股经营现金流': 'ocfps',
    '净资产收益率-摊薄': 'roe',
    '净利润': 'net_profit',
    '营业总收入': 'revenue',
    '净利润同比增长率': 'profit_yoy',
    '营业总收入同比增长率': 'revenue_yoy',
    '销售净利率': 'net_margin',
    '销售毛利率': 'gross_margin',
    '资产负债率': 'debt_ratio',
}
_OUT_COLS = list(dict.fromkeys(_COLMAP.values()))


def _parse_num(x):
    """解析同花顺数值：'6.29亿'/'42万'/'14.92%'/'False'/'--' -> float。"""
    if x is None or isinstance(x, bool):
        return np.nan
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip()
    if s in ('', 'False', '--', 'nan', 'None'):
        return np.nan
    mult = 1.0
    if s.endswith('%'):
        s = s[:-1]
    elif s.endswith('亿'):
        s = s[:-1]; mult = 1e8
    elif s.endswith('万'):
        s = s[:-1]; mult = 1e4
    try:
        return float(s) * mult
    except ValueError:
        return np.nan


def _all_symbols() -> list[tuple[str, str]]:
    """Enumerate the current stock list plus all local delisted A shares."""
    from data.db.stock_list import get_all_stock_code_list

    codes = sorted(get_all_stock_code_list())
    if not codes:
        raise RuntimeError("current stock_list + delist 股票全集为空")
    return [(code, code[:6]) for code in codes]


def _validate_snapshot(
    frame: pd.DataFrame,
    expected_codes: set[str],
) -> pd.DataFrame:
    required = {"stock_code", "report_period", *_OUT_COLS}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"deep_indicators 缺少列: {sorted(missing)}")
    if frame.empty:
        raise ValueError("deep_indicators 不得为空")

    result = frame.loc[:, ["stock_code", "report_period", *_OUT_COLS]].copy()
    result["stock_code"] = result["stock_code"].astype(str).str.strip().str.upper()
    unexpected = sorted(set(result["stock_code"]).difference(expected_codes))
    if unexpected:
        raise ValueError(
            "deep_indicators 包含股票全集之外的代码: "
            + ", ".join(unexpected[:20])
        )
    periods = pd.to_datetime(
        result["report_period"].astype(str), format="%Y%m%d", errors="raise"
    )
    result["report_period"] = periods.dt.strftime("%Y%m%d").astype(np.int64)
    if result.duplicated(["stock_code", "report_period"]).any():
        raise ValueError("deep_indicators 存在重复 (stock_code, report_period)")
    for column in _OUT_COLS:
        result[column] = pd.to_numeric(result[column], errors="raise")
    return result.sort_values(
        ["stock_code", "report_period"], kind="stable"
    ).reset_index(drop=True)


def _save_snapshot_atomic(
    frame: pd.DataFrame,
    expected_codes: set[str],
) -> None:
    canonical = _validate_snapshot(frame, expected_codes)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    temp_path = OUT_PATH.with_name(
        f".{OUT_PATH.name}.{os.getpid()}.tmp.parquet"
    )
    try:
        canonical.to_parquet(temp_path, index=False)
        _validate_snapshot(pd.read_parquet(temp_path), expected_codes)
        temp_path.replace(OUT_PATH)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def _parse_one(symbol6: str) -> pd.DataFrame:
    import akshare as ak

    df = ak.stock_financial_abstract_ths(symbol=symbol6, indicator='按报告期')
    if df is None or df.empty or '报告期' not in df.columns:
        return pd.DataFrame()
    out = pd.DataFrame()
    out['report_period'] = pd.to_datetime(df['报告期'], errors='coerce').dt.strftime('%Y%m%d')
    for src, dst in _COLMAP.items():
        out[dst] = df[src].map(_parse_num) if src in df.columns else np.nan
    out = out.dropna(subset=['report_period'])
    out['report_period'] = out['report_period'].astype(int)
    return out


def main(*, refresh: bool = False):
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    symbols = _all_symbols()
    expected_codes = {full for full, _symbol in symbols}
    existing = pd.DataFrame(columns=["stock_code", "report_period", *_OUT_COLS])
    if OUT_PATH.exists():
        existing = _validate_snapshot(pd.read_parquet(OUT_PATH), expected_codes)
        logger.info('已存在 %d 只股票', existing['stock_code'].nunique())

    done = set(existing['stock_code'].unique())
    todo = symbols if refresh else [(f, s) for f, s in symbols if f not in done]
    logger.info('待抓取 %d / 共 %d 只', len(todo), len(symbols))

    replacements: dict[str, pd.DataFrame] = {}

    def _combined() -> pd.DataFrame:
        replaced = set(replacements)
        parts = [existing[~existing["stock_code"].isin(replaced)]]
        parts.extend(replacements.values())
        return pd.concat(parts, ignore_index=True)

    t0 = time.time()
    failures: list[tuple[str, Exception]] = []
    for i, (full, sym6) in enumerate(todo):
        one = pd.DataFrame()
        for attempt in range(4):
            try:
                one = _parse_one(sym6)
                break
            except Exception as e:  # noqa: BLE001 — 下载模块允许网络重试
                if attempt == 3:
                    failures.append((full, e))
                    logger.warning('  %s 失败: %r', full, e)
                else:
                    time.sleep(2.0 * (attempt + 1))
        if not one.empty:
            one.insert(0, 'stock_code', full)
            replacements[full] = one
        time.sleep(0.12)
        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(todo) - i - 1) / rate / 60
            logger.info('进度 %d/%d  %.2f stk/s  ETA %.1f min  fail=%d  最近=%s',
                        i + 1, len(todo), rate, eta, len(failures), full)
            if replacements:
                _save_snapshot_atomic(_combined(), expected_codes)

    if replacements or not OUT_PATH.exists():
        _save_snapshot_atomic(_combined(), expected_codes)
    combined = _validate_snapshot(pd.read_parquet(OUT_PATH), expected_codes)
    logger.info('完成：%d 只股票, %d 行, fail=%d -> %s',
                combined['stock_code'].nunique(), len(combined), len(failures), OUT_PATH)
    if failures:
        shown = ", ".join(code for code, _exc in failures[:20])
        suffix = f" ...(+{len(failures) - 20})" if len(failures) > 20 else ""
        raise RuntimeError(
            f"deep_indicators 更新失败 {len(failures)} 只: {shown}{suffix}"
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="重新请求股票全集；成功代码替换旧记录，失败代码保留旧记录并阻断更新链",
    )
    main(refresh=parser.parse_args().refresh)
