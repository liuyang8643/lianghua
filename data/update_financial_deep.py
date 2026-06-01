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
import sys
import time
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import akshare as ak

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


def _ordered_symbols() -> list[tuple[str, str]]:
    """返回 [(full_code, symbol6)]，按首个交易日升序（早上市优先）。"""
    rt = sorted((DATA_DIR / 'runtime').glob('runtime_*.npz'))[-1]
    d = np.load(rt, allow_pickle=False)
    codes = d['stock_codes']
    open_ = d['open']
    first_idx = np.argmax(np.isfinite(open_), axis=0)
    has_open = np.isfinite(open_).any(axis=0)
    order = np.argsort(first_idx)
    return [(str(codes[j]), str(codes[j])[:6]) for j in order if has_open[j]]


def _parse_one(symbol6: str) -> pd.DataFrame:
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


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    parts = []
    if OUT_PATH.exists():
        existing = pd.read_parquet(OUT_PATH)
        done = set(existing['stock_code'].unique())
        parts.append(existing)
        logger.info('已存在 %d 只股票，续传跳过', len(done))

    symbols = _ordered_symbols()
    todo = [(f, s) for f, s in symbols if f not in done]
    logger.info('待抓取 %d / 共 %d 只', len(todo), len(symbols))

    new_parts = []
    t0 = time.time()
    fail = 0
    for i, (full, sym6) in enumerate(todo):
        one = pd.DataFrame()
        for attempt in range(4):
            try:
                one = _parse_one(sym6)
                break
            except Exception as e:  # noqa: BLE001 — 下载模块允许网络重试
                if attempt == 3:
                    fail += 1
                    logger.warning('  %s 失败: %r', full, e)
                else:
                    time.sleep(2.0 * (attempt + 1))
        if not one.empty:
            one.insert(0, 'stock_code', full)
            new_parts.append(one)
        time.sleep(0.12)
        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(todo) - i - 1) / rate / 60
            logger.info('进度 %d/%d  %.2f stk/s  ETA %.1f min  fail=%d  最近=%s',
                        i + 1, len(todo), rate, eta, fail, full)
            if new_parts:
                pd.concat(parts + new_parts, ignore_index=True).to_parquet(OUT_PATH, index=False)

    pd.concat(parts + new_parts, ignore_index=True).to_parquet(OUT_PATH, index=False)
    combined = pd.read_parquet(OUT_PATH)
    logger.info('完成：%d 只股票, %d 行, fail=%d -> %s',
                combined['stock_code'].nunique(), len(combined), fail, OUT_PATH)


if __name__ == '__main__':
    sys.exit(main())
