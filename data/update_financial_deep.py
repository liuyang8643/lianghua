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
import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

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


def _fetch_ths(symbol6: str) -> pd.DataFrame:
    """THS 报告期页的有界请求；AkShare 对此接口不暴露 timeout 参数。"""
    with requests.Session() as session:
        session.trust_env = False
        response = session.get(f'https://basic.10jqka.com.cn/new/{symbol6}/finance.html',
                               headers={'User-Agent': 'Mozilla/5.0'}, timeout=15)
        response.raise_for_status()
    node = BeautifulSoup(response.text, 'lxml').find('p', id='main')
    if node is None or node.string is None:
        raise ValueError(f'THS 财务页缺少报告: {symbol6}')
    payload = json.loads(node.string)
    names = [item[0] if isinstance(item, list) else item for item in payload['title']]
    reports = payload['report']
    return (pd.DataFrame(reports[1:], columns=reports[0], index=names[1:]).T
            .rename_axis('报告期').reset_index())


def _parse_one(symbol6: str) -> pd.DataFrame:
    try:
        df = _fetch_ths(symbol6)
    except (AttributeError, ValueError, requests.RequestException):
        # 老退市代码的 THS 页面已迁移，新浪仍保留历史报告，不能当下载成功跳过。
        return _parse_sina(symbol6)
    if df is None or df.empty or '报告期' not in df.columns:
        return _parse_sina(symbol6)
    out = pd.DataFrame()
    out['report_period'] = pd.to_datetime(df['报告期'], errors='coerce').dt.strftime('%Y%m%d')
    for src, dst in _COLMAP.items():
        out[dst] = df[src].map(_parse_num) if src in df.columns else np.nan
    out = out.dropna(subset=['report_period'])
    out['report_period'] = out['report_period'].astype(int)
    if out.empty or not np.isfinite(out[_OUT_COLS].to_numpy(dtype=float)).any():
        raise RuntimeError(f'THS 财务报告全为空值: {symbol6}')
    return out


def _parse_sina(symbol6: str) -> pd.DataFrame:
    """同口径报告期指标备用源；未提供的指标保持 NaN，不用零或最新值填充。"""
    prefix = 'sh' if symbol6.startswith('6') else 'sz'
    with requests.Session() as session:
        session.trust_env = False
        response = session.get(
            'https://quotes.sina.cn/cn/api/openapi.php/CompanyFinanceService.getFinanceReport2022',
            params={'paperCode': prefix + symbol6, 'source': 'gjzb',
                    'type': '0', 'page': '1', 'num': '1000'}, timeout=15)
        response.raise_for_status()
        result = response.json()['result']
    if result['status']['code'] != 0:
        raise RuntimeError(f'新浪财务请求失败: {symbol6}')
    reports = result['data']['report_list']
    if not reports:
        raise RuntimeError(f'新浪财务无报告: {symbol6}')
    mapping = {**_COLMAP, '净资产收益率': 'roe', '销售净利率': 'net_margin',
               '销售毛利率': 'gross_margin'}
    rows = []
    for period, report in reports.items():
        row = {key: np.nan for key in _OUT_COLS}
        row['report_period'] = int(pd.to_datetime(period, format='%Y%m%d').strftime('%Y%m%d'))
        for item in report['data']:
            title = item['item_title']
            if title in mapping:
                row[mapping[title]] = _parse_num(item['item_value'])
            if title in ('净利润', '营业总收入'):
                row['profit_yoy' if title == '净利润' else 'revenue_yoy'] = _parse_num(item.get('item_tongbi'))
        rows.append(row)
    frame = pd.DataFrame(rows)
    if not np.isfinite(frame[_OUT_COLS].to_numpy(dtype=float)).any():
        raise RuntimeError(f'新浪财务报告全为空值: {symbol6}')
    return frame.sort_values('report_period').reset_index(drop=True)


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
    if fail:
        # 保留已成功股票作为断点，但必须让 update_all 知道本次
        # 并未完整成功；否则会误写 checkpoint 并用不完整数据构建 Runtime。
        raise RuntimeError(f"[深历史财务] {fail}/{len(todo)} 只下载失败")


if __name__ == '__main__':
    sys.exit(main())
