"""单回测入口：加载 NPZ → 因子计算 → 回测 → 报告。不依赖 core.ga。"""
import argparse
import sys
from datetime import date, datetime
from pathlib import Path

from core.backtest import run_single_mode
from core.runtime import load_runtime_stock_codes
from testback.logger import testback_logger
from utils.stock.time import get_trading_date_span


def _parse_date(s):
    if s is None:
        return None
    s = s.replace('-', '')
    if len(s) == 8:
        return date(int(s[:4]), int(s[4:6]), int(s[6:8]))
    raise ValueError(f'日期格式错误: {s}')


def main():
    from loguru import logger as loguru_logger

    parser = argparse.ArgumentParser(description='WBR 单回测')
    parser.add_argument('--individual-config', type=str, default='configs/config.json', help='最终策略 config JSON 文件')
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--start-date', type=str, default='20240101')
    parser.add_argument('--end-date', type=str, default='20241231')
    parser.add_argument('--no-live-sim', action='store_true',
                        help='跳过分钟价格实盘模拟，仅运行历史回测和报告')
    args = parser.parse_args()

    loguru_logger.remove()
    loguru_logger.add(sys.stderr, level='INFO')

    start_date = _parse_date(args.start_date)
    end_date = _parse_date(args.end_date)

    backtest_datetime_list = [
        datetime.combine(d, datetime.min.time())
        for d in get_trading_date_span(start_date, end_date)
    ]

    all_stocks = load_runtime_stock_codes()
    testback_logger.info(f"股票池(runtime历史全集): {len(all_stocks)} 只, 区间: {start_date} ~ {end_date}")

    mode_config = {'desc': '单回测', 'log_level': 'INFO', 'save_charts': True}
    from core.trend_timing import compute_configured_timing_multipliers
    return run_single_mode(
        args, mode_config, backtest_datetime_list, all_stocks,
        live_sim=not args.no_live_sim,
        timing_multiplier_builder=compute_configured_timing_multipliers,
    )


if __name__ == '__main__':
    main()
