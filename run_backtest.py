"""单回测入口示例：构建 runtime 数据 → 因子计算 → 逐日回测 → 生成报告

用法:
  python run_backtest.py --start 2024-01-01 --end 2024-12-31
  python run_backtest.py --start 2024-01-01 --end 2024-12-31 --config my_config.json
"""
import argparse
import json
import sys
import tempfile
from datetime import date, datetime
from pathlib import Path


def build_runtime(start_date: str, end_date: str):
    """步骤1: 拉取全量数据并构建 runtime NPZ"""
    from data.build_runtime import main as build_main
    import subprocess
    subprocess.run([sys.executable, '-u', 'data/build_runtime.py',
                    '--start', start_date, '--end', end_date], check=True)


def run_backtest(args):
    """步骤2: 运行单因子回测"""
    from core.backtest import run_single_mode
    from core.factors.registry import get_factor_class
    from data.db import allow_buy_stock_code_list
    from utils.stock.time import get_trading_date_span
    from testback.logger import testback_logger

    # 构建股票池和日期范围
    start = date.fromisoformat(args.start_date)
    end = date.fromisoformat(args.end_date)
    backtest_datetime_list = [datetime.combine(d, datetime.min.time())
                              for d in get_trading_date_span(start, end)]
    all_stocks = list(allow_buy_stock_code_list())

    # 构建 individual_config
    if args.config:
        with open(args.config, 'r') as f:
            individual_config = json.load(f)
    else:
        # 默认示例：TrueMarketCap 因子，持仓 5 只，日均仓再平衡
        individual_config = {
            'weights': {'TrueMarketCap': 1.0},
            'buy_n': 5,
            'sell_m': 5,
            'temperatures': {'TrueMarketCap': 1.0},
            'timing_enabled': False,
            'rebalance': True,
        }

    mode_config = {
        'desc': '单回测示例',
        'log_level': 'INFO',
        'save_charts': True,
    }

    # 写入临时 JSON 文件供 run_single_mode 读取
    config_data = {
        'individual_config': individual_config,
    }
    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
    json.dump(config_data, tmp)
    tmp.close()

    # 构造 args namespace
    class Args:
        individual_config = tmp.name
        output_dir = args.output_dir

    print(f"回测区间: {start} ~ {end}, 股票池: {len(all_stocks)} 只")
    print(f"因子: {list(individual_config['weights'].keys())}, "
          f"持仓: {individual_config['buy_n']} 只")
    print(f"开始回测...")

    result = run_single_mode(Args, mode_config, backtest_datetime_list, all_stocks)
    Path(tmp.name).unlink(missing_ok=True)
    return result


def main():
    parser = argparse.ArgumentParser(description='WBR 单回测示例')
    parser.add_argument('--start-date', default='2024-01-01')
    parser.add_argument('--end-date', default='2024-12-31')
    parser.add_argument('--config', default=None, help='individual_config JSON 文件（可选）')
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--build', action='store_true', help='先拉数据构建 runtime NPZ 再回测')
    args = parser.parse_args()

    if args.build:
        build_runtime(args.start_date, args.end_date)

    return run_backtest(args)


if __name__ == '__main__':
    main()
