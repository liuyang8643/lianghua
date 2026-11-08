"""File-only production entry plus retained offline diagnostic helpers."""
from datetime import datetime, timedelta
import numpy as np

from utils.stock.time import get_last_trading_day


def _load_prev_eod_baseline(trade_date):
    """T-1 收盘基线：昨日持仓股数 + 昨日现金，用于把「今日目标」锚定到昨日收盘状态。

    动机：base_target 必须与「当天运行几次」无关——否则重复运行会用实时持仓/资金重算目标，
    叠加「低买被开盘价高估权益」的正反馈，导致重跑不断追价 churn（见 603810 case）。
    数据源与回测 seed-replay 同口径：positions_{T-1}.parquet + daily_summary[T-1].cash。

    Returns: (prev_cash, {code: yesterday_shares}) 或 (None, None)（缺快照时回退实时）。
    """
    from datetime import timedelta
    from utils.stock.time import get_last_trading_day
    from trading.persistence import _TRADE_DIR

    prev = get_last_trading_day(trade_date - timedelta(days=1))
    pos_path = _TRADE_DIR / f"positions_{prev.isoformat()}.parquet"
    summary_path = _TRADE_DIR / "daily_summary.parquet"
    if not pos_path.exists() or not summary_path.exists():
        return None, None
    import pandas as pd
    sdf = pd.read_parquet(summary_path)
    prev_rows = sdf[sdf['date'] == prev]
    if prev_rows.empty or 'cash' not in prev_rows.columns:
        return None, None
    prev_cash = float(prev_rows['cash'].iloc[-1])
    pdf = pd.read_parquet(pos_path)
    y_shares = {r['code']: int(r['volume'])
                for _, r in pdf.iterrows() if int(r['volume']) > 0}
    return prev_cash, y_shares


def _parse_skip_datetime(skip_arg: str) -> datetime:
    """解析 --skip 的快进时间；分钟级 09:25 自动对齐到开盘调仓触发秒。"""
    from trading.scheduler import PREPARE_REBALANCE_START
    if len(skip_arg) == 12:
        skip_dt = datetime.strptime(skip_arg, '%Y%m%d%H%M')
        if skip_dt.hour == 9 and skip_dt.minute == 25 and skip_dt.second == 0:
            return datetime.combine(skip_dt.date(), PREPARE_REBALANCE_START)
        return skip_dt
    if len(skip_arg) == 14:
        return datetime.strptime(skip_arg, '%Y%m%d%H%M%S')
    raise ValueError(f'--skip 快进时间格式错误: {skip_arg}')


def _require_previous_session(trade_date):
    from core.runtime import latest_runtime_npz_path
    previous = get_last_trading_day(trade_date - timedelta(days=1))
    with np.load(latest_runtime_npz_path(), allow_pickle=False) as runtime:
        dates = runtime['trade_dates'].astype('datetime64[D]')
    if np.datetime64(previous) not in dates:
        raise RuntimeError(f'盘前数据未就绪: Runtime 缺少上一交易日 {previous}；请完成盘后更新')


if __name__ == '__main__':
    from trading.planner_watchdog import cli
    cli()
