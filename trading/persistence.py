"""实盘成交持久化 — parquet 存储，AL-5 Diff 模块的数据基础。

目录: data/live_trades/
  plan_{date}.parquet      盘前调仓计划（候选股+订单计划+合法性状态）
  fills_{date}.parquet     逐笔成交（含 est_price / slippage_pct）
  positions_{date}.parquet 日终持仓快照（含 daily_pnl / daily_return_pct）
  daily_summary.parquet    累计日终摘要（追加）
  cash_flows.parquet       出入金记录（追加）
"""
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from trading.logger import trading_logger

_TRADE_DIR = Path(__file__).resolve().parents[1] / "data" / "live_trades"

FILL_COLS = ['date', 'code', 'name', 'direction', 'price', 'shares', 'amount',
             'fee_est', 'order_id', 'fill_time', 'est_price', 'slippage_pct']

PLAN_COLS = ['date', 'code', 'name', 'direction', 'est_price', 'est_volume',
             'est_amount', 'factor_score', 'limit_status', 'reason', 'plan_seq']

# 统一事件流：覆盖 4 种回调（order / trade / order_error / cancel_error）。
# 任何 QMT 推送或主动查询补的成交都必须先落 events_{T}.parquet，作为原始事实来源。
# fills_{T}.parquet 是 events 中 type='trade' 的派生 view（向后兼容 PostCloseReport / dim3 / dim5）。
EVENT_COLS = [
    'date', 'ts', 'event_type', 'source',
    'order_id', 'code', 'order_type', 'direction',
    'order_status', 'order_volume', 'traded_volume',
    'price', 'traded_price', 'amount',
    'status_msg', 'name',
]

# 事件类型常量
EVT_ORDER = 'order'
EVT_TRADE = 'trade'
EVT_ORDER_ERROR = 'order_error'
EVT_CANCEL_ERROR = 'cancel_error'

# 事件来源
SRC_CALLBACK = 'watcher_callback'   # watcher 的 QMT 推送回调
SRC_QMT_BACKFILL = 'qmt_backfill'   # post_close 调 query_stock_trades 补的
SRC_MANUAL = 'manual'               # 手动脚本（一次性数据修复等）

POSITION_COLS = [
    'date', 'code', 'name', 'volume', 'can_use_volume', 'yesterday_volume',
    'market_value', 'avg_price', 'last_price', 'open_cost', 'float_profit',
    'bought_today', 'sold_today', 'buy_amount_today', 'sell_amount_today',
    'fee_today', 'daily_pnl', 'daily_return_pct',
]


def _position_last_price(p) -> float:
    """从 XtPosition 取 last_price，缺失时用 market_value/volume 反算。"""
    lp = float(getattr(p, 'last_price', 0) or 0)
    if lp > 0:
        return lp
    vol = int(p.volume)
    return float(p.market_value) / vol if vol > 0 else 0.0


def _load_cash_flows() -> pd.DataFrame:
    path = _TRADE_DIR / "cash_flows.parquet"
    if path.exists():
        return pd.read_parquet(path)
    return pd.DataFrame(columns=['date', 'amount', 'type', 'note'])


class LiveTradeRecorder:
    def __init__(self):
        _TRADE_DIR.mkdir(parents=True, exist_ok=True)
        self._today_fills: list[dict] = []
        # 盘前 plan 的预估价缓存，watcher 在成交回调中读它算 slippage
        self._today_plan_prices: dict[str, float] = {}

    def get_plan_est_price(self, code: str) -> float | None:
        """供 watcher 在成交回调里查 est_price（已记录 plan 后才有值）。"""
        v = self._today_plan_prices.get(code)
        return float(v) if v and v > 0 else None

    def record_cash_flow(self, amount: float, flow_type: str = 'deposit',
                         note: str = ''):
        """记录出入金: amount>0=入金, amount<0=出金。"""
        row = {'date': date.today(), 'amount': amount, 'type': flow_type,
               'note': note}
        path = _TRADE_DIR / "cash_flows.parquet"
        df = pd.concat([_load_cash_flows(), pd.DataFrame([row])], ignore_index=True)
        df.to_parquet(path, index=False)
        trading_logger.info(f"[LiveTrade] 出入金: {flow_type} ¥{amount:+,.0f} {note}")

    def get_today_cash_flows(self, trade_date: date | None = None) -> float:
        """返回指定交易日净入金（入金-出金）。默认 date.today()。"""
        target = trade_date or date.today()
        df = _load_cash_flows()
        if df.empty:
            return 0.0
        today_rows = df[df['date'] == target]
        return float(today_rows['amount'].sum()) if len(today_rows) > 0 else 0.0

    def sync_bank_transfers_from_qmt(self, trader, trade_date: date | None = None) -> int:
        """从 QMT 同步指定日银证流水到 cash_flows.parquet（去重）。

        xtquant `BankTransferStream` 字段：
          - success: bool          是否成功（False = 占位 / 查询为空，跳过）
          - balance: float         转账金额
          - transfer_direction:    "1" 入金 / "2" 出金 (字符串)
          - transfer_no: str       流水号（唯一标识，去重 key）
          - date / time / bank_name / remark / ...

        Args:
            trader: trading.trader.Trader 实例
            trade_date: 目标日，默认 date.today()

        Returns: 新增条数（已存在的会去重跳过）。
        """
        target = trade_date or date.today()
        date_str = target.strftime('%Y%m%d')
        streams = trader.query_bank_transfers(date_str, date_str)
        if not streams:
            return 0

        # 已有记录（QMT 同步出来的）去重 key: (date, transfer_no)
        df_old = _load_cash_flows()
        existing_nos: set[str] = set()
        if not df_old.empty and 'note' in df_old.columns:
            for _, r in df_old.iterrows():
                note = str(r.get('note', '') or '')
                if 'transfer_no=' in note:
                    no = note.split('transfer_no=')[1].split(' ')[0]
                    if no:
                        existing_nos.add(no)

        new_rows = []
        valid_streams = 0
        for s in streams:
            # 跳过占位/失败记录
            if not bool(getattr(s, 'success', False)):
                continue
            valid_streams += 1
            amount = float(getattr(s, 'balance', 0) or 0)
            if amount <= 0:
                continue
            direction = str(getattr(s, 'transfer_direction', '')).strip()
            # 方向编码："1"入金 / "2"出金 / 其他兜底用金额符号
            if direction == '1' or '入' in direction.lower():
                signed = abs(amount)
            elif direction == '2' or '出' in direction.lower():
                signed = -abs(amount)
            else:
                signed = amount  # 默认按正

            no = str(getattr(s, 'transfer_no', '') or '')
            if no and no in existing_nos:
                continue

            tm = str(getattr(s, 'time', '') or '')
            bank = str(getattr(s, 'bank_name', '') or '')
            remark = str(getattr(s, 'remark', '') or '')
            note = f"QMT流水 transfer_no={no} dir={direction} t={tm} bank={bank} remark={remark}"

            new_rows.append({
                'date': target, 'amount': signed,
                'type': 'qmt_sync', 'note': note,
            })

        if new_rows:
            path = _TRADE_DIR / "cash_flows.parquet"
            df = pd.concat([df_old, pd.DataFrame(new_rows)], ignore_index=True)
            df.to_parquet(path, index=False)
            trading_logger.info(
                f"[LiveTrade] QMT 银证流水同步: 新增 {len(new_rows)} 条 "
                f"(QMT 共返 {len(streams)} 条, 有效 {valid_streams} 条)"
            )
        else:
            trading_logger.info(
                f"[LiveTrade] QMT 银证流水无新增 "
                f"(QMT 共返 {len(streams)} 条, 有效 {valid_streams} 条)"
            )
        return len(new_rows)

    # ── 统一事件流落地 ─────────────────────────────────────────
    # 设计要点：所有 QMT 回调（order/trade/order_error/cancel_error）和主动查询
    # 补回来的成交（qmt_backfill）必须先经此接口落 events_{T}.parquet。
    # `record_fill` 退化为 record_event(type='trade') 的内部派生路径，保持向后兼容。

    def record_event(self, event_type: str, *, source: str = SRC_CALLBACK,
                     trade_date: date | None = None, **payload) -> dict:
        """统一事件入口。任何 watcher 回调都应通过这里落盘。

        Args (payload 中可能包含的字段)：
            order_id, code, order_type, direction,
            order_status, order_volume, traded_volume,
            price, traded_price, amount, status_msg, name,
            est_price (仅 trade 派生 fill 时用), fee_est (同上)

        Returns: 已写入的 event 行（dict）。
        """
        now = datetime.now()
        target_date = trade_date or now.date()
        row = {
            'date': target_date, 'ts': now,
            'event_type': event_type, 'source': source,
            'order_id': int(payload.get('order_id', 0) or 0),
            'code': payload.get('code', '') or '',
            'order_type': int(payload['order_type']) if payload.get('order_type') is not None else None,
            'direction': payload.get('direction'),
            'order_status': int(payload['order_status']) if payload.get('order_status') is not None else None,
            'order_volume': int(payload.get('order_volume', 0) or 0),
            'traded_volume': int(payload.get('traded_volume', 0) or 0),
            'price': float(payload.get('price', 0) or 0),
            'traded_price': float(payload.get('traded_price', 0) or 0),
            'amount': float(payload.get('amount', 0) or 0),
            'status_msg': (payload.get('status_msg') or '').strip(),
            'name': (payload.get('name') or '').strip(),
        }
        self._append_event(row)

        # 派生 fill: trade 事件同步更新 fills_{T}.parquet
        if event_type == EVT_TRADE:
            est_price = payload.get('est_price')
            if est_price is None:
                est_price = self.get_plan_est_price(row['code'])
            slippage_pct = None
            if est_price is not None and est_price > 0 and row['traded_price'] > 0:
                slippage_pct = round((row['traded_price'] - est_price) / est_price * 100, 4)
            fee = payload.get('fee_est')
            if fee is None:
                amt = row['amount']
                fee = max(amt * 0.0000854, 0.1) + amt * 0.00002
                if row['direction'] == 'sell':
                    fee += amt * 0.0005
            fill = {
                'date': target_date, 'code': row['code'], 'name': row['name'],
                'direction': row['direction'],
                'price': round(row['traded_price'], 4),
                'shares': row['traded_volume'],
                'amount': round(row['amount'], 2),
                'fee_est': round(float(fee), 4),
                'order_id': row['order_id'],
                'fill_time': now,
                'est_price': round(float(est_price), 4) if est_price else None,
                'slippage_pct': slippage_pct,
            }
            self._today_fills.append(fill)
            self._append_fill(fill)
        return row

    def record_fill(self, code: str, direction: str, price: float,
                    shares: int, amount: float, order_id: int,
                    name: str = '', fee: float = 0.0,
                    est_price: float | None = None):
        """兼容旧入口 —— 内部转 record_event(trade)。"""
        self.record_event(
            EVT_TRADE, source=SRC_CALLBACK,
            code=code, direction=direction,
            traded_price=price, traded_volume=shares, amount=amount,
            order_id=order_id, name=name, fee_est=fee, est_price=est_price,
        )

    def plan_path(self, trade_date: date | None = None) -> Path:
        target_date = trade_date or date.today()
        return _TRADE_DIR / f"plan_{target_date.isoformat()}.parquet"

    def record_plan(self, plan_rows: list[dict], trade_date: date | None = None):
        """落地盘前调仓计划。plan_rows 应是已经组装好、列与 PLAN_COLS 对齐的字典列表。
        同时把 est_price 写入 _today_plan_prices 缓存供 watcher 计算 slippage。
        """
        if not plan_rows:
            trading_logger.info("[LiveTrade] 计划为空，跳过 record_plan")
            return
        target_date = trade_date or date.today()
        path = _TRADE_DIR / f"plan_{target_date.isoformat()}.parquet"
        df = pd.DataFrame(plan_rows, columns=PLAN_COLS)
        df.to_parquet(path, index=False)
        # 刷新 est_price 缓存（每只股票取首次出现的非零 est_price，buy 行优先于 sell）
        self._today_plan_prices = {}
        for row in plan_rows:
            code = row['code']
            ep = row.get('est_price', 0) or 0
            if code not in self._today_plan_prices and ep > 0:
                self._today_plan_prices[code] = float(ep)
        trading_logger.info(
            f"[LiveTrade] 计划落地: {len(df)} 行 → {path.name} "
            f"(est_price 缓存 {len(self._today_plan_prices)} 只)"
        )

    def snapshot_positions(self, positions: list, fills_df: pd.DataFrame | None = None,
                           trade_date: date | None = None):
        """保存日终持仓快照 + 计算个股当日 P&L。

        公式：daily_pnl = (T_lp × T_vol) - (Y_lp × Y_vol) + S_amt - B_amt - fees
              daily_return_pct = daily_pnl / (Y_lp × Y_vol) × 100  （若昨日持仓>0）
                                = daily_pnl / B_amt × 100         （否则按今日开仓金额）

        Args:
            positions: List[XtPosition]
            fills_df: 今日 fills（必须列：code, direction, price, shares, amount, fee_est）
            trade_date: 默认 date.today()
        """
        target_date = trade_date or date.today()

        # 1. 聚合今日 fills 到 per-code 字典 + 收集 fills 中的 name
        fill_agg: dict[str, dict] = {}
        name_map: dict[str, str] = {}
        if fills_df is not None and not fills_df.empty:
            for code, grp in fills_df.groupby('code'):
                buys = grp[grp['direction'] == 'buy']
                sells = grp[grp['direction'] == 'sell']
                fill_agg[code] = {
                    'bought': int(buys['shares'].sum()) if not buys.empty else 0,
                    'sold': int(sells['shares'].sum()) if not sells.empty else 0,
                    'buy_amount': float(buys['amount'].sum()) if not buys.empty else 0.0,
                    'sell_amount': float(sells['amount'].sum()) if not sells.empty else 0.0,
                    'fee': float(grp['fee_est'].sum()),
                }
                if 'name' in grp.columns:
                    for n in grp['name']:
                        if isinstance(n, str) and n.strip():
                            name_map[code] = n.strip()
                            break

        # 2. 从 plan_{T}.parquet 收集 name（更全，含未成交的候选）
        plan_path = _TRADE_DIR / f"plan_{target_date.isoformat()}.parquet"
        if plan_path.exists():
            plan_df = pd.read_parquet(plan_path)
            if not plan_df.empty and 'name' in plan_df.columns:
                for _, row in plan_df.iterrows():
                    if row['code'] in name_map:
                        continue
                    n = row.get('name')
                    if isinstance(n, str) and n.strip():
                        name_map[row['code']] = n.strip()

        # 3. 加载昨日快照（用于计算 daily_pnl）
        from utils.stock.time import get_last_trading_day
        from datetime import timedelta
        yesterday = get_last_trading_day(target_date - timedelta(days=1))
        yesterday_path = _TRADE_DIR / f"positions_{yesterday.isoformat()}.parquet"
        y_map: dict[str, dict] = {}
        if yesterday_path.exists():
            ydf = pd.read_parquet(yesterday_path)
            for _, r in ydf.iterrows():
                if r['volume'] > 0:
                    avg_raw = r['avg_price'] if 'avg_price' in r.index else 0
                    y_avg = float(avg_raw) if pd.notna(avg_raw) else 0.0
                    y_map[r['code']] = {
                        'volume': int(r['volume']),
                        'last_price': float(r['last_price']),
                        'avg_price': y_avg,
                    }
                    if 'name' in r and isinstance(r['name'], str) and r['name'].strip():
                        name_map.setdefault(r['code'], r['name'].strip())

        # 4. 逐持仓组装行 — 用 QMT 真实 (vol, avg_price, last_price) 反推
        # cost basis, 不依赖 fills 完整性。fills 缺记时仍能算准。
        from data.db.stock_name import get_stock_name_at_date
        rows = []
        for p in positions:
            code = p.stock_code
            vol_t = int(p.volume)
            lp_t = _position_last_price(p)
            mv_t = float(p.market_value) if p.market_value else lp_t * vol_t
            t_avg = float(p.avg_price) if vol_t > 0 else 0.0

            agg = fill_agg.get(code, {})
            bought = agg.get('bought', 0)
            sold = agg.get('sold', 0)
            buy_amt = agg.get('buy_amount', 0.0)
            sell_amt = agg.get('sell_amount', 0.0)
            fee = agg.get('fee', 0.0)

            y = y_map.get(code) or {}
            y_vol = y.get('volume', 0)
            y_lp = y.get('last_price', 0.0)
            y_avg = y.get('avg_price', 0.0)
            y_mv = y_lp * y_vol

            # vol 平衡反推 bought_real / sold_real（QMT 持仓为准）：
            # vol_t - y_vol = bought_real - sold_real
            # 假设当日单向（要么纯买要么纯卖；同日双向罕见但下方仍能处理）
            expected_vol = y_vol + bought - sold
            gap = vol_t - expected_vol  # >0 → buy fills 缺，<0 → sell fills 缺
            if gap > 0:
                bought_real, sold_real = bought + gap, sold
            elif gap < 0:
                bought_real, sold_real = bought, sold - gap
            else:
                bought_real, sold_real = bought, sold

            # cost basis 反推（核心，独立于 fills）：
            #   T_cost - Y_cost = buy_amt - y_avg × sold_real
            #   ⇒ buy_amt_real = T_cost - Y_cost + y_avg × sold_real
            t_cost = t_avg * vol_t
            y_cost = y_avg * y_vol
            buy_amt_real = max(0.0, t_cost - y_cost + y_avg * sold_real)

            # sell_amt：优先用 fills 的均价 × 真实 sold_real，兜底用今日 close
            if sold_real > 0:
                if sold > 0 and sell_amt > 0:
                    avg_sell_price = sell_amt / sold
                else:
                    avg_sell_price = lp_t if lp_t > 0 else y_lp
                sell_amt_real = avg_sell_price * sold_real
            else:
                sell_amt_real = 0.0

            # fee 按真实金额比例放大（fills 缺时同步放大）
            total_fills = buy_amt + sell_amt
            total_real = buy_amt_real + sell_amt_real
            if total_fills > 0 and total_real > 0:
                fee_real = fee * (total_real / total_fills)
            else:
                # 兜底估算：买 0.01%, 卖 0.06%
                fee_real = buy_amt_real * 0.0001 + sell_amt_real * 0.0006

            # Guard: 当日无任何 fills + 无昨日快照 + 仍有持仓 → 持仓来源未知（历史继承）
            # cost basis 反推会得到错误的「假当日开仓」，更诚实地标 None
            if vol_t > 0 and y_vol == 0 and buy_amt == 0 and sell_amt == 0:
                daily_pnl = None
                daily_ret = None
            else:
                daily_pnl = (lp_t * vol_t) - y_mv + sell_amt_real - buy_amt_real - fee_real
                if y_mv > 0:
                    daily_ret = daily_pnl / y_mv * 100
                elif buy_amt_real > 0:
                    daily_ret = daily_pnl / buy_amt_real * 100
                else:
                    daily_ret = None

            # name 多层兜底：plan/fills > XtPosition.stock_name > db lookup
            name = name_map.get(code, '')
            if not name:
                name = (getattr(p, 'stock_name', '') or '').strip()
            if not name:
                name = get_stock_name_at_date(code, target_date) or ''

            rows.append({
                'date': target_date, 'code': code,
                'name': name,
                'volume': vol_t,
                'can_use_volume': int(p.can_use_volume),
                'yesterday_volume': int(getattr(p, 'yesterday_volume', 0) or 0),
                'market_value': round(mv_t, 2),
                'avg_price': round(float(p.avg_price), 4),
                'last_price': round(lp_t, 4),
                'open_cost': round(float(getattr(p, 'open_cost', 0) or 0), 2),
                'float_profit': round(float(getattr(p, 'float_profit', 0) or 0), 2),
                'bought_today': bought,
                'sold_today': sold,
                'buy_amount_today': round(buy_amt, 2),
                'sell_amount_today': round(sell_amt, 2),
                'fee_today': round(fee, 4),
                'daily_pnl': round(daily_pnl, 2) if daily_pnl is not None else None,
                'daily_return_pct': round(daily_ret, 4) if daily_ret is not None else None,
            })

        # 5. 加上「昨日持仓但今日已清空（fully sold）」的 zero-volume 行 —
        # 不变量保证：sum(per_stock_daily_pnl) == 账户日变化（减去 cash_flow）。
        # 若漏掉这些股票，sum 就会少算它们的 (sell_amt - Y_mv - fee) 部分。
        today_codes = {p.stock_code for p in positions}
        sold_to_zero = set(y_map.keys()) - today_codes
        for code in sold_to_zero:
            y = y_map[code]
            y_mv = y['last_price'] * y['volume']
            agg = fill_agg.get(code, {})
            sell_amt = agg.get('sell_amount', 0.0)
            buy_amt = agg.get('buy_amount', 0.0)
            fee = agg.get('fee', 0.0)
            sold = agg.get('sold', 0)
            bought = agg.get('bought', 0)
            # 公式：T_mv=0（已清仓），daily_pnl = -Y_mv + sell_amt - buy_amt - fee
            daily_pnl = -y_mv + sell_amt - buy_amt - fee
            daily_ret = (daily_pnl / y_mv * 100) if y_mv > 0 else None
            name = name_map.get(code) or get_stock_name_at_date(code, target_date) or ''
            rows.append({
                'date': target_date, 'code': code, 'name': name,
                'volume': 0, 'can_use_volume': 0,
                'yesterday_volume': y['volume'],
                'market_value': 0.0, 'avg_price': 0.0, 'last_price': 0.0,
                'open_cost': 0.0, 'float_profit': 0.0,
                'bought_today': bought, 'sold_today': sold,
                'buy_amount_today': round(buy_amt, 2),
                'sell_amount_today': round(sell_amt, 2),
                'fee_today': round(fee, 4),
                'daily_pnl': round(daily_pnl, 2),
                'daily_return_pct': round(daily_ret, 4) if daily_ret is not None else None,
            })

        if rows:
            path = _TRADE_DIR / f"positions_{target_date.isoformat()}.parquet"
            pd.DataFrame(rows, columns=POSITION_COLS).to_parquet(path, index=False)
            n_pnl = sum(1 for r in rows if r['daily_pnl'] is not None)
            n_named = sum(1 for r in rows if r['name'])
            n_sold = len(sold_to_zero)
            trading_logger.info(
                f"[LiveTrade] 持仓快照 + 日 P&L: {len(rows)} 只 "
                f"({n_pnl} 只可算 P&L, {n_named} 只有名称, {n_sold} 只昨日持仓已清空) → {path.name}"
            )

    def write_daily_summary(self, total_asset: float, cash: float,
                            market_value: float, trade_date: date | None = None):
        """写日终摘要。

        Args:
            total_asset / cash / market_value: 来自 query_asset() 的当日终值
            trade_date: 默认 date.today()；--skip 模拟模式下应传入模拟日期
        """
        target = trade_date or date.today()
        buys = sum(1 for r in self._today_fills if r['direction'] == 'buy')
        sells = sum(1 for r in self._today_fills if r['direction'] == 'sell')
        fees = sum(r['fee_est'] for r in self._today_fills)
        net_cf = self.get_today_cash_flows(trade_date=target)

        summary_path = _TRADE_DIR / "daily_summary.parquet"
        prev_asset = None
        if summary_path.exists():
            prev = pd.read_parquet(summary_path)
            prev_rows = prev[prev['date'] < target]  # 用 < target 而非 iloc[-1]，避免重跑当日污染
            if not prev_rows.empty:
                prev_asset = float(prev_rows['total_asset'].iloc[-1])

        daily_ret = 0.0
        daily_pnl = 0.0
        if prev_asset and prev_asset > 0:
            daily_pnl = total_asset - prev_asset - net_cf
            daily_ret = daily_pnl / prev_asset * 100

        row = {
            'date': target, 'total_asset': round(total_asset, 2),
            'cash': round(cash, 2), 'market_value': round(market_value, 2),
            'daily_return_pct': round(daily_ret, 4),
            'daily_pnl': round(daily_pnl, 2),
            'net_cash_flow': round(net_cf, 2),
            'total_fees': round(fees, 2), 'buy_count': buys, 'sell_count': sells,
        }

        df_new = pd.DataFrame([row])
        if summary_path.exists():
            df_old = pd.read_parquet(summary_path)
            df_old = df_old[df_old['date'] != target]  # 去重：删除同日旧行
            df_all = pd.concat([df_old, df_new], ignore_index=True)
        else:
            df_all = df_new
        df_all.to_parquet(summary_path, index=False)
        trading_logger.info(f"[LiveTrade] 日终摘要: daily_ret={daily_ret:+.2f}% pnl={daily_pnl:+.0f} cf={net_cf:+.0f}")

    def backfill_fills_from_qmt(self, trader, trade_date: date | None = None) -> int:
        """从 QMT 当日全部成交回填 fills_{T}.parquet，弥补 watcher 漏接的回调。

        QMT `query_stock_trades` 返回的每条 XtTrade 字段：
          - traded_id: str       成交编号（唯一去重 key）
          - order_id: int        委托编号
          - stock_code: str
          - order_type: int      xtconstant.STOCK_BUY / STOCK_SELL
          - traded_volume: int
          - traded_price: float
          - traded_amount: float
          - traded_time: int     unix 时间戳
          - strategy_name: str

        去重策略：按 `traded_id` 做集合差集，已有的不重复写。

        Returns: 新增 fill 条数。
        """
        from xtquant import xtconstant
        target = trade_date or date.today()
        try:
            trades = trader.query_all_trades()
        except Exception as e:
            trading_logger.warning(f"[LiveTrade] QMT 当日成交查询失败，跳过回填: {e}")
            return 0
        if not trades:
            return 0

        path = _TRADE_DIR / f"fills_{target.isoformat()}.parquet"
        df_old = pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=FILL_COLS)
        existing_ids: set[str] = set()
        if not df_old.empty:
            # order_id 不唯一（同一委托可拆多笔成交），用 (order_id, price, shares) 做 key
            for _, r in df_old.iterrows():
                existing_ids.add(f"{r['order_id']}|{r['price']}|{r['shares']}")

        n_new = 0
        for t in trades:
            order_id = int(t.order_id)
            price = round(float(t.traded_price), 4)
            shares = int(t.traded_volume)
            key = f"{order_id}|{price}|{shares}"
            if key in existing_ids:
                continue
            amount = float(t.traded_amount) if t.traded_amount else price * shares
            direction = 'buy' if int(t.order_type) == xtconstant.STOCK_BUY else 'sell'
            self.record_event(
                EVT_TRADE, source=SRC_QMT_BACKFILL, trade_date=target,
                code=t.stock_code,
                order_type=int(t.order_type),
                direction=direction,
                traded_price=price, traded_volume=shares, amount=amount,
                order_id=order_id, name='',
            )
            existing_ids.add(key)
            n_new += 1

        if n_new:
            trading_logger.info(
                f"[LiveTrade] QMT 成交回填: 新增 {n_new} 条 / QMT 共返 {len(trades)} 条 → {path.name}"
            )
        else:
            trading_logger.info(
                f"[LiveTrade] QMT 成交回填: 无新增 (QMT 共返 {len(trades)} 条, fills 已完整)"
            )
        return n_new

    def get_today_fills_df(self, trade_date: date | None = None) -> pd.DataFrame:
        """获取指定日的 fills DataFrame。

        优先返回内存里 `_today_fills`（仅当其日期与 trade_date 一致），
        否则回退读 `fills_{date}.parquet`，缺失返回空 DataFrame。

        这样保证 sim 模式、进程重启、盘后对账等场景都能拿到数据。
        """
        target = trade_date or date.today()
        # 内存数据匹配同一日时优先用（最快，含未刷盘的数据）
        if self._today_fills:
            first_date = self._today_fills[0].get('date')
            if first_date == target:
                return pd.DataFrame(self._today_fills, columns=FILL_COLS)
        # 回退读 parquet
        path = _TRADE_DIR / f"fills_{target.isoformat()}.parquet"
        if path.exists():
            return pd.read_parquet(path)
        return pd.DataFrame(columns=FILL_COLS)

    def _append_fill(self, record: dict):
        """追加一行到每日 parquet，按 (order_id, price, shares) 去重。

        修：原来用 (order_id, code, price) 做 key——同一委托拆 2 笔成交 price 相同时会被吞掉。
        改为 (order_id, price, shares)，覆盖更稳健（一笔成交 = 唯一三元组）。
        """
        target = record.get('date') or date.today()
        path = _TRADE_DIR / f"fills_{target.isoformat()}.parquet"
        df_new = pd.DataFrame([record])
        if path.exists():
            df_old = pd.read_parquet(path)
            key = ['order_id', 'price', 'shares']
            mask = ~df_old.set_index(key).index.isin(df_new.set_index(key).index)
            df_all = pd.concat([df_old[mask], df_new], ignore_index=True)
        else:
            df_all = df_new
        df_all.to_parquet(path, index=False)

    def _append_event(self, row: dict):
        """统一事件流追加。

        按 (ts, event_type, order_id, traded_volume, price) 去重；
        同一回调可能因网络重传/进程重启被推送多次，需保证幂等。
        """
        target = row.get('date') or date.today()
        path = _TRADE_DIR / f"events_{target.isoformat()}.parquet"
        df_new = pd.DataFrame([row], columns=EVENT_COLS)
        if path.exists():
            df_old = pd.read_parquet(path)
            # 去重 key：同一笔事件的所有字段（ts 精确到微秒，自然唯一）
            key = ['ts', 'event_type', 'order_id', 'traded_volume', 'price']
            try:
                mask = ~df_old.set_index(key).index.isin(df_new.set_index(key).index)
                df_all = pd.concat([df_old[mask], df_new], ignore_index=True)
            except KeyError:
                df_all = pd.concat([df_old, df_new], ignore_index=True)
        else:
            df_all = df_new
        df_all.to_parquet(path, index=False)


live_trade_recorder = LiveTradeRecorder()
