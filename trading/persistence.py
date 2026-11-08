"""实盘成交持久化 — parquet 存储，AL-5 Diff 模块的数据基础。

目录: data/live_trades/
  plan_{date}.parquet      盘前调仓计划（候选股+订单计划+合法性状态）
  fills_{date}.parquet     逐笔成交（含 est_price / slippage_pct）
  positions_{date}.parquet 日终持仓快照（含 daily_pnl / daily_return_pct）
  daily_summary.parquet    累计日终摘要（追加）
  cash_flows.parquet       出入金记录（追加）
"""
import os
import threading
import time
import uuid
from collections import Counter
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from filelock import FileLock

from utils.fees import COMMISSION_RATE, MIN_COMMISSION, STAMP_TAX_RATE, TRANSFER_FEE_RATE
from trading.logger import trading_logger

_TRADE_DIR = Path(__file__).resolve().parents[1] / "data" / "live_trades"

# 进程内 RLock 配合每个目标文件的 FileLock，序列化全部
# 「读全量→concat→原子替换」事务；多个 trading 进程也不能并发覆盖同一文件。
_WRITE_LOCK = threading.RLock()


@contextmanager
def _parquet_append_lock(path: Path):
    """串行化 parquet 的读改写事务，覆盖同进程线程和多个 trading 进程。"""
    lock_path = path.with_suffix(path.suffix + ".lock")
    with _WRITE_LOCK:
        with FileLock(str(lock_path), timeout=30):
            yield


def _atomic_write_parquet(df: pd.DataFrame, path: Path):
    """原子写 parquet:先写唯一临时文件再 os.replace,避免写一半 / 并发写导致文件损坏。

    tmp 名带 pid+uuid:即便多线程/多进程同时写同一目标文件,各自写各自的 tmp,
    不会互相写花;os.replace 本身原子,最终落地的一定是某个完整文件。
    """
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}.{uuid.uuid4().hex[:8]}")
    try:
        df.to_parquet(tmp, index=False)
        # Windows 上若另一个进程刚释放 parquet 句柄，replace 可能短暂返回
        # WinError 5。跨进程锁负责写者互斥，这里只等待系统释放文件句柄。
        for attempt in range(20):
            try:
                os.replace(tmp, path)
                break
            except PermissionError:
                if attempt == 19:
                    raise
                time.sleep(0.05)
    finally:
        if tmp.exists():
            try:
                os.remove(tmp)
            except OSError:
                pass


def _safe_read_parquet(path: Path):
    """读 parquet;若文件损坏则隔离为 .corrupt 并返回 None(当作不存在),避免坏文件永久卡死追加。"""
    try:
        return pd.read_parquet(path)
    except Exception as e:
        corrupt = path.with_suffix(path.suffix + ".corrupt")
        try:
            os.replace(path, corrupt)
        except Exception as e2:
            trading_logger.warning(f"[LiveTrade] 损坏文件重命名失败 {path.name}: {e2}")
        trading_logger.error(
            f"[LiveTrade] {path.name} 损坏,已隔离为 {corrupt.name} 并重新开始追加: {e}")
        return None


def _event_dedupe_columns(row: dict) -> list[str]:
    """返回事件的稳定业务键；本地生成的 ts 不能用于回调幂等。"""
    event_type = row['event_type']
    if event_type == EVT_TRADE:
        if row.get('traded_id'):
            return ['event_type', 'traded_id']
        return [
            'event_type', 'order_id', 'code', 'traded_volume',
            'traded_price', 'amount',
        ]
    if event_type == EVT_ORDER:
        return [
            'event_type', 'order_id', 'code', 'order_status',
            'order_volume', 'traded_volume', 'price', 'traded_price',
            'status_msg',
        ]
    if event_type in (EVT_ORDER_ERROR, EVT_CANCEL_ERROR):
        return ['event_type', 'order_id', 'code', 'status_msg']
    return ['ts', 'event_type', 'order_id', 'traded_volume', 'price']


def _coarse_key(order_id, price, shares) -> str:
    return f"{int(order_id)}|{round(float(price), 4)}|{int(shares)}"


def _build_existing_fill_index(df_old: pd.DataFrame):
    """从既有 fills 建去重索引:返回 (traded_id 集合, 粗键计数 Counter)。

    粗键 = (order_id, price, shares);用计数(而非集合)是为了「按重数消费」——
    既能识别从 events 重建出来、traded_id 为空的行,又不会把多笔等量同价成交误判为重复。
    """
    tids: set[str] = set()
    coarse: Counter = Counter()
    if df_old is None or df_old.empty:
        return tids, coarse
    has_tid = 'traded_id' in df_old.columns
    for _, r in df_old.iterrows():
        tid = str(r['traded_id']) if has_tid and pd.notna(r.get('traded_id')) and str(r['traded_id']) else ''
        if tid:
            tids.add(tid)
        coarse[_coarse_key(r['order_id'], r['price'], r['shares'])] += 1
    return tids, coarse


def _consume_existing_fill(tid, order_id, price, shares, tids: set, coarse: Counter) -> bool:
    """判断这笔成交是否已在既有 fills 中:traded_id 命中,或消费掉一个粗键存量。返回是否重复。"""
    if tid and tid in tids:
        return True
    key = _coarse_key(order_id, price, shares)
    if coarse.get(key, 0) > 0:
        coarse[key] -= 1
        return True
    return False

FILL_COLS = ['date', 'code', 'name', 'direction', 'price', 'shares', 'amount',
             'fee_est', 'order_id', 'traded_id', 'fill_time', 'est_price', 'slippage_pct']

PLAN_COLS = ['date', 'code', 'name', 'direction', 'est_price', 'est_volume',
             'est_amount', 'factor_score', 'limit_status', 'reason', 'plan_seq']

# 统一事件流：覆盖 4 种回调（order / trade / order_error / cancel_error）。
# 任何 QMT 推送或主动查询补的成交都必须先落 events_{T}.parquet，作为原始事实来源。
# fills_{T}.parquet 是 events 中 type='trade' 的派生 view（向后兼容 PostCloseReport / dim3 / dim5）。
EVENT_COLS = [
    'date', 'ts', 'event_type', 'source',
    'order_id', 'traded_id', 'code', 'order_type', 'direction',
    'order_status', 'order_volume', 'traded_volume',
    'price', 'traded_price', 'amount',
    'status_msg', 'name',
]

# 事件类型常量
EVT_ORDER = 'order'
EVT_TRADE = 'trade'
EVT_ORDER_ERROR = 'order_error'
EVT_CANCEL_ERROR = 'cancel_error'
EVT_EXECUTION_ACTION = 'execution_action'

# 事件来源
SRC_CALLBACK = 'watcher_callback'   # watcher 的 QMT 推送回调
SRC_QMT_BACKFILL = 'qmt_backfill'   # post_close 调 query_stock_trades 补的
SRC_MANUAL = 'manual'               # 手动脚本（一次性数据修复等）
SRC_EXECUTOR = 'executor_monitor'    # executor monitor 的提交/撤单/熔断/资金检查等本地动作

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


def _parse_qmt_date(v) -> date | None:
    """解析 QMT 流水里的日期字段（'YYYYMMDD' / 'YYYY-MM-DD' / date / datetime）。"""
    if v is None:
        return None
    if isinstance(v, datetime):
        return v.date()
    if isinstance(v, date):
        return v
    s = str(v).strip()
    if not s:
        return None
    for fmt in ('%Y%m%d', '%Y-%m-%d'):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    return None


class LiveTradeRecorder:
    def __init__(self):
        _TRADE_DIR.mkdir(parents=True, exist_ok=True)
        self._today_fills: list[dict] = []
        # 盘前 plan 的预估价缓存，watcher 在成交回调中读它算 slippage
        self._today_plan_prices: dict[str, float] = {}

    def _upsert_today_fill(self, fill: dict):
        """按成交号（旧数据退回粗键）更新内存成交，避免回调重传重复计盈亏。"""
        traded_id = str(fill.get('traded_id', '') or '')
        coarse = _coarse_key(fill['order_id'], fill['price'], fill['shares'])
        with _WRITE_LOCK:
            for idx, old in enumerate(self._today_fills):
                if old.get('date') != fill.get('date'):
                    continue
                old_id = str(old.get('traded_id', '') or '')
                same = (
                    old_id == traded_id if traded_id and old_id
                    else _coarse_key(old['order_id'], old['price'], old['shares']) == coarse
                )
                if same:
                    self._today_fills[idx] = fill
                    return
            self._today_fills.append(fill)

    def get_plan_est_price(self, code: str) -> float | None:
        """供 watcher 在成交回调里查 est_price（已记录 plan 后才有值）。"""
        v = self._today_plan_prices.get(code)
        return float(v) if v and v > 0 else None

    def record_cash_flow(self, amount: float, flow_type: str = 'deposit',
                         note: str = '', trade_date: date | None = None, reference: str | None = None):
        """记录出入金: amount>0=入金, amount<0=出金。

        Args:
            trade_date: 流水归属日，默认今天。补登历史日请显式传入。
        """
        target = trade_date or date.today()
        if reference:
            note = f'reference={reference} ' + note
        row = {'date': target, 'amount': amount, 'type': flow_type, 'note': note}
        path = _TRADE_DIR / "cash_flows.parquet"
        df_old = _load_cash_flows()
        if reference and not df_old.empty and df_old['note'].str.contains(
                'reference=' + reference + ' ', regex=False, na=False).any():
            return
        df_new = pd.DataFrame([row])
        df = df_new if df_old.empty else pd.concat([df_old, df_new], ignore_index=True)
        df.to_parquet(path, index=False)
        trading_logger.debug(
            f"[LiveTrade] 出入金 {target}: {flow_type} ¥{amount:+,.0f} {note}")

    def get_today_cash_flows(self, trade_date: date | None = None) -> float:
        """返回指定交易日净入金（入金-出金）。默认 date.today()。"""
        target = trade_date or date.today()
        df = _load_cash_flows()
        if df.empty:
            return 0.0
        today_rows = df[df['date'] == target]
        return float(today_rows['amount'].sum()) if len(today_rows) > 0 else 0.0

    def sync_bank_transfers_from_qmt(self, trader, trade_date: date | None = None,
                                     lookback_days: int = 5) -> int:
        """从 QMT 同步银证流水到 cash_flows.parquet（回看窗口 + 去重）。

        xtquant `BankTransferStream` 字段：
          - success: bool          是否成功（False = 占位 / 查询为空，跳过）
          - balance: float         转账金额
          - transfer_direction:    "1" 入金 / "2" 出金 (字符串)
          - transfer_no: str       流水号（唯一标识，去重 key）
          - date / time / bank_name / remark / ...

        为什么要回看窗口：入金可能发生在 T 日 15:00 盘后同步之后（当天查不到），
        若只查单日，次日盘后只会查次日，这笔流水将永久漏记。改为查
        [trade_date - lookback_days, trade_date]，并**按流水自身日期记账**，
        使迟到流水能在后续任一盘后被补登到其真实日期。

        去重双保险：
          1. transfer_no（note 内嵌）—— 主键；
          2. 无 transfer_no 时用 (日期, 金额) 兜底，避免回看窗口重复入库。

        Args:
            trader: 已确认的 FileAccount / 客户端账户适配器
            trade_date: 目标日（窗口右端），默认 date.today()
            lookback_days: 回看天数，默认 5（覆盖周末 + 迟到流水）

        Returns: 新增条数（已存在的会去重跳过）。
        """
        from datetime import timedelta
        target = trade_date or date.today()
        start = target - timedelta(days=max(0, lookback_days))
        streams = trader.query_bank_transfers(start.strftime('%Y%m%d'),
                                              target.strftime('%Y%m%d'))
        if not streams:
            return 0

        df_old = _load_cash_flows()
        existing_nos: set[str] = set()          # 去重1: transfer_no
        existing_fallback: set[tuple] = set()   # 去重2: (date_iso, amount) for qmt_sync
        if not df_old.empty:
            for _, r in df_old.iterrows():
                note = str(r.get('note', '') or '')
                if 'transfer_no=' in note:
                    no = note.split('transfer_no=')[1].split(' ')[0]
                    if no:
                        existing_nos.add(no)
                if str(r.get('type', '')) == 'qmt_sync':
                    d = r.get('date')
                    d_iso = d.isoformat() if hasattr(d, 'isoformat') else str(d)
                    try:
                        existing_fallback.add((d_iso, round(float(r.get('amount', 0)), 2)))
                    except (TypeError, ValueError):
                        pass

        new_rows = []
        matched_user_rows = set()
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

            # 按流水自身日期记账（不是运行日），迟到流水才能回填到正确日期
            transfer_date = _parse_qmt_date(getattr(s, 'date', None)) or target
            no = str(getattr(s, 'transfer_no', '') or '')
            fb_key = (transfer_date.isoformat(), round(signed, 2))
            if no and no in existing_nos:
                continue
            if not no and fb_key in existing_fallback:
                continue

            tm = str(getattr(s, 'time', '') or '')
            bank = str(getattr(s, 'bank_name', '') or '')
            remark = str(getattr(s, 'remark', '') or '')
            note = f"QMT流水 transfer_no={no} dir={direction} t={tm} bank={bank} remark={remark}"

            # Upgrade one explicitly user-confirmed transfer to broker evidence,
            # rather than counting the same withdrawal again on a later poll.
            matches = df_old[(df_old['date'] == transfer_date)
                & (df_old['amount'].round(2) == round(signed, 2))
                & (df_old['type'] == 'user_confirmed')].index if not df_old.empty else []
            matches = [i for i in matches if i not in matched_user_rows]
            if matches:
                index = matches[0]
                df_old.loc[index, 'note'] = str(df_old.loc[index, 'note']) + ' | ' + note
                df_old.loc[index, 'type'] = 'qmt_sync'
                matched_user_rows.add(index)
                if no:
                    existing_nos.add(no)
                existing_fallback.add(fb_key)
                continue

            new_rows.append({
                'date': transfer_date, 'amount': signed,
                'type': 'qmt_sync', 'note': note,
            })
            # 同批次内去重，避免一次查询返回重复流水
            if no:
                existing_nos.add(no)
            existing_fallback.add(fb_key)

        if new_rows or matched_user_rows:
            path = _TRADE_DIR / "cash_flows.parquet"
            df_new = pd.DataFrame(new_rows)
            df = df_old if not new_rows else df_new if df_old.empty else pd.concat([df_old, df_new], ignore_index=True)
            df.to_parquet(path, index=False)
            trading_logger.info(
                f"[LiveTrade] QMT 银证流水同步: 新增 {len(new_rows)} 条 "
                f"(窗口 {start.isoformat()}~{target.isoformat()}, "
                f"QMT 共返 {len(streams)} 条, 有效 {valid_streams} 条)"
            )
        else:
            trading_logger.info(
                f"[LiveTrade] QMT 银证流水无新增 "
                f"(窗口 {start.isoformat()}~{target.isoformat()}, "
                f"QMT 共返 {len(streams)} 条, 有效 {valid_streams} 条)"
            )
        return len(new_rows)

    # ── 统一事件流落地 ─────────────────────────────────────────
    # 设计要点：所有 QMT 回调（order/trade/order_error/cancel_error）和主动查询
    # 补回来的成交（qmt_backfill）必须先经此接口落 events_{T}.parquet。
    # `record_fill` 退化为 record_event(type='trade') 的内部派生路径，保持向后兼容。

    def record_event(self, event_type: str, *, source: str = SRC_CALLBACK,
                     trade_date: date | None = None, event_time: datetime | None = None,
                     **payload) -> dict:
        """统一事件入口。任何 watcher 回调都应通过这里落盘。

        Args (payload 中可能包含的字段)：
            order_id, code, order_type, direction,
            order_status, order_volume, traded_volume,
            price, traded_price, amount, status_msg, name,
            est_price (仅 trade 派生 fill 时用), fee_est (同上)

        Returns: 已写入的 event 行（dict）。
        """
        now = event_time or datetime.now()
        target_date = trade_date or now.date()
        row = {
            'date': target_date, 'ts': now,
            'event_type': event_type, 'source': source,
            'order_id': int(payload.get('order_id', 0) or 0),
            'traded_id': str(payload.get('traded_id', '') or ''),
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
        trading_logger.info(
            f"[LiveTradeEvent] type={row['event_type']} source={row['source']} "
            f"order_id={row['order_id']} code={row['code']} order_type={row['order_type']} "
            f"status={row['order_status']} order_vol={row['order_volume']} "
            f"traded_vol={row['traded_volume']} price={row['price']:.4f} "
            f"traded_price={row['traded_price']:.4f} amount={row['amount']:.2f} "
            f"msg={row['status_msg']}"
        )

        # 派生 fill: trade 事件同步更新 fills_{T}.parquet
        if event_type == EVT_TRADE:
            est_price = payload.get('est_price')
            if est_price is None and target_date == date.today():
                est_price = self.get_plan_est_price(row['code'])
            slippage_pct = None
            if est_price is not None and est_price > 0 and row['traded_price'] > 0:
                slippage_pct = round((row['traded_price'] - est_price) / est_price * 100, 4)
            fee = payload.get('fee_est')
            if fee is None:
                amt = row['amount']
                fee = max(amt * COMMISSION_RATE, MIN_COMMISSION) + amt * TRANSFER_FEE_RATE
                if row['direction'] == 'sell':
                    fee += amt * STAMP_TAX_RATE
            fill = {
                'date': target_date, 'code': row['code'], 'name': row['name'],
                'direction': row['direction'],
                'price': round(row['traded_price'], 4),
                'shares': row['traded_volume'],
                'amount': round(row['amount'], 2),
                'fee_est': round(float(fee), 4),
                'order_id': row['order_id'],
                'traded_id': row['traded_id'],
                'fill_time': now,
                'est_price': round(float(est_price), 4) if est_price else None,
                'slippage_pct': slippage_pct,
            }
            self._upsert_today_fill(fill)
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
        # 链路断裂检测：T-1 快照缺失但存在更早快照 → 卖出无成本基线、持仓真伪不可辨，
        # 此时不能算 per-stock daily_pnl（否则把卖出金额误当利润）→ 全部标 None。
        # 无任何更早快照（真正首日）则正常计算（全是当日新开仓）。
        chain_broken = False
        if not yesterday_path.exists():
            for p in _TRADE_DIR.glob("positions_*.parquet"):
                try:
                    d = date.fromisoformat(p.stem.split('positions_')[1])
                except (ValueError, IndexError):
                    continue
                if d < target_date:
                    chain_broken = True
                    break
            if chain_broken:
                trading_logger.warning(
                    f"[LiveTrade] T-1 快照 {yesterday_path.name} 缺失但存在更早快照 → "
                    f"链路断裂，当日 per-stock P&L 标记为不可计算(None)")
        y_map: dict[str, dict] = {}
        if yesterday_path.exists():
            ydf = pd.read_parquet(yesterday_path)
            for _, r in ydf.iterrows():
                if r['volume'] > 0:
                    y_map[r['code']] = {
                        'volume': int(r['volume']),
                        'last_price': float(r['last_price']),
                    }
                    if 'name' in r and isinstance(r['name'], str) and r['name'].strip():
                        name_map.setdefault(r['code'], r['name'].strip())

        # 4. 用券商市值、昨日市值和真实成交现金流计算，数量不守恒时标未知。
        from data.db.stock_name import get_stock_name_at_date
        rows = []
        for p in positions:
            code = p.stock_code
            vol_t = int(p.volume)
            lp_t = _position_last_price(p)
            mv_t = float(p.market_value) if p.market_value else lp_t * vol_t

            agg = fill_agg.get(code, {})
            bought = agg.get('bought', 0)
            sold = agg.get('sold', 0)
            buy_amt = agg.get('buy_amount', 0.0)
            sell_amt = agg.get('sell_amount', 0.0)
            fee = agg.get('fee', 0.0)

            y = y_map.get(code) or {}
            y_vol = y.get('volume', 0)
            y_lp = y.get('last_price', 0.0)
            y_mv = y_lp * y_vol

            # 成交金额是现金流事实；券商成本价可能含费用、已实现损益或人工调整，
            # 不能据此反推买卖金额。数量不守恒时也禁止凭收盘价补造成交。
            balanced = vol_t == y_vol + bought - sold
            if chain_broken or not balanced:
                daily_pnl = daily_ret = None
            else:
                daily_pnl = mv_t - y_mv + sell_amt - buy_amt - fee
                base = y_mv if y_mv > 0 else buy_amt
                daily_ret = daily_pnl / base * 100 if base > 0 else None

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
        sold_to_zero = (set(y_map) | set(fill_agg)) - today_codes
        for code in sold_to_zero:
            y = y_map.get(code, {'last_price': 0.0, 'volume': 0})
            y_mv = y['last_price'] * y['volume']
            agg = fill_agg.get(code, {})
            sell_amt = agg.get('sell_amount', 0.0)
            buy_amt = agg.get('buy_amount', 0.0)
            fee = agg.get('fee', 0.0)
            sold = agg.get('sold', 0)
            bought = agg.get('bought', 0)
            # 公式：T_mv=0（已清仓），daily_pnl = -Y_mv + sell_amt - buy_amt - fee
            daily_pnl = (-y_mv + sell_amt - buy_amt - fee
                         if not chain_broken and y['volume'] + bought == sold else None)
            daily_ret = (daily_pnl / y_mv * 100) if daily_pnl is not None and y_mv > 0 else None
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
                'daily_pnl': round(daily_pnl, 2) if daily_pnl is not None else None,
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
                            market_value: float, trade_date: date | None = None,
                            per_stock_pnl: float | None = None):
        """写日终摘要。

        Args:
            total_asset / cash / market_value: 来自 query_asset() 的当日终值
            trade_date: 默认 date.today()；--skip 模拟模式下应传入模拟日期
            per_stock_pnl: 个股日 P&L 总和（报告口径）。给定时 daily_pnl 以它为准
                （免疫未记账的银证出入金）；为 None 时回退账户口径
                (total_asset - prev_asset - net_cash_flow)。账户口径始终另存 account_pnl。
        """
        target = trade_date or date.today()
        # parquet 是跨进程权威成交源。进程重启后内存 `_today_fills` 为空或只含
        # 重启后的增量，若只看内存会把真实成交数和费用写成 0。
        fills = self.get_today_fills_df(trade_date=target)
        buys = int((fills['direction'] == 'buy').sum()) if not fills.empty else 0
        sells = int((fills['direction'] == 'sell').sum()) if not fills.empty else 0
        fees = float(fills['fee_est'].sum()) if not fills.empty else 0.0
        net_cf = self.get_today_cash_flows(trade_date=target)

        summary_path = _TRADE_DIR / "daily_summary.parquet"
        prev_asset = None
        if summary_path.exists():
            from datetime import timedelta
            from utils.stock.time import get_last_trading_day
            previous_day = get_last_trading_day(target - timedelta(days=1))
            prev = pd.read_parquet(summary_path)
            prev_rows = prev[prev['date'] == previous_day]
            if not prev_rows.empty:
                prev_asset = float(prev_rows['total_asset'].iloc[-1])

        # 账户口径（含未剔除出入金风险）
        account_pnl = None
        if prev_asset and prev_asset > 0:
            account_pnl = total_asset - prev_asset - net_cf
        # daily_pnl 以个股口径为准（免疫未记账出入金），缺失回退账户口径
        daily_pnl = per_stock_pnl if per_stock_pnl is not None else account_pnl
        daily_ret = (daily_pnl / prev_asset * 100) if (daily_pnl is not None and prev_asset and prev_asset > 0) else None

        row = {
            'date': target, 'total_asset': round(total_asset, 2),
            'cash': round(cash, 2), 'market_value': round(market_value, 2),
            'daily_return_pct': round(daily_ret, 4) if daily_ret is not None else None,
            'daily_pnl': round(daily_pnl, 2) if daily_pnl is not None else None,
            'account_pnl': round(account_pnl, 2) if account_pnl is not None else None,
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
        trading_logger.info(f"[LiveTrade] 日终摘要: daily_ret={daily_ret} pnl={daily_pnl} cf={net_cf:+.0f}")

    def backfill_fills_from_qmt(self, trader, trade_date: date | None = None) -> int | None:
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

        Returns: 新增 fill 条数；QMT 查询失败时返回 ``None``。
        """
        from trading import broker_codes as xtconstant
        target = trade_date or date.today()
        try:
            trades = trader.query_all_trades()
        except Exception as e:
            trading_logger.warning(f"[LiveTrade] QMT 当日成交查询失败，跳过回填: {e}")
            return None
        if not trades:
            return 0

        path = _TRADE_DIR / f"fills_{target.isoformat()}.parquet"
        df_old = pd.read_parquet(path) if path.exists() else pd.DataFrame(columns=FILL_COLS)
        # 去重判定:既有行先建索引——traded_id 命中即重复;否则按 (order_id,price,shares)
        # 粗键「按重数消费」(兼容从 events 重建、traded_id 为空的行,且不会把多笔等量同价
        # 成交误判为重复)。修复:旧实现要求"无 tid 才查粗键",导致 QMT 带 tid 的回填撞不上
        # 重建出的空 tid 行 → 整批重复追加、成交翻倍。
        existing_tids, existing_coarse = _build_existing_fill_index(df_old)

        n_new = 0
        for t in trades:
            order_id = int(t.order_id)
            price = round(float(t.traded_price), 4)
            shares = int(t.traded_volume)
            tid = str(getattr(t, 'traded_id', '') or '')
            if _consume_existing_fill(tid, order_id, price, shares, existing_tids, existing_coarse):
                continue
            amount = float(t.traded_amount) if t.traded_amount else price * shares
            direction = 'buy' if int(t.order_type) == xtconstant.STOCK_BUY else 'sell'
            self.record_event(
                EVT_TRADE, source=SRC_QMT_BACKFILL, trade_date=target,
                code=t.stock_code,
                order_type=int(t.order_type),
                direction=direction,
                traded_price=price, traded_volume=shares, amount=amount,
                order_id=order_id, traded_id=tid, name='',
            )
            if tid:
                existing_tids.add(tid)  # 防止同一次 QMT 返回里重复的同 tid 再次入账
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

        parquet 是每笔成交同步落盘的权威源，应优先读取；仅在文件尚不存在时
        回退同日内存数据。这样进程重启后不会只看到重启后的部分成交。
        """
        target = trade_date or date.today()
        path = _TRADE_DIR / f"fills_{target.isoformat()}.parquet"
        if path.exists():
            return pd.read_parquet(path)
        # record_event 同步写 parquet；此分支只用于测试或写入前的瞬时状态。
        if self._today_fills:
            matching = [item for item in self._today_fills if item.get('date') == target]
            if matching:
                return pd.DataFrame(matching, columns=FILL_COLS)
        return pd.DataFrame(columns=FILL_COLS)

    def _append_fill(self, record: dict):
        """追加一行到每日 parquet，按唯一成交号 traded_id 去重。

        历史教训：用 (order_id, price, shares) 做 key，会把「同一委托拆成多笔等量同价成交」
        （如 800 股市价单成交为 4×200@同价）误判成重复而吞掉，导致少记 3/4 的量。
        QMT `XtTrade.traded_id` 是每笔成交的唯一编号，是正确的去重键。
        traded_id 缺失（旧数据 / 兜底）时退回 (order_id, price, shares) 粗键。
        """
        target = record.get('date') or date.today()
        path = _TRADE_DIR / f"fills_{target.isoformat()}.parquet"
        df_new = pd.DataFrame([record], columns=FILL_COLS)
        with _parquet_append_lock(path):
            df_old = _safe_read_parquet(path) if path.exists() else None
            if df_old is not None and not df_old.empty:
                tid = str(record.get('traded_id', '') or '')
                if tid and 'traded_id' in df_old.columns:
                    df_old = df_old[df_old['traded_id'].astype(str) != tid]
                else:
                    key = ['order_id', 'price', 'shares']
                    mask = ~df_old.set_index(key).index.isin(df_new.set_index(key).index)
                    df_old = df_old[mask]
                df_all = pd.concat([df_old, df_new], ignore_index=True)
            else:
                df_all = df_new
            _atomic_write_parquet(df_all, path)

    def _append_event(self, row: dict):
        """统一事件流追加。

        QMT 回调按稳定业务字段去重；executor action 才使用本地 ts 保留每次动作。
        同一回调可能因网络重传或多个 QMT 客户端被推送多次，必须保证幂等。
        """
        target = row.get('date') or date.today()
        path = _TRADE_DIR / f"events_{target.isoformat()}.parquet"
        df_new = pd.DataFrame([row], columns=EVENT_COLS)
        with _parquet_append_lock(path):
            df_old = _safe_read_parquet(path) if path.exists() else None
            if df_old is not None and not df_old.empty:
                key = _event_dedupe_columns(row)
                try:
                    mask = ~df_old.set_index(key).index.isin(df_new.set_index(key).index)
                    old_filtered = df_old[mask]
                    if old_filtered.empty:
                        df_all = df_new
                    else:
                        # 丢掉新行中全空的列再拼接，避免 pandas 对 all-NA 列的
                        # FutureWarning；最后恢复固定事件 schema。
                        new_values = df_new.dropna(axis=1, how='all')
                        df_all = pd.concat(
                            [old_filtered, new_values], ignore_index=True
                        ).reindex(columns=EVENT_COLS)
                except KeyError as e:
                    trading_logger.warning(
                        f"[LiveTrade] {path.name} 去重key缺失({e}), 回退直接拼接 (旧{len(df_old)}行+新{len(df_new)}行)")
                    df_all = pd.concat([df_old, df_new], ignore_index=True)
            else:
                df_all = df_new
            _atomic_write_parquet(df_all, path)


live_trade_recorder = LiveTradeRecorder()
