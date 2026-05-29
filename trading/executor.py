"""调仓下单执行器。

把 `before_trade` 算好的 `pending_rebalance` 计划落到 QMT 委托。集中承载所有
「真实买卖动作」相关逻辑，后续的撤单重试、分批下单、择时拆单等都挂在这里。

与回测口径对齐的关键约束（防止实盘买不足）：
  - 卖单并发提交。
  - 买单按 topN 顺序串行轮询 QMT 可用资金：回测里卖出回款是「瞬时到账」、买入用更新后
    的现金买满 base_target；而实盘卖单是 T+0 异步成交、回款随成交逐步到账，所以不能在
    卖单尚未成交时就用当下 asset.cash 一次性给买单降量（会严重买不足）。改为边回款边买，
    每只买到计划量后转下一只；卖单仍在途时等待最高优先级标的回款（不跳到低优先级），
    卖单全部终态后再用剩余现金买后续可买标的。
"""
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from xtquant import xtconstant

from data.db import get_stock_detail
from trading.helper import get_order_status_label
from trading.logger import trading_logger
from utils.recorder import recorder
from utils.stock.info import min_buy_shares

TERMINAL_STATUS = {
    xtconstant.ORDER_SUCCEEDED, xtconstant.ORDER_CANCELED,
    xtconstant.ORDER_JUNK, xtconstant.ORDER_PART_CANCEL,
}


class RebalanceExecutor:
    """调仓下单执行器，依赖一个 `Trader` 实例进行实际委托。"""

    SLIPPAGE = 1.01       # 市价单实际成交价高于估算开盘价的缓冲，留 1% 防临界废单
    BUY_TIMEOUT_SEC = 120  # 买入轮询兜底超时，避免卖单挂死导致死循环
    SETTLE_WAIT_SEC = 30   # 等待所有委托进入终态的最长秒数

    def __init__(self, trader):
        self.trader = trader

    def _submit_sell_order(self, code, shares, signal_date, trade_date):
        """提交卖出委托。shares=-1 表示全部清仓。返回 {code, order_type, order_id, shares} 或 None。"""
        remark = f'rebalance signal={signal_date.isoformat()} trade={trade_date.isoformat()}'
        try:
            if shares < 0:
                order_id = self.trader.clear_position(code, reason=remark)
            else:
                order_id = self.trader.order(
                    xtconstant.STOCK_SELL, code, shares, None, order_remark=remark)
            if order_id is None:
                trading_logger.info(f"{code} 无需卖出或委托未发出")
                return None
            trading_logger.info(
                f"已提交卖出委托: {code} {'全仓' if shares < 0 else f'{shares}股'} order_id={order_id}")
            recorder.mark("提交卖出委托")
            return {'code': code, 'order_type': 'SELL', 'order_id': order_id, 'shares': shares}
        except ValueError as e:
            trading_logger.info(f"{code} 卖出前校验拦截: {e}")
        except Exception as e:
            trading_logger.exception(f"{code} 卖出委托失败: {e}")
        return None

    def _submit_buy_order(self, code, shares, signal_date, trade_date):
        """提交买入委托。返回 {code, order_type, order_id, shares} 或 None。"""
        remark = f'rebalance signal={signal_date.isoformat()} trade={trade_date.isoformat()}'
        try:
            order_id = self.trader.order(
                xtconstant.STOCK_BUY, code, shares, None, order_remark=remark)
            trading_logger.info(f"已提交买入委托: {code} * {shares} 股 order_id={order_id}")
            recorder.mark("提交买入委托")
            return {'code': code, 'order_type': 'BUY', 'order_id': order_id, 'shares': shares}
        except ValueError as e:
            trading_logger.info(f"{code} 买入前校验拦截: {e}")
        except Exception as e:
            trading_logger.exception(f"{code} 买入委托失败: {e}")
        return None

    def _sells_all_terminal(self, sell_submitted):
        for s in sell_submitted:
            o = self.trader.query_order(s['order_id'])
            if o and o.order_status not in TERMINAL_STATUS:
                return False
        return True

    def _execute_sells(self, sell_orders, signal_date, trade_date):
        """并发提交所有卖单，返回已提交委托列表。"""
        submitted = []
        with ThreadPoolExecutor(max_workers=min(16, len(sell_orders))) as executor:
            futures = [
                executor.submit(self._submit_sell_order, code, shares, signal_date, trade_date)
                for code, shares in sell_orders
            ]
            for future in as_completed(futures):
                r = future.result()
                if r:
                    submitted.append(r)
        return submitted

    def _order_progress(self, order_id):
        """返回某委托的 (已成交股数, 在途未成股数)。

        废单/已撤/部撤的未成部分视为「释放」(在途计 0)，从而其缺口会被重新计入、自动补单。
        在途(未报/待报/已报/部成/待撤)的未成部分计入「在途」，避免重复下单。
        """
        o = self.trader.query_order(order_id)
        if not o:
            return 0, 0
        if o.order_status in (xtconstant.ORDER_JUNK, xtconstant.ORDER_CANCELED,
                              xtconstant.ORDER_PART_CANCEL, xtconstant.ORDER_SUCCEEDED):
            return o.traded_volume, 0
        return o.traded_volume, o.order_volume - o.traded_volume

    def _execute_buys(self, buy_allocations, buy_n_stocks, prices,
                      sell_submitted, signal_date, trade_date):
        """两阶段买入，目标是让实盘成交逼近回测的多退少补计划量。

        回测只要过合法性检查就一定精确达到目标；实盘受卖单成交延迟/滑点/废单影响会偏离，
        故单独用执行循环把实盘补回到同一计划量(不改变计划本身，不碰回测对齐)：

          阶段1(不拆单)：按 topN 顺序，凡当前现金能一次性覆盖整笔计划量的直接整单打出；
            遇到第一只现金不够的就停(保持 topN 优先级，不越级买低优先级)。
          阶段2(拆单)：while 轮询，按 topN 优先级用已到账现金以「板块最小手整数倍」把每只
            补到计划量；卖单回款逐步到账时等高优先级标的、不越级；废单/未成缺口会被重算并
            重试，直到全部补满 / 卖单终态且无钱可买 / 超时。

        以「我方委托的实际成交+在途量」对账(见 _order_progress)，既不重复下单，也能补回废单。
        """
        submitted = []
        buy_seq = [c for c in buy_n_stocks if c in buy_allocations and prices.get(c, 0) > 0]
        targets = {c: int(buy_allocations[c]) for c in buy_seq}
        orders_by_code = {c: [] for c in buy_seq}

        def commit(code):
            """该 code 已成交 + 在途量。"""
            filled = inflight = 0
            for oid in orders_by_code[code]:
                f, i = self._order_progress(oid)
                filled += f
                inflight += i
            return filled, inflight

        def do_submit(code, shares):
            r = self._submit_buy_order(code, shares, signal_date, trade_date)
            if r:
                orders_by_code[code].append(r['order_id'])
                submitted.append(r)

        # ---- 阶段1：不拆单 ----
        for code in buy_seq:
            rem = targets[code] - sum(commit(code))
            if rem <= 0:
                continue
            asset = self.trader.query_asset()
            cash = float(asset.cash) if asset else 0.0
            if cash >= rem * prices[code] * self.SLIPPAGE:
                do_submit(code, rem)
                time.sleep(0.3)
            else:
                break  # 现金不够，保持 topN 优先级，剩余交给阶段2

        # ---- 阶段2：拆单(最小手整数倍)轮询补满 ----
        deadline = time.time() + self.BUY_TIMEOUT_SEC
        while time.time() < deadline:
            sells_done = self._sells_all_terminal(sell_submitted)
            rem = {}
            inflight_total = 0
            for code in buy_seq:
                filled, inflight = commit(code)
                rem[code] = targets[code] - filled - inflight
                inflight_total += inflight

            if all(v <= 0 for v in rem.values()):
                if inflight_total > 0:
                    time.sleep(0.5)  # 等在途订单落定(可能废单后需重挂)
                    continue
                break

            asset = self.trader.query_asset()
            cash = float(asset.cash) if asset else 0.0
            pick = None
            for code in buy_seq:
                min_lot = min_buy_shares(code)
                if rem[code] < min_lot:
                    continue  # 已满足，或不足一个最小手无法再下单(接受微小缺口)
                afford = int(cash / (prices[code] * self.SLIPPAGE) / 100) * 100
                buy = (min(rem[code], afford) // 100) * 100
                if buy >= min_lot:
                    pick = (code, buy)
                    break
                # 最高优先级标的现金不够：卖单仍在途则等回款，不越级买低优先级标的
                if not sells_done:
                    break

            if pick is None:
                if sells_done and inflight_total == 0:
                    short = [c for c in buy_seq if rem[c] >= min_buy_shares(c)]
                    if short:
                        trading_logger.warning(f"卖单已全部成交，剩余资金无法补满: {short}")
                    break
                time.sleep(0.5)
                continue

            do_submit(*pick)
            time.sleep(0.3)  # 等 QMT 冻结资金、刷新可用现金后再算下一笔
        return submitted

    def _wait_terminal(self, submitted):
        """轮询等待所有委托进入终态。"""
        waited = 0
        pending_orders = []
        while waited < self.SETTLE_WAIT_SEC:
            pending_orders = [
                s['code'] for s in submitted
                if (o := self.trader.query_order(s['order_id'])) and o.order_status not in TERMINAL_STATUS
            ]
            if not pending_orders:
                return
            time.sleep(1)
            waited += 1
        trading_logger.warning(f"等待成交超时, 仍有{len(pending_orders)}笔未终态: {pending_orders[:5]}...")

    def _summarize(self, submitted):
        """汇总成交 / 未完成 / 失败并打印。"""
        ok_list, fail_list, partial_list = [], [], []
        for s in submitted:
            o = self.trader.query_order(s['order_id'])
            status = get_order_status_label(o.order_status) if o else '查询失败'
            msg = o.status_msg if o else ''
            traded = o.traded_volume if o else 0
            price = o.traded_price if o else 0
            vol = o.order_volume if o else 0
            detail = get_stock_detail(s['code'])
            name = (detail.get('InstrumentName', '') if detail else '').strip()
            label = f"{s['code']} {name}".strip()
            line = f"{s['order_type']:4s} {label} {s['shares']}股 → {status}"
            if traded and traded != vol:
                line += f" {traded}/{vol}股"
            if price:
                line += f" @{price:.2f}"
            if status == '已成':
                ok_list.append(line)
            elif status in ('废单', '已撤', '部撤'):
                fail_list.append(f"{line} {msg}")
            else:
                partial_list.append(f"{line} {msg}")

        if ok_list:
            trading_logger.info(f"=== 成交 {len(ok_list)} 笔 ===")
            for l in ok_list:
                trading_logger.info(f"  {l}")
        if partial_list:
            trading_logger.warning(f"=== 未完成 {len(partial_list)} 笔 ===")
            for l in partial_list:
                trading_logger.warning(f"  {l}")
        if fail_list:
            trading_logger.error(f"=== 失败 {len(fail_list)} 笔 ===")
            for l in fail_list:
                trading_logger.error(f"  {l}")

    def execute(self, pending, *, execute_sell=True, execute_buy=True):
        """执行一次调仓：提交卖买委托、等待终态、汇总。

        Args:
            pending: before_trade 产出的 pending_rebalance 字典。
            execute_sell / execute_buy: 是否执行卖 / 买（分时段调仓时使用）。
        """
        signal_date = pending['signal_date']
        trade_date = pending['trade_date']
        sell_orders = pending.get('sell_orders', [])
        buy_allocations = pending.get('buy_allocations', {})
        buy_n_stocks = pending.get('buy_n_stocks', list(buy_allocations.keys()))
        prices = pending.get('prices', {})

        trading_logger.info(
            f"开始调仓: sell={len(sell_orders)} buy={len(buy_allocations)} "
            f"signal={signal_date.isoformat()} trade={trade_date.isoformat()}")

        sell_submitted, buy_submitted = [], []
        if execute_sell and sell_orders:
            sell_submitted = self._execute_sells(sell_orders, signal_date, trade_date)
        if execute_buy and buy_allocations:
            buy_submitted = self._execute_buys(
                buy_allocations, buy_n_stocks, prices, sell_submitted, signal_date, trade_date)

        submitted = sell_submitted + buy_submitted
        self._wait_terminal(submitted)
        self._summarize(submitted)
        return submitted
