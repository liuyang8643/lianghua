"""回测与实盘共用的多退少补调仓计划。"""
import numpy as np

from core.fees import (
    BUY_FEE_RATE,  # Backward-compatible public re-export.
    SELL_FEE_RATE,  # Backward-compatible public re-export.
    SIM_SLIPPAGE_BPS,
    slippage_rate_from_bps,
)
from core.sim.account import (
    calculate_buy_total_cost,
    calculate_sell_net_proceeds,
)
from utils.stock.info import (
    board_limit_ratio, floor_buy_shares, is_kcb_stock, limit_up_price,
    min_sell_shares, round_buy_shares,
)

OVER_TARGET_TOLERANCE = 1.01    # cv > target×1.01 才多退
UNDER_TARGET_TOLERANCE = 0.99   # cv < target×0.99 才少补


def freeze_unit_price(code: str, open_price: float, prev_close: float) -> float:
    """市价买单资金冻结单价 = 涨停价（前收×(1+板块涨跌幅)）。

    除权日前收与开盘价不同口径（跳空超板块涨跌幅）→ 用开盘价作冻结基准，
    避免虚高涨停价误判资金不足。
    """
    pc = prev_close
    if pc and pc > 0 and open_price > 0 and abs(open_price - pc) / pc > board_limit_ratio(code):
        pc = open_price
    return limit_up_price(code, pc) or open_price


def select_tradable_buys(checker, *, buy_n_stocks, prices, stock_indices,
                         trade_idx, signal_date, buy_n,
                         limit_up_protection=False,
                         final_score_arr=None, valid_stocks=None) -> list[str]:
    """买入合法性闸门 + 一字涨停保护补位，保持 topN 顺序。回测与实盘共用。"""
    valid_buy, valid_idx = [], []
    for s in buy_n_stocks:
        if s in prices:
            valid_buy.append(s)
            valid_idx.append(stock_indices[s])
    tradable: list[str] = []
    if valid_buy:
        ok, _ = checker.check(valid_idx, trade_idx, signal_date, is_buy=True)
        tradable = [s for s, o in zip(valid_buy, ok) if o]

    # 一字涨停保护：被过滤的标的从排名中补位，保持 topN 满额
    if limit_up_protection and len(tradable) < buy_n and final_score_arr is not None:
        held = set(tradable)
        ranked_all = [valid_stocks[i] for i in np.argsort(-final_score_arr)]
        for code in ranked_all:
            if len(tradable) >= buy_n:
                break
            if code in held or code not in prices:
                continue
            ok_single, _ = checker.check([stock_indices[code]], trade_idx, signal_date, is_buy=True)
            if ok_single[0]:
                tradable.append(code)
                held.add(code)
    return tradable


def compute_rebalance_plan(*, positions, sellable_volumes, pos_vals, cash,
                           buy_n_stocks, tradable_buy_stocks, sellable_ok,
                           prices, limit_prices, base_target,
                           keep_stocks=None, rebalance=True,
                           slippage_bps=SIM_SLIPPAGE_BPS,
                           rebalance_band_pct=0.01):
    """多退少补：每只持仓目标 = base_target，超出则卖、不足则补。

    Args:
        positions: {code: 持仓股数>0}
        sellable_volumes: {code: 当前可卖股数}（回测=volume；实盘=can_use_volume）
        pos_vals: {code: 持仓股数×open[T]}
        cash: 起始现金（回测=账户现金；实盘=QMT 可用资金）
        buy_n_stocks: topN 顺序（含已持有标的）
        tradable_buy_stocks: 已过买入合法性闸门的 topN 子集（保持 topN 顺序）
        sellable_ok: 已过卖出合法性闸门的代码集合
        prices: {code: open[T]}
        limit_prices: {code: 冻结单价}（freeze_unit_price 产出；缺失回退 open）
        base_target: 单只目标市值 = total_eq×timing/(buy_n+reserve_L)
        keep_stocks: 保留不卖的名单（sell_m），持仓不动但不一定补到 target
        rebalance: True=多退少补；False=仅替换（只清不在 keep_set 的持仓 +
                   现金均分买入 topN 中未持有的标的）

    Returns:
        (sell_orders, buy_orders, skip_reasons)
        sell_orders: [(code, shares)]，shares=-1 表示全清
        buy_orders: {code: shares}，按买入优先级有序
        skip_reasons: {code: 原因}，topN 内未下买单的原因（已达标/未触发少补/冻结资金不足）
    """
    if not 0 <= float(rebalance_band_pct) < 1:
        raise ValueError('rebalance_band_pct must be in [0, 1)')
    slippage_rate = slippage_rate_from_bps(slippage_bps)
    over_target_tolerance = 1.0 + float(rebalance_band_pct)
    under_target_tolerance = 1.0 - float(rebalance_band_pct)

    buy_n_set = set(buy_n_stocks)
    keep_set = set(keep_stocks if keep_stocks is not None else buy_n_stocks)
    sell_orders: list[tuple[str, int]] = []
    cash_sim = cash

    if rebalance:
        sell_seq = list(buy_n_stocks) + [c for c in positions if c not in buy_n_set]
        for code in sell_seq:
            if code not in positions or code not in prices or code not in sellable_ok:
                continue
            cv = pos_vals[code]
            tgt = base_target if code in buy_n_set else (
                cv if code in keep_set else 0.0
            )
            if cv <= tgt * over_target_tolerance:
                continue
            sellable = int(sellable_volumes[code])
            if sellable <= 0:
                continue
            if tgt == 0:
                sell_orders.append((code, -1))
                cash_sim += calculate_sell_net_proceeds(
                    sellable * prices[code],
                    slippage_rate=slippage_rate,
                )
                continue
            sell_step = 1 if is_kcb_stock(code) else 100
            sv = int((cv - tgt) / prices[code] / sell_step) * sell_step
            sv = min(sv, sellable) // sell_step * sell_step
            if sv < min_sell_shares(code):
                continue
            sell_orders.append((code, sv))
            cash_sim += calculate_sell_net_proceeds(
                sv * prices[code],
                slippage_rate=slippage_rate,
            )
    else:
        for code in positions:
            if code in keep_set or code not in prices or code not in sellable_ok:
                continue
            if int(sellable_volumes[code]) <= 0:
                continue
            sell_orders.append((code, -1))
            cash_sim += calculate_sell_net_proceeds(
                int(sellable_volumes[code]) * prices[code],
                slippage_rate=slippage_rate,
            )

    buy_orders: dict[str, int] = {}
    skip_reasons: dict[str, str] = {}

    def _affordable_buy_shares(code: str, unit_price: float) -> int:
        if cash_sim <= 0 or unit_price <= 0:
            return 0
        low = 0
        high = int(cash_sim / unit_price) + 1
        while low + 1 < high:
            middle = (low + high) // 2
            required_cash = calculate_buy_total_cost(
                middle * unit_price,
                slippage_rate=slippage_rate,
            )
            if required_cash <= cash_sim:
                low = middle
            else:
                high = middle
        return floor_buy_shares(code, low)

    def _try_buy(code: str, bv: int):
        nonlocal cash_sim
        bv = round_buy_shares(code, bv)
        if bv <= 0:
            skip_reasons[code] = '未触发少补'
            return
        unit = limit_prices[code] if code in limit_prices else prices[code]
        budget_price = max(unit, prices[code])
        affordable = _affordable_buy_shares(code, budget_price)
        bv = min(bv, affordable)
        if bv <= 0:
            skip_reasons[code] = '冻结资金不足'
            return
        buy_orders[code] = bv
        cash_sim -= calculate_buy_total_cost(
            bv * prices[code],
            slippage_rate=slippage_rate,
        )

    if rebalance:
        for code in tradable_buy_stocks:
            if code not in prices:
                continue
            cv = pos_vals[code] if code in pos_vals else 0.0
            if cv >= base_target * under_target_tolerance:
                skip_reasons[code] = '已达标'
                continue
            _try_buy(code, int((base_target - cv) / prices[code]))
    else:
        new_codes = [c for c in tradable_buy_stocks
                     if c not in positions and c in prices]
        if new_codes:
            cash_per_new = cash_sim / len(new_codes)
            for code in new_codes:
                _try_buy(code, int(cash_per_new / prices[code]))

    return sell_orders, buy_orders, skip_reasons
