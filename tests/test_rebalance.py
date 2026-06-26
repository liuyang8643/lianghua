"""core/rebalance.py 多退少补统一实现的单元测试（纯函数，不依赖 NPZ/QMT）。

覆盖 2026-06-11 盘后 diff 复盘出的全部决策类问题：
1. 科创板部分卖出 <200 股不生成委托（实盘会废单）
2. 序贯涨停价冻结校验：末位标的现金不够冻结 → 不下买单（与回测一致）
3. 多退量恰好等于可卖量时照常卖出（旧实盘代码 sv < can_use 漏卖边界）
4. min_buy_shares：创业板 100 股、科创板 200 股
"""
from core.rebalance import (
    BUY_FEE_RATE, compute_rebalance_plan, freeze_unit_price,
)
from utils.stock.info import min_buy_shares, min_sell_shares


def _plan(**kw):
    defaults = dict(
        positions={}, sellable_volumes={}, pos_vals={}, cash=0.0,
        buy_n_stocks=[], tradable_buy_stocks=[], sellable_ok=set(),
        prices={}, limit_prices={}, base_target=0.0, rebalance=True,
    )
    defaults.update(kw)
    return compute_rebalance_plan(**defaults)


# ── 卖出侧 ───────────────────────────────────────────────

def test_kcb_partial_sell_below_200_is_skipped():
    """回归 2026-06-11:688296/688026 多退 100 股 → QMT「委托数量100必须大于等于
    最小可委托数量200」废单,而回测照常成交 → 持仓 diff。统一实现必须不生成该委托。"""
    code = '688296.SH'
    sells, _, _ = _plan(
        positions={code: 2686}, sellable_volumes={code: 2686},
        pos_vals={code: 2686 * 13.72}, cash=1000.0,
        buy_n_stocks=[code], sellable_ok={code},
        prices={code: 13.72}, base_target=35_000.0,
    )
    # 超出目标 ~1856 元 → 多退 100 股,但科创板部分卖出最低 200 股 → 跳过
    assert sells == []


def test_kcb_partial_sell_at_or_above_200_is_allowed():
    code = '688296.SH'
    # cv=41160, target=38400 → 多退约 201 股,科创板部分卖出 200 股起、1 股递增
    sells, _, _ = _plan(
        positions={code: 3000}, sellable_volumes={code: 3000},
        pos_vals={code: 3000 * 13.72}, cash=0.0,
        buy_n_stocks=[code], sellable_ok={code},
        prices={code: 13.72}, base_target=38_400.0,
    )
    assert sells == [(code, 201)]


def test_kcb_full_clear_not_limited_by_min_200():
    """全清（不在 topN）不受 200 股限制：余额一次性卖出合法。"""
    code = '688026.SH'
    sells, _, _ = _plan(
        positions={code: 100}, sellable_volumes={code: 100},
        pos_vals={code: 100 * 12.99}, cash=0.0,
        buy_n_stocks=[], sellable_ok={code},
        prices={code: 12.99}, base_target=0.0,
    )
    assert sells == [(code, -1)]


def test_partial_sell_capped_to_sellable_rounds_to_lot():
    """多退量超过可卖量时 cap 到可卖量并取整百——部分卖出不可申报零股。"""
    code = '600000.SH'
    # 多退想卖 1000 股,但可卖只有 950(非整百) → 900
    sells, _, _ = _plan(
        positions={code: 2000}, sellable_volumes={code: 950},
        pos_vals={code: 20_000.0}, cash=0.0,
        buy_n_stocks=[code], sellable_ok={code},
        prices={code: 10.0}, base_target=10_000.0,
    )
    assert sells == [(code, 900)]


def test_mainboard_oversell_equal_to_sellable_is_kept():
    """多退量恰好等于可卖量 → 照常卖出（旧实盘代码 `sv < can_use` 漏卖该边界）。"""
    code = '600000.SH'
    # cv=20000, target=10000 → 多退 1000 股(@10) == can_use 1000
    sells, _, _ = _plan(
        positions={code: 2000}, sellable_volumes={code: 1000},
        pos_vals={code: 20_000.0}, cash=0.0,
        buy_n_stocks=[code], sellable_ok={code},
        prices={code: 10.0}, base_target=10_000.0,
    )
    assert sells == [(code, 1000)]


def test_sell_blocked_by_legality_gate():
    code = '600000.SH'
    sells, _, _ = _plan(
        positions={code: 2000}, sellable_volumes={code: 2000},
        pos_vals={code: 20_000.0}, cash=0.0,
        buy_n_stocks=[], sellable_ok=set(),   # 跌停禁卖
        prices={code: 10.0}, base_target=0.0,
    )
    assert sells == []


# ── 买入侧:序贯冻结校验 ─────────────────────────────────

def test_sequential_freeze_check_downsizes_last_stock():
    """复刻 2026-06-12:末位标的按涨停价冻结口径买不起全额 → 减量买入
    （而非整只跳过留 5% 现金闲置）;回测与 plan 同决策。"""
    a, b = '600100.SH', '600200.SH'
    sells, buys, skip = _plan(
        positions={}, sellable_volumes={}, pos_vals={},
        cash=22_000.0,
        buy_n_stocks=[a, b], tradable_buy_stocks=[a, b], sellable_ok=set(),
        prices={a: 10.0, b: 10.0},
        limit_prices={a: 11.0, b: 11.0},
        base_target=10_000.0,
    )
    # a: 1000股,冻结 11000×(1+fee) < 22000 ✓;扣开盘价成本后剩 ~11990
    # b: 1000股,冻结 11000×(1+fee) ≈ 11012 < 11990 ✓ → 都买得起
    assert buys == {a: 1000, b: 1000}

    _, buys2, skip2 = _plan(
        cash=20_000.0,
        buy_n_stocks=[a, b], tradable_buy_stocks=[a, b],
        prices={a: 10.0, b: 10.0}, limit_prices={a: 11.0, b: 11.0},
        base_target=10_000.0,
    )
    # a 扣完剩 ~9990,b 全额冻结需 11012 → 按冻结口径减量到 900 股
    assert buys2 == {a: 1000, b: 900}
    assert b not in skip2

    _, buys3, skip3 = _plan(
        cash=1_000.0,
        buy_n_stocks=[b], tradable_buy_stocks=[b],
        prices={b: 10.0}, limit_prices={b: 11.0},
        base_target=10_000.0,
    )
    # 连一手都冻结不起 → 跳过并记录原因
    assert buys3 == {}
    assert skip3[b] == '冻结资金不足'


def test_buy_skip_reasons_reported():
    a, b = '600100.SH', '600200.SH'
    _, buys, skip = _plan(
        positions={a: 1000}, sellable_volumes={a: 1000},
        pos_vals={a: 10_000.0}, cash=50_000.0,
        buy_n_stocks=[a, b], tradable_buy_stocks=[a, b], sellable_ok={a},
        prices={a: 10.0, b: 10.0}, limit_prices={a: 11.0, b: 11.0},
        base_target=10_000.0,
    )
    assert skip[a] == '已达标'      # cv == target
    assert buys == {b: 1000}


def test_kcb_buy_bumped_to_min_lot_with_freeze_check():
    code = '688528.SH'
    _, buys, skip = _plan(
        cash=3_000.0,
        buy_n_stocks=[code], tradable_buy_stocks=[code],
        prices={code: 10.44}, limit_prices={code: 12.84},
        base_target=1_500.0,   # 100 股 → 上调到科创最小 200 股
    )
    # 冻结 200×12.84×(1+fee) ≈ 2571 < 3000 → 可买
    assert buys == {code: 200}

    _, buys2, skip2 = _plan(
        cash=2_500.0,
        buy_n_stocks=[code], tradable_buy_stocks=[code],
        prices={code: 10.44}, limit_prices={code: 12.84},
        base_target=1_500.0,
    )
    assert buys2 == {}
    assert skip2[code] == '冻结资金不足'


def test_sell_proceeds_fund_buys():
    """卖出回款计入模拟现金后,买单才买得起 —— 多退少补的资金闭环。

    回款 ≈ 40000×(1-费率) ≈ 39939 + 现金 100 → 3400 股按涨停价冻结需 ≈ 37441 ✓;
    若无回款(现金只有 100)则整只买不进。
    """
    old, new = '600300.SH', '600400.SH'
    sells, buys, _ = _plan(
        positions={old: 1000}, sellable_volumes={old: 1000},
        pos_vals={old: 40_000.0}, cash=100.0,
        buy_n_stocks=[new], tradable_buy_stocks=[new], sellable_ok={old},
        prices={old: 40.0, new: 10.0}, limit_prices={new: 11.0},
        base_target=34_000.0,
    )
    assert sells == [(old, -1)]
    assert buys == {new: 3400}

    _, buys2, skip2 = _plan(
        cash=100.0,
        buy_n_stocks=[new], tradable_buy_stocks=[new],
        prices={new: 10.0}, limit_prices={new: 11.0}, base_target=34_000.0,
    )
    assert buys2 == {}
    assert skip2[new] == '冻结资金不足'


def test_freeze_downsize_replays_20260612_meiteng():
    """回归 2026-06-12 实景:卖出回款后剩 ~3.6w,美腾科技目标 1500 股(3.4w 按开盘价
    买得起),但科创板涨停价冻结口径需 ~4.07w → 旧逻辑整只跳过、3.5w 现金闲置一天。
    减量后按科创板 200 股起、1 股递增规则买入 1326 股。"""
    code = '688420.SH'
    open_px, pre_close = 22.63, 22.59
    unit = freeze_unit_price(code, open_px, pre_close)   # 22.59×1.2 = 27.108
    _, buys, skip = _plan(
        cash=35_985.0,
        buy_n_stocks=[code], tradable_buy_stocks=[code],
        prices={code: open_px}, limit_prices={code: unit},
        base_target=34_000.0,
    )
    assert buys == {code: 1326}
    assert code not in skip


# ── 仅替换模式 ───────────────────────────────────────────

def test_replace_mode_clears_non_topn_and_splits_cash():
    old, a, b = '600500.SH', '600600.SH', '600700.SH'
    sells, buys, _ = _plan(
        positions={old: 1000, a: 500}, sellable_volumes={old: 1000, a: 500},
        pos_vals={old: 10_000.0, a: 5_000.0}, cash=0.0,
        buy_n_stocks=[a, b], tradable_buy_stocks=[a, b], sellable_ok={old, a},
        prices={old: 10.0, a: 10.0, b: 10.0}, limit_prices={b: 11.0},
        base_target=7_500.0, rebalance=False,
    )
    assert sells == [(old, -1)]      # 只清不在 topN 的;topN 内已持有的不动
    assert a not in buys             # 已持有 → 不补
    assert buys[b] > 0               # 回款均分买入新标的


# ── 基础规则 ─────────────────────────────────────────────

def test_min_lot_rules():
    assert min_buy_shares('688528.SH') == 200   # 科创板 200 股起
    assert min_buy_shares('300876.SZ') == 100   # 创业板 100 股(修正:此前误为 200)
    assert min_buy_shares('600000.SH') == 100
    assert min_sell_shares('688296.SH') == 200  # 科创板部分卖出 ≥200
    assert min_sell_shares('300876.SZ') == 100


def test_freeze_unit_price_normal_and_exdiv():
    # 常规:前收×(1+板块幅)
    assert abs(freeze_unit_price('600000.SH', 10.0, 10.0) - 11.0) < 1e-9
    assert abs(freeze_unit_price('688528.SH', 10.44, 10.70) - 10.70 * 1.2) < 1e-9
    # 除权日:开盘 5.0 vs 前收 10.0 跳空超板块幅 → 用开盘价作冻结基准
    assert abs(freeze_unit_price('600000.SH', 5.0, 10.0) - 5.5) < 1e-9
    # 缺前收 → 回退开盘价
    assert freeze_unit_price('600000.SH', 10.0, 0.0) == 10.0


def test_buy_fee_rate_matches_account_mocker():
    from core.sim.account import StockAccountMocker
    acc = StockAccountMocker(cash=0)
    assert abs(BUY_FEE_RATE - (acc.commission + acc.transfer_fee + acc.slippage)) < 1e-12
