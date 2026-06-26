from core import fees
from core.rebalance import BUY_FEE_RATE, SELL_FEE_RATE, compute_rebalance_plan
from core.sim.account import StockAccountMocker
from utils.stock.info import floor_buy_shares, round_buy_shares


def test_rebalance_fee_rates_use_core_fees():
    assert BUY_FEE_RATE == fees.BUY_FEE_RATE
    assert SELL_FEE_RATE == fees.SELL_FEE_RATE


def test_account_default_fees_use_core_fees():
    account = StockAccountMocker(cash=100_000)
    assert account.commission == fees.COMMISSION_RATE
    assert account.min_commission == fees.MIN_COMMISSION
    assert account.stamp_tax == fees.STAMP_TAX_RATE
    assert account.transfer_fee == fees.TRANSFER_FEE_RATE
    assert account.slippage == fees.SIM_SLIPPAGE_RATE


def test_kcb_buy_lot_allows_300_share_rebalance_plan():
    code = '688420.SH'
    open_price = 22.33
    previous_shares = 1200

    _, buy_orders, skip_reasons = compute_rebalance_plan(
        positions={code: previous_shares},
        sellable_volumes={code: 0},
        pos_vals={code: previous_shares * open_price},
        cash=100_000.0,
        buy_n_stocks=[code],
        tradable_buy_stocks=[code],
        sellable_ok=set(),
        prices={code: open_price},
        limit_prices={code: open_price * 1.2},
        base_target=1500 * open_price,
        rebalance=True,
    )

    assert buy_orders == {code: 300}
    assert code not in skip_reasons


def test_kcb_buy_lot_floor_and_target_rounding():
    code = '688420.SH'
    assert floor_buy_shares(code, 199) == 0
    assert floor_buy_shares(code, 200) == 200
    assert floor_buy_shares(code, 300) == 300
    assert round_buy_shares(code, 1) == 200
    assert round_buy_shares(code, 300) == 300


def test_main_board_buy_lot_stays_round_hundred():
    code = '600000.SH'
    assert floor_buy_shares(code, 99) == 0
    assert floor_buy_shares(code, 100) == 100
    assert floor_buy_shares(code, 150) == 100
    assert floor_buy_shares(code, 300) == 300
    assert round_buy_shares(code, 1) == 100
