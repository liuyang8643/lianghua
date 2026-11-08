from datetime import date

import pandas as pd
import pytest

from trading.execution_reconciliation import reconcile_execution
from trading.report import PostCloseReport


def scenario(bought=100, sold=100, cost=999):
    previous = pd.DataFrame([
        {'code': 'A', 'volume': 100, 'market_value': 900},
        {'code': 'B', 'volume': 100, 'market_value': 2100}])
    current = pd.DataFrame([
        {'code': 'A', 'volume': 100 + bought, 'market_value': (100 + bought) * 12,
         'avg_price': cost, 'daily_pnl': -999, 'daily_return_pct': 0},
        {'code': 'B', 'volume': 100 - sold, 'market_value': (100 - sold) * 19,
         'avg_price': cost, 'daily_pnl': -999, 'daily_return_pct': 0}])
    targets = pd.DataFrame([
        {'code': 'A', 'direction': 'buy', 'est_volume': 100, 'est_price': 10},
        {'code': 'B', 'direction': 'sell', 'est_volume': 100, 'est_price': 20}])
    fills = pd.DataFrame([
        {'code': 'A', 'direction': 'buy', 'shares': bought, 'amount': bought * 10.1,
         'fee_est': 1, 'est_price': 10},
        {'code': 'B', 'direction': 'sell', 'shares': sold, 'amount': sold * 19.9,
         'fee_est': 2, 'est_price': 20}])
    return previous, current, fills, targets, {'A': 12, 'B': 19}


def test_full_execution_only_has_wear_regardless_of_cost_basis():
    result = reconcile_execution(*scenario())
    assert result['all_filled'] and result['within_tolerance']
    assert result['live_total_pnl'] == pytest.approx(377)
    assert result['bt_total_pnl'] == pytest.approx(400)
    assert result['slippage_cost'] == pytest.approx(20)
    assert result['fee_cost'] == 3
    assert result['residual'] == pytest.approx(0)


@pytest.mark.parametrize('bought,sold', [(50, 100), (100, 40), (0, 0), (200, 100)])
def test_partial_and_extra_trades_have_quantity_attribution(bought, sold):
    result = reconcile_execution(*scenario(bought, sold))
    assert not result['all_filled']
    assert result['within_tolerance']
    assert result['quantity_pnl'] == pytest.approx((bought - 100) * 2 + (sold - 100))
    assert result['residual'] == pytest.approx(0)


@pytest.mark.parametrize('cash_change', [-100000, 0, 100000, 123.45])
def test_account_cash_changes_do_not_change_execution_comparison(cash_change, monkeypatch):
    monkeypatch.setattr(PostCloseReport, '_chain_broken', lambda self: False)
    previous, current, fills, targets, closes = scenario()
    report = PostCloseReport(date(2026, 9, 29))
    report.feed_positions_df(current)
    report.feed_fills_df(fills)
    report.feed_asset(10000 + 377 + cash_change, 10000)
    report.feed_execution_reference(previous, targets, closes)
    result = report.build()
    assert result['summary']['live_daily_pnl'] == pytest.approx(377)
    assert result['summary']['bt_daily_pnl'] == pytest.approx(400)
    assert result['reconcile']['diff'] == pytest.approx(-cash_change)
    assert result['execution']['within_tolerance']


def test_stock_transfer_is_not_fabricated_as_fill():
    previous, current, fills, targets, closes = scenario()
    current.loc[0, 'volume'] += 100
    current.loc[0, 'market_value'] += 1200
    result = reconcile_execution(previous, current, fills, targets, closes)
    assert not result['within_tolerance']
    assert not result['available']
    assert result['position_gap'] == 100
    assert 'live_total_pnl' not in result


def test_missing_baseline_or_close_is_unverified():
    previous, current, fills, targets, closes = scenario()
    assert not reconcile_execution(None, current, fills, targets, closes)['available']
    assert not reconcile_execution(previous, current, fills, targets, {})['available']


def test_zero_position_placeholder_does_not_require_price():
    previous, current, fills, targets, closes = scenario()
    previous.loc[len(previous)] = {'code': 'DELISTED', 'volume': 0, 'market_value': 0}
    result = reconcile_execution(previous, current, fills, targets, closes)
    assert result['available'] and result['within_tolerance']


def test_valuation_drift_is_not_hidden_in_wear():
    previous, current, fills, targets, closes = scenario()
    current.loc[0, 'market_value'] += 100
    result = reconcile_execution(previous, current, fills, targets, closes)
    assert result['all_filled'] and not result['within_tolerance']
    assert result['residual'] == pytest.approx(100)


def test_journal_overrides_rewritten_plan_and_rejects_stale_close(tmp_path):
    import json
    from trading.execution_reconciliation import load_execution_reference
    previous, current, fills, targets, closes = scenario()
    previous.to_parquet(tmp_path / 'positions_2026-09-28.parquet')
    current.to_parquet(tmp_path / 'positions_2026-09-29.parquet')
    targets.to_parquet(tmp_path / 'plan_2026-09-29.parquet')
    fills.to_parquet(tmp_path / 'fills_2026-09-29.parquet')
    journal = {'version': 1, 'pending': {'trade_date': '2026-09-29',
               'buy_allocations': {'A': 200}, 'sell_orders': [['B', 100]],
               'prices': {'A': 10, 'B': 20}}}
    (tmp_path / 'execution_2026-09-29.json').write_text(json.dumps(journal))
    kline_dir = tmp_path / 'kline'
    kline_dir.mkdir()
    # 不能将昨日收盘当作今天清仓股票的估值；仍持有的 A 可用券商市值。
    pd.DataFrame({'time': [pd.Timestamp('2026-09-28 15:00')], 'close': [20]}).to_parquet(kline_dir / 'B.parquet')
    y, plan, prices = load_execution_reference(date(2026, 9, 29), tmp_path, kline_dir)
    assert plan.loc[plan['code'] == 'A', 'est_volume'].iloc[0] == 200
    assert prices == {'A': 12}


def test_report_html_card_and_slippage_share_execution_prices(monkeypatch):
    monkeypatch.setattr(PostCloseReport, '_chain_broken', lambda self: False)
    previous, current, fills, targets, closes = scenario()
    fills['est_price'] = 1  # 重启后的错误估价不能覆盖原始执行计划
    report = PostCloseReport(date(2026, 9, 29))
    report.feed_positions_df(current)
    report.feed_fills_df(fills)
    report.feed_execution_reference(previous, targets, closes)
    data = report.build()
    html = report._build_html(data)
    assert '同计划零磨损回放' in html and '未解释残差' in html
    assert data['dim4_slippage']['total_slippage_cost'] == pytest.approx(20)
    assert sum(r['slippage_cost'] for r in report._slippage_agg_rows()) == pytest.approx(20)
    cards = []
    monkeypatch.setattr('trading.report.lark_sender.send_table_card', lambda **kw: cards.append(kw))
    report._send_pnl_slippage_card(data)
    assert any(t['element_id'] == 'execution_tbl' for t in cards[0]['tables'])


def test_unplanned_reverse_trade_uses_same_reference_in_all_views(monkeypatch):
    monkeypatch.setattr(PostCloseReport, '_chain_broken', lambda self: False)
    previous, current, fills, targets, closes = scenario()
    fills.loc[len(fills)] = {'code': 'A', 'direction': 'sell', 'shares': 100,
                            'amount': 1100, 'fee_est': 1, 'est_price': 11}
    current.loc[0, 'volume'] -= 100
    current.loc[0, 'market_value'] -= 1200
    report = PostCloseReport(date(2026, 9, 29))
    report.feed_positions_df(current)
    report.feed_fills_df(fills)
    report.feed_execution_reference(previous, targets, closes)
    result = report.build()
    assert result['execution']['slippage_cost'] == pytest.approx(-80)
    assert result['dim4_slippage']['total_slippage_cost'] == pytest.approx(-80)
    assert sum(r['slippage_cost'] for r in report._slippage_agg_rows()) == pytest.approx(-80)
    assert result['execution']['within_tolerance'] and not result['execution']['all_filled']


def test_all_cash_with_empty_plan_is_verified():
    empty = pd.DataFrame()
    result = reconcile_execution(empty, empty, empty, empty, {})
    assert result['available'] and result['all_filled'] and result['within_tolerance']
    assert result['live_total_pnl'] == result['bt_total_pnl'] == 0


def test_snapshot_uses_fills_not_broker_cost(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from trading import persistence
    monkeypatch.setattr(persistence, '_TRADE_DIR', tmp_path)
    pd.DataFrame([{'code': 'A', 'volume': 100, 'last_price': 9,
                   'avg_price': 8, 'market_value': 900}]).to_parquet(
        tmp_path / 'positions_2026-09-28.parquet')
    pos = SimpleNamespace(stock_code='A', volume=200, last_price=12,
                          market_value=2400, avg_price=999, can_use_volume=100)
    fills = scenario()[2].iloc[:1]
    persistence.LiveTradeRecorder().snapshot_positions([pos], fills, date(2026, 9, 29))
    row = pd.read_parquet(tmp_path / 'positions_2026-09-29.parquet').iloc[0]
    assert row['daily_pnl'] == pytest.approx(489)
