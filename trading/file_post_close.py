"""15:35 facts/reconciliation from files only; no broker SDK or calls."""
from datetime import datetime, timedelta
import json
import hashlib
from pathlib import Path
from types import SimpleNamespace

from trading.bridge_wire import atomic_json
from trading.file_account import FileAccount
from trading.target_protocol import broker_fact_date


def reconcile_from_files(trade_date, config_path, directory):
    directory = Path(directory)
    config = json.loads((directory / 'config.json').read_text(encoding='utf-8'))
    account = FileAccount(config['account'], directory, trade_date=trade_date, eod=True)
    facts = account._snapshot()
    if datetime.fromtimestamp(facts['captured_at']).time() < datetime.strptime('15:35', '%H:%M').time():
        raise RuntimeError('A real post-close account snapshot is required')
    # Even a missing/conflicting yesterday baseline must not prevent today's
    # broker facts from reaching the original audit files. The later pass uses
    # this same fixed snapshot and idempotently adds P&L after baseline binding.
    from trading.post_close import _capture_live_close_facts
    captured = _capture_live_close_facts(SimpleNamespace(trader=account), trade_date, skip_account=False)
    if not captured['ok']:
        raise RuntimeError('Post-close broker facts could not be persisted: ' + '; '.join(captured['errors']))
    from trading.persistence import _TRADE_DIR
    journal_path = directory / 'journal' / ('execution_' + trade_date.isoformat() + '.json')
    if not journal_path.exists():
        raise RuntimeError('Original execution journal is missing; no final report can be verified')
    if journal_path.exists():
        journal = json.loads(journal_path.read_text(encoding='utf-8'))
        if journal['account'] != config['account'] or journal['pending']['trade_date'] != trade_date.isoformat():
            raise RuntimeError('Client execution journal account/date mismatch')
        mirror = _TRADE_DIR / journal_path.name
        if mirror.exists():
            previous = json.loads(mirror.read_text(encoding='utf-8'))
            if previous['id'] != journal['id']:
                raise RuntimeError('Another execution journal already exists for this date')
        atomic_json(mirror, journal)
        mirror_frozen_plan(directory, journal, _TRADE_DIR)
    from core.strategy_config import load_strategy_config
    from trading.post_close import run_post_close
    strategy = load_strategy_config(config_path)
    store = SimpleNamespace(trader=account, _now=lambda: datetime.combine(trade_date, datetime.min.time()),
        _skip_update=False, _historical_replay=False, _file_mode=True, _local_reports_only=True,
        _individual_config=strategy['individual_config'], _factor_classes=strategy['factor_classes'])
    def prepare_valuation():
        from trading.close_valuation import value_at_close
        kline_directory = directory.parent / 'k-line'
        mirror_previous_baseline(directory, config['account'], trade_date, _TRADE_DIR,
                                 kline_directory=kline_directory)
        valued, proof = value_at_close(facts, kline_directory)
        proof['source_sha256'] = hashlib.sha256((directory / ('account_eod_' + trade_date.isoformat() + '.json')).read_bytes()).hexdigest()
        atomic_json(directory / 'daily' / ('valuation_' + trade_date.isoformat() + '.json'), proof)
        account._fixed_snapshot = valued
        store._valuation_proof = proof
    store._prepare_close_valuation = prepare_valuation
    result = run_post_close(store)
    if result is not None:
        if not result['execution']['available']:
            raise RuntimeError('Final execution report is unverifiable: ' + result['execution']['reason'])
        report = store._post_close_report
        html = _TRADE_DIR / 'reports' / ('diff_' + trade_date.isoformat() + '.html')
        packet = dict(version=1, account=config['account'], trade_date=trade_date.isoformat(),
            target_id=journal['target_document']['id'], execution_id=journal['id'],
            summary=result['summary'], execution=result['execution'], reconcile=result['reconcile'],
            holdings={p['stock_code']: p['volume'] for p in facts['positions'] if p['volume'] > 0},
            captured_at=facts['captured_at'],
            valuation=store._valuation_proof,
            cash_flow_evidence=json.loads((directory / ('cash_flows_' + trade_date.isoformat() + '.json')).read_text()),
            orders=[o for o in facts['orders'] if broker_fact_date(o['order_time']) == trade_date],
            trades=[vars(t) for t in account.query_all_trades()],
            detail_card=report._send_pnl_slippage_card(result, render_only=True),
            html_path=str(html.resolve()), html_sha256=hashlib.sha256(html.read_bytes()).hexdigest())
        atomic_json(directory / 'daily' / ('close_' + trade_date.isoformat() + '.json'), packet)
    return result


def mirror_previous_baseline(directory, account, trade_date, trade_dir, *, kline_directory=None):
    """Bind legacy report inputs to genuine same-account EOD facts, never P&L guesses."""
    import pandas as pd
    from trading.plan_publisher import previous_baseline
    from trading.persistence import POSITION_COLS, _atomic_write_parquet, _parquet_append_lock
    from utils.stock.time import get_last_trading_day
    previous = get_last_trading_day(trade_date - timedelta(days=1))
    previous_baseline(directory, previous, account)  # Account, date, capture and completeness checks.
    source = directory / ('account_eod_' + previous.isoformat() + '.json')
    facts = json.loads(source.read_text(encoding='utf-8'))
    from trading.close_valuation import value_at_close
    facts, valuation = value_at_close(facts, kline_directory or directory.parent / 'k-line')
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    proof_path = trade_dir / ('baseline_' + previous.isoformat() + '.json')
    rows = []
    for p in facts['positions']:
        if p['volume'] <= 0:
            continue
        rows.append(dict(date=previous, code=p['stock_code'], name=p.get('instrument_name', ''),
            volume=p['volume'], can_use_volume=p['can_use_volume'],
            yesterday_volume=p.get('yesterday_volume'), market_value=p['market_value'],
            avg_price=p['avg_price'], last_price=p['market_value'] / p['volume'],
            open_cost=p.get('open_cost'), float_profit=p.get('float_profit')))
    expected = pd.DataFrame(rows, columns=POSITION_COLS)
    path = trade_dir / ('positions_' + previous.isoformat() + '.parquet')
    with _parquet_append_lock(path):
        if path.exists():
            old = pd.read_parquet(path)
            columns = ['code', 'volume']
            a = old[old.volume > 0][columns].sort_values('code').reset_index(drop=True)
            b = expected[columns].sort_values('code').reset_index(drop=True)
            pd.testing.assert_frame_equal(a, b, check_dtype=False, atol=0.02, rtol=0)
            if proof_path.exists():
                prior_proof = json.loads(proof_path.read_text())
                if prior_proof['account'] != account or prior_proof['source_sha256'] != source_hash:
                    raise ValueError('Previous derived baseline has a conflicting source')
            else:
                # An unbound legacy valuation may be adopted only if it already
                # agrees with the independently dated closing prices.
                cols = ['code', 'market_value', 'last_price']
                pd.testing.assert_frame_equal(old[old.volume > 0][cols].sort_values('code').reset_index(drop=True),
                    expected[cols].sort_values('code').reset_index(drop=True), check_dtype=False, atol=0.02, rtol=0)
            refreshed = old.copy()
            for row in rows:
                mask = refreshed.code == row['code']
                if abs(float(refreshed.loc[mask, 'market_value'].iloc[0]) - row['market_value']) > 0.01:
                    refreshed.loc[mask, ['daily_pnl', 'daily_return_pct']] = None
                refreshed.loc[mask, 'market_value'] = row['market_value']
                refreshed.loc[mask, 'last_price'] = row['last_price']
            _atomic_write_parquet(refreshed, path)
        else:
            _atomic_write_parquet(expected, path)
    summary_path = trade_dir / 'daily_summary.parquet'
    with _parquet_append_lock(summary_path):
        summary = pd.read_parquet(summary_path) if summary_path.exists() else pd.DataFrame()
        asset = facts['asset']
        row = dict(date=previous, total_asset=asset['total_asset'], cash=asset['cash'],
                   market_value=asset['market_value'])
        existing = summary[summary.date == previous] if not summary.empty else summary
        if not existing.empty:
            for key in ('total_asset', 'cash'):
                if abs(float(existing.iloc[-1][key]) - row[key]) > 0.02:
                    raise ValueError('Legacy account baseline conflicts with bound EOD JSON: ' + key)
            summary.loc[summary.date == previous, 'market_value'] = row['market_value']
            _atomic_write_parquet(summary, summary_path)
        else:
            # Unknown past returns, trade counts and fees stay null; no fake empty day.
            merged = pd.concat([summary, pd.DataFrame([row])], ignore_index=True).sort_values('date')
            _atomic_write_parquet(merged, summary_path)
    atomic_json(proof_path, dict(
        account=account, trade_date=previous.isoformat(), captured_at=facts['captured_at'],
        source=str(source.resolve()), source_sha256=source_hash,
        valuation=valuation, asset_component_gap=valuation['asset_component_gap']))


def mirror_frozen_plan(directory, journal, trade_dir):
    import pandas as pd
    from trading.daily_records import ensure_planner_record
    from trading.day_board import file_plan_rows
    from trading.persistence import PLAN_COLS, _atomic_write_parquet, _parquet_append_lock
    day = journal['target_document']['trade_date']
    planner = json.loads(Path(ensure_planner_record(directory, day)).read_text(encoding='utf-8'))
    rows = file_plan_rows(planner)
    for row in rows:
        row['date'] = datetime.fromisoformat(day).date()
    path = trade_dir / ('plan_' + day + '.parquet')
    expected = pd.DataFrame(rows, columns=PLAN_COLS)
    with _parquet_append_lock(path):
        if path.exists():
            old = pd.read_parquet(path)
            cols = ['code', 'direction', 'est_volume', 'est_price']
            pd.testing.assert_frame_equal(old[cols].reset_index(drop=True),
                                          expected[cols], check_dtype=False)
        else:
            _atomic_write_parquet(expected, path)
