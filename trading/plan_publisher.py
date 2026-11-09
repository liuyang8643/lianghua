"""Local strategy -> immutable absolute targets. Never imports/calls QMT.

Account inputs are files exported by the client; market inputs are local
Runtime plus Tencent T-open. Selection and legality use the shared core.
"""
import json
import time
import uuid
import logging
from datetime import datetime, time as daytime, timedelta
from pathlib import Path
from types import SimpleNamespace

from trading.account_snapshot import validate_account_snapshot
from trading.target_protocol import read_document, validate_target, write_immutable_target


def load_account_file(path, account, trade_date, now=None):
    snapshot, _ = read_document(Path(path))
    # Read first, then timestamp: the client atomically refreshes this file.
    now = time.time() if now is None else now() if callable(now) else now
    if (snapshot['version'] != 1 or snapshot['account'] != account
            or snapshot['trade_date'] != trade_date.isoformat()
            or not 0 <= now - snapshot['captured_at'] <= 30):
        raise ValueError('Client account snapshot is stale, future or wrong account')
    asset = SimpleNamespace(**snapshot['asset'])
    positions = [SimpleNamespace(**row) for row in snapshot['positions']]
    validate_account_snapshot(asset, positions)
    return snapshot, asset, {p.stock_code: p for p in positions}


def previous_baseline(directory, previous, account):
    path = directory / ('account_eod_' + previous.isoformat() + '.json')
    if path.exists():
        data, _ = read_document(path)
        if data['account'] != account or data['trade_date'] != previous.isoformat():
            raise ValueError('Previous account snapshot account/date mismatch')
        captured = datetime.fromtimestamp(data['captured_at'])
        if captured.date() != previous or captured.time() < daytime(15, 35):
            raise ValueError('Previous account baseline is not a real post-close snapshot')
        validate_account_snapshot(SimpleNamespace(**data['asset']),
                                  [SimpleNamespace(**p) for p in data['positions']])
        return data['asset']['cash'], {p['stock_code']: p['volume'] for p in data['positions']}
    raise FileNotFoundError('Account-bound previous real baseline is missing; legacy unbound files are insufficient')


def document_from_day_plan(plan, positions, account, trade_date, generated_at, snapshot_time):
    targets = {code: int(p.volume) for code, p in positions.items() if code in plan.prices}
    for code, shares in plan.sell_orders:
        quantity = int(positions[code].can_use_volume) if shares < 0 else int(shares)
        targets[code] = max(0, int(positions[code].volume) - quantity)
    for code, shares in plan.buy_orders.items():
        targets[code] = targets.get(code, 0) + int(shares)
    rows = {code: {'shares': shares, 'open_price': float(plan.prices[code]),
                   'freeze_price': float(plan.limit_prices.get(code, plan.prices[code]))}
            for code, shares in targets.items()}
    return dict(version=1, id=uuid.uuid4().hex[:12], account=account,
                trade_date=trade_date.isoformat(), generated_at=generated_at,
                expires=datetime.combine(trade_date, daytime(15)).timestamp(),
                account_snapshot_at=snapshot_time, targets=rows,
                planning_positions={c:int(p.volume) for c,p in positions.items()},
                buy_priority=[c for c in plan.buy_n_stocks if c in rows])


def ensure_local_history(previous):
    import numpy as np
    from core.runtime import latest_runtime_npz_path
    from utils.stock.time import get_trading_date_span
    from trading.startup_data import validate_startup_runtime
    try:
        validate_startup_runtime(previous)
    except (OSError, ValueError, KeyError):
        # Deliberately bypass update_kline's native-QMT fallback. The local
        # strategy process has no brokerage connection or process control.
        from trading.post_close import run_update_all
        expected = np.array(get_trading_date_span(previous - timedelta(days=60), previous),
                            dtype='datetime64[D]')
        try:
            with np.load(latest_runtime_npz_path(), allow_pickle=False) as data:
                missing = expected[~np.isin(expected, data['trade_dates'].astype('datetime64[D]'))]
        except (OSError, ValueError, KeyError):
            missing = expected
        first = missing[0].item() if len(missing) else previous
        days = max(1, len(get_trading_date_span(first, previous)))
        if not run_update_all(SimpleNamespace(_skip_update=False), history_end=previous,
                              history_days=days, allow_qmt=False):
            raise RuntimeError('QMT-free startup data update failed; target generation blocked')
        validate_startup_runtime(previous)


def publish_target(config_path, directory, *, now=None, now_fn=None, overlay=None, simulation=False):
    import numpy as np
    from core.backtest import _compute_factor_scores
    from core.legality import LegalityChecker
    from core.prefilter import prefilter_quote_codes, select_prefilter_candidates
    from core.strategy import build_rebalance_day
    from core.strategy_config import load_strategy_config
    from core.timing import compute_position_multiplier_for_date
    from data.db.stock_list import allow_buy_stock_code_list
    from utils.stock.time import get_last_trading_day, _get_trading_calendar_state

    if now is not None and now_fn is not None:
        raise ValueError('Provide one clock source')
    clock = now_fn or ((lambda: now) if now is not None else datetime.now)
    now = clock()
    today = now.date()
    calendar, _, _ = _get_trading_calendar_state()
    if today not in calendar:
        raise ValueError('Today is not covered as a trading day by the verified calendar')
    if not daytime(9, 25, 10) <= now.time() < daytime(10):
        raise ValueError('New target generation is restricted to the preparation window')
    directory = Path(directory)
    if simulation and directory.resolve() == Path(__file__).resolve().parents[1] / 'data/live_file':
        raise ValueError('Simulation targets cannot be published to the production directory')
    deployment = json.loads((directory / 'config.json').read_text(encoding='utf-8'))
    target_path = directory / ('targets_' + today.isoformat() + '.json')
    journal_path = directory / 'journal' / ('execution_' + today.isoformat() + '.json')
    frozen = None
    if journal_path.exists():
        journal = json.loads(journal_path.read_text(encoding='utf-8'))
        if journal['account'] != deployment['account'] or journal['pending']['trade_date'] != today.isoformat():
            raise RuntimeError('Existing client journal account/date mismatch')
        frozen = journal['target_document']
    if target_path.exists():
        existing, _ = read_document(target_path)
        validate_target(existing, deployment['account'], today, now.timestamp(), allow_simulation=simulation)
        if frozen is not None and existing != frozen:
            raise RuntimeError('Target file differs from the already-started client journal')
        _archive_daily(directory, existing, journal=journal if frozen is not None else None, simulation=simulation)
        return target_path
    if frozen is not None:
        write_immutable_target(target_path, frozen)
        _archive_daily(directory, frozen, journal=journal, simulation=simulation)
        return target_path
    previous = get_last_trading_day(today - timedelta(days=1))
    from trading.startup_data import validate_startup_runtime
    validate_startup_runtime(previous)
    snapshot, asset, positions = load_account_file(directory / 'account_latest.json',
                                                  deployment['account'], today,
                                                  lambda: clock().timestamp())
    budget_metadata = None
    if (not (directory / ('account_eod_' + previous.isoformat() + '.json')).exists()
            and (directory / ('resume_request_' + today.isoformat() + '.json')).exists()):
        from trading.resume_baseline import load_resume_budget
        old_cash, old_positions, budget_metadata = load_resume_budget(
            directory, snapshot, config_path, clock())
    else:
        old_cash, old_positions = previous_baseline(directory, previous, deployment['account'])
    strategy = load_strategy_config(config_path)
    config = strategy['individual_config']
    weights, classes = config['weights'], strategy['factor_classes']
    candidates = allow_buy_stock_code_list(target_date=today)
    data = None
    count = config['prefilter_n']
    if count:
        prior = _compute_factor_scores([datetime.combine(previous, daytime())], candidates, weights, classes)
        if prior is None:
            raise ValueError('Previous full ranking is missing')
        data, scores, _, _, indices, codes, columns = prior
        candidates = select_prefilter_candidates(scores, indices[0], codes, columns,
                                                  weights, count, list(positions))
    if overlay is None:
        from data.update_live import update_live_quick
        overlay = update_live_quick(patch_npz=False, anchor_date=today,
                                   codes=prefilter_quote_codes(candidates, positions))
    result = _compute_factor_scores([datetime.combine(today, daytime())], candidates, weights,
                                    classes, kline_data=overlay, data=data)
    if result is None:
        raise ValueError('Today T-open overlay is unavailable')
    data, scores, _, _, indices, codes, columns = result
    index = indices[0]
    protection = config.get('limit_up_protection', False)
    checker = LegalityChecker(data, columns, limit_up_protection=protection)
    plan = build_rebalance_day(data=data, all_scores=scores, date_idx=index, trade_idx=index,
        signal_date=today, valid_stocks=codes,
        valid_cols=np.array([columns[c] for c in codes], dtype=np.intp),
        stock_indices=columns, weights=weights, buy_n=config['buy_n'], sell_m=config['sell_m'],
        checker=checker, positions={c: p.volume for c, p in positions.items()},
        sellable_volumes={c: p.can_use_volume for c, p in positions.items()}, cash=asset.cash,
        rebalance=config['rebalance'],
        position_multiplier=compute_position_multiplier_for_date(config, today),
        target_cash=old_cash, target_positions=old_positions,
        price_codes_extra=set(old_positions), limit_up_protection=protection,
        rank_within_candidates=bool(count))
    completed = clock()
    if completed.date() != today or not daytime(9, 25, 10) <= completed.time() < daytime(10):
        raise RuntimeError('Target calculation finished outside the new-plan window')
    document = document_from_day_plan(plan, positions, deployment['account'], today,
                                      completed.timestamp(), snapshot['captured_at'])
    if budget_metadata is not None:
        if completed.timestamp() - snapshot['captured_at'] > 30:
            raise RuntimeError('Recovery account expired during calculation; retry with fresh facts')
        document['budget_baseline'] = budget_metadata
    if simulation:
        document['simulation'] = True
    validate_target(document, deployment['account'], today, completed.timestamp(), allow_simulation=simulation)
    directory.mkdir(parents=True, exist_ok=True)
    write_immutable_target(target_path, document)
    _archive_daily(directory, document, snapshot=snapshot, simulation=simulation)
    return target_path


def _archive_daily(directory, document, snapshot=None, journal=None, simulation=False):
    try:
        from trading.daily_records import write_planner_record
        write_planner_record(directory, document, snapshot=snapshot, journal=journal, simulated=simulation)
    except Exception:
        logging.exception('Planner daily JSON failed; published execution target remains valid')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/config.json')
    parser.add_argument('--directory', default='data/live_file')
    args = parser.parse_args()
    from trading.planner_watchdog import install_no_qmt_guard
    install_no_qmt_guard()
    print(publish_target(args.config, args.directory))
