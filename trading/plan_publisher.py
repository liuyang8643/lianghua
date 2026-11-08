"""Local strategy -> immutable absolute targets. Never imports/calls QMT.

Account inputs are files exported by the client; market inputs are local
Runtime plus Tencent T-open. Selection and legality use the sole strategy_release/env implementation.
"""
import json
import time
import uuid
import logging
from datetime import datetime, time as daytime, timedelta
from pathlib import Path
from types import SimpleNamespace

from trading.account_snapshot import validate_account_snapshot
from trading.target_protocol import FILE_EXECUTOR_PROTOCOL, read_document, validate_target, write_immutable_target


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
    asset,positions=validate_account_snapshot(asset, positions)
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
        quantity=int(shares)
        if quantity<=0 or quantity>int(positions[code].volume):
            raise ValueError('Canonical order plan contains an invalid sale')
        targets[code] = int(positions[code].volume) - quantity
    for code, shares in plan.buy_orders.items():
        if int(shares)<=0:
            raise ValueError('Canonical order plan contains an invalid purchase')
        targets[code] = targets.get(code, 0) + int(shares)
    rows = {code: {'shares': shares, 'open_price': float(plan.prices[code]),
                   'freeze_price': float(plan.limit_prices.get(code, plan.prices[code]))}
            for code, shares in targets.items()}
    return dict(version=1, id=uuid.uuid4().hex[:12], account=account,
                executor_protocol=FILE_EXECUTOR_PROTOCOL,
                trade_date=trade_date.isoformat(), generated_at=generated_at,
                expires=datetime.combine(trade_date, daytime(15)).timestamp(),
                account_snapshot_at=snapshot_time, targets=rows,
                planning_positions={c:int(p.volume) for c,p in positions.items()},
                buy_priority=[c for c in plan.buy_n_stocks if c in rows])


def ensure_local_history(previous):
    import numpy as np
    from offline_data import latest_runtime_npz_path
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
    from offline_data import latest_runtime_npz_path, load_runtime_slice
    from data.db.stock_list import load_current_stock_codes
    from data.strategy_inputs import replace_open_quotes
    from trading.strategy_plan import calculate_plan
    from trading.position_age import holding_days
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
    if snapshot.get('executor_protocol')!=FILE_EXECUTOR_PROTOCOL:
        raise RuntimeError('Reload WBR_EXEC_BRIDGE before publishing the new strategy: executor protocol is stale')
    previous_baseline(directory, previous, deployment['account'])
    settings = json.loads(Path(config_path).read_text(encoding='utf8'))
    inputs_directory = Path(settings['strategy_inputs'])
    unmanaged = set(settings['unmanaged_codes'])
    managed_positions = {c:p for c,p in positions.items() if c not in unmanaged}
    from trading.preopen import prepare
    prepared=prepare(today,config_path,directory,allow_calculation=False)
    path=prepared.identity['runtime_path']
    history=prepared.cache.runtime
    active = set(load_current_stock_codes()) | set(managed_positions)
    if not active.issubset(history.stock_codes):
        raise ValueError('Current stock universe is missing historical factor inputs')
    if overlay is None:
        from data.update_live import update_live_quick
        overlay = update_live_quick(patch_npz=False,anchor_date=today,codes=sorted(active))
    missing = active-set(overlay)
    if missing:
        raise ValueError('Full cross-section quotes are incomplete: '+str(sorted(missing)))
    import numpy as np
    for code in active:
        quote=overlay[code]
        quote_day=np.datetime64(int(quote['time'][-1]),'ms').astype('datetime64[D]')
        if (quote_day!=np.datetime64(today) or not np.isfinite(quote['open'][-1])
                or quote['open'][-1]<0 or not np.isfinite(quote['preClose'][-1])
                or quote['preClose'][-1]<=0):
            raise ValueError('Invalid same-day opening quote: '+code)
    runtime = replace_open_quotes(history,overlay)
    quote_received_at=clock().timestamp()
    signal_id=uuid.uuid4().hex[:12]
    signal_audit={}
    def save_signal(index,raw,normalized,heads,scores,eligible):
        try:
            from trading.bridge_wire import atomic_json
            from utils.atomic_file import file_sha256
            audit=directory/'audit';audit.mkdir(parents=True,exist_ok=True)
            stem='signal_'+today.isoformat()+'_'+signal_id
            path=audit/(stem+'.npz');temporary=path.with_suffix('.tmp.npz')
            np.savez_compressed(temporary,stock_codes=np.asarray(runtime.stock_codes),
                feature_names=np.asarray(prepared.assembly.strategy.names),raw=raw,normalized=normalized,
                predictions=heads,scores=scores,eligible=eligible)
            temporary.replace(path)
            provenance=audit/(stem+'.json')
            atomic_json(provenance,dict(trade_date=today.isoformat(),identity=prepared.identity,
                quote_received_at=quote_received_at,prediction_completed_at=clock().timestamp(),
                quotes={c:dict(open=float(q['open'][-1]),preClose=float(q['preClose'][-1])) for c,q in overlay.items()},
                factor_snapshot=path.name,factor_sha256=file_sha256(path)))
            signal_audit.update(path=str(provenance.relative_to(directory)),sha256=file_sha256(provenance))
        except Exception:
            logging.exception('Signal audit failed; execution target remains independently journaled')
    scores=prepared.assembly.predict(runtime,[runtime.index_of(today)],prepared.inputs,
                                     opening_cache=prepared.cache,on_prediction=save_signal)
    # Prediction is account-independent. Bind allocation to a fresh confirmed
    # account after it finishes, rather than aging the account during factors.
    snapshot,asset,positions=load_account_file(directory/'account_latest.json',deployment['account'],
                                              today,lambda:clock().timestamp())
    if snapshot.get('executor_protocol')!=FILE_EXECUTOR_PROTOCOL:
        raise RuntimeError('The confirmed executor protocol changed during calculation')
    managed_positions={c:p for c,p in positions.items() if c not in unmanaged}
    if not set(managed_positions).issubset(active):
        raise RuntimeError('Account acquired an unquoted security during calculation')
    ages = holding_days(directory,deployment['account'],today,managed_positions,calendar,
                        unmanaged_codes=unmanaged)
    plan = calculate_plan(path,today,managed_positions,asset.cash,ages,
                          inputs_directory=inputs_directory,runtime=runtime,assembly=prepared.assembly,
                          inputs=prepared.inputs,opening_cache=prepared.cache,scores=scores)
    from trading.preopen import identity
    if identity(today,config_path)!=prepared.identity:
        raise RuntimeError('Strategy or input identity changed during the daily calculation')
    completed = clock()
    if not 0<=completed.timestamp()-snapshot['captured_at']<=30:
        raise RuntimeError('Confirmed account aged beyond the publication freshness limit')
    if completed.date() != today or not daytime(9, 25, 10) <= completed.time() < daytime(10):
        raise RuntimeError('Target calculation finished outside the new-plan window')
    document = document_from_day_plan(plan, positions, deployment['account'], today,
                                      completed.timestamp(), snapshot['captured_at'])
    document.update(strategy_id=plan.strategy_id, model_sha256=plan.model_sha256,
                    max_positions=plan.max_positions, holding_days=ages,
                    unmanaged_codes=sorted(unmanaged),strategy_open_nav=plan.strategy_open_nav,
                    broker_snapshot_nav=float(asset.total_asset),signal_audit=signal_audit)
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
