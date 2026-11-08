"""24x7 local file planner. Has no brokerage SDK or QMT process control."""
import argparse
import importlib.abc
from datetime import datetime, time as daytime, timedelta
import logging
from pathlib import Path
import time
import sys

from filelock import FileLock, Timeout
from trading.bridge_wire import atomic_json


def install_no_qmt_guard():
    class NoQmt(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if (fullname == 'xtquant' or fullname.startswith('xtquant.')
                    or fullname in ('trading.qmt', 'trading.trader', 'trading.bridge_client')):
                raise RuntimeError('Local file planner prohibits QMT access: ' + fullname)
    if any(name == 'xtquant' or name.startswith('xtquant.') for name in sys.modules):
        raise RuntimeError('Local planner was started in a process that already loaded QMT')
    sys.meta_path.insert(0, NoQmt())


def run(config_path, directory, *, manual_start=False, preserve_halt=False):
    from trading.plan_publisher import publish_target, ensure_local_history
    from utils.stock.time import get_last_trading_day, _get_trading_calendar_state

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with FileLock(str(directory / 'planner.lock'), timeout=0):
        apply_start_markers(directory, manual_start=manual_start, preserve_halt=preserve_halt)
        next_publish_attempt = 0
        last_preflight = 0
        history_ready = None
        daily_update_done = None
        post_close_done = None
        last_close_attempt = 0
        last_calendar_attempt = 0
        last_notification_attempt = 0
        while True:
            now = datetime.now()
            try:
                calendar, _, calendar_end = _get_trading_calendar_state()
                if calendar_end is None or calendar_end < now.date():
                    raise ValueError('Trading calendar does not cover today')
            except Exception as error:
                if time.time() - last_calendar_attempt >= 300:
                    last_calendar_attempt = time.time()
                    try:
                        from data.update_all import _update_trading_calendar
                        _update_trading_calendar()
                    except Exception:
                        logging.exception('Trading calendar repair failed')
                    atomic_json(directory / 'planner_status.json',
                                dict(updated=time.time(), stage='calendar_blocked', error=str(error)))
                time.sleep(1)
                continue
            trading_today = now.date() in calendar
            if (trading_today and now.time() >= daytime(15, 35)
                    and post_close_done != now.date() and time.time() - last_close_attempt >= 300):
                last_close_attempt = time.time()
                try:
                    from trading.file_post_close import reconcile_from_files
                    if reconcile_from_files(now.date(), config_path, directory) is not None:
                        post_close_done = now.date()
                except Exception:
                    logging.exception('File-based post-close reconciliation blocked')
            if time.time() - last_preflight >= 300:
                last_preflight = time.time()
                try:
                    atomic_json(directory / 'planner_status.json',
                                dict(updated=time.time(), stage='checking_history'))
                    previous = get_last_trading_day(now.date() - timedelta(days=1))
                    if now.time() >= daytime(16) and trading_today:
                        previous = now.date()
                        if daily_update_done != now.date():
                            from types import SimpleNamespace
                            from trading.post_close import run_update_all
                            # File mode never persists intraday bars; refresh the
                            # completed session, with prior-gap checks in the updater.
                            if run_update_all(SimpleNamespace(_skip_update=False), history_days=1, allow_qmt=False):
                                daily_update_done = now.date()
                            else:
                                raise RuntimeError('QMT-free daily full update failed')
                    # Do not pull market-wide history inside the open ranking window.
                    if not daytime(9, 15) <= now.time() < daytime(10):
                        if history_ready != previous:
                            ensure_local_history(previous)
                            history_ready = previous
                    atomic_json(directory / 'planner_status.json', dict(updated=time.time(),
                        stage='history_ready' if history_ready == previous else 'awaiting_history',
                        history_end=str(history_ready)))
                except Exception as error:
                    logging.exception('Local history preflight blocked; no target will be published')
                    atomic_json(directory / 'planner_status.json',
                                dict(updated=time.time(), stage='history_blocked', error=str(error)))
            now = datetime.now()  # A data step can take substantial time.
            if (now.date() in calendar
                    and daytime(9, 25, 10) <= now.time() < daytime(10)
                    and time.time() >= next_publish_attempt):
                try:
                    atomic_json(directory / 'planner_status.json',
                                dict(updated=time.time(), stage='calculating_target', trade_date=now.date().isoformat()))
                    target = publish_target(config_path, directory)
                    atomic_json(directory / 'planner_status.json',
                                dict(updated=time.time(), stage='published', target=str(target)))
                    next_publish_attempt = time.time() + 300
                except Exception as error:
                    logging.exception('Local target generation blocked')
                    atomic_json(directory / 'planner_status.json',
                                dict(updated=time.time(), stage='blocked', error=str(error)))
                    next_publish_attempt = time.time() + 5
            if time.time() - last_notification_attempt >= 60:
                last_notification_attempt = time.time()
                try:
                    from trading.daily_notify import deliver_daily
                    deliver_daily(directory, now.date())
                except Exception:
                    logging.exception('Daily JSON/Feishu relay failed; trading files remain unchanged')
            time.sleep(1)


def apply_start_markers(directory, *, manual_start=False, preserve_halt=False):
    """Called only after acquiring the planner lock; duplicate starts cannot unhalt."""
    directory = Path(directory)
    markers = (directory.parent / 'live_trades/trading_watchdog.halt', directory / 'stop.request')
    if preserve_halt and any(path.exists() for path in markers):
        raise RuntimeError('Execution stop marker exists; automated startup refused')
    if manual_start and not preserve_halt:
        for path in markers:
            path.unlink(missing_ok=True)


def cli():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/config.json')
    parser.add_argument('--directory', default='data/live_file')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--manual-start', action='store_true')
    mode.add_argument('--preserve-halt', action='store_true')
    args = parser.parse_args()
    install_no_qmt_guard()
    logging.basicConfig(level=logging.INFO)
    try:
        run(args.config, args.directory, manual_start=args.manual_start, preserve_halt=args.preserve_halt)
    except Timeout:
        raise SystemExit('Another local planner is already running')


if __name__ == '__main__':
    cli()
