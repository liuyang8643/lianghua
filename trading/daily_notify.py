"""Local JSON-to-Feishu relay. Never connects to the broker or computes targets."""
from datetime import date
import hashlib
import json
from pathlib import Path
import time
import uuid

from filelock import FileLock
from trading.bridge_wire import atomic_json
from trading.daily_records import daily_path, ensure_planner_record

RENDER_VERSION = b'original-TradingDayBoard-v2'


def _deliver_close_attachments(directory, day, packet, sender):
    """Persist each acknowledgement; an ambiguous send never triggers another create."""
    receipt_path = directory / 'daily' / ('feishu_close_' + day + '.json')
    receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
    html = Path(packet['html_path'])
    if hashlib.sha256(html.read_bytes()).hexdigest() != packet['html_sha256']:
        raise ValueError('Post-close HTML changed after report generation')
    for kind in ('html', 'detail'):
        if kind == 'detail' and packet['detail_card'] is None:
            receipt[kind] = dict(state='skipped', reason='original_report_has_no_detail')
            atomic_json(receipt_path, receipt)
            continue
        state = receipt.get(kind, {})
        digest = (packet['html_sha256'] if kind == 'html' else
                  hashlib.sha256(json.dumps(packet['detail_card'], sort_keys=True).encode()).hexdigest())
        if state.get('state') == 'sent' and state.get('digest') == digest:
            continue
        if state.get('state') == 'sent' and not state.get('digest'):
            raise RuntimeError('Legacy attachment content hash must be reconciled before correction')
        if state.get('state') in ('sending', 'unknown'):
            raise RuntimeError('Post-close ' + kind + ' acknowledgement unknown; inspect sender audit')
        if state.get('state') == 'sent' and kind == 'detail':
            if not sender.update_card(state['message_id'], packet['detail_card']):
                raise RuntimeError('Post-close detail correction failed; existing message retained')
            state.update(digest=digest, updated_at=time.time())
            atomic_json(receipt_path, receipt)
            continue
        history = state.get('history', [])
        if state.get('state') == 'sent':
            history = history + [{k:v for k,v in state.items() if k != 'history'}]
        receipt[kind] = dict(state='sending', attempted_at=time.time(), digest=digest, history=history)
        atomic_json(receipt_path, receipt)
        if kind == 'html':
            name = html.stem + '_corrected_' + digest[:8] + html.suffix if history else html.name
            response = sender.send_file(str(html), file_name=name)
            ok = response is not None and response.success()
            message_id = response.data.message_id if ok else None
            ok = ok and bool(message_id)
        else:
            key = uuid.uuid5(uuid.NAMESPACE_URL, packet['account'] + ':' + day + ':post_close_detail').hex
            message_id = sender.send_card(packet['detail_card'], idempotency_key=key)
            ok = bool(message_id)
        receipt[kind].update(state='sent' if ok else 'unknown', message_id=message_id, updated_at=time.time())
        atomic_json(receipt_path, receipt)
        if not ok:
            raise RuntimeError('Post-close ' + kind + ' delivery failed or acknowledgement unknown')


def deliver_daily(directory, trade_date=None, *, sender=None):
    directory = Path(directory)
    day = (trade_date or date.today()).isoformat()
    ensure_planner_record(directory, day)  # Archival is independent of notification settings.
    settings_path = directory / 'notification_config.json'
    if not settings_path.exists() or not json.loads(settings_path.read_text())['enabled']:
        return None
    config = json.loads((directory / 'config.json').read_text(encoding='utf-8'))
    planner_path = daily_path(directory, 'planner', day)
    executor_path = daily_path(directory, 'executor', day)
    daily = directory / 'daily'
    daily.mkdir(exist_ok=True)
    with FileLock(str(daily / 'notification.lock'), timeout=0):
        if not planner_path.exists():
            return None
        planner = json.loads(planner_path.read_text(encoding='utf-8'))
        execution = json.loads(executor_path.read_text(encoding='utf-8')) if executor_path.exists() else None
        if (planner['producer'] != 'local_python' or planner['simulated'] or planner['account'] != config['account']
                or planner['trade_date'] != day):
            raise ValueError('Invalid local daily summary')
        if execution is not None and (execution['producer'] != 'broker_python' or execution['simulated']
                or execution['account'] != planner['account'] or execution['trade_date'] != day
                or execution['target_document'] != planner['target_document']):
            raise ValueError('Invalid or simulated client daily summary')
        close_path = daily / ('close_' + day + '.json')
        close = json.loads(close_path.read_text(encoding='utf-8')) if close_path.exists() else None
        if close is not None and (close['account'] != config['account'] or close['trade_date'] != day
                or close['target_id'] != planner['target_id'] or execution is None
                or close['execution_id'] != execution['execution_id']):
            raise ValueError('Post-close report account/date/execution mismatch')
        digest = hashlib.sha256(RENDER_VERSION + planner_path.read_bytes()
            + (executor_path.read_bytes() if execution else b'')
            + (close_path.read_bytes() if close else b'')).hexdigest()
        receipt_path = daily / ('feishu_' + day + '.json')
        receipt = json.loads(receipt_path.read_text()) if receipt_path.exists() else {}
        from trading.day_board import build_file_daily_card
        if sender is None:
            from trading.lark.sender import lark_sender
            sender = lark_sender
        if receipt.get('digest') == digest and receipt.get('state') == 'sent':
            if close:
                _deliver_close_attachments(directory, day, close, sender)
            return receipt
        card = build_file_daily_card(planner, execution, close)
        atomic_json(daily / ('card_' + day + '.json'), card)
        key = uuid.uuid5(uuid.NAMESPACE_URL, config['account'] + ':' + day + ':file_daily').hex
        if not receipt.get('message_id') and receipt.get('state') in ('sending', 'unknown'):
            # Recover a successful send from the existing sender audit after a crash.
            audit = directory.parent / 'live_trades/lark_audit' / (day.replace('-', '') + '.jsonl')
            if audit.exists():
                for line in audit.read_text(encoding='utf-8').splitlines():
                    row = json.loads(line)
                    if row.get('idempotency_key') == key and row.get('ok') and row.get('message_id'):
                        receipt['message_id'] = row['message_id']
            if not receipt.get('message_id'):
                raise RuntimeError('Feishu delivery receipt unknown; automatic duplicate creation refused')
        if receipt.get('message_id'):
            if not sender.update_card(receipt['message_id'], card):
                raise RuntimeError('Feishu daily card update failed; existing message retained')
        else:
            receipt = dict(state='sending', account=config['account'], trade_date=day,
                           idempotency_key=key, attempted_at=time.time())
            atomic_json(receipt_path, receipt)
            message_id = sender.send_card(card, idempotency_key=key)
            if not message_id:
                receipt['state'] = 'unknown'
                atomic_json(receipt_path, receipt)
                raise RuntimeError('Feishu send failed or acknowledgement unknown; inspect sender audit')
            receipt['message_id'] = message_id
        receipt.update(state='sent', digest=digest, phase='close' if close else 'execution' if execution else 'plan', sent_at=time.time())
        atomic_json(receipt_path, receipt)
        if close:
            _deliver_close_attachments(directory, day, close, sender)
        return receipt


if __name__ == '__main__':
    from trading.planner_watchdog import install_no_qmt_guard
    install_no_qmt_guard()
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', default='data/live_file')
    args = parser.parse_args()
    print(json.dumps(deliver_daily(args.directory), ensure_ascii=False))
