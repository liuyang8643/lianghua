"""Run the actual local publisher against archived inputs, with no network/SDK.

This rechecks strategy computation; the old T-1 file reader is a simulated
boundary. Production still requires the account-bound EOD JSON contract.
"""
from datetime import date, datetime
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from trading.bridge_wire import atomic_json
from trading.planner_watchdog import install_no_qmt_guard


def main():
    install_no_qmt_guard()
    root = Path(__file__).resolve().parents[1]
    fixture = json.loads((root / 'data/live_file/replays/2026-11-02/input.json').read_text(encoding='utf-8'))
    output = root / 'tmp' / ('publisher_replay_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
    output.mkdir()
    atomic_json(output / 'config.json', {'account': fixture['account']})
    atomic_json(output / 'account_latest.json', dict(version=1, account=fixture['account'],
        trade_date=fixture['trade_date'], captured_at=fixture['account_captured_at'],
        asset=fixture['asset'], positions=fixture['positions']))
    prior = pd.read_parquet(root / 'data/live_trades/positions_2026-10-30.parquet')
    previous_positions = {r['code']: int(r['volume']) for r in prior.to_dict('records') if r['volume'] > 0}
    summary = pd.read_parquet(root / 'data/live_trades/daily_summary.parquet')
    cash = float(summary.loc[pd.to_datetime(summary['date']).dt.date == date(2026, 10, 30), 'cash'].iloc[-1])
    assert previous_positions == {r['stock_code']: r['volume'] for r in fixture['positions']}
    cache = json.loads((root / 'data/market_status/tencent_close_2026-11-02.json').read_text(encoding='utf-8'))
    texts = [p['text'] for p in cache['packets']]
    texts.append(json.loads((root / 'data/market_status/closing_probe_raw.json').read_text(encoding='utf-8'))['text'])
    statements = {}
    for text in texts:
        for line in text.split(';'):
            line = line.strip()
            if line.startswith('v_') and '="' in line:
                symbol = line.split('=', 1)[0][2:]
                statements[symbol] = line + ';'
    requested = []
    def quote_get(url, **kwargs):
        assert url.startswith('https://qt.gtimg.cn/q=')
        symbols = url.split('q=', 1)[1].split(',')
        requested.extend(symbols)
        payload = '\n'.join(statements[s] for s in symbols)
        return SimpleNamespace(content=payload.encode('gbk'), raise_for_status=lambda: None)
    from trading.plan_publisher import publish_target
    with patch('trading.plan_publisher.previous_baseline', return_value=(cash, previous_positions)), \
         patch('data.update_live.requests.get', side_effect=quote_get):
        path = publish_target(root / 'configs/config.json', output,
                              now_fn=lambda: datetime(2026, 11, 2, 9, 25, 30), simulation=True)
    document = json.loads(path.read_text(encoding='utf-8'))
    actual = {c: r['shares'] - previous_positions.get(c, 0) for c, r in document['targets'].items()
              if r['shares'] != previous_positions.get(c, 0)}
    expected = {r['code']: r['shares'] * (1 if r['direction'] == 'BUY' else -1)
                for r in fixture['expected_orders']}
    report = dict(passed=actual == expected, actual_differences=actual, original_plan_differences=expected,
        requested_quote_count=len(requested), network_calls=0, qmt_imports=0,
        baseline_cash=cash, preopen_cash=fixture['asset']['cash'],
        quote_input='Archived completed-session response parsed ONLY as T-open/preClose overlay',
        baseline='Original T-1 Parquet values, numerically matched to bound pre-open receipt; simulated file-reader boundary',
        target_path=str(path))
    atomic_json(output / 'report.json', report)
    print(json.dumps(report, ensure_ascii=False))
    assert report['passed'], 'Publisher differs from original 1102 plan'


if __name__ == '__main__':
    main()
