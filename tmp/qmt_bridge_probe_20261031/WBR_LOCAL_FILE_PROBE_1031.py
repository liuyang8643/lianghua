# coding: utf-8
"""Paper-only local file probe. No broker queries, orders, or cancellations."""
import json
import os
import sys

ROOT = 'D:/coding/WBR-live/tmp/qmt_bridge_probe_20261031/'


def differences(case):
    orders = []
    blocked = []
    for code in sorted(case['targets']):
        target = case['targets'][code]
        held = case['positions'].get(code, 0)
        working_buy = case['working_buy'].get(code, 0)
        working_sell = case['working_sell'].get(code, 0)
        delta = target - held - working_buy + working_sell
        if delta > 0:
            orders.append({'code': code, 'side': 'BUY', 'shares': delta})
        elif delta < 0:
            available = max(0, case['sellable'].get(code, 0) - working_sell)
            shares = min(-delta, available)
            if shares:
                orders.append({'code': code, 'side': 'SELL', 'shares': shares})
            if shares < -delta:
                blocked.append({'code': code, 'shares': -delta - shares})
    return {'orders': orders, 'blocked': blocked}


def run_probe(context):
    with open(ROOT + 'request.json', 'r', encoding='utf-8') as stream:
        request = json.load(stream)
    if request['mode'] != 'PAPER_ONLY' or request['version'] != 1:
        raise ValueError('Unsupported probe request')
    results = []
    for case in request['cases']:
        result = differences(case)
        passed = result == case['expected']
        item = {'name': case['name'], 'result': result, 'passed': passed}
        if not result['blocked']:
            # Simulate fills in a private dictionary, never in an account.
            paper = dict(case['positions'])
            for code, shares in case['working_buy'].items():
                paper[code] = paper.get(code, 0) + shares
            for code, shares in case['working_sell'].items():
                paper[code] = paper.get(code, 0) - shares
            for order in result['orders']:
                sign = 1 if order['side'] == 'BUY' else -1
                code = order['code']
                paper[code] = paper.get(code, 0) + sign * order['shares']
            settled = dict(case)
            settled.update(positions=paper, sellable=paper,
                           working_buy={}, working_sell={})
            second = differences(settled)
            item['paper_positions_after_fills'] = paper
            item['second_pass'] = second
            item['passed'] = passed and second == {'orders': [], 'blocked': []}
        results.append(item)
        print('WBR_PROBE_CASE', json.dumps(item, sort_keys=True))
    response = {
        'mode': 'PAPER_ONLY', 'version': 1,
        'challenge': request['challenge'],
        'source': 'QMT_INIT_CALLBACK' if context is not None else 'APP_SCRIPT_MAIN',
        'python_version': sys.version,
        'python_executable': sys.executable,
        'pid': os.getpid(), 'parent_pid': os.getppid(),
        'context_type': type(context).__name__,
        'embedded_trade_api_present': callable(globals().get('passorder')),
        'cases': results,
        'all_passed': all(item['passed'] for item in results),
        'broker_orders_sent': 0,
    }
    with open(ROOT + 'response.json', 'w', encoding='utf-8') as stream:
        json.dump(response, stream, indent=2, sort_keys=True)
    # Read back the file written by the embedded interpreter.
    with open(ROOT + 'response.json', 'r', encoding='utf-8') as stream:
        saved = json.load(stream)
    if saved != response:
        raise RuntimeError('Local response readback mismatch')
    print('WBR_PROBE_FINISHED', request['challenge'], response['all_passed'])
    if not response['all_passed']:
        raise AssertionError('Paper reconciliation failed')


def init(ContextInfo):
    run_probe(ContextInfo)


def handlebar(ContextInfo):
    pass


if __name__ == '__main__':
    run_probe(None)
