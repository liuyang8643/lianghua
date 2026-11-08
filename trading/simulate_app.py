"""Run the execution engine with FakeSDK in the installed app's Python 3.6."""
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess

from trading.bridge_setup import bootstrap_source


def launch_module(root, module, argument, directory):
    config = json.loads((root / 'data/live_bridge/config.json').read_text(encoding='utf-8'))
    bootstrap = directory / 'run_fake_app.py'
    bootstrap.write_text(bootstrap_source(root, module), encoding='utf-8')
    runtime = root / 'data/qmt_bridge_runtime'
    env = dict(os.environ)
    env['PYTHONPATH'] = os.pathsep.join(map(str, (runtime / 'stdlib', runtime,
                                               runtime / 'site-packages', root)))
    env['PYTHONIOENCODING'] = 'utf-8'
    result = subprocess.run([config['broker_python'], str(bootstrap), str(argument)],
                            cwd=root, env=env, capture_output=True, timeout=60,
                            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    (directory / 'console.log').write_bytes(result.stdout)
    return result


def simulate(root=None):
    root = Path(root or Path(__file__).resolve().parents[1])
    directory = root / 'tmp' / ('app_simulation_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    directory.mkdir(parents=True)
    report_path = directory / 'report.json'
    result = launch_module(root, 'trading/app_simulation_worker.py', report_path, directory)
    if not report_path.exists():
        raise RuntimeError('Simulation interpreter did not produce a report: ' +
                           result.stderr.decode('utf-8', errors='replace'))
    report = json.loads(report_path.read_text(encoding='utf-8'))
    print(json.dumps({'report': str(report_path), 'python': report['python'],
                      'passed': report['passed'], 'real_orders_or_cancels': 0}, ensure_ascii=False))
    if not report['passed']:
        raise RuntimeError(report['error'])
    return report_path


def replay(trade_date, root=None):
    from trading.replay_day import prepare_replay
    root = Path(root or Path(__file__).resolve().parents[1])
    fixture = prepare_replay(root, trade_date)
    directory = root / 'tmp' / ('historical_replay_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    directory.mkdir(parents=True)
    started = datetime.now()
    result = launch_module(root, 'trading/replay_worker.py', fixture, directory)
    latest = json.loads((fixture.parent / 'latest.json').read_text(encoding='utf-8'))
    if datetime.fromisoformat(latest['actual_run_time']) < started:
        raise RuntimeError('Historical replay did not produce a new result: ' + result.stderr.decode('utf-8', errors='replace'))
    report = json.loads(Path(latest['report']).read_text(encoding='utf-8'))
    print(json.dumps(latest, ensure_ascii=False))
    if not report['passed']:
        raise RuntimeError(report['error'])
    return Path(latest['report'])


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replay-date', help='Replay a dated real pre-open snapshot and saved plan')
    args = parser.parse_args()
    if args.replay_date:
        replay(datetime.strptime(args.replay_date, '%Y-%m-%d').date())
    else:
        simulate()
