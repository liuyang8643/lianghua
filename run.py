"""Run the local file planner; start WBR_EXEC_BRIDGE in the broker app separately."""
import argparse
from datetime import datetime
import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parent


def launch(*, preserve_halt=False, restart=False):
    root = ROOT
    config = root / 'configs/config.json'
    directory = root / 'data/live_file'
    markers = (root / 'data/live_trades/trading_watchdog.halt', directory / 'stop.request')
    if not config.is_file() or not (directory / 'config.json').is_file():
        raise RuntimeError('Missing strategy or file execution configuration')
    if preserve_halt and any(path.exists() for path in markers):
        raise RuntimeError('Execution stop marker exists; automated startup refused')

    # Fail before clearing stop markers or starting anything if sync fails.
    subprocess.run(['uv', 'sync', '--locked'], cwd=root, check=True)
    python = root / '.venv/Scripts/python.exe'
    if not python.is_file():
        raise RuntimeError('Missing project Python: ' + str(python))
    if restart:
        subprocess.run([str(python),'-m','trading.planner_control','--root',str(root)],
                       cwd=root,check=True,creationflags=subprocess.CREATE_NO_WINDOW)

    logs = root / 'configs/logs'
    logs.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    stdout = logs / ('watchdog-' + stamp + '.out.log')
    stderr = logs / ('watchdog-' + stamp + '.err.log')
    env = dict(os.environ, PYTHONPATH=str(root), PYTHONIOENCODING='utf-8', PYTHONUTF8='1')
    env.update({key:'1' for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS')})
    command = [str(python), '-u', '-m', 'trading.planner_watchdog',
               '--config', str(config), '--directory', str(directory),
               '--preserve-halt' if preserve_halt else '--manual-start']
    with stdout.open('wb') as out, stderr.open('wb') as err:
        process = subprocess.Popen(command, cwd=root, env=env, stdin=subprocess.DEVNULL,
                                   stdout=out, stderr=err,
                                   creationflags=subprocess.CREATE_NO_WINDOW)
    try:
        code = process.wait(timeout=1)
    except subprocess.TimeoutExpired:
        print('Local planner started in background (PID={}).'.format(process.pid))
        print('Start WBR_EXEC_BRIDGE in the logged-in broker app separately.')
        print('stdout: ' + str(stdout))
        print('stderr: ' + str(stderr))
        return
    raise RuntimeError('Planner exited (code={}); inspect {}'.format(code, stderr))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preserve-halt', action='store_true',
                        help='For automated startup: refuse to clear execution stop markers')
    parser.add_argument('--restart',action='store_true',help='Replace only the verified local planner process')
    args = parser.parse_args()
    launch(preserve_halt=args.preserve_halt,restart=args.restart)


if __name__ == '__main__':
    main()
