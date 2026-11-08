"""Prepare the isolated Python 3.6 runtime used by the client-side broker bridge."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
import urllib.request
import zipfile


PYTHON_URL = 'https://www.python.org/ftp/python/3.6.8/python-3.6.8-embed-amd64.zip'
PSUTIL36_URL = ('https://files.pythonhosted.org/packages/0b/58/'
                'bcffb5ab03ec558e565d2871c01215dde74e11f583fb71e7d2b107200caa/'
                'psutil-5.9.8-cp36-cp36m-win_amd64.whl')
PSUTIL36_SHA256 = '27cc40c3493bb10de1be4b3f07cae4c010ce715290a5be22b98493509c6299e2'


def prepare_runtime(root=None):
    import xtquant
    root = Path(root or Path(__file__).resolve().parents[1])
    target = root / 'data' / 'qmt_bridge_runtime'
    target.mkdir(parents=True, exist_ok=True)
    archive = target / 'python-3.6.8-embed-amd64.zip'
    if not archive.exists():
        temporary = archive.with_suffix('.download')
        with urllib.request.urlopen(PYTHON_URL, timeout=30) as response:
            with temporary.open('wb') as stream:
                shutil.copyfileobj(response, stream)
        temporary.replace(archive)
    with zipfile.ZipFile(archive) as bundle:
        bad = bundle.testzip()
        if bad:
            raise ValueError('Corrupt official Python runtime: ' + bad)
        for name in bundle.namelist():
            # Keep the broker's executable and Python DLL intact.
            if '/' in name or '\\' in name:
                raise ValueError('Unexpected embedded Python archive path')
            if name.endswith(('.pyd', '.zip', '.dll')) and name != 'python36.dll':
                (target / name).write_bytes(bundle.read(name))
    # The broker's reduced codecs can recurse when opening another compressed
    # stdlib zip. An unpacked official stdlib avoids that bootstrap dependency.
    with zipfile.ZipFile(target / 'python36.zip') as stdlib:
        for name in stdlib.namelist():
            destination = (target / 'stdlib' / name).resolve()
            if not destination.is_relative_to((target / 'stdlib').resolve()):
                raise ValueError('Unexpected standard library path')
        stdlib.extractall(target / 'stdlib')
    packages = target / 'site-packages'
    packages.mkdir(exist_ok=True)
    wheel = target / 'psutil-5.9.8-cp36-cp36m-win_amd64.whl'
    if not wheel.exists():
        with urllib.request.urlopen(PSUTIL36_URL, timeout=30) as response:
            content = response.read()
        if hashlib.sha256(content).hexdigest() != PSUTIL36_SHA256:
            raise ValueError('Python 3.6 psutil wheel checksum mismatch')
        wheel.write_bytes(content)
    if hashlib.sha256(wheel.read_bytes()).hexdigest() != PSUTIL36_SHA256:
        raise ValueError('Python 3.6 psutil wheel checksum mismatch')
    with zipfile.ZipFile(wheel) as bundle:
        for name in bundle.namelist():
            if name.startswith('psutil/') and not name.endswith('/'):
                destination = (packages / name).resolve()
                if not destination.is_relative_to(packages.resolve()):
                    raise ValueError('Unexpected psutil archive path')
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(bundle.read(name))
    source = Path(xtquant.__file__).parent
    shutil.copytree(source, packages / 'xtquant', dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    manifest = {
        'python_url': PYTHON_URL,
        'python_archive_sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
        'xtquant_version': xtquant.__version__,
        'psutil36_url': PSUTIL36_URL,
        'psutil36_sha256': PSUTIL36_SHA256,
        'files': {str(p.relative_to(target)): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in target.rglob('*') if p.is_file() and p.name != 'manifest.json'},
    }
    (target / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return target


def bootstrap_source(root, module='trading/qmt_bridge.py'):
    root = Path(root).resolve().as_posix()
    return '''# coding: utf-8
import sys
ROOT = %r
RUNTIME = ROOT + '/data/qmt_bridge_runtime'
sys.path[:0] = [RUNTIME + '/stdlib', RUNTIME,
               RUNTIME + '/site-packages', ROOT]
# encodings is imported before this entry point; extend its package path too.
import encodings
encodings.__path__.append(RUNTIME + '/stdlib/encodings')
SOURCE = ROOT + '/' + %r
import hashlib
def execute_source():
    with open(SOURCE, 'r', encoding='utf-8') as stream:
        code = stream.read()
    source_hash = hashlib.sha256(code.encode('utf-8')).hexdigest()
    exec(compile(code, SOURCE, 'exec'), {'__name__': '__main__', '__file__': SOURCE,
                                      '__source_sha256__': source_hash})
if %r == 'trading/qmt_bridge.py':
    import time
    while True:
        try:
            execute_source()
            break
        except Exception as error:
            print('WBR_BRIDGE_RECONNECT', type(error).__name__, str(error), flush=True)
            time.sleep(5)
else:
    execute_source()
''' % (root, module, module)


def prepare_config(root=None, *, live_enabled=False, individual_config='configs/config.json'):
    from configs import TRADE_ACCOUNT
    from configs.env import QMT_ROOT_DIR
    import pyarrow.parquet as pq
    from xtquant import xtconstant
    root = Path(root or Path(__file__).resolve().parents[1])
    individual_config = Path(individual_config)
    if not individual_config.is_absolute():
        individual_config = root / individual_config
    if not individual_config.is_file():
        raise FileNotFoundError(individual_config)
    broker_bin = next(Path(path) for path in QMT_ROOT_DIR
                      if (Path(path) / 'pythonw.exe').exists())
    directory = root / 'data/live_bridge'
    directory.mkdir(parents=True, exist_ok=True)
    from trading.bridge_wire import atomic_json
    config = {
        'root': str(root), 'account': TRADE_ACCOUNT,
        'individual_config': str(individual_config.resolve()),
        'userdata': str(broker_bin.parent / 'userdata_mini'),
        'project_python': str(root / '.venv/Scripts/python.exe'),
        'project_python_images': [sys.executable, sys._base_executable],
        'broker_python': str(broker_bin / 'pythonw.exe'),
        'client_images': [str(broker_bin / name) for name in
                          ('miniquote.exe', 'XtItClient.exe')],
        'strategy_name': 'FlashMan', 'live_enabled': live_enabled,
        'price_types': [xtconstant.FIX_PRICE, xtconstant.MARKET_PEER_PRICE_FIRST],
        'trading_dates': [str(value) for value in
                          pq.read_table(root / 'data/trading_calendar.parquet')['trade_date'].to_pylist()],
    }
    atomic_json(directory / 'config.json', config)
    return directory


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--configure', action='store_true')
    parser.add_argument('--live', action='store_true')
    parser.add_argument('--install-client-entry', action='store_true')
    parser.add_argument('--individual-config', default='configs/config.json')
    args = parser.parse_args()
    if args.live and not args.configure:
        parser.error('--live requires --configure')
    print(prepare_runtime())
    if args.configure:
        print(prepare_config(live_enabled=args.live, individual_config=args.individual_config))
    if args.install_client_entry:
        from configs.env import QMT_ROOT_DIR
        broker_bin = next(Path(path) for path in QMT_ROOT_DIR if (Path(path) / 'pythonw.exe').exists())
        entry = broker_bin.parent / 'mpython/WBR_EXEC_BRIDGE.py'
        entry.write_text(bootstrap_source(Path(__file__).resolve().parents[1]), encoding='utf-8')
        print(entry)
    if args.configure or args.install_client_entry:
        print('Start/restart WBR_EXEC_BRIDGE from the logged-in QMT client to apply changes.')
