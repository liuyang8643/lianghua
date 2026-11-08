# coding: utf-8
import sys
ROOT = 'D:/coding/WBR-live'
RUNTIME = ROOT + '/data/qmt_bridge_runtime'
sys.path[:0] = [RUNTIME + '/stdlib', RUNTIME,
               RUNTIME + '/site-packages', ROOT]
# encodings is imported before this entry point; extend its package path too.
import encodings
encodings.__path__.append(RUNTIME + '/stdlib/encodings')
SOURCE = ROOT + '/' + 'trading/qmt_bridge.py'
import hashlib
def execute_source():
    with open(SOURCE, 'r', encoding='utf-8') as stream:
        code = stream.read()
    source_hash = hashlib.sha256(code.encode('utf-8')).hexdigest()
    exec(compile(code, SOURCE, 'exec'), {'__name__': '__main__', '__file__': SOURCE,
                                      '__source_sha256__': source_hash})
if 'trading/qmt_bridge.py' == 'trading/qmt_bridge.py':
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
