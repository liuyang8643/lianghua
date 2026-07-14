from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent

class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

if __name__ == "__main__":
    address = ("127.0.0.1", 8765)
    print(f"Coverage dashboard: http://{address[0]}:{address[1]}")
    ThreadingHTTPServer(address, Handler).serve_forever()
