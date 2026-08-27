"""Rate-limited Eastmoney client shared by research pre-download entries."""

from __future__ import annotations

import random
import time

import requests

URL = "https://datacenter-web.eastmoney.com/api/data/v1/get"
SESSION = requests.Session()
SESSION.headers.update(
    {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "Chrome/117.0.0.0 Safari/537.36"
        ),
        "Referer": "https://data.eastmoney.com/",
    }
)
LAST_CALL = [0.0]


def em_get(params: dict[str, str]) -> requests.Response:
    wait = 1.0 - (time.time() - LAST_CALL[0])
    if wait > 0.0:
        time.sleep(wait + random.uniform(0.1, 0.5))
    try:
        return SESSION.get(URL, params=params, timeout=20)
    finally:
        LAST_CALL[0] = time.time()
