"""One throttled request implementation with caller-owned request state."""
import random
import time


def throttled_get(session, *, url, params, last_request_time, minimum_interval, maximum_jitter, timeout=15):
    target_interval = minimum_interval + random.uniform(0.0, maximum_jitter)
    elapsed = time.monotonic() - last_request_time
    if elapsed < target_interval:
        time.sleep(target_interval - elapsed)
    response = session.get(url, params=params, timeout=timeout)
    return response, time.monotonic()
