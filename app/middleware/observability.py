import time
from functools import wraps
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

REQ_TOTAL = Counter("req_total", "Total requests")
CACHE_HIT = Counter("cache_hit_total", "Cache hits")
CACHE_MISS = Counter("cache_miss_total", "Cache misses")
PRIMARY_CALL = Counter("primary_calls_total", "Primary model calls")
FALLBACK_CALL = Counter("fallback_calls_total", "Fallback model calls")
BLOCKED = Counter("blocked_total", "Blocked by safety gate")

LATENCY = Histogram("chat_latency_seconds", "Chat latency (seconds)")


def timed(fn):
    @wraps(fn)
    async def wrapper(*args, **kwargs):
        start = time.time()
        try:
            return await fn(*args, **kwargs)
        finally:
            LATENCY.observe(time.time() - start)
    return wrapper


def metrics_response():
    return generate_latest(), CONTENT_TYPE_LATEST
