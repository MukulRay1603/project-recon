"""Rate limiter, retry/backoff and parallel retrieval tests. No network."""
import threading
import time

import pytest
import requests

from src import retriever_utils as ru
from src.agents import retriever
from src.state import Paper, WebResult


class FakeClock:
    def __init__(self):
        self.t = 0.0
        self.sleeps = []

    def clock(self):
        return self.t

    def sleep(self, d):
        self.sleeps.append(round(d, 3))
        self.t += d


def test_rate_limiter_spaces_requests():
    c = FakeClock()
    lim = ru.RateLimiter(1.0, clock=c.clock, sleep=c.sleep)
    for _ in range(3):
        lim.wait()
    assert c.sleeps == [1.0, 1.0]  # first call free, then 1s apart


def test_rate_limiter_is_shared_across_threads():
    lim = ru.RateLimiter(0.05)
    starts = []
    lock = threading.Lock()

    def worker():
        lim.wait()
        with lock:
            starts.append(time.monotonic())

    threads = [threading.Thread(target=worker) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    starts.sort()
    gaps = [b - a for a, b in zip(starts, starts[1:])]
    assert all(g >= 0.045 for g in gaps), gaps


class Resp:
    def __init__(self, code, body=None, headers=None):
        self.status_code = code
        self._body = body or {}
        self.headers = headers or {}

    def json(self):
        return self._body


class Session:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def get(self, url, headers, params, timeout):
        self.calls += 1
        r = self.responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r


NOLIMIT = ru.RateLimiter(0)


def test_retry_on_429_then_success_honors_retry_after():
    sleeps = []
    s = Session([Resp(429, headers={"Retry-After": "5"}), Resp(200, {"data": [1]})])
    out = ru._get_with_retry("u", headers={}, params={}, limiter=NOLIMIT, sleep=sleeps.append, session=s)
    assert out == {"data": [1]}
    assert sleeps == [5.0]


def test_retry_on_timeout_with_exponential_backoff():
    sleeps = []
    s = Session([requests.exceptions.Timeout("t"), requests.exceptions.ConnectionError("c"),
                 Resp(503), Resp(200, {"ok": True})])
    out = ru._get_with_retry("u", headers={}, params={}, limiter=NOLIMIT, sleep=sleeps.append, session=s)
    assert out == {"ok": True}
    assert sleeps == [1.0, 2.0, 4.0]


def test_gives_up_after_max_retries():
    sleeps = []
    s = Session([Resp(500)] * 4)
    assert ru._get_with_retry("u", headers={}, params={}, limiter=NOLIMIT,
                              sleep=sleeps.append, session=s, max_retries=3) is None
    assert s.calls == 4


def test_no_retry_on_client_error():
    s = Session([Resp(400)])
    assert ru._get_with_retry("u", headers={}, params={}, limiter=NOLIMIT, sleep=lambda d: None, session=s) is None
    assert s.calls == 1


def _fake_sources(monkeypatch, delay=0.0):
    def fake_s2(query, limit, decay_config):
        time.sleep(delay)
        # Overlapping ids across questions to exercise dedup
        return [Paper(title=f"{query}-{k}", abstract="a", year=2023, citation_count=1,
                      paper_id=f"id-{k}" if k == 0 else f"{query}-{k}", hybrid_score=0.5)
                for k in range(2)]

    def fake_web(q, limit):
        time.sleep(delay)
        return [WebResult(url=f"http://{q}", snippet="s", title="t")]

    monkeypatch.setattr(retriever, "search_semantic_scholar", fake_s2)
    monkeypatch.setattr(retriever, "search_web", fake_web)
    monkeypatch.setattr("src.openalex_utils.search_openalex", lambda q, max_results: [])


def test_parallel_retrieval_matches_sequential(monkeypatch):
    _fake_sources(monkeypatch)
    state = {"sub_questions": ["alpha beta", "gamma delta", "epsilon zeta"]}
    monkeypatch.setattr(retriever, "RETRIEVAL_WORKERS", 1)
    seq = retriever.retriever_node(dict(state))
    monkeypatch.setattr(retriever, "RETRIEVAL_WORKERS", 3)
    par = retriever.retriever_node(dict(state))
    assert [p.paper_id for p in seq["retrieved_papers"]] == [p.paper_id for p in par["retrieved_papers"]]
    assert [r.url for r in seq["web_results"]] == [r.url for r in par["web_results"]]
    # id-0 is returned for every question but kept once
    assert sum(p.paper_id == "id-0" for p in par["retrieved_papers"]) == 1


def test_parallel_retrieval_is_faster_with_io_latency(monkeypatch):
    _fake_sources(monkeypatch, delay=0.1)
    state = {"sub_questions": ["a", "b", "c"]}
    monkeypatch.setattr(retriever, "RETRIEVAL_WORKERS", 1)
    t0 = time.perf_counter(); retriever.retriever_node(dict(state)); seq = time.perf_counter() - t0
    monkeypatch.setattr(retriever, "RETRIEVAL_WORKERS", 3)
    t0 = time.perf_counter(); retriever.retriever_node(dict(state)); par = time.perf_counter() - t0
    assert par < seq * 0.6, (seq, par)
