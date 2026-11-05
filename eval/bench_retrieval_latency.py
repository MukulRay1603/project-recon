"""
Retrieval latency benchmark: sequential (RECON_RETRIEVAL_WORKERS=1) vs parallel.

Two modes:

  python eval/bench_retrieval_latency.py --simulate
      No network. Replaces S2 / web / OpenAlex with sleeps of a fixed,
      configurable latency, but uses the REAL rate limiter and the REAL
      retriever_node. Shows the effect of the code change in isolation.
      These are not production numbers.

  python eval/bench_retrieval_latency.py --live
      Real APIs (needs network; S2_API_KEY recommended). Disables the disk
      cache so every call hits the network. No LLM key needed: the retriever
      node makes no LLM calls.
"""
import argparse
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import retriever_utils as ru  # noqa: E402
from src.agents import retriever  # noqa: E402
from src.state import Paper, WebResult  # noqa: E402

SUB_QUESTIONS = [
    "foundational papers on KV cache compression for LLMs",
    "recent advances in KV cache quantization",
    "open challenges in long-context inference memory",
]


def _simulate(s2_latency, web_latency, oa_latency):
    def fake_s2(query, limit, decay_config):
        ru.S2_RATE_LIMITER.wait()  # same limiter the real call uses
        time.sleep(s2_latency)
        return [Paper(title=f"{query} {k}", abstract="a", year=2023, citation_count=5,
                      paper_id=f"{query}-{k}", hybrid_score=0.5) for k in range(4)]

    def fake_web(q, limit):
        time.sleep(web_latency)
        return [WebResult(url=f"http://{q}/{k}", snippet="s", title="t") for k in range(3)]

    def fake_oa(q, max_results):
        time.sleep(oa_latency)
        return []

    retriever.search_semantic_scholar = fake_s2
    retriever.search_web = fake_web
    import src.openalex_utils as oa
    oa.search_openalex = fake_oa


def _simulate_legacy(s2_latency, web_latency):
    """Emulates the pre-change retriever: fixed time.sleep(3) before every S2
    call, sequential sub-questions, and a 1s pause between sub-questions."""
    calls = {"web": 0}

    def fake_s2(query, limit, decay_config):
        time.sleep(3)
        time.sleep(s2_latency)
        return [Paper(title=f"{query} {k}", abstract="a", year=2023, citation_count=5,
                      paper_id=f"{query}-{k}", hybrid_score=0.5) for k in range(4)]

    def fake_web(q, limit):
        time.sleep(web_latency)
        calls["web"] += 1
        if calls["web"] % len(SUB_QUESTIONS) != 0:
            time.sleep(1)  # old inter-question pause
        return [WebResult(url=f"http://{q}/{k}", snippet="s", title="t") for k in range(3)]

    retriever.search_semantic_scholar = fake_s2
    retriever.search_web = fake_web


def _run(workers, live):
    retriever.RETRIEVAL_WORKERS = workers
    ru.S2_RATE_LIMITER._next_allowed = 0.0  # fresh limiter state per run
    state = {"sub_questions": SUB_QUESTIONS, "decay_config": "linear"}
    t0 = time.perf_counter()
    retriever.retriever_node(state)
    return time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--simulate", action="store_true")
    g.add_argument("--live", action="store_true")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--s2-interval", type=float, default=None,
                    help="override S2 min spacing (default: 1.05s with key, 3.0s without)")
    ap.add_argument("--s2-latency", type=float, default=0.8)
    ap.add_argument("--web-latency", type=float, default=1.5)
    ap.add_argument("--oa-latency", type=float, default=0.6)
    args = ap.parse_args()

    if args.s2_interval is not None:
        ru.S2_RATE_LIMITER.min_interval = args.s2_interval
    if args.simulate:
        _simulate_legacy(args.s2_latency, args.web_latency)
        import src.openalex_utils as oa
        oa.search_openalex = lambda q, max_results: []
        legacy = [_run(1, False) for _ in range(args.repeats)]
        print(f"legacy emulation (old code path): median {statistics.median(legacy):.2f}s")
        _simulate(args.s2_latency, args.web_latency, args.oa_latency)
    else:
        # Never read the disk cache, so every run hits the network.
        ru._cache_get = lambda key: None

    print(f"mode={'simulate' if args.simulate else 'live'} "
          f"s2_interval={ru.S2_RATE_LIMITER.min_interval}s sub_questions={len(SUB_QUESTIONS)}")
    for workers in (1, 3):
        times = [_run(workers, args.live) for _ in range(args.repeats)]
        print(f"workers={workers}: median {statistics.median(times):.2f}s  "
              f"runs={[round(t, 2) for t in times]}")


if __name__ == "__main__":
    main()
