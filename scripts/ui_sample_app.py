"""
Launch the RECON UI with SAMPLE DATA, for UI review and screenshots only.

Nothing here is a real result: no LLM or API is called, the papers and
verdict are invented placeholders, and a banner says so on the page. The
normal app (python app.py) never imports this file.

    python scripts/ui_sample_app.py            # http://127.0.0.1:7862
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import CSS, build_demo
from src.state import Claim, Paper

BANNER = "SAMPLE DATA for UI review. These papers, verdicts and answers are placeholders, not RECON output."


def _p(pid, title, year, cites, score, abstract):
    return Paper(title=title, abstract=abstract, year=year, citation_count=cites, paper_id=pid,
                 authors=["A. Example", "B. Placeholder", "C. Sample"], hybrid_score=score)


SUPPORT = [
    _p("s1", "Sample paper: attention-score eviction for KV caches", 2023, 410, 0.71,
       "Placeholder abstract. Describes evicting cache entries with low accumulated attention."),
    _p("s2", "Sample paper: layer-wise KV cache merging", 2024, 95, 0.66,
       "Placeholder abstract. Merges adjacent cache entries to reduce memory."),
    _p("s3", "Sample paper: fixed-window cache truncation", 2020, 1200, 0.52,
       "Placeholder abstract. Keeps only a recent window of tokens."),
]
OPPOSING = [
    _p("o1", "Sample paper: revisiting cache eviction on long reasoning tasks", 2025, 30, 0.58,
       "Placeholder abstract. Reports accuracy drops for eviction methods on multi-step reasoning."),
]
RELIABILITY = {
    "s1": {"dominant_signal": "CURRENT"},
    "s2": {"dominant_signal": "CURRENT"},
    "s3": {"dominant_signal": "SUPERSEDED"},
}


def sample_stream(query, session_id, decay_config="linear", delay=0.0):
    st = {"original_query": query, "retry_count": 0}
    steps = [
        ("session_loader", {}),
        ("planner", {"sub_questions": ["q1", "q2", "q3"]}),
        ("retriever", {"retrieved_papers": SUPPORT, "web_results": [1, 2, 3]}),
        ("counter_evidence", {"opposing_papers": OPPOSING}),
        ("critic", {"critic_verdict": "STALE", "retry_count": 1, "paper_reliability_scores": RELIABILITY,
                    "critic_notes": "Sample note: mean reliability below threshold.",
                    "rewritten_questions": ["sample rewritten query about recent KV cache methods"]}),
        ("retry_retriever", {}),
        ("critic", {"critic_verdict": "PASS", "rewritten_questions": [],
                    "critic_notes": "Sample note: evidence passes all checks."}),
        ("synthesizer", {
            "synthesized_position": "**Sample answer.** This placeholder text shows how an answer with "
                                    "citations is laid out [Example et al., 2024].",
            "claim_confidences": [
                Claim("Sample claim with high confidence.", SUPPORT[0].title, 2023, "high"),
                Claim("Sample contested claim.", OPPOSING[0].title, 2025, "low", flagged=True),
            ],
            "export_md": "# Sample export\n"}),
    ]
    for node, delta in steps:
        time.sleep(delay)
        st.update(delta)
        yield node, dict(st)
    st["latency_ms"] = 4200
    yield "__done__", st


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "7862"))
    delay = float(os.environ.get("SAMPLE_DELAY", "0"))
    demo = build_demo(stream_fn=lambda **k: sample_stream(**k, delay=delay),
                      status={"llm": True, "s2_key": False}, banner=BANNER)
    demo.launch(server_port=port, css=CSS)
