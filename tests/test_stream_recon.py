"""stream_recon yields one update per node and the merged final state."""
import src.graph as g


def test_stream_recon_yields_each_node(monkeypatch, tmp_path):
    monkeypatch.setattr(g, "init_db", lambda: None)
    monkeypatch.setattr(g, "planner_node", lambda s: {**s, "sub_questions": ["a"]})
    monkeypatch.setattr(g, "retriever_node", lambda s: {**s, "retrieved_papers": []})
    monkeypatch.setattr(g, "counter_evidence_node", lambda s: {**s, "opposing_papers": []})
    monkeypatch.setattr(g, "critic_node", lambda s: {"critic_verdict": "PASS", "retry_count": 0})
    monkeypatch.setattr(g, "synthesizer_node", lambda s: {**s, "synthesized_position": "done"})
    events = list(g.stream_recon("q", "", "linear"))
    nodes = [n for n, _ in events]
    assert nodes == ["session_loader", "planner", "retriever", "counter_evidence", "critic", "synthesizer", "__done__"]
    final = events[-1][1]
    assert final["synthesized_position"] == "done" and final["critic_verdict"] == "PASS"
    assert final["latency_ms"] > 0
