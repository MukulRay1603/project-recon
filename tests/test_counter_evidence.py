"""Unit tests for counter-evidence retrieval. No network, no LLM."""
from src.state import Paper
from src.agents import counter_evidence as ce
from src.agents import critic


def _paper(pid, year=2020, title=None, score=0.5):
    return Paper(title=title or f"Paper {pid}", abstract=f"Abstract {pid}", year=year,
                 citation_count=10, paper_id=pid, hybrid_score=score)


def test_negate_claim_simple():
    assert ce.negate_claim("Dropout improves generalization") == "dropout does not improve generalization"
    assert ce.negate_claim("Does chain-of-thought outperform direct answers?") == \
        "chain-of-thought does not outperform direct answers"
    assert ce.negate_claim("What are the open problems in RLHF?") is None


def test_build_counter_queries_adds_critique_terms_and_caps():
    qs = ce.build_counter_queries(
        "Batch normalization improves training",
        ["foundational papers on batch normalization",
         "recent advances in batch normalization alternatives",
         "open challenges in normalization layers",
         "a fourth sub question about layer norm",
         "a fifth"],
    )
    assert len(qs) <= ce.MAX_QUERIES
    assert qs[0].startswith("batch normalization does not improve")
    assert any(term in q for q in qs[1:] for term in ce.CRITIQUE_TERMS)
    assert len(set(qs)) == len(qs)


def test_keywords_use_whole_word_stopwords():
    # "there"/"whether" must be dropped whole, not turned into "re"/"whe"
    q = "Is there consensus on whether KV cache compression degrades reasoning in LLMs?"
    assert ce._keywords(q) == "kv cache compression degrades reasoning"


def test_build_counter_queries_falls_back_to_original_query():
    qs = ce.build_counter_queries("KV cache compression", [])
    assert qs and "kv cache compression" in qs[0]


def test_node_excludes_supporting_papers_and_dedupes(monkeypatch):
    calls = []

    def fake_search(q, limit, decay_config):
        calls.append(q)
        return [_paper("support-1"), _paper("opp-1", 2022), _paper("opp-2", 2023, score=0.9)]

    monkeypatch.setattr("src.retriever_utils.search_semantic_scholar", fake_search)
    state = {"original_query": "x improves y", "sub_questions": ["dropout regularization", "batch normalization"],
             "retrieved_papers": [_paper("support-1")]}
    out = ce.counter_evidence_node(state)
    ids = [p.paper_id for p in out["opposing_papers"]]
    assert ids == ["opp-2", "opp-1"]           # sorted by score, deduped, supporting removed
    assert all(p.source.endswith("+counter") for p in out["opposing_papers"])
    assert out["retrieved_papers"] == state["retrieved_papers"]  # supporting set untouched
    assert len(calls) == 3


def test_node_survives_search_errors(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("network down")
    monkeypatch.setattr("src.retriever_utils.search_semantic_scholar", boom)
    out = ce.counter_evidence_node({"original_query": "q", "sub_questions": ["q"]})
    assert out["opposing_papers"] == []


def test_node_can_be_disabled(monkeypatch):
    monkeypatch.setenv("RECON_COUNTER_EVIDENCE", "0")
    monkeypatch.setattr("src.retriever_utils.search_semantic_scholar",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not search")))
    assert ce.counter_evidence_node({"original_query": "q"})["opposing_papers"] == []


def test_critic_pairs_supporting_with_opposing(monkeypatch):
    checked = []

    def fake_check(a, b):
        checked.append((a.paper_id, b.paper_id))
        return (b.paper_id == "opp-new", "B reports the effect does not replicate")

    monkeypatch.setattr(critic, "_check_contradiction", fake_check)
    supporting = [_paper("s1", 2021), _paper("s2", 2021), _paper("s3", 2021)]
    opposing = [_paper("opp-new", 2021), _paper("opp-old", 2015), _paper("opp-3", 2024)]
    found = critic._detect_counter_contradictions(supporting, opposing)

    # Only top-2 supporting x top-2 opposing; older opposing paper skipped;
    # same-year critique allowed (no 2-year gap rule here).
    assert checked == [("s1", "opp-new"), ("s2", "opp-new")]
    assert len(found) == 2 and found[0][1] == "Paper opp-new"


def test_critic_node_stale_logic_ignores_opposing_papers(monkeypatch):
    """Opposing papers must not change the staleness signal: reliability is
    scored on retrieved_papers only."""
    scored = {}

    class RS:
        def __init__(self, s): self.score = s

    def fake_score(papers, query, use_llm):
        scored["ids"] = [p.paper_id for p in papers]
        return {p.paper_id: RS(0.9) for p in papers}

    monkeypatch.setattr(critic, "score_papers", fake_score)
    monkeypatch.setattr(critic, "_check_contradiction", lambda a, b: (False, ""))
    supporting = [_paper(f"s{i}", 2024, score=0.6) for i in range(4)]
    state = {"retrieved_papers": supporting, "opposing_papers": [_paper("o1", 2025)],
             "retry_count": 0, "sub_questions": ["q"], "original_query": "q"}
    out = critic.critic_node(state)
    assert scored["ids"] == ["s0", "s1", "s2", "s3"]
    assert out["critic_verdict"] == "PASS"
