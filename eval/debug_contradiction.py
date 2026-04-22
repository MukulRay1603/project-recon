import sys, os, json, time
sys.path.insert(0, '.')
from dotenv import load_dotenv
load_dotenv()

from src.agents.critic import _detect_contradictions, _check_contradiction
from src.agents.retriever import retriever_node
from src.agents.planner import planner_node
from src.state import ResearchState

# Load one Cat C question
questions = json.load(open('eval/questions.json'))
cat_c = [q for q in questions if q.get('category') == 'C']
test_q = cat_c[0]

print(f"Testing: [{test_q['id']}] {test_q['question']}")
print("=" * 70)

# Build minimal state and run planner + retriever
state = {
    "original_query": test_q["question"],
    "session_id": "debug-001",
    "session_context": None,
    "sub_questions": [],
    "retrieved_papers": [],
    "citation_graph": {},
    "web_results": [],
    "critic_verdict": "",
    "critic_notes": "",
    "rewritten_questions": [],
    "retry_count": 0,
    "synthesized_position": "",
    "claim_confidences": [],
    "session_update": None,
    "export_md": "",
    "decay_config": "linear",
    "calibration_bin": "",
    "latency_ms": 0.0,
    "paper_reliability_scores": {},
    "reliability_dominant_signals": {},
    "is_followup": False,
    "previous_papers": [],
    "previous_verdict": "",
    "previous_claims": [],
    "turn_number": 1,
    "explore_mode": False,
    "explorer_findings": [],
}

print("Step 1: Running planner...")
state = planner_node(state)
print(f"Sub-questions: {state.get('sub_questions', [])}")

print("\nStep 2: Running retriever...")
state = retriever_node(state)
papers = state.get("retrieved_papers", [])
print(f"Papers retrieved: {len(papers)}")
print("\nTop 6 papers:")
for i, p in enumerate(papers[:6]):
    print(f"  [{i+1}] {p.year} | citations={p.citation_count} | score={p.hybrid_score:.3f}")
    print(f"       {p.title[:80]}")

print("\nStep 3: Running _detect_contradictions on top 4 papers...")
top4 = papers[:4]
print("Paper pairs being checked (must have >= 2 year gap):")
checked = 0
for i, pa in enumerate(top4):
    for pb in top4[i+1:]:
        gap = abs((pa.year or 0) - (pb.year or 0))
        print(f"  Pair ({i},{i+1}): {pa.year} vs {pb.year} -- gap={gap}yrs -- {'WILL CHECK' if gap >= 2 else 'SKIPPED (gap < 2)'}")
        if gap >= 2:
            checked += 1
            older = pa if (pa.year or 0) < (pb.year or 0) else pb
            newer = pa if (pa.year or 0) > (pb.year or 0) else pb
            print(f"    older: [{older.year}] {older.title[:60]}")
            print(f"    newer: [{newer.year}] {newer.title[:60]}")
            print(f"    older abstract: {(older.abstract or 'NONE')[:200]}")
            print(f"    newer abstract: {(newer.abstract or 'NONE')[:200]}")
            result, reason = _check_contradiction(older, newer)
            print(f"    LLM result: contradicts={result}, reason={reason}")
            print()

print(f"\nTotal pairs actually LLM-checked: {checked}")
contradictions = _detect_contradictions(papers)
print(f"Final _detect_contradictions output: {len(contradictions)} contradictions found")
