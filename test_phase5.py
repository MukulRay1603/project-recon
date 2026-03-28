import sys, logging
sys.path.insert(0, ".")
logging.basicConfig(level=logging.WARNING)

import requests, os
from dotenv import load_dotenv
load_dotenv()

from src.state import ResearchState, SessionContext
from src.agents.planner import planner_node
from src.agents.retriever import retriever_node
from src.memory import init_db

init_db()
print("=== Phase 5: Retriever Agent ===\n")

# Build initial state
state: ResearchState = {
    "original_query": "What is the current state of speculative decoding in LLMs?",
    "session_id": "test-session-005",
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
}

print("Step 1: Running planner...")
state = planner_node(state)
print(f"  Sub-questions: {len(state['sub_questions'])}")
for i, q in enumerate(state['sub_questions'], 1):
    print(f"    {i}. {q}")

# -------------------------------------------------------------------
# Raw S2 API debug — bypasses all our code
# -------------------------------------------------------------------
print("\n--- Raw S2 API debug ---")
s2_key = os.getenv("S2_API_KEY")
print(f"  Key present: {bool(s2_key)}")
print(f"  Key preview: {s2_key[:8] if s2_key else 'NONE'}...")

first_q = state['sub_questions'][0]
print(f"  Query: {first_q}")

headers = {"x-api-key": s2_key} if s2_key else {}
params = {
    "query": first_q,
    "limit": 3,
    "fields": "title,abstract,year,citationCount,paperId",
}

r = requests.get(
    "https://api.semanticscholar.org/graph/v1/paper/search",
    headers=headers,
    params=params,
    timeout=15,
)
print(f"  HTTP status: {r.status_code}")
data = r.json()
print(f"  Total in S2: {data.get('total', 0)}")
print(f"  Items in data[]: {len(data.get('data', []))}")

for i, p in enumerate(data.get("data", [])[:3]):
    has_abstract = "YES" if p.get("abstract") else "NONE"
    print(f"  [{i+1}] abstract:{has_abstract} | {p.get('title','?')[:55]}")

# -------------------------------------------------------------------
# Now test our wrapper directly
# -------------------------------------------------------------------
print("\n--- Our wrapper (use_cache=False) ---")
from src.retriever_utils import search_semantic_scholar
papers = search_semantic_scholar(first_q, limit=3, use_cache=False)
print(f"  Wrapper returned: {len(papers)} papers")
for p in papers:
    print(f"    [{p.hybrid_score:.3f}] {p.title[:55]} ({p.year})")

# -------------------------------------------------------------------
# Full retriever agent
# -------------------------------------------------------------------
print("\nStep 2: Running full retriever agent...")
state = retriever_node(state)

print(f"\n--- Final Results ---")
print(f"  Papers retrieved: {len(state['retrieved_papers'])}")
print(f"  Web results:      {len(state['web_results'])}")
print(f"  Citation edges:   {sum(len(v) for v in state['citation_graph'].values())}")

if state['retrieved_papers']:
    print(f"\n  Top 3 papers by hybrid score:")
    for p in state['retrieved_papers'][:3]:
        print(f"    [{p.hybrid_score:.3f}] {p.title[:65]} ({p.year})")

if state['web_results']:
    print(f"\n  Sample web results:")
    for r in state['web_results'][:3]:
        print(f"    [{r.source}] {r.title[:65]}")

print("\n✅ Phase 5 complete")