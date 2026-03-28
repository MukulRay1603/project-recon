import sys, logging
sys.path.insert(0, ".")
logging.basicConfig(level=logging.WARNING)

from src.state import ResearchState, SessionContext
from src.agents.planner import planner_node
from src.memory import init_db

init_db()
print("=== Phase 4: Planner Agent ===\n")

# Test 1: Fresh query, no session context
state: ResearchState = {
    "original_query": "What is the current state of speculative decoding in LLMs?",
    "session_id": "test-session-001",
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

result = planner_node(state)
print("Query: What is the current state of speculative decoding in LLMs?")
print(f"Sub-questions generated: {len(result['sub_questions'])}")
for i, q in enumerate(result['sub_questions'], 1):
    print(f"  {i}. {q}")

# Test 2: Query with session context (should avoid repeating)
print("\n--- With session context ---")
ctx = SessionContext(
    prior_queries=["What is the current state of speculative decoding in LLMs?"],
    prior_positions=["Speculative decoding reduces latency by 2-3x..."],
    flagged_contradictions=[]
)
state2 = {**state,
    "original_query": "What are the limitations of speculative decoding?",
    "session_context": ctx,
}
result2 = planner_node(state2)
print("Query: What are the limitations of speculative decoding?")
print(f"Sub-questions generated: {len(result2['sub_questions'])}")
for i, q in enumerate(result2['sub_questions'], 1):
    print(f"  {i}. {q}")

print("\n✅ Phase 4 complete")