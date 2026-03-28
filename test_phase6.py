import sys, logging
sys.path.insert(0, ".")
logging.basicConfig(level=logging.WARNING)

from src.state import ResearchState, Verdict
from src.agents.planner import planner_node
from src.agents.retriever import retriever_node
from src.agents.critic import critic_node
from src.memory import init_db

init_db()
print("=== Phase 6: Critic Agent ===\n")

state: ResearchState = {
    "original_query": "What is the current state of speculative decoding in LLMs?",
    "session_id": "test-session-006",
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

print("Step 1: Planner...")
state = planner_node(state)
for i, q in enumerate(state['sub_questions'], 1):
    print(f"  {i}. {q}")

print("\nStep 2: Retriever...")
state = retriever_node(state)
print(f"  Papers: {len(state['retrieved_papers'])}")
if state['retrieved_papers']:
    print(f"  Top paper: {state['retrieved_papers'][0].title[:60]}")
    print(f"  Score range: {state['retrieved_papers'][-1].hybrid_score:.3f} - {state['retrieved_papers'][0].hybrid_score:.3f}")

print("\nStep 3: Critic...")
state = critic_node(state)

print(f"\n--- Critic Results ---")
print(f"  Verdict:         {state['critic_verdict']}")
print(f"  Notes:           {state['critic_notes']}")
print(f"  Calibration bin: {state['calibration_bin']}")
print(f"  Retry count:     {state['retry_count']}")

if state['rewritten_questions']:
    print(f"  Rewritten questions:")
    for q in state['rewritten_questions']:
        print(f"    - {q}")

# Test INSUFFICIENT path
print("\n--- Testing INSUFFICIENT path ---")
state_insufficient = {
    **state,
    "retrieved_papers": [],
    "retry_count": 0,
    "sub_questions": ["speculative decoding methods"],
}
result = critic_node(state_insufficient)
print(f"  Verdict: {result['critic_verdict']} (expected: INSUFFICIENT)")
print(f"  Notes:   {result['critic_notes']}")

# Test FORCED_PASS path
print("\n--- Testing FORCED_PASS path ---")
state_forced = {**state, "retry_count": 2}
result2 = critic_node(state_forced)
print(f"  Verdict: {result2['critic_verdict']} (expected: FORCED_PASS)")

print("\n✅ Phase 6 complete")