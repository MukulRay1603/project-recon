import sys, logging
sys.path.insert(0, ".")
logging.basicConfig(level=logging.WARNING)

from src.state import ResearchState
from src.agents.planner import planner_node
from src.agents.retriever import retriever_node
from src.agents.critic import critic_node
from src.agents.synthesizer import synthesizer_node
from src.memory import init_db, load_session

init_db()
print("=== Phase 7: Synthesizer Agent ===\n")

state: ResearchState = {
    "original_query": "What is the current state of speculative decoding in LLMs?",
    "session_id": "test-session-007",
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
print(f"  {len(state['sub_questions'])} sub-questions generated")

print("Step 2: Retriever...")
state = retriever_node(state)
print(f"  {len(state['retrieved_papers'])} papers retrieved")

print("Step 3: Critic...")
state = critic_node(state)
print(f"  Verdict: {state['critic_verdict']}")

print("Step 4: Synthesizer (takes ~15s)...")
state = synthesizer_node(state)

print(f"\n--- Synthesizer Results ---")
print(f"  Position length: {len(state['synthesized_position'])} chars")
print(f"  Claims extracted: {len(state['claim_confidences'])}")
print(f"  Export MD length: {len(state['export_md'])} chars")

print(f"\n--- Position preview (first 500 chars) ---")
print(state['synthesized_position'][:500])

print(f"\n--- Claims ---")
for c in state['claim_confidences']:
    flag = " ⚠️" if c.flagged else ""
    print(f"  [{c.confidence.upper()}] {c.text[:70]}{flag}")
    print(f"    Source: {c.source_title[:50]} ({c.source_year})")

print(f"\n--- Session check ---")
session_ctx = load_session("test-session-007")
print(f"  Prior positions in DB: {len(session_ctx.prior_positions)}")
print(f"  Prior queries in DB: {len(session_ctx.prior_queries)}")

print(f"\n--- Export MD preview ---")
print(state['export_md'][:300])

print("\n✅ Phase 7 complete")