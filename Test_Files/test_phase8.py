import sys, logging
sys.path.insert(0, ".")
logging.basicConfig(level=logging.WARNING)

import uuid
from src.graph import run_recon

print("=== Phase 8: Full LangGraph Pipeline ===\n")

session_id = str(uuid.uuid4())
print(f"Session ID: {session_id}")
print("Running full pipeline (takes ~60s)...\n")

result = run_recon(
    query="What is the current state of speculative decoding in LLMs?",
    session_id=session_id,
    decay_config="linear",
)

print(f"=== Pipeline Results ===")
print(f"  Verdict:        {result['critic_verdict']}")
print(f"  Calibration:    {result['calibration_bin']}")
print(f"  Papers used:    {len(result['retrieved_papers'])}")
print(f"  Claims:         {len(result['claim_confidences'])}")
print(f"  Latency:        {result['latency_ms']:.0f}ms")
print(f"  Retry count:    {result['retry_count']}")

print(f"\n=== Position preview ===")
print(result['synthesized_position'][:600])

print(f"\n=== Claims ===")
for c in result['claim_confidences']:
    flag = " ⚠️" if c.flagged else ""
    print(f"  [{c.confidence.upper()}] {c.text[:65]}{flag}")

print(f"\n=== Multi-turn test ===")
print("Running second query in same session...")
result2 = run_recon(
    query="What are the limitations of speculative decoding?",
    session_id=session_id,
    decay_config="linear",
)
print(f"  Verdict: {result2['critic_verdict']}")
print(f"  Claims:  {len(result2['claim_confidences'])}")
print(f"  Session context used: {len(result2.get('session_context').prior_positions if result2.get('session_context') else [])} prior positions")

print("\n✅ Phase 8 complete")