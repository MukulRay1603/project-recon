import sys
sys.path.insert(0, ".")
from src.graph import run_recon
import uuid

r = run_recon("What are the open problems in RLHF for LLMs?", str(uuid.uuid4()))
print(f"Verdict:  {r['critic_verdict']}")
print(f"Papers:   {len(r['retrieved_papers'])}")
print(f"Claims:   {len(r['claim_confidences'])}")
print(f"Latency:  {r['latency_ms']:.0f}ms")
print(f"Retry:    {r['retry_count']}")
print("Top 3 papers:")
for p in r['retrieved_papers'][:3]:
    print(f"  [{p.hybrid_score:.3f}] {p.title[:60]} ({p.year})")