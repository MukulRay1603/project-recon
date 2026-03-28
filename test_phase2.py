import sys
sys.path.insert(0, ".")

from src.state import ResearchState, Paper, Claim, Verdict, SessionContext
from src.memory import init_db, load_session, save_turn, export_session_md, delete_session
from src.state import SessionUpdate
import uuid

print("=== Phase 2: State + Memory ===")

# Test 1: State dataclasses
p = Paper(
    title="Test Paper",
    abstract="This is a test abstract.",
    year=2024,
    citation_count=100,
    paper_id="abc123"
)
print(f"✓ Paper dataclass: {p.title} ({p.year})")

c = Claim(text="Test claim", source_title="Test Paper", source_year=2024, confidence="high")
print(f"✓ Claim dataclass: [{c.confidence}] {c.text}")

print(f"✓ Verdict constants: {Verdict.PASS} / {Verdict.STALE} / {Verdict.CONTRADICTED}")

# Test 2: SQLite memory
init_db()
print("✓ Database initialized")

session_id = str(uuid.uuid4())

# Load empty session
ctx = load_session(session_id)
print(f"✓ Empty session loaded: {len(ctx.prior_positions)} prior positions")

# Save a turn
update = SessionUpdate(
    query="What is the state of KV cache compression?",
    position="KV cache compression has advanced significantly with methods like H2O and StreamingLLM.",
    claim_confidences=[
        Claim("H2O reduces KV cache size by 20x", "H2O Paper", 2023, "high"),
        Claim("StreamingLLM enables infinite context", "StreamingLLM", 2023, "medium"),
    ],
    contradictions_found=["StreamingLLM contradicted by later infinite attention work (2024)"]
)
save_turn(session_id, update)
print("✓ Turn saved to database")

# Reload and verify
ctx2 = load_session(session_id)
print(f"✓ Session reloaded: {len(ctx2.prior_positions)} prior position(s)")
print(f"  Prior query: {ctx2.prior_queries[0][:60]}...")

# Export markdown
md = export_session_md(session_id)
print(f"✓ Markdown export: {len(md)} characters")
print(f"  Preview: {md[:120].strip()}")

# Cleanup
delete_session(session_id)
print("✓ Session deleted")

print("\n✅ Phase 2 complete")