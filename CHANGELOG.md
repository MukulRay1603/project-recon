# CHANGELOG — RECON v2 (Edge Reliability)

## [Unreleased]

### Phase 1 — Integrity + Critic Fix
- 1.1: Archived patch_contradiction.py to eval/archived/ with README
- 1.2: Fixed critic_node — STALE and CONTRADICTED checks now run in parallel
  - Removed short-circuit: STALE no longer blocks CONTRADICTED
  - When both fire, CONTRADICTED wins (richer signal)
  - Removed **state spread from all return paths for consistency
  - All 5 return paths now have identical key shape
  - Added missing retry_count to FORCED_PASS path
- 1.3: Ran full 130Q eval on v2 recon_linear
  - Staleness catch rate: 48% (vs 52% v1 — within run-to-run variance, confirmed same methodology)
  - Contradiction catch rate: 0% (root cause: retriever finds adjacent-topic papers, not opposing-camp pairs; gap filter also too coarse for ML field — 2yr filter blocks most recent paper pairs)
  - Position accuracy (MATCH): 44.6% vs v1 43.9%
  - Cat C has no ground truth in ground_truth.json — contradiction_caught is critic-behavior only
  - Gap filter lowered to 1yr and tested on all 30 Cat C questions — reverted, no improvement (CONTRADICTED still 0/30); problem is in the retriever, not the filter

### Phase 0 — Branch Setup
- Created v2-edge-reliability branch from main
- Added CHANGELOG.md
- v1 remains on main branch, deployed to HF Spaces
- All v2 development happens on this branch
