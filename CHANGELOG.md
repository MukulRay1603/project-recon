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

### Phase 0 — Branch Setup
- Created v2-edge-reliability branch from main
- Added CHANGELOG.md
- v1 remains on main branch, deployed to HF Spaces
- All v2 development happens on this branch
