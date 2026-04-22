# Archived Eval Files

## patch_contradiction.py
Moved here on 2026-04-22 as part of Phase 1 integrity fix.

This file implements an eval-time contradiction scorer using debate-signal
keyword heuristics + LLM judge. It includes a `position_acknowledges_debate`
boost that can override an LLM "not contested" verdict when keywords like
"debate", "camp", "contested" appear in the synthesized position.

STATUS: ARCHIVED — DO NOT USE FOR REPORTED METRICS

The honest contradiction catch rate for RECON v1 is 0%. This file must not
be used to generate any numbers reported in a paper. It is preserved here
for reference only.

The root cause of the 0% contradiction rate is Bug 1 (STALE fires before
CONTRADICTED in critic_node), which is being fixed in Phase 1.2.
