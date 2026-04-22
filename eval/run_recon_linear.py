import sys, os, json
sys.path.insert(0, '.')
from dotenv import load_dotenv
load_dotenv()

from eval.run_eval import run_architecture, run_recon_full

questions = json.load(open('eval/questions.json'))
ground_truth_list = json.load(open('eval/ground_truth.json'))
gt = {e['id']: e for e in ground_truth_list}

OUTPUT = 'eval/results/recon_linear_v2.csv'

print(f"Running recon_linear v2 -- {len(questions)} questions")
print(f"Output: {OUTPUT}")
print(f"Crash-resume: enabled (will skip already-done question IDs)")
print("=" * 60)

run_architecture(
    arch_name='recon_linear_v2',
    decay_config='linear',
    runner_fn=run_recon_full,
    questions=questions,
    gt_map=gt,
    output_path=OUTPUT,
)

print("\nRun complete. Computing summary...")
import csv
with open(OUTPUT) as f:
    rows = list(csv.DictReader(f))

verdicts = [r.get('critic_verdict', '') for r in rows]
verdict_counts = {v: verdicts.count(v) for v in set(verdicts)}

cat_b = [r for r in rows if r.get('category') == 'B']
cat_c = [r for r in rows if r.get('category') == 'C']

stale_rows = [r for r in rows if r.get('critic_verdict', '') == 'STALE']
contra_rows = [r for r in rows if r.get('critic_verdict', '') == 'CONTRADICTED']

acc_rows = [r for r in rows if r.get('position_accuracy') in ('MATCH', 'PARTIAL', 'MISMATCH')]
match_rows = [r for r in acc_rows if r.get('position_accuracy') == 'MATCH']

staleness_caught = sum(
    1 for r in cat_b
    if r.get('critic_verdict', '') in ('STALE', 'CONTRADICTED')
)
contradiction_caught = sum(
    1 for r in cat_c
    if r.get('critic_verdict', '') == 'CONTRADICTED'
)

print(f"\nTotal rows: {len(rows)}")
print(f"Verdict distribution: {verdict_counts}")
print(f"\nSTALE: {len(stale_rows)} | CONTRADICTED: {len(contra_rows)}")
print(f"\nCat B (staleness, n={len(cat_b)}): staleness_caught={staleness_caught} ({staleness_caught/max(len(cat_b),1)*100:.1f}%)")
print(f"Cat C (contradiction, n={len(cat_c)}): contradiction_caught={contradiction_caught} ({contradiction_caught/max(len(cat_c),1)*100:.1f}%)")
print(f"\nPosition accuracy (MATCH): {len(match_rows)}/{len(acc_rows)} = {len(match_rows)/max(len(acc_rows),1)*100:.1f}%")
print(f"\nv1 baseline comparison:")
print(f"  Contradiction catch rate: v1=0.0%  v2={contradiction_caught/max(len(cat_c),1)*100:.1f}%")
print(f"  Position accuracy:        v1=43.9% v2={len(match_rows)/max(len(acc_rows),1)*100:.1f}%")
