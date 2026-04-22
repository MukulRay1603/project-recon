import sys, os, json
sys.path.insert(0, '.')
from dotenv import load_dotenv
load_dotenv()

from eval.run_eval import run_architecture, run_recon_full

questions = json.load(open('eval/questions.json'))
ground_truth_list = json.load(open('eval/ground_truth.json'))
gt = {e['id']: e for e in ground_truth_list}

smoke_qs = questions[:5]

print(f"Running smoke test: {len(smoke_qs)} questions, recon_linear architecture")
print("=" * 60)

run_architecture(
    arch_name='recon_linear_smoke',
    decay_config='linear',
    runner_fn=run_recon_full,
    questions=smoke_qs,
    gt_map=gt,
    output_path='eval/results/recon_linear_smoke.csv',
)

print("Smoke test complete. Reading results...")
import csv
with open('eval/results/recon_linear_smoke.csv') as f:
    rows = list(csv.DictReader(f))

print(f"\n{'ID':<8} {'VERDICT':<15} {'ACCURACY':<10} QUESTION[:60]")
print("-" * 80)
for r in rows:
    print(f"{r.get('question_id','?'):<8} {r.get('critic_verdict','?'):<15} {r.get('position_accuracy','?'):<10} {r.get('question','')[:60]}")

verdicts = [r.get('critic_verdict', '') for r in rows]
print(f"\nVerdict counts: { {v: verdicts.count(v) for v in set(verdicts)} }")
