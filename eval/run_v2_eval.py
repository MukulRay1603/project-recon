import sys, os, json
sys.path.insert(0, '.')
from dotenv import load_dotenv
load_dotenv()
from eval.run_eval import run_architecture, run_recon_full

questions = json.load(open('eval/questions.json'))
gt = {e['id']: e for e in json.load(open('eval/ground_truth.json'))}

OUTPUT = 'eval/results/recon_linear_v2_full.csv'
print(f"Running v2 recon_linear -- {len(questions)} questions")
print(f"Output: {OUTPUT}")
print(f"Key changes vs v1: edge reliability scorer, OpenAlex augmentation, trust summary")
print("=" * 60)

run_architecture(
    arch_name='recon_linear_v2_full',
    decay_config='linear',
    runner_fn=run_recon_full,
    questions=questions,
    gt_map=gt,
    output_path=OUTPUT,
)

import csv
rows = list(csv.DictReader(open(OUTPUT, encoding='utf-8')))
verdicts = [r.get('critic_verdict', '') for r in rows]
match_rows = [r for r in rows if r.get('position_accuracy', '').upper() == 'MATCH']
stale_rows = [r for r in rows if r.get('critic_verdict') == 'STALE']
contra_rows = [r for r in rows if r.get('critic_verdict') == 'CONTRADICTED']
cat_b = [r for r in rows if r.get('category') == 'B']
cat_b_stale = [r for r in cat_b if r.get('staleness_caught', '') == '1']

print(f"\n=== v2 FINAL RESULTS ({len(rows)}/130) ===")
print(f"Verdict distribution: { {v: verdicts.count(v) for v in sorted(set(verdicts))} }")
print(f"STALE: {len(stale_rows)}/130 | CONTRADICTED: {len(contra_rows)}/130")
print(f"Staleness catch rate (Cat B): {len(cat_b_stale)}/{len(cat_b)} = {len(cat_b_stale)/max(len(cat_b),1)*100:.1f}%")
print(f"Position accuracy (MATCH): {len(match_rows)}/130 = {len(match_rows)/130*100:.1f}%")
print(f"\n--- vs baselines ---")
print(f"v1 staleness catch rate: 52.0% | v2: {len(cat_b_stale)/max(len(cat_b),1)*100:.1f}%")
print(f"v1 position accuracy:    43.9% | v2: {len(match_rows)/130*100:.1f}%")
