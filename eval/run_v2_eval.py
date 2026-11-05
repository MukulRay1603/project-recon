import sys, os, json
sys.path.insert(0, '.')
from dotenv import load_dotenv
load_dotenv()
from eval.run_eval import run_architecture, run_recon_full

questions = json.load(open('eval/questions.json'))
gt = {e['id']: e for e in json.load(open('eval/ground_truth.json'))}

# Override to keep ablation runs separate, e.g.
#   RECON_COUNTER_EVIDENCE=0 RECON_EVAL_OUTPUT=eval/results/v2_no_counter.csv python eval/run_v2_eval.py
OUTPUT = os.environ.get('RECON_EVAL_OUTPUT', 'eval/results/recon_linear_v2_full.csv')
print(f"Counter-evidence retrieval: {'OFF' if os.environ.get('RECON_COUNTER_EVIDENCE', '1') in ('0', 'false', 'no') else 'ON'}")
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

# Reporting only (no scoring change). staleness_caught() counts STALE *or*
# CONTRADICTED, so after the counter-evidence change report them separately
# to see where any movement in the headline number comes from.
cat_a = [r for r in rows if r.get('category') == 'A']
cat_c = [r for r in rows if r.get('category') == 'C']
cat_b_stale_only = [r for r in cat_b if r.get('critic_verdict') == 'STALE']
cat_b_contra = [r for r in cat_b if r.get('critic_verdict') == 'CONTRADICTED']
cat_c_contra = [r for r in cat_c if r.get('critic_verdict') == 'CONTRADICTED']
cat_a_flagged = [r for r in cat_a if r.get('critic_verdict') in ('STALE', 'CONTRADICTED')]
print(f"\n--- breakdown ---")
print(f"Cat B caught via STALE: {len(cat_b_stale_only)} | via CONTRADICTED: {len(cat_b_contra)}")
print(f"Contradiction catch rate (Cat C): {len(cat_c_contra)}/{len(cat_c)} = {len(cat_c_contra)/max(len(cat_c),1)*100:.1f}%")
print(f"Cat A flagged STALE/CONTRADICTED (false-positive proxy): {len(cat_a_flagged)}/{len(cat_a)}")
