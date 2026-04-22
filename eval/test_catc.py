import sys, os, json
sys.path.insert(0, '.')
from dotenv import load_dotenv
load_dotenv()
from eval.run_eval import run_architecture, run_recon_full

questions = json.load(open('eval/questions.json'))
ground_truth_list = json.load(open('eval/ground_truth.json'))
gt = {e['id']: e for e in ground_truth_list}

cat_c = [q for q in questions if q.get('category') == 'C']
print(f"Running {len(cat_c)} Cat C questions with gap filter = 1 year")

run_architecture(
    arch_name='recon_catc_gap1yr',
    decay_config='linear',
    runner_fn=run_recon_full,
    questions=cat_c,
    gt_map=gt,
    output_path='eval/results/recon_catc_gap1yr.csv',
)

import csv
rows = list(csv.DictReader(open('eval/results/recon_catc_gap1yr.csv', encoding='utf-8')))
verdicts = [r.get('critic_verdict', '') for r in rows]
contra = [r for r in rows if r.get('critic_verdict') == 'CONTRADICTED']
print(f"\nCat C results ({len(rows)} questions, gap filter = 1yr):")
print(f"Verdict distribution: { {v: verdicts.count(v) for v in sorted(set(verdicts))} }")
print(f"CONTRADICTED: {len(contra)}/{len(rows)} = {len(contra)/max(len(rows),1)*100:.1f}%")
if contra:
    print("CONTRADICTED hits:")
    for r in contra:
        print(f"  [{r.get('question_id','?')}] {r.get('question','')[:70]}")
