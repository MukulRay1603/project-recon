import csv, os, sys

path = 'eval/results/recon_linear_v2.csv'

if not os.path.exists(path):
    print('No results file yet — eval may still be starting up')
    sys.exit(0)

with open(path, newline='', encoding='utf-8') as f:
    rows = list(csv.DictReader(f))

if not rows:
    print('File exists but is empty — eval just started')
    sys.exit(0)

verdicts = [r.get('critic_verdict', '') for r in rows]
correct = [r for r in rows if r.get('position_accuracy', '').strip().upper() == 'MATCH']
contra = [r for r in rows if r.get('critic_verdict') == 'CONTRADICTED']
stale = [r for r in rows if r.get('critic_verdict') == 'STALE']

print(f"Progress: {len(rows)} / 130 questions done")
print(f"Verdict distribution: { {v: verdicts.count(v) for v in sorted(set(verdicts))} }")
print(f"CONTRADICTED: {len(contra)} ({len(contra)/len(rows)*100:.1f}%)")
print(f"STALE: {len(stale)} ({len(stale)/len(rows)*100:.1f}%)")
print(f"Position accuracy: {len(correct)}/{len(rows)} = {len(correct)/len(rows)*100:.1f}%")

if contra:
    print(f"\nFirst CONTRADICTED hits:")
    for r in contra[:5]:
        print(f"  [{r.get('question_id','?')}] {r.get('question','')[:70]}")
