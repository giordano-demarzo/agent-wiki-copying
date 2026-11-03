"""Re-score the stored replies whose JSON did not parse. The answer experiments (data/answers/replicate_*.jsonl):
replicate.parse, the answer field read with a pattern. The copying experiment (data/replies_*.jsonl): run.parse, the
first complete JSON object of the reply. Files are rewritten in place; records that parsed are unchanged.

    python rescore.py                                  # every file
    python rescore.py replicate_ministral-3-14b.jsonl  # only these (not a file still being written)
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ONLY = sys.argv[1:]
sys.argv = [sys.argv[0], '--model', 'x']
import replicate as R  # noqa: E402

for path in sorted((ROOT / 'data' / 'answers').glob('replicate_*.jsonl')):
    if ONLY and path.name not in ONLY:
        continue
    rows, fixed = [json.loads(l) for l in open(path)], 0
    for r in rows:
        if r.get('raw') and not r.get('error') and r.get('answer') is None:
            p = R.parse(r['raw'])
            if p.get('answer') is not None:
                v = R.fp.value(p['answer'])
                right, wrong = R.cr.COUNTRIES[r['country']][:2]
                r.update(answer=p['answer'], note=str(p.get('wiki_note') or ''), is_right=v == float(right),
                         is_wrong=v == float(wrong), parsed_by='pattern')
                fixed += 1
    with open(path, 'w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
    print(f'{path.name}: {fixed} replies re-scored, {sum(1 for r in rows if r.get("raw") and r.get("answer") is None and not r.get("error"))} still without an answer')

import run  # noqa: E402

for path in sorted((ROOT / 'data').glob('replies_*.jsonl')):
    if path.name.endswith('_failed.jsonl') or (ONLY and path.name not in ONLY):
        continue
    rows, fixed = [json.loads(l) for l in open(path)], 0
    for r in rows:
        if r['reply'] is None and r.get('raw'):
            p = run.parse(r['raw'])
            if p:
                r.update(reply=p, parsed_by='first object')
                fixed += 1
    with open(path, 'w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
    print(f'{path.name}: {fixed} replies re-parsed, {sum(r["reply"] is None for r in rows)} still unparsed')
