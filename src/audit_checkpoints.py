"""P0 audit: the sample sizes and counts that the main text, the Extended Data
and the SI state, recomputed from the data and the caches, against the values
of the previous draft (main(8).pdf). Writes reports/checkpoints.json and
reports/checkpoints.md. Run after run_all.py.
"""
import collections
import json
import os

import numpy as np

from common import ROOT, Dataset, load_cache, load_raw

DRAFT = {  # value stated in main(8).pdf / si(2).pdf
    'revisions in the release': 14591, 'pages in the release': 4579,
    'edits after cleaning': 13661, 'handles after cleaning': 3099,
    'handles of the population': 1201, 'edits of the population': 5929,
    'task pages': 679, 'edits on task pages': 3807, 'task families': 41,
    'arrivals': 2612, 'arrivals that create': 679, 'arrivals that append': 1933,
    'Fig. 2a candidates': 95342, 'Table 1 page candidates': 89855, 'Table 1 page decisions': 1933,
    'Fig. 2b records': 15776, 'Table 1 name records': 9728,
    'Fig. 2c first uses': 2679, 'Table 1 form first uses': 1052,
    'candidate conventions': 52, 'kept conventions': 16,
    'OECD uses': 325, 'OECD pages': 97, 'OECD handles': 125, 'padded uses after the first': 159,
    'chain decisions': 2400, 'Fig. 4a agents per point': 100, 'Fig. 4b agents per point (20 notes)': 125,
    'family conflicts (forms)': 212,
}


def main():
    raw = load_raw()
    d = load_cache('data.pkl')
    ds = Dataset()
    v = {}
    v['revisions in the release'] = len(d['revs'])
    v['pages in the release'] = len(d['pages'])
    v['edits after cleaning'] = len(raw['revs'])
    v['handles after cleaning'] = len({r['label'] for r in raw['revs']})
    v['handles of the population'] = len(ds.handles)
    v['edits of the population'] = len(ds.revs)
    task = [r for r in ds.revs if ds.is_task(r)]
    v['task pages'] = len({r['page_id'] for r in task})
    v['edits on task pages'] = len(task)
    v['task families'] = len({ds.family(r) for r in task})
    pages = load_cache('pages.pkl')
    cr = pages['creation'] if isinstance(pages.get('creation'), dict) else {}
    seen, arrivals, creates = collections.defaultdict(set), 0, 0
    for r in task:
        if r['page_id'] not in seen[r['label']]:
            seen[r['label']].add(r['page_id'])
            arrivals += 1
            creates += r['diff_base_reason'] == 'page_created'
    v['arrivals'], v['arrivals that create'], v['arrivals that append'] = arrivals, creates, arrivals - creates
    v['Fig. 2a candidates'] = len(pages['response'])
    idf = load_cache('identification.pkl')
    v['Table 1 page candidates'] = idf['pages']['n']
    v['Table 1 page decisions'] = idf['pages']['n_decisions']
    v['page appends left out (chosen page not a candidate)'] = idf['pages'].get('n_outside')
    names = load_cache('names.pkl')
    v['Fig. 2b records'] = int((~np.isnan(names['records']['view'])).sum())
    v['Table 1 name records'] = idf['names']['n']
    forms = load_cache('forms.pkl')
    v['Fig. 2c first uses'] = sum(len(x) for x in forms['by_class'].values())
    v['Table 1 form first uses'] = idf['forms']['n']
    v['candidate conventions'] = len(forms['rows'])
    v['kept conventions'] = len(forms['keep'])
    ans = load_cache('answers_record.pkl')
    v['OECD uses'], v['OECD pages'], v['OECD handles'] = ans['n_uses'], ans['n_pages'], ans['n_handles']
    v['padded uses after the first'] = ans['uses_before'] - 1
    lab = load_cache('si_lab.pkl')
    v['chain decisions'] = sum(c['agents'] for c in lab['chains'].values())
    v['Fig. 4a agents per point'] = sorted({a['oss']['valid'] for a in lab['agreement'].values()})
    v['Fig. 4b agents per point (20 notes)'] = sorted({c['valid'] for c in lab['mixed'][20]['curve'].values()})
    v['Fig. 4b countries'] = lab['mixed'][20]['countries']
    v['family conflicts (forms)'] = load_cache('si_forms.pkl')['family']['conflict']['pooled'][3]
    rows = []
    for k, val in v.items():
        old = DRAFT.get(k)
        same = (old is None) or (val == old) or (isinstance(val, list) and val == [old])
        rows.append(dict(quantity=k, found=val, draft=old, status='ok' if same else 'DIFFERENT'))
    out = os.path.join(ROOT, 'reports')
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, 'checkpoints.json'), 'w') as fh:
        json.dump(rows, fh, indent=1, default=lambda x: x.item() if hasattr(x, 'item') else str(x))
    with open(os.path.join(out, 'checkpoints.md'), 'w') as fh:
        fh.write('# Checkpoints, recomputed by `src/audit_checkpoints.py`\n\n| Quantity | Found | Previous draft | Status |\n|---|---:|---:|---|\n')
        for r in rows:
            fh.write(f'| {r["quantity"]} | {r["found"]} | {"" if r["draft"] is None else r["draft"]} | {r["status"]} |\n')
    for r in rows:
        print(f'{r["status"]:9} {r["quantity"]:48} {r["found"]}  (draft {r["draft"]})')


if __name__ == '__main__':
    main()
