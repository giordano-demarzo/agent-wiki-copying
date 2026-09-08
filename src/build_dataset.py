"""Parse the collusion.wiki release into cache/data.pkl.

Reads data/{labels,pages,revisions,events}.jsonl and stores, for every
revision, the lines that the edit added, which is what we treat as the
message written by that edit.

    python src/build_dataset.py
"""
import collections
import json
import os
import pickle
import sys

from common import DATA, cache_path

REQUIRED = ['labels.jsonl', 'pages.jsonl', 'revisions.jsonl', 'events.jsonl']


def added_lines(rev):
    """The lines this edit added, from the hunks of its diff against its base."""
    lines = rev['body'].split('\n')
    if rev['diff_base_reason'] == 'page_created' or not rev.get('hunks'):
        return '\n'.join(lines)
    added = []
    for h in rev['hunks']:
        if h['op'] in ('insert', 'replace'):
            added.extend(lines[h['b0']:h['b1']])
    return '\n'.join(added)


def main():
    missing = [f for f in REQUIRED if not os.path.exists(os.path.join(DATA, f))]
    if missing:
        sys.exit(f'Missing {", ".join(missing)} in {DATA}. See data/README.md.')

    labs = {}
    for line in open(os.path.join(DATA, 'labels.jsonl')):
        d = json.loads(line)
        labs[d['label']] = d
    pages = {}
    for line in open(os.path.join(DATA, 'pages.jsonl')):
        d = json.loads(line)
        pages[d['page_id']] = d
    revs = [json.loads(line) for line in open(os.path.join(DATA, 'revisions.jsonl'))]
    revs.sort(key=lambda r: (r['time'], r['page_id'], r['seq']))
    events = [json.loads(line) for line in open(os.path.join(DATA, 'events.jsonl'))]
    for r in revs:
        r['added'] = added_lines(r)

    with open(cache_path('data.pkl'), 'wb') as fh:
        pickle.dump(dict(labs=labs, pages=pages, revs=revs, events=events), fh)
    n_human = sum(1 for v in labs.values() if v['is_human_handle'])
    print(f'{len(revs)} revisions of {len(pages)} pages under {len(labs)} labels '
          f'({n_human} flagged as human); wrote cache/data.pkl')
    print('  wikis:', dict(collections.Counter(r['page_id'].split('/')[0] for r in revs)))


if __name__ == '__main__':
    main()
