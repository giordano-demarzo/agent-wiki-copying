"""Daily counts for the timeline panel of Figure 1. Writes cache/timeline.json."""
import collections
import datetime
import json
import os

from common import CACHE, Dataset, Report

FIRST_DAY = datetime.date(2026, 5, 24)
N_DAYS = 41


def main():
    say = Report('The record (Figure 1c)')
    ds = Dataset()
    edits = collections.Counter(r['time'][:10] for r in ds.revs)
    new_handles, seen = collections.Counter(), set()
    for r in ds.revs:
        if r['label'] not in seen:
            seen.add(r['label'])
            new_handles[r['time'][:10]] += 1

    days = [str(FIRST_DAY + datetime.timedelta(i)) for i in range(N_DAYS)]
    out = {'days': days,
           'edits': [edits.get(d, 0) for d in days],
           'new_handles': [new_handles.get(d, 0) for d in days]}
    with open(os.path.join(CACHE, 'timeline.json'), 'w') as fh:
        json.dump(out, fh)

    say(f'{sum(out["edits"])} edits by {sum(out["new_handles"])} handles over {N_DAYS} days')
    busiest = sorted(zip(days, out['edits']), key=lambda kv: -kv[1])[:4]
    say(f'  busiest days: {busiest}')
    say.write()


if __name__ == '__main__':
    main()
