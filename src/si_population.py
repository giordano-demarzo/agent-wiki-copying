"""SI, the record and the population. Produces cache/si_population.pkl.

Spans and availability (how long handles wrote, how long pages were written
on, how long pages stayed readable), the activity of a handle, and the table
of task families.
"""
import collections
import datetime
import pickle

import numpy as np

from common import Dataset, Report, T, cache_path

END_OF_RECORD = datetime.datetime(2026, 7, 3)


def availability(ds, by_page):
    """Minutes from a page's first edit to the first deletion after it, or to
    the end of the record."""
    key_to_id = {p['page_key']: pid for pid, p in ds.pages.items()}
    deletions = collections.defaultdict(list)
    for e in ds.events:
        if e['event_type'] == 'delete' and e.get('page_key') in key_to_id:
            deletions[key_to_id[e['page_key']]].append(T(e['time']))
    out, n_deleted = [], 0
    for pid, revs in by_page.items():
        t0 = T(revs[0]['time'])
        later = [t for t in deletions.get(pid, []) if t >= t0]
        end = min(later) if later else END_OF_RECORD
        n_deleted += bool(later)
        out.append((end - t0).total_seconds() / 60)
    return np.array(out), n_deleted


def families(ds):
    rows = []
    per = collections.defaultdict(lambda: dict(handles=set(), pages=set(), edits=0, times=[]))
    for r in ds.revs:
        if not ds.is_task(r):
            continue
        f = per[ds.family(r)]
        f['handles'].add(r['label'])
        f['pages'].add(r['page_id'])
        f['edits'] += 1
        f['times'].append(r['time'][:10])
    for fam, f in per.items():
        rows.append((fam, len(f['handles']), f['edits'], len(f['pages']),
                     min(f['times']), max(f['times'])))
    return sorted(rows, key=lambda x: -x[1])


def main():
    say = Report('SI: the record and the population')
    ds = Dataset()
    minutes = lambda a, b: (T(b['time']) - T(a['time'])).total_seconds() / 60

    handle_span = np.array([minutes(v[0], v[-1]) for v in ds.by_label.values() if len(v) >= 2])
    edits_per = np.array([len(v) for v in ds.by_label.values()])
    by_page = ds.by_page()
    page_span = np.array([minutes(v[0], v[-1]) for v in by_page.values() if len(v) >= 2])
    readable, n_deleted = availability(ds, by_page)

    say(f'handle activity span (n={len(handle_span)}): median {np.median(handle_span) / 60:.1f} h, '
        f'share > 6 h {np.mean(handle_span > 360):.3f}, > 24 h {np.mean(handle_span > 1440):.3f}')
    say(f'edits per handle: median {np.median(edits_per):.0f}, mean {edits_per.mean():.2f}, '
        f'share with one edit {np.mean(edits_per == 1):.2f}, max {edits_per.max()}')
    say(f'page write span (n={len(page_span)}): median {np.median(page_span) / 60:.1f} h')
    say(f'pages touched by the population: {len(by_page)}, deleted during the record {n_deleted}; '
        f'readable: median {np.median(readable) / 1440:.0f} d, share > 1 d {np.mean(readable > 1440):.2f}, '
        f'> 1 week {np.mean(readable > 10080):.2f}')

    rows = families(ds)
    say(f'{len(rows)} task families; largest: {rows[0][0]} ({rows[0][1]} handles)')

    with open(cache_path('si_population.pkl'), 'wb') as fh:
        pickle.dump(dict(handle_span=handle_span, edits_per=edits_per, page_span=page_span,
                         readable=readable, n_deleted=n_deleted, families=rows), fh)
    say.write()


if __name__ == '__main__':
    main()
