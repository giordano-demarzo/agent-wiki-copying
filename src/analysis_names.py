"""How to sign: the pieces of names, the 16 name features, and what each
newcomer could see. Writes cache/names.pkl.

1. The pieces carried by at least 15 handles and their classification
   (Table S2): generic, specific to one task, or a month.
2. The name features and, for each newcomer, the share of each feature among
   its task-mates on its page, in the feed, out of view, and among the handles
   of other tasks in the feed (Fig. 3b, Table 1).
3. For every piece: its prior strength, fitted on the share among the task-mates
   in view, its co-occurrence among co-authors, and its volatility (Fig. 4,
   Table S2).
"""
import collections

import numpy as np

import names as nm
from common import Dataset, Report, fit, save_cache


def main():
    say = Report('How to sign')
    ds = Dataset()
    n = len(ds.handles)
    piece_rows = nm.classify_pieces(ds)
    classes = collections.Counter(r['cls'] for r in piece_rows)
    say(f'pieces carried by at least {nm.MIN_CARRIERS} handles: {len(piece_rows)} '
        f'({classes["month"]} months, {classes["task-specific"]} task-specific, {classes["generic"]} generic)')
    over = sorted(r['over'] for r in piece_rows if r['cls'] != 'month')
    below = max(o for o in over if o < nm.TASK_SPECIFIC)
    above = min(o for o in over if o >= nm.TASK_SPECIFIC)
    say(f'  over-representation in one task: largest below the cut {below:.1f}, smallest above {above:.1f}')
    say('  generic: ' + ', '.join(f'{r["name"]} ({r["carriers"]})' for r in piece_rows if r['cls'] == 'generic'))
    say('  task-specific: ' + ', '.join(f'{r["name"]} ({r["carriers"]})' for r in piece_rows if r['cls'] == 'task-specific'))

    # what a name is made of: occurrences of pieces by class
    cls_of = {r['name']: r['cls'] for r in piece_rows}
    occ = collections.Counter()
    for h in ds.handles:
        for p in nm.pieces(h):
            c = cls_of.get(p, 'rare')
            occ['stamp' if (p in nm.AFFILIATION or c == 'month') else c] += 1
    total = sum(occ.values())
    say(f'piece occurrences: {total}; affiliation and date stamp {occ["stamp"] / total:.2f}, '
        f'task-specific {occ["task-specific"] / total:.2f}, generic role pieces {occ["generic"] / total:.2f}, '
        f'rare pieces (fewer than {nm.MIN_CARRIERS} carriers) {occ["rare"] / total:.2f}')

    feats = nm.features(ds, piece_rows)
    say(f'name features: {len(feats)} ({", ".join(feats)}); at least one in '
        f'{int((sum(feats.values()) > 0).sum())} of {n} handles')
    say('  carriers: ' + ', '.join(f'{t} {int(v.sum())}' for t, v in feats.items()))
    affil = nm.indicators(ds, nm.AFFILIATION)
    say(f'  affiliation stamp (Open, AI or OAI): {int(affil.sum())} handles; date stamp {int(feats["date"].sum())}')

    groups = nm.exposure_groups(ds)
    rec = nm.records(ds, feats, groups)
    in_sample = sum(nm.in_sample(g) for g in groups)
    say(f'newcomers in the name analysis (complete feed, at least {nm.MIN_MATES} task-mates in '
        f'{nm.MATES_H} h): {in_sample} of {n}; mean task-mates in view '
        f'{np.mean([len(g["in_view"]) for g in groups if nm.in_sample(g)]):.1f}, out of view '
        f'{np.mean([len(g["out_of_view"]) for g in groups if nm.in_sample(g)]):.1f}')

    # the two curves of Fig. 3b: joint fit on the shares in view and out of view
    m = ~np.isnan(rec['view']) & ~np.isnan(rec['out'])
    b, se, _ = fit([rec['view'][m], rec['out'][m]], rec['y'][m], (rec['feature'][m],), cluster=rec['handle'][m])
    say(f'joint fit, feature fixed effects (n={m.sum()}): task-mates in view {b[0]:.2f}+-{1.96 * se[0]:.2f}, '
        f'out of view {b[1]:.2f}+-{1.96 * se[1]:.2f} (95% intervals, clustered by handle)')

    # every piece and feature: prior strength, co-occurrence, volatility
    A = nm.coauthors(ds)
    stats = {}
    for r in piece_rows:
        v = nm.indicators(ds, [r['name']])
        stats[r['name']] = dict(r, **nm.feature_statistics(v, groups, A))
    stats['date'] = dict(name='date', cls='generic', carriers=int(feats['date'].sum()),
                         **nm.feature_statistics(feats['date'], groups, A))
    for cls in ('generic', 'task-specific', 'month'):
        rows = [s for s in stats.values() if s['cls'] == cls]
        gaps = np.array([s['gap'] for s in rows if s['gap'] > 0])
        say(f'{cls:13}: prior strength {min(s["s"] for s in rows):.2f}-{max(s["s"] for s in rows):.2f}; '
            f'co-occurrence among co-authors, geometric mean {np.exp(np.log(gaps).mean()):.1f} '
            f'({gaps.min():.1f}-{gaps.max():.1f}, {len(gaps)} pieces with at least {nm.MIN_CARRIERS_COOCCURRENCE} carriers)')
    for t in feats:
        s = stats[t]
        say(f'  feature {t:10}: carriers {int(feats[t].sum()):4d}  mu_A {s["muA"]:.3f} mu_B {s["muB"]:.3f} '
            f's {s["s"]:.2f}  co-occurrence {s["gap"]:.2f}  volatility {s["vol"]:.2f}')

    save_cache('names.pkl', dict(pieces=piece_rows, stats=stats, features=feats, affiliation=affil,
                                 groups=groups, records=rec))
    say.write()


if __name__ == '__main__':
    main()
