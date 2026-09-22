"""SI, how to sign (Section S4). Writes cache/si_names.pkl.

1. The succession of tasks: the share of newborns carrying a piece, in blocks
   of 100, against the share of newborns belonging to the piece's task.
2. The visibility test feature by feature, with and without fixed effects for
   the task and the three-hour block.
3. The skeleton of a name, the order of the classes of its pieces, and how often
   a newcomer shares it with task-mates in and out of view.
4. Who arrives first: whether a piece carried by one of the first five arrivals
   of a task spreads in the task, with the placebo of the last five.
5. Fashions: four role pieces through the population, against a copying replay.
"""
import collections
import re

import numpy as np

import names as nm
from common import Dataset, Report, fit, load_cache, save_cache

BLOCK = 100
SUCCESSION = (('Watcher', 'datausa-poverty-county'), ('Agent', 'datausa-sector61-state'),
              ('Open', 'datausa-clothing-workforce'), ('Scout', 'oecd-equity'))
FASHIONS = ('Scout', 'Agent', 'Research', 'Helper')
FIRST_K, MIN_TASK = 5, 40


def skeleton(handle, cls_of):
    """Classes of the pieces of a name in order: A affiliation, R generic role
    piece, T task-specific piece, M month, N number, X anything else; runs of
    the same class are merged."""
    out = []
    for p in re.findall(r'[A-Z][a-z]+|[A-Z]{2,}(?![a-z])|\d+', handle):
        if p.isdigit():
            k = 'N'
        elif p in nm.AFFILIATION:
            k = 'A'
        elif p in nm.MONTHS:
            k = 'M'
        else:
            k = {'generic': 'R', 'task-specific': 'T'}.get(cls_of.get(p), 'X')
        if not out or out[-1] != k or k == 'X':
            out.append(k)
    return ''.join(out)


def replay(v, mu, rng, k=30, seed_share=0.05):
    """Each newborn carries the piece with probability mu_A + (1 - mu_A - mu_B) rho,
    rho the share among the last k simulated names; the first 5% are the record."""
    n = len(v)
    n_seed = int(seed_share * n)
    sim = np.zeros(n)
    sim[:n_seed] = v[:n_seed]
    for i in range(n_seed, n):
        sim[i] = rng.random() < mu[0] + (1 - mu[0] - mu[1]) * sim[max(0, i - k):i].mean()
    return sim


def blocks(v):
    n = len(v) // BLOCK * BLOCK
    return np.asarray(v, float)[:n].reshape(-1, BLOCK).mean(1)


def main():
    say = Report('SI: how to sign')
    ds = Dataset()
    d = load_cache('names.pkl')
    fam = np.array([ds.first_family[h] for h in ds.handles])
    birth = np.array([ds.birth[h].timestamp() for h in ds.handles])
    groups, feats, stats = d['groups'], d['features'], d['stats']
    cls_of = {p['name']: p['cls'] for p in d['pieces']}
    out = {}

    # 1. the succession of tasks
    for piece, family in SUCCESSION:
        v = nm.indicators(ds, [piece])
        inf = (fam == family).astype(float)
        g, f = blocks(v), blocks(inf)
        n = len(g) * BLOCK
        within = np.array([v[i:i + BLOCK][inf[i:i + BLOCK] == 1].mean() if inf[i:i + BLOCK].sum() >= 5 else np.nan
                           for i in range(0, n, BLOCK)])
        outside = np.array([v[i:i + BLOCK][inf[i:i + BLOCK] == 0].mean() for i in range(0, n, BLOCK)])
        say(f'succession, {piece} / {family}: correlation of the share of newborns with the piece and with the task '
            f'{np.corrcoef(g, f)[0, 1]:.2f}; inside the task {" ".join("-" if np.isnan(x) else f"{x:.2f}" for x in within)}; '
            f'outside {" ".join(f"{x:.2f}" for x in outside)}')

    # 2. the visibility test feature by feature
    rec = d['records']
    m0 = ~np.isnan(rec['view']) & ~np.isnan(rec['out'])
    per_feature = {}
    for t in feats:
        m = m0 & (rec['feature'] == t)
        tb = np.array([f'{a}|{b}' for a, b in zip(rec['family'][m], rec['block'][m])])
        b0, se0, _ = fit([rec['view'][m], rec['out'][m]], rec['y'][m])
        b1, se1, _ = fit([rec['view'][m], rec['out'][m]], rec['y'][m], (tb,))
        per_feature[t] = (b0, se0, b1, se1, int(m.sum()))
        say(f'  {t:10} in view {b0[0]:5.2f} out of view {b0[1]:5.2f} | with task x 3 h FE: in view {b1[0]:5.2f}'
            f'+-{se1[0]:.2f} out of view {b1[1]:5.2f}+-{se1[1]:.2f} (n={m.sum()})')
    out['per_feature'] = per_feature

    # 3. the skeleton of a name
    sk = [skeleton(h, cls_of) for h in ds.handles]
    say('most common skeletons: ' + ', '.join(f'{s} ({c})' for s, c in collections.Counter(sk).most_common(6)))
    rows = []
    for i, g in enumerate(groups):
        if not nm.in_sample(g):
            continue
        others = [j for j in g['feed'] if fam[j] != fam[i] and birth[i] - 6 * 3600 <= birth[j] < birth[i]]
        if len(g['in_view']) >= 3 and len(g['out_of_view']) >= 3 and len(others) >= 3:
            same = lambda js: np.mean([sk[j] == sk[i] for j in js])
            rows.append((same(g['in_view']), same(g['out_of_view']), same(others)))
    R = np.array(rows)
    diff = R[:, 0] - R[:, 1]
    rng = np.random.default_rng(0)
    boot = [rng.choice(diff, len(diff)).mean() for _ in range(2000)]
    say(f'same skeleton as a task-mate in view {R[:, 0].mean():.3f}, out of view {R[:, 1].mean():.3f}, as a handle of '
        f'another task in the feed {R[:, 2].mean():.3f}; difference in - out {diff.mean():.3f} '
        f'[{np.percentile(boot, 2.5):.3f}, {np.percentile(boot, 97.5):.3f}] over {len(R)} newcomers')

    # 4. who arrives first
    sets = [set(nm.pieces(h)) for h in ds.handles]
    generic = {p for p, c in cls_of.items() if c == 'generic'}
    rare_or_generic = lambda p: cls_of.get(p) not in ('task-specific', 'month') and p not in nm.MONTHS
    res = {}
    for placebo in (False, True):
        rows = []
        for f, n in collections.Counter(fam).most_common():
            if n < MIN_TASK or f == 'none':
                continue
            members = np.where(fam == f)[0]
            probe, rest = (members[-FIRST_K:], members[:-FIRST_K]) if placebo else (members[:FIRST_K], members[FIRST_K:])
            for p in {p for i in members for p in sets[i]}:
                if not rare_or_generic(p) or p.lower() in f.lower():
                    continue
                outside = np.mean([p in s for j, s in enumerate(sets) if fam[j] != f])
                if outside > 0.15:
                    continue
                rows.append((any(p in sets[i] for i in probe), np.mean([p in sets[i] for i in rest])))
        r = np.array(rows, float)
        present = r[:, 0] == 1
        res[placebo] = (r, present)
        if not placebo:
            say(f'first {FIRST_K} arrivals, tasks with at least {MIN_TASK} handles, {len(r)} task-piece pairs: a piece '
                f'carried by one of them is later carried by {r[present, 1].mean():.3f} of the task (n={present.sum()}), '
                f'otherwise {r[~present, 1].mean():.3f}; reaches a fifth of the task {np.mean(r[present, 1] >= 0.2):.2f} '
                f'against {np.mean(r[~present, 1] >= 0.2):.3f}; ratio {r[present, 1].mean() / r[~present, 1].mean():.1f}')
        else:
            say(f'placebo, last {FIRST_K} arrivals looking backwards: ratio {r[present, 1].mean() / r[~present, 1].mean():.1f}')

    # 5. fashions
    fashions = {}
    for t in FASHIONS:
        v = feats[t]
        mu = (stats[t]['muA'], stats[t]['muB'])
        runs = np.array([blocks(replay(v, mu, np.random.default_rng(seed))) for seed in range(20)])
        obs = blocks(v)
        med = np.median(runs, 0)
        fashions[t] = dict(observed=obs, runs=runs, mu=mu)
        say(f'fashion {t:9}: mu_A {mu[0]:.3f} mu_B {mu[1]:.3f}; observed first block {obs[0]:.2f}, last {obs[-1]:.2f}, '
            f'max {obs.max():.2f}; replay first {med[0]:.2f}, last {med[-1]:.2f}; mean absolute error {np.abs(obs - med).mean():.2f}')
    n = len(ds.handles) // BLOCK * BLOCK
    out['fashions'] = fashions
    out['block_days'] = [ds.birth[ds.handles[i + BLOCK // 2]].strftime('%d Jun') for i in range(0, n, BLOCK)]
    save_cache('si_names.pkl', out)
    say.write()


if __name__ == '__main__':
    main()
