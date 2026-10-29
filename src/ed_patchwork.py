"""Extended Data Fig. 1b: the patchwork of conventions, with two permutation nulls.

For each kept convention, the probability that two uses take the same form when
they are on the same page and when they are on different pages (as in
analysis_forms.agreement), with
- the time null of the paper: forms permuted among the uses of the same
  three-hour block, which keeps the timing of every use and removes the page;
- a task-and-time null: forms permuted among the uses of the same task family
  (the family of the page) and three-hour block, which also keeps what agents of
  one task write at one time; only uses in a family-block group with at least
  two pages can move between pages, and their share is reported;
- a 95% interval of the observed same-page agreement from a bootstrap over
  pages, the unit within which uses are dependent.
Each null uses N_PERM permutations and gives its median, its 2.5-97.5
percentiles and a one-sided permutation p-value (1 + #null >= observed) / (1 + N_PERM).
Writes cache/ed_patchwork.pkl.
"""
import collections

import numpy as np

from analysis_forms import agreement, by_page
from common import Dataset, Report, T, block_of, load_cache, save_cache
from conventions import CONV, sequence

N_PERM = 1000
N_BOOT = 1000


def _groups(keys):
    out = collections.defaultdict(list)
    for i, k in enumerate(keys):
        out[k].append(i)
    return [np.array(v) for v in out.values() if len(v) > 1]


def permuted(pages, a, keys, rng, n=N_PERM):
    groups = _groups(keys)
    out = np.empty(n)
    for k in range(n):
        f2 = a.copy()
        for g in groups:
            f2[g] = a[rng.permutation(g)]
        out[k] = agreement(by_page(pages, f2))[0]
    return out


def bootstrap_pages(pages, a, rng, n=N_BOOT):
    forms = list(by_page(pages, a).values())
    out = np.empty(n)
    for k in range(n):
        pick = rng.integers(len(forms), size=len(forms))
        out[k] = agreement({j: forms[i] for j, i in enumerate(pick)})[0]
    return out


def summary(null, observed):
    return dict(median=float(np.median(null)), lo=float(np.percentile(null, 2.5)), hi=float(np.percentile(null, 97.5)),
                p=float((1 + np.sum(null >= observed)) / (1 + len(null))))


def main():
    say = Report('Extended Data Fig. 1b: the patchwork with two nulls')
    ds = Dataset()
    keep = load_cache('forms.pkl')['keep']
    ts = np.array([T(r['time']).timestamp() for r in ds.revs])
    rng = np.random.default_rng(1)
    rows = []
    for n in keep:
        idx, form = sequence(ds.revs, n)
        pages = [ds.revs[i]['page_id'] for i in idx]
        fams = [ds.family(ds.revs[i]) for i in idx]
        a = (form == 0).astype(int)
        blocks = [block_of(t) for t in ts[idx]]
        same, diff = agreement(by_page(pages, a))
        time_null = permuted(pages, a, blocks, rng)
        task_keys = list(zip(fams, blocks))
        task_null = permuted(pages, a, task_keys, rng)
        pages_in = collections.defaultdict(set)
        for p, k in zip(pages, task_keys):
            pages_in[k].add(p)
        movable = float(np.mean([len(pages_in[k]) >= 2 for k in task_keys]))
        boot = bootstrap_pages(pages, a, rng)
        rows.append(dict(name=n, cls=CONV[n][0], uses=len(a), pages=len(set(pages)), same=same, diff=diff,
                         same_lo=float(np.percentile(boot, 2.5)), same_hi=float(np.percentile(boot, 97.5)),
                         time=summary(time_null, same), task=summary(task_null, same), movable=movable))
        r = rows[-1]
        say(f'{n:26} uses {r["uses"]:5} pages {r["pages"]:3} same {same:.3f} [{r["same_lo"]:.3f}, {r["same_hi"]:.3f}] '
            f'different {diff:.3f} | time null {r["time"]["median"]:.3f} [{r["time"]["lo"]:.3f}, {r["time"]["hi"]:.3f}] '
            f'p {r["time"]["p"]:.4f} | task-time null {r["task"]["median"]:.3f} [{r["task"]["lo"]:.3f}, '
            f'{r["task"]["hi"]:.3f}] p {r["task"]["p"]:.4f}, movable uses {movable:.2f}')
    above = lambda k: sum(r['same'] > r[k]['hi'] for r in rows)
    say(f'observed same-page agreement above the 97.5th percentile of the null: time {above("time")} of {len(rows)}, '
        f'task and time {above("task")} of {len(rows)}; above the median: time '
        f'{sum(r["same"] > r["time"]["median"] for r in rows)}, task and time {sum(r["same"] > r["task"]["median"] for r in rows)}')
    say(f'excess over the null, median over conventions: time {np.median([r["same"] - r["time"]["median"] for r in rows]):.3f}, '
        f'task and time {np.median([r["same"] - r["task"]["median"] for r in rows]):.3f}; uses that can move between pages '
        f'under the task-and-time null: median {np.median([r["movable"] for r in rows]):.2f} '
        f'(from {min(r["movable"] for r in rows):.2f} to {max(r["movable"] for r in rows):.2f})')
    save_cache('ed_patchwork.pkl', dict(rows=rows, n_perm=N_PERM, n_boot=N_BOOT))
    say.write()


if __name__ == '__main__':
    main()
