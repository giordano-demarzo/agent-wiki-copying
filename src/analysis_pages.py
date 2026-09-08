"""Where to write: how agents pick a page. Produces cache/pages.pkl.

Three things are computed here.

1. The response curve. Every time a handle writes on a task page that is new
   to it, each distinct page appearing in the last 100 lines of the feed is a
   candidate; we record the share of those lines it holds and whether it was
   the page chosen.
2. The visibility-against-audience control. The same candidates, with the
   number of distinct handles that had already edited each of them.
3. The inputs of the stylized model: how many edits and how many distinct
   pages each real handle produced, the observed number of handles per page,
   and the creation probability c.
"""
import collections
import pickle

import numpy as np

from common import Dataset, Report, T, cache_path, wilson

FEED_LINES = 100   # the feed an agent is assumed to scroll


def landing_records(ds):
    """Candidate pages for every decision to write on a page new to the writer."""
    ts_all = np.array([T(r['time']).timestamp() for r in ds.revs_all])
    pid_all = [r['page_id'] for r in ds.revs_all]

    seen = collections.defaultdict(set)
    response, control = [], []
    n_create = n_append = n_in_window = 0
    handles_on = collections.defaultdict(int)

    for r in ds.revs:
        handle, page = r['label'], r['page_id']
        if not ds.is_task(r) or page in seen[handle]:
            continue
        seen[handle].add(page)
        if r['diff_base_reason'] == 'page_created':
            n_create += 1
            handles_on[page] += 1
            continue
        n_append += 1
        k = np.searchsorted(ts_all, T(r['time']).timestamp())
        window = collections.Counter(pid_all[max(0, k - FEED_LINES):k])
        if page in window:
            n_in_window += 1
            for cand, lines in window.items():
                control.append((lines, handles_on[cand], int(cand == page)))
        for cand, lines in window.items():
            response.append((lines / FEED_LINES, int(cand == page)))
        handles_on[page] += 1

    return (np.array(response), np.array(control),
            dict(n_create=n_create, n_append=n_append, n_in_window=n_in_window))


def first_landings(ds):
    """For each handle, whether its first task edit created a page or appended,
    and how far back in the feed the page's previous edit was."""
    ts_all = np.array([T(r['time']).timestamp() for r in ds.revs_all])
    pid_all = [r['page_id'] for r in ds.revs_all]
    seen, out = set(), []
    for r in ds.revs:
        handle = r['label']
        if handle in seen or not ds.is_task(r):
            continue
        seen.add(handle)
        if r['diff_base_reason'] == 'page_created':
            out.append(('creates', np.nan, np.nan))
            continue
        t = T(r['time']).timestamp()
        k = np.searchsorted(ts_all, t)
        prev = [j for j in range(k - 1, -1, -1) if pid_all[j] == r['page_id']]
        if not prev:
            out.append(('appends (no prior edit)', np.nan, np.nan))
            continue
        out.append(('appends', k - 1 - prev[0], (t - ts_all[prev[0]]) / 3600))
    return out


def model_inputs(ds):
    """Per-handle activity, observed handles per page, and the creation rate."""
    task_revs = [r for r in ds.revs if ds.is_task(r)]
    by_page = collections.defaultdict(set)
    pages_of, edits_of = collections.defaultdict(set), collections.Counter()
    n_create = 0
    for r in task_revs:
        by_page[r['page_id']].add(r['label'])
        pages_of[r['label']].add(r['page_id'])
        edits_of[r['label']] += 1
        n_create += r['diff_base_reason'] == 'page_created'
    handles = list(pages_of)
    edits_per = [edits_of[h] for h in handles]
    pages_per = [len(pages_of[h]) for h in handles]
    handles_per_page = np.array([len(v) for v in by_page.values()])
    return dict(handles_per_page=handles_per_page, edits_per=edits_per,
                pages_per=pages_per, c=n_create / sum(pages_per),
                n_pages=len(by_page), n_edits=len(task_revs), n_create=n_create)


def feed_model(n_handles, edits_per, pages_per, c, m=FEED_LINES, seed=0):
    """The stylized page-choice model.

    Handles arrive in sequence. Each makes as many edits and touches as many
    distinct pages as a real handle did. For every new page it creates one
    with probability c, and otherwise picks uniformly among the last m lines
    of the feed, which means proportionally to the share of the feed that page
    holds. Remaining edits go to pages it has already touched.
    """
    rng = np.random.default_rng(seed)
    feed, handles_on, next_id = [], collections.defaultdict(set), 0
    for h in range(n_handles):
        mine = []
        n_pages = max(1, int(round(pages_per[h % len(pages_per)])))
        n_edits = max(n_pages, int(round(edits_per[h % len(edits_per)])))
        for _ in range(n_edits):
            if len(mine) < n_pages:
                if not feed or rng.random() < c:
                    next_id += 1
                    page = next_id
                else:
                    page = feed[rng.integers(min(len(feed), m))]
                mine.append(page)
            else:
                page = mine[rng.integers(len(mine))]
            handles_on[page].add(h)
            feed.insert(0, page)
            feed = feed[:m]
    return np.array([len(v) for v in handles_on.values()])


def main():
    say = Report('Where to write (Figure 2)')
    ds = Dataset()
    response, control, counts = landing_records(ds)
    inputs = model_inputs(ds)

    slope = np.polyfit(response[:, 0], response[:, 1], 1)[0]
    total = counts['n_create'] + counts['n_append']
    say(f'decisions on a page new to the writer: {total} '
        f'({counts["n_create"] / total:.2f} creations)')
    say(f'appends landing on a page inside the last {FEED_LINES} feed lines: '
        f'{counts["n_in_window"] / counts["n_append"]:.2f}')
    say(f'response of P(write there) to the page share of the feed: slope {slope:.2f}')

    land = first_landings(ds)
    kinds = collections.Counter(k for k, _, _ in land)
    rank = np.array([r for k, r, _ in land if k == 'appends'])
    say(f"newcomers' first task edit: {dict(kinds)}")
    say(f"  appends: the page's last edit was {np.median(rank):.0f} edits back in the "
        f"feed (median); beyond 30 {np.mean(rank > 30):.2f}, beyond 100 {np.mean(rank > 100):.2f}")

    say(f'task pages {inputs["n_pages"]}, task edits {inputs["n_edits"]}, '
        f'creations {inputs["n_create"]}; creation probability c={inputs["c"]:.2f}')

    sims = [feed_model(len(inputs['edits_per']), inputs['edits_per'],
                       inputs['pages_per'], inputs['c'], seed=s) for s in range(20)]
    ks = (5, 10, 20, 40)
    obs = [np.mean(inputs['handles_per_page'] >= k) for k in ks]
    mod = [np.mean([np.mean(h >= k) for h in sims]) for k in ks]
    say(f'P(handles per page >= {ks}): observed {[round(v, 3) for v in obs]}, '
        f'model {[round(v, 3) for v in mod]}')

    # the control, reported as in Figure 2b
    say('control (each curve normalised to its first bin):')
    for label, col, other, lo, hi, bins in [
            ('lines held now', 0, 1, 3, 12, [(1, 2), (2, 3), (3, 5), (5, 9), (9, 15), (15, 25)]),
            ('handles already there', 1, 0, 3, 9, [(1, 3), (3, 6), (6, 12), (12, 25), (25, 45), (45, 200)])]:
        sel = (control[:, other] >= lo) & (control[:, other] < hi)
        xs, ys = [], []
        for a, b in bins:
            m = sel & (control[:, col] >= a) & (control[:, col] < b)
            if m.sum() >= 150 and control[m, 2].sum() >= 10:
                p, _, _ = wilson(control[m, 2].sum(), m.sum())
                xs.append(round(control[m, col].mean(), 1))
                ys.append(p)
        say(f'  {label}: {[round(v / ys[0], 2) for v in ys]} at {xs}')

    with open(cache_path('pages.pkl'), 'wb') as fh:
        pickle.dump(dict(response=response, control=control, counts=counts,
                         inputs=inputs, sims=sims), fh)
    say.write()


if __name__ == '__main__':
    main()
