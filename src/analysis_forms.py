"""How to write: how agents pick between two forms. Produces cache/forms.pkl.

Fits the copying response of every convention, measures which exposure the
agents follow when two of them disagree, and compares the observed patchwork
of pages with the model.
"""
import collections
import pickle

import numpy as np

from common import Dataset, Report, cache_path, wilson
from conventions import CONV, FEED_USES, exposure, fit_mu, sequence

MIN_USES = 100        # a convention needs this many uses
MIN_PAGES = 10        # and this many pages carrying five or more uses
MIN_MINORITY = 0.05   # and a minority form used at least this often
MIN_IN_VIEW = 3       # a record needs this many instances in view
CLASSES = ('coined', 'semantic', 'habit')


def fit_all(ds):
    """Fit the two error rates of every candidate convention and keep the
    ones with enough data."""
    label = [r['label'] for r in ds.revs]
    page = [r['page_id'] for r in ds.revs]
    fits, keep = {}, []
    for name in CONV:
        idx, form = sequence(ds.revs, name)
        if len(idx) < MIN_USES:
            continue
        share, share_page = exposure(idx, form, label, page)
        took_a = (form == 0).astype(int)
        mu = fit_mu(share, took_a)
        if np.isnan(mu[0]):
            continue
        by_page = collections.defaultdict(list)
        for k in range(len(idx)):
            by_page[page[idx[k]]].append(form[k])
        n_pages = sum(1 for v in by_page.values() if len(v) >= 5)
        minority = min(np.mean(form == 0), 1 - np.mean(form == 0))
        fits[name] = dict(cls=CONV[name][0], uses=len(idx), mu=mu, n_pages=n_pages,
                          minority=minority)
        if n_pages >= MIN_PAGES and minority >= MIN_MINORITY:
            keep.append(name)
    return fits, keep


def response_and_conflicts(ds, keep):
    """The response curve by class, and the two conflict tests."""
    label = [r['label'] for r in ds.revs]
    page = [r['page_id'] for r in ds.revs]
    by_class = {c: [] for c in CLASSES}
    versus_feed = {c: [] for c in CLASSES}
    versus_own = {c: [] for c in CLASSES}

    for name in keep:
        cls = CONV[name][0]
        idx, form = sequence(ds.revs, name)
        _, share_page = exposure(idx, form, label, page)
        took_a = (form == 0).astype(int)
        page_hist, seen, previous = {}, set(), {}
        for k in range(len(idx)):
            handle, pg = label[idx[k]], page[idx[k]]
            on_page = page_hist.get(pg, np.zeros(2))
            window = [j for j in range(max(0, k - FEED_USES), k) if label[idx[j]] != handle]
            feed = (np.array([np.sum(form[window] == 0), np.sum(form[window] == 1)], float)
                    if window else np.zeros(2))
            first_use = handle not in seen

            if first_use and not np.isnan(share_page[k]):
                by_class[cls].append((share_page[k], took_a[k]))
            if (first_use and feed.sum() >= MIN_IN_VIEW and on_page.sum() >= MIN_IN_VIEW
                    and feed[0] != feed[1] and on_page[0] != on_page[1]
                    and np.argmax(feed) != np.argmax(on_page)):
                versus_feed[cls].append(int(form[k] == np.argmax(on_page)))
            if (handle in previous and on_page.sum() >= MIN_IN_VIEW
                    and on_page[0] != on_page[1]
                    and int(np.argmax(on_page)) != previous[handle]):
                versus_own[cls].append((on_page.max() / on_page.sum(),
                                        int(form[k] == np.argmax(on_page))))

            seen.add(handle)
            previous[handle] = form[k]
            step = np.zeros(2)
            step[form[k]] = 1
            page_hist[pg] = on_page + step
    return by_class, versus_feed, versus_own


def agreement(pages_forms):
    """P(two uses take the same form), for two uses on the same page and on
    two different pages."""
    same_n = same_a = 0
    total_n = total_a = 0
    for forms in pages_forms.values():
        n, n1 = len(forms), sum(forms)
        if n >= 2:
            same_n += n * (n - 1)
            same_a += n1 * (n1 - 1) + (n - n1) * (n - n1 - 1)
        total_n += n
        total_a += n1
    all_n = total_n * (total_n - 1)
    all_a = total_a * (total_a - 1) + (total_n - total_a) * (total_n - total_a - 1)
    diff_n, diff_a = all_n - same_n, all_a - same_a
    return (same_a / same_n if same_n else np.nan,
            diff_a / diff_n if diff_n else np.nan)


def simulate(mu, n_handles, edits_per, pages_per, c, seed=0, m=100,
             k_feed=FEED_USES, p_use=0.5):
    """The model of Figure 4c.

    Handles arrive and land on pages exactly as in the page model, and each
    edit uses the convention with a fixed probability p_use. A use takes form
    A with probability mu_A + (1 - mu_A - mu_B) rho, where rho is the share of
    form A already on that page, or in the last k_feed uses if the page carries
    neither form. Nothing from the record enters the simulation.
    """
    rng = np.random.default_rng(seed)
    feed, uses, recent, next_id = [], collections.defaultdict(list), [], 0
    for h in range(n_handles):
        mine = []
        n_pages = max(1, int(round(pages_per[h % len(pages_per)])))
        n_edits = max(n_pages, int(round(edits_per[h % len(edits_per)])))
        for _ in range(n_edits):
            if len(mine) < n_pages:
                if not feed or rng.random() < c:
                    next_id += 1
                    page = ('new', next_id)
                else:
                    page = feed[rng.integers(min(len(feed), m))]
                mine.append(page)
            else:
                page = mine[rng.integers(len(mine))]
            if rng.random() < p_use:
                source = uses[page] if uses[page] else recent[:k_feed]
                rho = np.mean(source) if len(source) else 0.5
                form = int(rng.random() < mu[0] + (1 - mu[0] - mu[1]) * rho)
                uses[page].append(form)
                recent.insert(0, form)
                recent = recent[:k_feed]
            feed.insert(0, page)
            feed = feed[:m]
    return uses


def main():
    say = Report('How to write (Figure 4)')
    ds = Dataset()
    with open(cache_path('pages.pkl'), 'rb') as fh:
        page_inputs = pickle.load(fh)['inputs']

    fits, keep = fit_all(ds)
    say(f'{len(keep)} conventions kept of {len(fits)} with enough uses:')
    for name in keep:
        f = fits[name]
        say(f'  {name:26} {f["cls"]:9} uses {f["uses"]:5} '
            f'mu_A {f["mu"][0]:.3f} mu_B {f["mu"][1]:.3f} pages {f["n_pages"]:3}')

    by_class, versus_feed, versus_own = response_and_conflicts(ds, keep)
    for cls in CLASSES:
        v = np.array(by_class[cls])
        slope, intercept = np.polyfit(v[:, 0], v[:, 1], 1)
        mus = [fits[n]['mu'] for n in keep if fits[n]['cls'] == cls]
        say(f'{cls:9}: response slope {slope:.2f}, intercept {intercept:.2f} '
            f'(n={len(v)}); mean mu_A + mu_B {np.mean([m[0] + m[1] for m in mus]):.2f}')
    pooled = []
    for cls in CLASSES:
        v = versus_feed[cls]
        p, lo, hi = wilson(sum(v), len(v))
        pooled += v
        say(f'{cls:9}: page against feed, n={len(v):4} follow the page '
            f'{p:.2f} [{lo:.2f}, {hi:.2f}]')
    p, lo, hi = wilson(sum(pooled), len(pooled))
    say(f'pooled   : page against feed, n={len(pooled)} follow the page '
        f'{p:.2f} [{lo:.2f}, {hi:.2f}]')
    for cls in CLASSES:
        v = np.array(versus_own[cls])
        p, lo, hi = wilson(v[:, 1].sum(), len(v))
        say(f'{cls:9}: page against the handle\'s own past, n={len(v):4} follow the page '
            f'{p:.2f} [{lo:.2f}, {hi:.2f}]')

    label = [r['label'] for r in ds.revs]
    page = [r['page_id'] for r in ds.revs]
    rows = []
    for name in keep:
        idx, form = sequence(ds.revs, name)
        observed_pages = collections.defaultdict(list)
        for k in range(len(idx)):
            observed_pages[page[idx[k]]].append(1 - form[k])
        same_obs, diff_obs = agreement(observed_pages)
        same_mod, diff_mod = [], []
        for seed in range(3):
            uses = simulate(fits[name]['mu'], len(page_inputs['edits_per']),
                            page_inputs['edits_per'], page_inputs['pages_per'],
                            page_inputs['c'], seed=seed)
            s, d = agreement({p_: v for p_, v in uses.items() if v})
            same_mod.append(s)
            diff_mod.append(d)
        rows.append((name, CONV[name][0], same_obs, float(np.mean(same_mod)),
                     diff_obs, float(np.mean(diff_mod))))
        say(f'  {name:26} same page {same_obs:.3f} / {np.mean(same_mod):.3f}   '
            f'different pages {diff_obs:.3f} / {np.mean(diff_mod):.3f}')
    a = np.array([[r[2], r[3], r[4], r[5]] for r in rows])
    say(f'same page: correlation {np.corrcoef(a[:, 0], a[:, 1])[0, 1]:.2f}, '
        f'mean absolute error {np.mean(np.abs(a[:, 0] - a[:, 1])):.3f}')
    say(f'different pages: correlation {np.corrcoef(a[:, 2], a[:, 3])[0, 1]:.2f}, '
        f'mean absolute error {np.mean(np.abs(a[:, 2] - a[:, 3])):.3f}')
    say(f'gap between the two: observed {np.mean(a[:, 0] - a[:, 2]):.3f}, '
        f'model {np.mean(a[:, 1] - a[:, 3]):.3f}')

    with open(cache_path('forms.pkl'), 'wb') as fh:
        pickle.dump(dict(fits=fits, keep=keep, by_class=by_class,
                         versus_feed=versus_feed, versus_own=versus_own,
                         agreement=rows), fh)
    say.write()


if __name__ == '__main__':
    main()
