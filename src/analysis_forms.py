"""How to write: the conventions, their selection and the response of an agent
to the form on its page. Writes cache/forms.pkl (cache/forms_full.pkl with
--full, the robustness check on all handles).

1. Every candidate convention, its base probabilities with the profile interval
   of the prior strength, and the selection (Table S3).
2. The patchwork (Fig. 2c): agreement of two uses on the same page and on
   different pages, and the same-page agreement with the forms shuffled among
   the uses of the same three-hour block.
3. The response of a handle's first use to the share of a form on the page,
   by class (Fig. 3c), and the two conflict tests (page against feed, page
   against the handle's own past).
4. For every kept convention, the statistics placed on the prior-strength axis
   of Fig. 4, and the usage rate that the model takes from the record.
"""
import collections
import sys

import numpy as np

from common import Dataset, Report, T, block_of, co_occurrence, save_cache, volatility, wilson
from conventions import CLASSES, CONV, FEED_USES, MIN_IN_VIEW, candidates, exposure, sequence

N_SHUFFLES = 50
MIN_MINORITY_USES = 10     # a convention enters Fig. 4b with at least this many minority uses
MIN_FIRST_USES = 150       # ... and Fig. 4c with at least this many first uses
VOL_BLOCK = 50             # first uses per block in the volatility of a form


def agreement(pages_forms):
    """P(two uses take the same form), for two uses on the same page and on
    two different pages."""
    same_n = same_a = total_n = total_a = 0
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
    return (same_a / same_n if same_n else np.nan, diff_a / diff_n if diff_n else np.nan)


def by_page(page_ids, forms):
    out = collections.defaultdict(list)
    for p, f in zip(page_ids, forms):
        out[p].append(int(f))
    return out


def shuffled_agreement(page_ids, forms, blocks, rng, n=N_SHUFFLES):
    """Median same-page agreement over n shuffles of the forms among the uses
    of the same three-hour block, which keeps the time and destroys the page."""
    out = []
    for _ in range(n):
        f2 = forms.copy()
        for b in np.unique(blocks):
            m = np.where(blocks == b)[0]
            f2[m] = rng.permutation(forms[m])
        out.append(agreement(by_page(page_ids, f2))[0])
    return float(np.median(out))


def responses(ds, keep):
    """First-use response by class, and the two conflict tests."""
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
            if (handle in previous and on_page.sum() >= MIN_IN_VIEW and on_page[0] != on_page[1]
                    and int(np.argmax(on_page)) != previous[handle]):
                versus_own[cls].append(int(form[k] == np.argmax(on_page)))
            seen.add(handle)
            previous[handle] = form[k]
            step = np.zeros(2)
            step[form[k]] = 1
            page_hist[pg] = on_page + step
    return ({c: np.array(v) for c, v in by_class.items()}, versus_feed, versus_own)


def axis_statistics(ds, name, mu):
    """What Fig. 4 places on the prior-strength axis for one convention."""
    idx, form = sequence(ds.revs, name)
    a = (form == 0).astype(float)
    pages = [ds.revs[i]['page_id'] for i in idx]
    seen, first = set(), []
    for k, i in enumerate(idx):
        if ds.revs[i]['label'] not in seen:
            seen.add(ds.revs[i]['label'])
            first.append(k)
    s = mu[0] + mu[1]
    minority_uses = min(a.sum(), len(a) - a.sum())
    return dict(name=name, cls=CONV[name][0], s=min(s, 1.0), muA=mu[0], muB=mu[1], f=a.mean(),
                n_first=len(first), minority_uses=int(minority_uses),
                gap=co_occurrence(pages, a) if minority_uses >= MIN_MINORITY_USES else np.nan,
                vol=volatility(a[first], VOL_BLOCK) if len(first) >= MIN_FIRST_USES else np.nan,
                vol_all=volatility(a, VOL_BLOCK),
                dev=max(abs(a.mean() - mu[0] / s), 1e-3) if s > 0 else np.nan)


def main(full=False):
    say = Report('How to write' + (', all handles' if full else ''))
    ds = Dataset(full=full)
    rows = candidates(ds)
    keep = [n for n in CONV if rows[n]['kept']]
    enough = [n for n in CONV if rows[n]['uses'] >= 100]
    say(f'{len(rows)} candidate conventions, {len(enough)} with at least 100 uses, {len(keep)} kept')
    fails = collections.Counter(tuple(r['fails']) for r in rows.values() if not r['kept'])
    say(f'  dropped: fewer than 100 uses {sum(1 for r in rows.values() if r["uses"] < 100)}; '
        f'with 100 uses, the rarer form never used {sum(1 for n in enough if rows[n]["minority"] == 0)}; '
        f'other failures {sum(1 for n in enough if 0 < rows[n]["minority"] and not rows[n]["kept"])}')
    widths = lambda names: np.median([rows[n]['s_interval'][2] - rows[n]['s_interval'][1] for n in names])
    dropped_measured = [n for n in enough if not rows[n]['kept'] and rows[n]['s_interval'] is not None]
    say(f'  median width of the 95% interval of s: kept {widths(keep):.2f}, dropped with both forms used '
        f'{widths(dropped_measured):.2f}')
    say(f'  kept: uses {min(rows[n]["uses"] for n in keep)}-{max(rows[n]["uses"] for n in keep)}, '
        f'pages {min(rows[n]["n_pages"] for n in keep)}-{max(rows[n]["n_pages"] for n in keep)}, '
        f'prior strength {min(sum(rows[n]["mu"]) for n in keep):.2f}-{max(sum(rows[n]["mu"]) for n in keep):.2f}')
    for n in keep:
        r = rows[n]
        say(f'  {n:26} {r["cls"]:9} uses {r["uses"]:5} pages {r["n_pages"]:3} minority {r["minority"]:.2f} '
            f'mu_A {r["mu"][0]:.3f} mu_B {r["mu"][1]:.3f} s {sum(r["mu"]):.2f} '
            f'[{r["s_interval"][1]:.2f}, {r["s_interval"][2]:.2f}]')

    by_class, versus_feed, versus_own = responses(ds, keep)
    for cls in CLASSES:
        v = by_class[cls]
        slope, intercept = np.polyfit(v[:, 0], v[:, 1], 1)
        say(f'{cls:9}: first-use response slope {slope:.2f}, intercept {intercept:.2f} (n={len(v)}); '
            f'P(A) when the page has under 10% of A {v[v[:, 0] < 0.1, 1].mean():.2f}')
    pooled = np.vstack(list(by_class.values()))
    say(f'pooled   : slope {np.polyfit(pooled[:, 0], pooled[:, 1], 1)[0]:.2f} (n={len(pooled)})')
    for test, d in (('page against feed', versus_feed), ("page against the handle's own past", versus_own)):
        for cls in CLASSES + ('pooled',):
            v = d[cls] if cls != 'pooled' else [x for c in CLASSES for x in d[c]]
            p, lo, hi = wilson(sum(v), len(v))
            say(f'{test}, {cls:9}: follow the page {p:.2f} [{lo:.2f}, {hi:.2f}] (n={len(v)})')

    ts = np.array([T(r['time']).timestamp() for r in ds.revs])
    rng = np.random.default_rng(0)
    patchwork = []
    for n in keep:
        idx, form = sequence(ds.revs, n)
        pages = [ds.revs[i]['page_id'] for i in idx]
        a = (form == 0).astype(int)
        same, diff = agreement(by_page(pages, a))
        blocks = np.array([block_of(t) for t in ts[idx]])
        patchwork.append(dict(name=n, cls=CONV[n][0], same=same, diff=diff,
                              shuffled=shuffled_agreement(pages, a, blocks, rng)))
    pw = {k: np.array([p[k] for p in patchwork]) for k in ('same', 'diff', 'shuffled')}
    say(f'patchwork: two uses agree {pw["same"].mean():.3f} on the same page and {pw["diff"].mean():.3f} '
        f'on different pages (means over conventions); shuffled within {3}-h blocks {pw["shuffled"].mean():.3f}; '
        f'observed above the shuffle for {int((pw["same"] > pw["shuffled"]).sum())} of {len(keep)}')
    for p in patchwork:
        say(f'  {p["name"]:26} same {p["same"]:.2f} different {p["diff"]:.2f} shuffled {p["shuffled"]:.2f}')

    n_task = sum(ds.is_task(r) for r in ds.revs)
    usage = {n: sum(ds.is_task(ds.revs[i]) for i in sequence(ds.revs, n)[0]) / n_task for n in keep}
    say(f'usage rate: mean share of task edits using a convention {np.mean(list(usage.values())):.3f} '
        f'(from {min(usage.values()):.2f}, {min(usage, key=usage.get)}, to {max(usage.values()):.2f}, '
        f'{max(usage, key=usage.get)})')

    axis = [axis_statistics(ds, n, rows[n]['mu']) for n in keep]
    say(f'on the prior-strength axis: {sum(np.isfinite(e["gap"]) for e in axis)} conventions in Fig. 4b '
        f'(at least {MIN_MINORITY_USES} minority uses), {sum(np.isfinite(e["vol"]) for e in axis)} in Fig. 4c '
        f'(at least {MIN_FIRST_USES} first uses)')
    va = np.array([e['vol_all'] for e in axis if np.isfinite(e['vol'])])
    vf = np.array([e['vol'] for e in axis if np.isfinite(e['vol'])])
    say(f'  volatility, mean over those: all uses {va.mean():.2f}, first uses {vf.mean():.2f}')

    save_cache('forms_full.pkl' if full else 'forms.pkl',
               dict(rows=rows, keep=keep, fits={n: rows[n]['mu'] for n in keep}, by_class=by_class,
                    versus_feed=versus_feed, versus_own=versus_own, patchwork=patchwork,
                    usage=usage, axis=axis))
    say.write()


if __name__ == '__main__':
    main(full='--full' in sys.argv)
