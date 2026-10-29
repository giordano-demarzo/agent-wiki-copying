"""SI: sensitivity of the writing results to the convention selection rule.
The rule keeps a candidate with at least MIN_USES uses, a minority form in at
least MIN_MINORITY of them and at least MIN_PAGES pages with five or more uses.
Each threshold is varied in turn around its value, and for every rule the
number of conventions kept, the four coefficients of the forms regression of
Table 1 (task x 3 h specification) and the pooled first-use slope of Fig. 3c
are recomputed. Writes cache/si_thresholds.pkl.

Also the checks of the prior fit of Fig. 3b (cache/si_prior.pkl): the record
points of Fig. 3b under every rule; the habit of repeating one's own earlier
form, which the fit on all uses counts as copying; and the fit on first uses
only, with the distance normalized by the effective number of independent
writers, (sum n_h)^2 / sum n_h^2 over the uses n_h of each handle.
"""
import collections

import numpy as np

from analysis_forms import responses
from common import Dataset, Report, save_cache
from conventions import CONV, MIN_MINORITY, MIN_PAGES, MIN_USES, candidates, fit_prior, prior_exposure, sequence, NOT_A_CHOICE
from fig4_model import normalized
from identification import fit_forms, form_records

GRID = [(u, MIN_MINORITY, MIN_PAGES) for u in (50, 100, 200)] + \
       [(MIN_USES, m, MIN_PAGES) for m in (0.025, 0.10)] + \
       [(MIN_USES, MIN_MINORITY, p) for p in (5, 20)]


def subset(rec, keep):
    m = np.isin(rec['convention'], keep)
    return {k: v[m] for k, v in rec.items()}


def habit(idx, form, label, share, mu):
    """Among the later uses of a handle, the share that repeat its previous
    form, and the share the fitted rule predicts from what was in view."""
    last, same, pred = {}, [], []
    for k, i in enumerate(idx):
        h = label[i]
        if h in last and np.isfinite(share[k]):
            p_a = mu[0] + (1 - mu[0] - mu[1]) * share[k]
            same.append(form[k] == last[h])
            pred.append(p_a if last[h] == 0 else 1 - p_a)
        last[h] = form[k]
    return float(np.mean(same)), float(np.mean(pred))


def prior_checks(ds, rows, union, keep_main, say):
    label = [r['label'] for r in ds.revs]
    page = [r['page_id'] for r in ds.revs]
    fits, conv = {}, {}
    for n in union:
        idx, form = sequence(ds.revs, n)
        a = (form == 0)
        mu, _, mu_floor, floored = (fit_prior(idx, form, label, page) if np.isnan(rows[n]['mu']).any()
                                    else (rows[n]['mu'], None, rows[n]['mu_floor'], rows[n]['floored']))
        fits[n] = dict(s=min(sum(mu_floor), 1.0), devn=normalized(a.mean(), *mu_floor, len(idx)), floored=floored)
        if n not in keep_main:
            continue
        met, first = set(), np.zeros(len(idx), bool)
        for k, i in enumerate(idx):
            first[k] = label[i] not in met
            met.add(label[i])
        uses = np.array(list(collections.Counter(label[i] for i in idx).values()), float)
        n_eff = uses.sum() ** 2 / (uses ** 2).sum()
        mu1, iv1, mu1_floor, floored1 = fit_prior(idx, form, label, page, first_only=True)
        repeat, predicted = habit(idx, form, label, prior_exposure(idx, form, label, page), mu)
        conv[n] = dict(cls=CONV[n][0], uses=len(idx), handles=len(uses), first=first.mean(), n_eff=n_eff,
                       f=a.mean(), f_first=a[first].mean(), f_later=a[~first].mean(),
                       s_all=sum(mu), s_first=sum(mu1), s_first_interval=iv1, floored_first=floored1,
                       pi_first=mu1_floor[0] / sum(mu1_floor), repeat=repeat, predicted=predicted,
                       devn_first=normalized(a.mean(), *mu1_floor, n_eff), s_first_floor=min(sum(mu1_floor), 1.0))
        c = conv[n]
        say(f'{n:24} first uses {c["first"]:.2f}; repeat own form {repeat:.2f} (rule {predicted:.2f}); '
            f's all {c["s_all"]:.2f}, first {c["s_first"]:.2f} [{iv1[1]:.2f}, {iv1[2]:.2f}]; N_eff {n_eff:.0f} of {len(idx)}; '
            f'share A first {c["f_first"]:.2f}, later {c["f_later"]:.2f}')
    more = sum(c['repeat'] > c['predicted'] for c in conv.values())
    say(f'later uses repeat their own previous form more often than the rule predicts in {more} of {len(conv)} conventions')
    return fits, conv


def main():
    say = Report('SI: the convention selection rule')
    ds = Dataset()
    rows = candidates(ds)
    # the same candidates under every rule: '& / and' is a quotation, not a choice (conventions.NOT_A_CHOICE)
    union = [n for n in CONV if rows[n]['uses'] >= 50 and rows[n]['minority'] > 0 and n not in NOT_A_CHOICE]
    rec = form_records(ds, union)
    fits, conv = prior_checks(ds, rows, union, [n for n in union if rows[n]['kept']], say)
    out, grid = [], []
    for uses, minority, pages in GRID:
        keep = [n for n in union if rows[n]['uses'] >= uses and rows[n]['minority'] >= minority
                and rows[n]['n_pages'] >= pages]
        f = fit_forms(subset(rec, keep))
        by_class, _, _ = responses(ds, keep)
        pooled = np.vstack(list(by_class.values()))
        slope = np.polyfit(pooled[:, 0], pooled[:, 1], 1)[0]
        grid.append(dict(uses=uses, minority=minority, pages=pages, points=[dict(name=n, **fits[n]) for n in keep],
                         main=(uses, minority, pages) == (MIN_USES, MIN_MINORITY, MIN_PAGES)))
        out.append(dict(uses=uses, minority=minority, pages=pages, kept=len(keep), n=f['n'], b=f['b1'], se=f['se1'],
                        slope=slope, main=(uses, minority, pages) == (MIN_USES, MIN_MINORITY, MIN_PAGES)))
        say(f'uses >= {uses:3}, minority >= {minority:.3f}, pages >= {pages:2}: {len(keep):2} kept, n = {f["n"]:4}; '
            + ', '.join(f'{b:.2f} +- {s:.2f}' for b, s in zip(f['b1'], f['se1'])) + f'; slope {slope:.2f}')
    save_cache('si_thresholds.pkl', out)
    save_cache('si_prior.pkl', dict(grid=grid, conventions=conv))
    say.write()


if __name__ == '__main__':
    main()
