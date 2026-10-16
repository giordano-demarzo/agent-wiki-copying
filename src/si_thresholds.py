"""SI: sensitivity of the writing results to the convention selection rule.
The rule keeps a candidate with at least MIN_USES uses, a minority form in at
least MIN_MINORITY of them and at least MIN_PAGES pages with five or more uses.
Each threshold is varied in turn around its value, and for every rule the
number of conventions kept, the four coefficients of the forms regression of
Table 1 (task x 3 h specification) and the pooled first-use slope of Fig. 3c
are recomputed. Writes cache/si_thresholds.pkl.
"""
import numpy as np

from analysis_forms import responses
from common import Dataset, Report, save_cache
from conventions import CONV, MIN_MINORITY, MIN_PAGES, MIN_USES, candidates, NOT_A_CHOICE
from identification import fit_forms, form_records

GRID = [(u, MIN_MINORITY, MIN_PAGES) for u in (50, 100, 200)] + \
       [(MIN_USES, m, MIN_PAGES) for m in (0.025, 0.10)] + \
       [(MIN_USES, MIN_MINORITY, p) for p in (5, 20)]


def subset(rec, keep):
    m = np.isin(rec['convention'], keep)
    return {k: v[m] for k, v in rec.items()}


def main():
    say = Report('SI: the convention selection rule')
    ds = Dataset()
    rows = candidates(ds)
    # the same candidates under every rule: '& / and' is a quotation, not a choice (conventions.NOT_A_CHOICE)
    union = [n for n in CONV if rows[n]['uses'] >= 50 and rows[n]['minority'] > 0 and n not in NOT_A_CHOICE]
    rec = form_records(ds, union)
    out = []
    for uses, minority, pages in GRID:
        keep = [n for n in union if rows[n]['uses'] >= uses and rows[n]['minority'] >= minority
                and rows[n]['n_pages'] >= pages]
        f = fit_forms(subset(rec, keep))
        by_class, _, _ = responses(ds, keep)
        pooled = np.vstack(list(by_class.values()))
        slope = np.polyfit(pooled[:, 0], pooled[:, 1], 1)[0]
        out.append(dict(uses=uses, minority=minority, pages=pages, kept=len(keep), n=f['n'], b=f['b1'], se=f['se1'],
                        slope=slope, main=(uses, minority, pages) == (MIN_USES, MIN_MINORITY, MIN_PAGES)))
        say(f'uses >= {uses:3}, minority >= {minority:.3f}, pages >= {pages:2}: {len(keep):2} kept, n = {f["n"]:4}; '
            + ', '.join(f'{b:.2f} +- {s:.2f}' for b, s in zip(f['b1'], f['se1'])) + f'; slope {slope:.2f}')
    save_cache('si_thresholds.pkl', out)
    say.write()


if __name__ == '__main__':
    main()
