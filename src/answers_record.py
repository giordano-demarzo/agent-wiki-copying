"""The OECD Equity answers in the record: a wrong consensus and its correction.
Writes cache/answers_record.pkl and the numbers of Results 1 and 4 that come
from the record side of the answers.

For five questions of the OECD Equity task the record carries two competing
values: the 'padded' one, the workbook's displayed value with a trailing zero
(9.70, 9.90, 16.40, 14.60, 23.10), and the right one to two decimals (9.69,
9.91, 16.38, 14.59, 23.13). Treated as a convention with form A = padded and
form B = right, the answer goes through the machinery of the conventions
(conventions.sequence: a use is an edit with strictly more instances of one
value than of the other). This module computes

  1. the wrong consensus: the uses before the first correction (the first use
     of a right value), how many handles and pages they span, and that every
     one of them is padded;
  2. the correction: the padded share per hour before and after it;
  3. the copy rule fitted on all uses, and separately before and after the
     correction, which shows that the pooled prior strength is a mixture of a
     regime with no prior (before) and one driven by the evidence on the page
     (after) and is not a property of the agents;
  4. the handles that changed answer between uses, from the right value to the
     padded one and the reverse;
  5. the notes written before the correction that carry both values: agents that
     answered the right value and then adopted the padded one, the examples of
     a switch from right to wrong for the SI.

    python src/answers_record.py
"""
import collections
import re

import numpy as np

import conventions as cv
from common import Dataset, Report, T, save_cache

PADDED = ['9.70', '9.90', '16.40', '14.60', '23.10']   # wrong: the displayed value padded with a zero
RAW = ['9.69', '9.91', '16.38', '14.59', '23.13']      # right: the stored value to two decimals
NAME = 'padded / raw (OECD answers)'
FIRST_CORRECTION = '2026-06-20T03:41:17'               # the first use of a right value (checked below)
HOURS_AFTER = 3


def num(values):
    """Matches any of the values as a number not preceded or followed by digits or a dot."""
    return cv.rx(r'(?<![\d.])(' + '|'.join(v.replace('.', r'\.') for v in values) + r')(?![\d])', 0)


def register():
    """Adds the answer convention to the candidate table, so that the machinery
    of the conventions applies to it; idempotent."""
    cv.CONV[NAME] = ('answer', num(PADDED), num(RAW))


def main():
    register()
    say = Report('The answers in the record')
    ds = Dataset()
    idx, form = cv.sequence(ds.revs, NAME)
    revs = [ds.revs[i] for i in idx]
    padded = form == 0
    t = np.array([T(r['time']).timestamp() for r in revs])
    label = [r['label'] for r in ds.revs]
    page = [r['page_id'] for r in ds.revs]
    out = dict(n_uses=len(idx), n_handles=len({r['label'] for r in revs}), n_pages=len({r['page_id'] for r in revs}),
               first_use=revs[0]['time'], first_correction=revs[int(np.argmax(~padded))]['time'])
    say(f'{NAME}: {out["n_uses"]} uses on {out["n_pages"]} pages by {out["n_handles"]} handles, padded in {padded.mean():.2f}; '
        f'first use {out["first_use"]} by {revs[0]["label"]}; first use of a right value {out["first_correction"]}')
    assert out['first_correction'] == FIRST_CORRECTION + 'Z', out['first_correction']
    alert = T(FIRST_CORRECTION).timestamp()

    # 1. the wrong consensus
    before = t < alert
    out.update(uses_before=int(before.sum()), padded_before=int(padded[before].sum()),
               handles_before=len({r['label'] for r, b in zip(revs, before) if b}),
               pages_before=len({r['page_id'] for r, b in zip(revs, before) if b}))
    say(f'before the first correction: {out["uses_before"]} uses, {out["padded_before"]} of them padded, by {out["handles_before"]} handles '
        f'on {out["pages_before"]} pages; the first use is the first note, so the next {out["uses_before"] - 1} answers all repeated it')

    # 2. the correction, by hour
    by_hour = {}
    for h in range(-1, HOURS_AFTER):
        sel = (t >= alert + h * 3600) & (t < alert + (h + 1) * 3600)
        by_hour[h + 1] = (int(sel.sum()), float(padded[sel].mean()) if sel.sum() else float('nan'))
    out['padded_by_hour'] = by_hour
    say('padded share by hour around the correction (hour 0 = the hour before): '
        + ', '.join(f'hour {h}: {n} uses, {p:.2f}' for h, (n, p) in by_hour.items()))

    # 3. the copy rule, pooled and split
    share, _ = cv.exposure(idx, form, label, page)
    a = padded.astype(float)
    fits = {}
    for name, sel in (('all', np.ones(len(idx), bool)), ('before', before), ('after', ~before)):
        m = sel & np.isfinite(share)
        mu = cv.fit_mu(share[m], a[m]) if m.sum() >= 30 else np.array([np.nan, np.nan])
        fits[name] = dict(n=int(m.sum()), mu=mu, s=float(mu.sum()), exposure=float(np.nanmean(share[sel])) if sel.sum() else np.nan)
        say(f'copy rule, {name} (uses with >= 3 in view, n={m.sum()}): mu_padded {mu[0]:.3f}, mu_right {mu[1]:.3f}, s {mu.sum():.2f}; '
            f'mean exposure padded {fits[name]["exposure"]:.2f}')
    bins = [(0, 0.2), (0.2, 0.5), (0.5, 0.8), (0.8, 0.95), (0.95, 1.01)]
    resp = []
    for lo, hi in bins:
        b = ~before & np.isfinite(share) & (share >= lo) & (share < hi)
        if b.sum() >= 3:
            resp.append((lo, hi, int(b.sum()), float(a[b].mean())))
    out['fits'], out['response_after'] = fits, resp
    say('after the correction, P(padded) by the padded share in view: '
        + ', '.join(f'{lo:.2f}-{hi:.2f}: n={n}, {p:.2f}' for lo, hi, n, p in resp))
    say('reading: before the correction every exposure is 1.00 and every answer padded, so mu_right is 0 and mu_padded is not '
        'identified; after it the agents take the right value at any share in view below 0.95, which is the evidence posted on the '
        'pages and not a prior; the pooled s mixes the two regimes')

    # 4. handles that changed answer
    seq = collections.defaultdict(list)
    for r, p in zip(revs, padded):
        seq[r['label']].append((r['time'], r['page_id'], bool(p)))
    right_to_wrong, wrong_to_right = [], []
    for h, uses in seq.items():
        forms = [p for _, _, p in uses]
        if any(not p1 and p2 for p1, p2 in zip(forms, forms[1:])):
            k = next(i for i, (p1, p2) in enumerate(zip(forms, forms[1:])) if not p1 and p2)
            right_to_wrong.append((h, uses[k][0], uses[k][1], uses[k + 1][0], uses[k + 1][1]))
        if any(p1 and not p2 for p1, p2 in zip(forms, forms[1:])):
            wrong_to_right.append(h)
    out['right_to_wrong'], out['wrong_to_right'] = right_to_wrong, wrong_to_right
    say(f'handles with more than one use: {sum(len(u) > 1 for u in seq.values())}; changed from the right value to the padded one: '
        f'{len(right_to_wrong)}; from the padded to the right one: {len(wrong_to_right)}')
    for h, t1, p1, t2, p2 in right_to_wrong[:10]:
        say(f'  {h}: right on {p1} at {t1}, then padded on {p2} at {t2}')
    # 5. the switch inside one note: edits before the correction that mention both values are agents that
    #    answered the right value, then adopted the padded one for the later rounds after reading the pages
    both = num(PADDED), num(RAW)
    mixed = [r for r in ds.revs if ds.is_task(r) and both[0](r['added'] or '') > 0 and both[1](r['added'] or '') > 0]
    mixed_before = [r for r in mixed if T(r['time']).timestamp() < alert]
    out['n_mixed'] = len(mixed)
    out['switch_examples'] = [dict(time=r['time'], label=r['label'], page=r['page_id'],
                                   text=re.sub(r'\s+', ' ', r['added'] or '')) for r in mixed_before]
    say(f'edits mentioning both a padded and a right value: {len(mixed)}, {len(mixed_before)} before the correction; '
        f'these are the agents that answered the right value and then adopted the padded one (examples for the SI):')
    for e in out['switch_examples'][:4]:
        say(f'  {e["time"]} {e["label"]} on {e["page"]}: {e["text"][:220]}')

    save_cache('answers_record.pkl', out)
    say.write()
    cv.CONV.pop(NAME)   # the answers are not a convention for the steps that follow in the same process


if __name__ == '__main__':
    main()
