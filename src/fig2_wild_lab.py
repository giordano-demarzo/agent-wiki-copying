"""Figure 2: agents copy what they see, in the wild and in the laboratory (models drawn at the record's bin positions).

The record (black) and two language models shown feeds and pages in which the
share of each option was assigned at random (lab/data):
(a) where to write: P(a candidate page is chosen) against its share of the last
    100 feed lines;
(b) how to sign: P(a name carries a feature) against its share among the
    task-mates in the feed (record) or among the handles of the feed (lab), the
    16 name features pooled, component plus residual of a fit with feature
    intercepts;
(c) how to write: P(a first use takes a form) against the share of that form
    on the page, the conventions pooled, first uses on pages with at least three
    instances.
In the lab the affiliation stamp, the date stamp and Agent were assigned at
random in the feed and the other name features varied with the handles drawn;
the share of each convention on the page was assigned at random.
"""
import json
import os
import re

import numpy as np

from common import ROOT, URL, Report, fit, load_cache, wilson
from conventions import CONV
from fig3_rule import NAME_EDGES, PAGE_BINS, binned_forms
from names import DATE, pieces
from style import GREY, INK, OSS, SLATE, SOL, panel, plt, save, use

LAB = os.path.join(ROOT, 'lab', 'data')
MODELS = (('gpt-oss-20b', OSS, 'gpt-oss-20b'), ('gpt-6.1-sol', SOL, 'GPT-6.1 Sol'))
FEED_HANDLE = re.compile(r'\. \. \. \d\d:\d\d:\d\d \. \. \. (\S+)')
SIGNATURE = re.compile(r'(--|\u2013|\u2014|Signed:?|Submitted by:?)\s*[A-Za-z0-9_]+\s*$')
OFFSET = {'record': 0.0, 'gpt-oss-20b': -1.0, 'gpt-6.1-sol': 1.0}   # small horizontal offsets, in units of dx


def has_feature(handle, feature):
    return bool(DATE.search(handle)) if feature == 'date' else feature in pieces(handle)


def counts(text, name):
    _, ca, cb = CONV[name]
    s = URL.sub(' ', text)
    return ca(s), cb(s)


def lab_records(model, features, keep):
    """The lab's three choices as records with the record's definitions."""
    trials = {}
    for line in open(os.path.join(LAB, 'trials.jsonl')):
        t = json.loads(line)
        trials[t['id']] = t
    pages, names, forms = [], [], []
    for line in open(os.path.join(LAB, f'replies_{model}.jsonl')):
        r = json.loads(line)
        t, rep = trials[r['id']], r['reply']
        if not rep:
            continue
        if t['kind'] == 'P':
            chosen = str(rep.get('page', '')).strip().lower()
            if chosen in {p.lower() for p in t['page_lines']}:
                for p, k in t['page_lines'].items():
                    pages.append((k / 100, float(p.lower() == chosen), t['id']))
            u = str(rep.get('username', ''))
            handles = sorted(set(FEED_HANDLE.findall(t['feed'])))
            for f in features:
                names.append((np.mean([has_feature(h, f) for h in handles]), float(has_feature(u, f)), f, t['id']))
        elif t['kind'] == 'W':
            text = SIGNATURE.sub('', str(rep.get('text', '')).strip())
            for n in keep:
                if n not in t['rho_target']:
                    continue                      # a convention the lab did not assign
                a, b = counts(text, n)
                pa, pb = counts(t['body'], n)
                if a != b and pa + pb >= 3:
                    forms.append((pa / (pa + pb), float(a > b), n, t['id']))
    return pages, names, forms


def adjusted(x, y, feat):
    """Component plus residual of y on x with feature intercepts, binned as in Fig. 3."""
    x, y, feat = (np.asarray(v) for v in (x, y, feat))
    x, y = x.astype(float), y.astype(float)
    levels = np.unique(feat)
    X = np.column_stack([np.ones(len(y)), x] + [(feat == lev).astype(float) for lev in levels[1:]])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    adj = y.mean() + beta[1] * (x - x.mean()) + (y - X @ beta)
    xs, ps, es = [], [], []
    for a, b in zip(NAME_EDGES[:-1], NAME_EDGES[1:]):
        s = (x >= a) & (x < b)
        if s.sum() >= 25:
            xs.append(x[s].mean())
            ps.append(adj[s].mean())
            es.append(1.96 * adj[s].std(ddof=1) / np.sqrt(s.sum()))
    return np.array(xs), np.array(ps), np.array(es)


def page_bins(x, y):
    x, y = np.asarray(x), np.asarray(y)
    xs, ps, lo, hi = [], [], [], []
    for a, b in PAGE_BINS:
        m = (x >= a) & (x < b)
        if m.sum() >= 20:
            p, l, h = wilson(y[m].sum(), m.sum())
            xs.append(x[m].mean())
            ps.append(p)
            lo.append(l)
            hi.append(h)
    return np.array(xs), np.array(ps), np.array(lo), np.array(hi)


def at_record(xr, xs, ps, elo, ehi):
    """A model's binned curve evaluated at the record's bin positions, by linear interpolation between
    the model's own bins (estimates and error bars alike)."""
    xs = np.asarray(xs, float)
    return (xr, np.interp(xr, xs, ps), np.interp(xr, xs, elo), np.interp(xr, xs, ehi))


def draw(ax, xs, ps, lo, hi, who, color, dx):
    """The record as filled points; a model as a thin line with small points."""
    xs = np.asarray(xs) + OFFSET[who] * dx
    if who == 'record':
        ax.errorbar(xs, ps, yerr=[lo, hi], fmt='o', color=INK, ms=3.6, lw=1.0, capsize=0, zorder=5)
    else:
        ax.errorbar(xs, ps, yerr=[lo, hi], fmt='o', color=color, ms=3.0, elinewidth=0.8, capsize=0,
                    alpha=0.95, zorder=4)


def main(MODELS=MODELS, name='fig2_wild_lab'):
    use()
    say = Report(f'Figure 2 ({name})')
    names_cache = load_cache('names.pkl')
    features = [r['name'] for r in names_cache['pieces'] if r['cls'] == 'generic'] + ['date']
    keep = load_cache('forms.pkl')['keep']
    lab = {m: lab_records(m, features, keep) for m, _, _ in MODELS}

    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.3))
    plt.subplots_adjust(wspace=0.55)

    # (a) where to write
    ax = axes[0]
    panel(ax, 'a')
    resp = load_cache('pages.pkl')['response']
    xs, ps, lo, hi = page_bins(resp[:, 0], resp[:, 1])
    xr = xs
    draw(ax, xs, ps, ps - lo, hi - ps, 'record', INK, 0.003)
    b, se, _ = fit([resp[:, 0]], resp[:, 1])
    say(f'(a) record: slope {b[0]:.2f} +- {1.96 * se[0]:.2f} (n={len(resp)} candidate lines)')
    for m, col, _ in MODELS:
        x, y, tid = (np.array(v) for v in zip(*lab[m][0]))
        x, y = x.astype(float), y.astype(float)
        xs, ps, lo, hi = page_bins(x, y)
        draw(ax, *at_record(xr, xs, ps, ps - lo, hi - ps), m, col, 0.003)
        b, se, _ = fit([x], y, cluster=tid)
        say(f'(a) {m}: slope {b[0]:.2f} +- {1.96 * se[0]:.2f} (n={len(np.unique(tid))} choices)')
    ax.plot([0, 0.21], [0, 0.21], color=GREY, lw=0.6, ls='--')
    ax.set(xlim=(-0.005, 0.185), ylim=(-0.007, 0.21), xticks=[0, 0.05, 0.1, 0.15], yticks=[0, 0.1, 0.2],
           xlabel='page’s share of the feed', ylabel='P(write there)')

    # (b) how to sign
    ax = axes[1]
    panel(ax, 'b')
    r = names_cache['records']
    m = ~np.isnan(r['view'])
    xs, ps, es = adjusted(r['view'][m], r['y'][m], r['feature'][m])
    xr = xs
    draw(ax, xs, ps, es, es, 'record', INK, 0.012)
    b, se, _ = fit([r['view'][m]], r['y'][m], (r['feature'][m],), cluster=r['handle'][m])
    say(f'(b) record: slope {b[0]:.2f} +- {1.96 * se[0]:.2f} (n={m.sum()} newcomer-feature records)')
    for mod, col, _ in MODELS:
        x, y, f, tid = (np.array(v) for v in zip(*lab[mod][1]))
        x, y = x.astype(float), y.astype(float)
        xs, ps, es = adjusted(x, y, f)
        draw(ax, *at_record(xr, xs, ps, es, es), mod, col, 0.012)
        b, se, _ = fit([x], y, (f,), cluster=tid)
        say(f'(b) {mod}: slope {b[0]:.2f} +- {1.96 * se[0]:.2f} (n={len(np.unique(tid))} names)')
    ax.plot([0, 1], [0, 1], color=GREY, lw=0.6, ls='--')
    ax.set(xlim=(-0.03, 1.03), ylim=(-0.03, 1.03), xlabel='share of task-mates with the feature', ylabel='P(name has the feature)')

    # (c) how to write
    ax = axes[2]
    panel(ax, 'c')
    by_class = load_cache('forms.pkl')['by_class']
    pooled = np.vstack([by_class[c] for c in ('coined', 'semantic', 'habit')])
    xs, ps, lo, hi = binned_forms(pooled)
    xr = xs
    draw(ax, xs, ps, ps - lo, hi - ps, 'record', INK, 0.012)
    b, se, _ = fit([pooled[:, 0]], pooled[:, 1])
    say(f'(c) record: slope {b[0]:.2f} +- {1.96 * se[0]:.2f} (n={len(pooled)} first uses)')
    for mod, col, _ in MODELS:
        x, y, conv, tid = (np.array(v) for v in zip(*lab[mod][2]))
        x, y = x.astype(float), y.astype(float)
        xs, ps, lo, hi = binned_forms(np.column_stack([x, y]))
        draw(ax, *at_record(xr, xs, ps, ps - lo, hi - ps), mod, col, 0.012)
        b, se, _ = fit([x], y, cluster=tid)
        say(f'(c) {mod}: slope {b[0]:.2f} +- {1.96 * se[0]:.2f} (n={len(y)} uses in {len(np.unique(tid))} notes, '
            f'{len(np.unique(conv))} conventions)')
    ax.plot([0, 1], [0, 1], color=GREY, lw=0.6, ls='--')
    ax.set(xlim=(-0.03, 1.03), ylim=(-0.03, 1.03), xlabel='share of the form on the page', ylabel='P(first use takes the form)')

    # one legend for the figure
    handles = [axes[0].errorbar([], [], yerr=[], fmt='o', color=INK, ms=3.6, capsize=0, label='record')]
    handles += [axes[0].errorbar([], [], yerr=[], fmt='o', color=c, ms=3.0, capsize=0,
                                 label=lab_) for _, c, lab_ in MODELS]
    handles += axes[0].plot([], [], color=GREY, lw=0.6, ls='--', label='proportional')
    axes[0].legend(handles=handles, frameon=False, fontsize=5.4, loc='upper left', handlelength=1.6)
    save(fig, name)
    say.write()


if __name__ == '__main__':
    main()
