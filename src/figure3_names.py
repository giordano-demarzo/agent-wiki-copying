"""Figure 3: what to call yourself."""
import pickle

import numpy as np

from common import cache_path, ccdf, wilson
from style import AMBER, GREY, INK, MODEL, SLATE, panel, plt, save, use


def binned(x, y, edges=(0, 0.05, 0.15, 0.3, 0.5, 1.01)):
    xs, ps, lo, hi = [], [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (x >= a) & (x < b)
        if m.sum() >= 20:
            p, l, h = wilson(y[m].sum(), m.sum())
            xs.append(x[m].mean())
            ps.append(p)
            lo.append(l)
            hi.append(h)
    return np.array(xs), np.array(ps), np.array(lo), np.array(hi)


def main():
    use()
    with open(cache_path('names.pkl'), 'rb') as fh:
        d = pickle.load(fh)
    feed, page, carries = d['feed'], d['page'], d['carries']

    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.2))
    plt.subplots_adjust(wspace=0.6)

    ax = axes[0]
    panel(ax, 'a')
    xs, ps, lo, hi = binned(feed, carries)
    ax.errorbar(xs, ps, yerr=[ps - lo, hi - ps], fmt='o-', color=INK, ms=3,
                capsize=0, label='last 30 names seen')
    m = ~np.isnan(page)
    xs, ps, lo, hi = binned(page[m], carries[m])
    ax.errorbar(xs, ps, yerr=[ps - lo, hi - ps], fmt='^-', color=AMBER, ms=3,
                capsize=0, label='authors of the page written on')
    ax.plot([0, 1], [0, 1], color=GREY, lw=0.6, ls=':')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('share of names in view with the token')
    ax.set_ylabel('P(name has the token)')
    ax.legend(frameon=False, fontsize=5.2, loc='upper left')

    ax = axes[1]
    panel(ax, 'b')
    bars = [('last 30\nnames', d['b_time'][1], d['se_time'][1], INK),
            ('30 before\nthose', d['b_time'][2], d['se_time'][2], SLATE),
            ('feed', d['b_src'][1], d['se_src'][1], INK),
            ("page's\nauthors", d['b_src'][2], d['se_src'][2], AMBER)]
    for j, (name, value, se, color) in enumerate(bars):
        ax.bar(j, value, color=color, width=0.65)
        ax.errorbar(j, value, yerr=1.96 * se, fmt='none', ecolor=INK, capsize=2, lw=0.8)
    ax.axvline(1.5, color=GREY, lw=0.5, ls=':')
    ax.set_xticks(range(4))
    ax.set_xticklabels([b[0] for b in bars], fontsize=4.6)
    ax.text(0.5, 1.02, 'how far back', ha='center', va='bottom', fontsize=5.4, color=SLATE)
    ax.text(2.5, 1.02, 'which source', ha='center', va='bottom', fontsize=5.4, color=SLATE)
    ax.set_ylabel('fitted coefficient')
    ax.set_ylim(-0.05, 1.0)

    ax = axes[2]
    panel(ax, 'c')
    observed = np.array(sorted(d['counts'].values(), reverse=True))
    runs = d['runs']
    kmax = int(max(observed.max(), max(f.max() for f in runs)))
    grid = np.arange(1, kmax + 1)
    cc = np.array([[np.mean(f >= k) for k in grid] for f in runs])
    ux, uy = ccdf(observed)
    ax.loglog(ux, uy, 'o', color=INK, ms=2.4, label='observed')
    med = np.median(cc, 0)
    ok = med > 0
    ax.loglog(grid[ok], med[ok], '-', color=MODEL, lw=1.2, label='model')
    ax.fill_between(grid[ok], np.percentile(cc, 10, 0)[ok],
                    np.percentile(cc, 90, 0)[ok], color=MODEL, alpha=0.25, lw=0)
    ax.set_xlabel('handles using a name piece, $k$')
    ax.set_ylabel('P($K \\geq k$)')
    ax.legend(frameon=False, fontsize=5.2, loc='lower left')

    save(fig, 'figB_names')


if __name__ == '__main__':
    main()
