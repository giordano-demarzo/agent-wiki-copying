"""Figure 2: where to write."""
import pickle

import numpy as np

from common import cache_path, ccdf, wilson
from style import AMBER, DATA, GREY, INK, MODEL, panel, plt, save, use

BINS = [(0, 0.015), (0.015, 0.033), (0.033, 0.065), (0.065, 0.13), (0.13, 0.5)]


def main():
    use()
    with open(cache_path('pages.pkl'), 'rb') as fh:
        d = pickle.load(fh)
    response, control, inputs, sims = d['response'], d['control'], d['inputs'], d['sims']

    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.2))
    plt.subplots_adjust(wspace=0.65)

    ax = axes[0]
    panel(ax, 'a')
    xs, ps, lo, hi = [], [], [], []
    for a, b in BINS:
        m = (response[:, 0] >= a) & (response[:, 0] < b)
        if m.sum() >= 20:
            p, l, h = wilson(response[m, 1].sum(), m.sum())
            xs.append(response[m, 0].mean())
            ps.append(p)
            lo.append(l)
            hi.append(h)
    ps, lo, hi = np.array(ps), np.array(lo), np.array(hi)
    ax.errorbar(xs, ps, yerr=[ps - lo, hi - ps], fmt='o-', color=INK, ms=3,
                capsize=0, label='observed')
    ax.plot([0, 0.22], [0, 0.22], color=GREY, lw=0.6, ls='--', label='proportional')
    ax.set_xlim(0, 0.22)
    ax.set_ylim(0, 0.22)
    ax.set_xlabel('page’s share of the last 100 feed lines')
    ax.set_ylabel('P(write there)')
    ax.legend(frameon=False, fontsize=5.2, loc='upper left')

    ax = axes[1]
    panel(ax, 'b')

    def curve(sel, col, bins, nmin=150):
        xs, ys, lo, hi = [], [], [], []
        for a, b in bins:
            m = sel & (control[:, col] >= a) & (control[:, col] < b)
            if m.sum() >= nmin and control[m, 2].sum() >= 10:
                p, l, h = wilson(control[m, 2].sum(), m.sum())
                xs.append(control[m, col].mean())
                ys.append(p)
                lo.append(l)
                hi.append(h)
        return (np.array(xs), np.array(ys), np.array(lo), np.array(hi))

    x1, y1, l1, h1 = curve((control[:, 1] >= 3) & (control[:, 1] < 12), 0,
                           [(1, 2), (2, 3), (3, 5), (5, 9), (9, 15), (15, 25)])
    x2, y2, l2, h2 = curve((control[:, 0] >= 3) & (control[:, 0] < 9), 1,
                           [(1, 3), (3, 6), (6, 12), (12, 25), (25, 45), (45, 200)])
    ax.errorbar(x1, y1 / y1[0], yerr=[(y1 - l1) / y1[0], (h1 - y1) / y1[0]],
                fmt='o-', color=INK, ms=3, capsize=0, label='lines held now')
    ax.errorbar(x2, y2 / y2[0], yerr=[(y2 - l2) / y2[0], (h2 - y2) / y2[0]],
                fmt='s-', color=AMBER, ms=3, capsize=0, label='handles already there')
    xx = np.array([1, 20])
    ax.plot(xx, xx / x1[0], color=GREY, lw=0.7, ls=':', label='linear')
    ax.axhline(1, color=GREY, lw=0.5, alpha=0.4)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(0.9, 70)
    ax.set_ylim(0.6, 20)
    ax.set_xlabel('lines held, or handles already there')
    ax.set_ylabel('relative P(write there)')
    ax.legend(frameon=False, fontsize=5.2, loc='upper left')

    ax = axes[2]
    panel(ax, 'c')
    u, y = ccdf(inputs['handles_per_page'])
    ax.loglog(u, y, 'o', color=DATA, ms=2.6, label='observed')
    grid = np.arange(1, 200)
    cc = np.array([[np.mean(h >= g) for g in grid] for h in sims])
    med = np.median(cc, 0)
    ok = med > 0
    ax.loglog(grid[ok], med[ok], '-', color=MODEL, lw=1.2, label='model')
    ax.fill_between(grid[ok], np.maximum(np.percentile(cc, 10, 0)[ok], 1e-4),
                    np.percentile(cc, 90, 0)[ok], color=MODEL, alpha=0.25, lw=0)
    ax.set_xlim(0.9, 200)
    ax.set_ylim(1e-3, 1.1)
    ax.set_xlabel('handles per page, $k$')
    ax.set_ylabel('P($K \\geq k$)')
    ax.legend(frameon=False, fontsize=5.2, loc='lower left')

    save(fig, 'figA_pages')


if __name__ == '__main__':
    main()
