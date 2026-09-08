"""Figure 4: how to write."""
import pickle

import numpy as np

from common import cache_path, wilson
from style import CLASS_COLOR, GREY, INK, SLATE, panel, plt, save, use


def binned(records, nb=5):
    v = np.array(records)
    edges = np.linspace(0, 1, nb + 1)
    xs, ps, lo, hi = [], [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (v[:, 0] >= a) & (v[:, 0] <= b) if b >= 1 else (v[:, 0] >= a) & (v[:, 0] < b)
        if m.sum() >= 10:
            p, l, h = wilson(v[m, 1].sum(), m.sum())
            xs.append(v[m, 0].mean())
            ps.append(p)
            lo.append(l)
            hi.append(h)
    return np.array(xs), np.array(ps), np.array(lo), np.array(hi)


def main():
    use()
    with open(cache_path('forms.pkl'), 'rb') as fh:
        d = pickle.load(fh)

    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.2))
    plt.subplots_adjust(wspace=0.6)

    ax = axes[0]
    panel(ax, 'a')
    for cls, (color, label) in CLASS_COLOR.items():
        xs, ps, lo, hi = binned(d['by_class'][cls])
        ax.errorbar(xs, ps, yerr=[ps - lo, hi - ps], fmt='o-', color=color, ms=2.6,
                    capsize=0, label=label)
    ax.plot([0, 1], [0, 1], color=GREY, lw=0.6, ls=':')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('share of the form on the page')
    ax.set_ylabel('P(use that form)')
    ax.legend(frameon=False, fontsize=5.2, loc='upper left')

    ax = axes[1]
    panel(ax, 'b')
    groups = [(d['versus_feed'], 'page vs feed'),
              ({c: [v[1] for v in d['versus_own'][c]] for c in CLASS_COLOR}, 'page vs own past')]
    for g, (source, _) in enumerate(groups):
        for j, (cls, (color, _)) in enumerate(CLASS_COLOR.items()):
            v = source[cls]
            p, lo, hi = wilson(sum(v), len(v))
            x = g * 4.0 + j
            ax.bar(x, p, color=color, width=0.8)
            ax.errorbar(x, p, yerr=[[p - lo], [hi - p]], fmt='none', ecolor=INK,
                        capsize=1.5, lw=0.7)
    ax.axhline(0.5, color=GREY, ls=':', lw=0.7)
    ax.set_xticks([1, 5.0])
    ax.set_xticklabels(['vs feed', 'vs own past'])
    ax.tick_params(axis='x', length=0)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('P(follow the page)')

    ax = axes[2]
    panel(ax, 'c')
    for _, cls, same_obs, same_mod, diff_obs, diff_mod in d['agreement']:
        color = CLASS_COLOR[cls][0]
        ax.plot(same_obs, same_mod, 'o', color=color, ms=4.5)
        ax.plot(diff_obs, diff_mod, 'o', color=color, ms=3.2, mfc='white', mew=1)
    ax.plot([0.4, 1.02], [0.4, 1.02], color=GREY, lw=0.7, ls='--')
    ax.set_xlim(0.4, 1.02)
    ax.set_ylim(0.4, 1.02)
    ax.set_xlabel('observed P(agree)')
    ax.set_ylabel('model P(agree)')
    ax.plot([], [], 'o', color=SLATE, ms=4.5, label='same page')
    ax.plot([], [], 'o', color=SLATE, mfc='white', mew=1, ms=3.2, label='different pages')
    for cls, (color, label) in CLASS_COLOR.items():
        ax.plot([], [], 's', color=color, ms=4, label=label)
    ax.legend(frameon=False, fontsize=5.2, loc='upper left')

    save(fig, 'figC_forms')


if __name__ == '__main__':
    main()
