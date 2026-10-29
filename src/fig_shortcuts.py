"""Draft figure: shortcuts and exploits, how they spread.
(a) cumulative handles mentioning each technique; (b) P(first mention) by what was in view;
(c) the task x 3 h regression on the shares on the page and in the feed."""
import datetime
import os

import matplotlib.dates as mdates
import numpy as np

from common import PREVIEW, load_cache
from style import AMBER, INDIGO, INK, MOSS, ROSE, SLATE, TEAL, panel, plt, use

COLORS = [INK, INDIGO, TEAL, AMBER, ROSE, MOSS, SLATE, '#8c5a1e']


def main():
    use()
    d = load_cache('shortcuts.pkl')
    names = [k for k in d if 'res' in d[k]]
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.5), gridspec_kw=dict(width_ratios=[1.2, 1.0, 1.0]))
    plt.subplots_adjust(wspace=0.55)
    ax = axes[0]; panel(ax, 'a')
    for c, k in zip(COLORS, names):
        t = sorted(datetime.datetime.utcfromtimestamp(x[0]) for x in d[k]['mentions'])
        first = {}
        for x in d[k]['mentions']:
            first.setdefault(x[1], x[0])
        t = sorted(datetime.datetime.utcfromtimestamp(v) for v in first.values())
        ax.step(t, np.arange(1, len(t) + 1), where='post', color=c, lw=1.1, label=k)
    ax.set_xlim(datetime.datetime(2026, 6, 16), datetime.datetime(2026, 6, 22, 12))
    ax.xaxis.set_major_locator(mdates.DayLocator(bymonthday=[16, 18, 20, 22]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d Jun'))
    ax.set_yscale('log'); ax.set_ylim(1, 400)
    ax.set_ylabel('handles that have mentioned it')
    ax.legend(frameon=False, fontsize=5.2, loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2, handlelength=1.4)
    ax = axes[1]; panel(ax, 'b')
    keys = ['out of view', 'in the feed only', 'on the page']
    w = 0.8 / len(names)
    for i, (c, k) in enumerate(zip(COLORS, names)):
        vals = [d[k]['raw'][q][0] for q in keys]
        ax.bar(np.arange(3) + (i - len(names) / 2 + 0.5) * w, vals, w, color=c)
    ax.set_yscale('log'); ax.set_ylim(5e-4, 0.5)
    ax.set_xticks(range(3)); ax.set_xticklabels(['out of\nview', 'feed\nonly', 'on the\npage'], fontsize=6)
    ax.set_ylabel('P(first mention) per edit')
    ax = axes[2]; panel(ax, 'c')
    for i, (c, k) in enumerate(zip(COLORS, names)):
        b, se, _ = d[k]['res']['task x 3 h']['share']
        ax.errorbar([0 + (i - len(names) / 2 + 0.5) * 0.09, 1 + (i - len(names) / 2 + 0.5) * 0.09], b, yerr=1.96 * se,
                    fmt='o', color=c, ms=3, lw=0.9, capsize=1.5)
    ax.axhline(0, color=SLATE, lw=0.6)
    ax.set_xticks([0, 1]); ax.set_xticklabels(['share on\nthe page', 'share in\nthe feed'], fontsize=6)
    ax.set_xlim(-0.6, 1.6); ax.set_ylim(-1.5, 1.0)
    ax.set_ylabel('coefficient (task x 3 h fixed effects)')
    fig.savefig(os.path.join(PREVIEW, 'figS_shortcuts.png'), dpi=200, bbox_inches='tight')
    print('wrote figures/figS_shortcuts.png')


if __name__ == '__main__':
    main()
