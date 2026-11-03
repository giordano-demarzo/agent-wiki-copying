"""Two candidate designs for Fig. 4c (the chains of gpt-oss-20b and GPT-6.1 Sol), each at the size of one panel of
the 3 x 1 figure, for choosing.

    2  trajectories in two short panels stacked in one column (gpt-oss-20b above, Sol below), every curve labelled
       with its starting page at its end
    3  the share of agents 11 to 30 that answer wrong against the starting page: the mean with 95% intervals over
       chains, and every chain as a dot, packed side by side (gpt-oss-20b to the left of each page, Sol to the right)
Writes ../figures/mock_fig4c_options.png.
"""
import collections
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / 'src'))
from style import INK, OSS, SLATE, SOL, panel, plt, use  # noqa: E402

MODELS = (('gpt-oss-20b', OSS, 'o', 'gpt-oss-20b', (1, 7, 19, 32)), ('gpt-6.1-sol', SOL, 'D', 'GPT-6.1 Sol', (1, 3, 7, 32)))


def chains(name):
    ch = collections.defaultdict(dict)
    for l in open(ROOT / 'data' / 'answers' / f'chains_{name}.jsonl'):
        r = json.loads(l)
        if r['scenario'].startswith('recstart') and not r.get('error'):
            ch[(int(re.match(r'recstart(\d+)', r['scenario']).group(1)), r['country'], r['chain'])][r['pos']] = r
    out = collections.defaultdict(list)
    for (n, _, _), v in ch.items():
        if len(v) == 30:
            out[n].append(np.array([v[p]['is_wrong'] for p in range(30)], float))
    return {n: np.array(w) for n, w in out.items()}


def smooth(W):
    return np.array([np.convolve(np.pad(w, 1, mode='edge'), np.ones(3) / 3, mode='valid') for w in W])


def spread_labels(ys, gap=0.11):
    """Label heights at least `gap` apart, kept as close as possible to the curves' ends."""
    order = np.argsort(ys)
    out = np.array(ys, float)
    for k in range(1, len(order)):
        if out[order[k]] - out[order[k - 1]] < gap:
            out[order[k]] = out[order[k - 1]] + gap
    return out


def option2(fig, cell):
    sub = cell.subgridspec(2, 1, hspace=0.22)
    pos = np.arange(1, 31)
    axes = []
    for k, (name, col, _, label, starts) in enumerate(MODELS):
        ax = fig.add_subplot(sub[k])
        axes.append(ax)
        C = chains(name)
        ends = []
        shade = [0.30, 0.55, 0.78, 1.0]          # the alpha of each starting page, from the fewest notes up
        for j, n in enumerate(starts):
            S = smooth(C[n])
            m, e = S.mean(0), 1.96 * S.std(0, ddof=1) / np.sqrt(len(S))
            ax.fill_between(pos, np.clip(m - e, 0, 1), np.clip(m + e, 0, 1), color=col, alpha=0.07 + 0.08 * shade[j], lw=0)
            ax.plot(pos, m, '-', color=col, alpha=shade[j], lw=0.9 + 0.45 * j, solid_capstyle='round')
            ends.append(m[-1])
        # labels in the order of the starting pages (fewest notes at the bottom), at least 0.15 apart and as close to
        # the curves' ends as that allows, each joined to its curve by a leader line
        ys = np.array(ends, float)
        for j in range(1, len(ys)):
            ys[j] = max(ys[j], ys[j - 1] + 0.15)
        if ys[-1] > 1.05:                       # keep the column inside the panel, shifting it down if needed
            ys -= ys[-1] - 1.05
            for j in range(len(ys) - 2, -1, -1):
                ys[j] = min(ys[j], ys[j + 1] - 0.15)
        for j, (n, y0, y) in enumerate(zip(starts, ends, ys)):
            ax.plot([30.2, 32.0], [y0, y], '-', color=col, alpha=shade[j], lw=0.5, clip_on=False)
            ax.text(32.4, y, str(n), fontsize=5.8, color=col, alpha=max(shade[j], 0.55), va='center', clip_on=False,
                    fontweight='bold' if j == len(starts) - 1 else 'normal')
        ax.text(2.0, 1.0 if k == 0 else 0.58, label, fontsize=6.0, color=col, va='top', fontweight='bold')
        ax.set(xlim=(1, 30), ylim=(-0.07, 1.07), xticks=[1, 10, 20, 30], yticks=[0, 0.5, 1])
        ax.set_yticklabels(['0', '0.5', '1'])
        ax.tick_params(labelsize=6.0)
    axes[0].set_xticklabels([])
    axes[1].set_xlabel('agent in the chain')
    axes[0].text(1.12, 1.04, 'wrong notes\nat the start', transform=axes[0].transAxes, fontsize=5.0, color=SLATE,
                 ha='center', va='bottom', linespacing=1.0)
    fig.text(axes[1].get_position().x0 - 0.075, 0.5, 'P(answers wrong)', rotation=90, va='center', ha='center',
             fontsize=7.4)
    panel(axes[0], 'c', x=-0.34)
    return axes


def packed(values, x0, side, cols=5, dx=0.017, dy=0.022):
    """Dot positions for one group of chains: equal values (to 0.05) packed in blocks of `cols` columns next to
    x0 (log10 units, to the left or right of the tick as `side` says), stacked inward from 0 and 1."""
    v = np.asarray(values, float)
    keys = np.round(v / 0.05) * 0.05
    xs, ys = np.zeros(len(v)), v.copy()
    for k in np.unique(keys):
        idx = np.where(keys == k)[0]
        inward = 1 if k < 0.5 else -1
        for m, i in enumerate(idx):
            row, col = divmod(m, cols)
            xs[i] = x0 + side * (col * dx + 0.012)
            ys[i] = v[i] + inward * row * dy
    return 10 ** xs, ys


def option3(fig, cell):
    ax = fig.add_subplot(cell)
    panel(ax, 'c', x=-0.34)
    rng = np.random.default_rng(4)
    for name, col, mk, label, _ in MODELS:
        C = chains(name)
        side = -1 if name == 'gpt-oss-20b' else 1
        xs, ys, lo, hi = [], [], [], []
        for n in sorted(C):
            late = C[n][:, 10:].mean(1)
            centre = np.log10(n) + side * 0.025
            px, py = packed(late, np.log10(n), side)
            ax.plot(px, py, mk, color=col, ms=1.9, alpha=0.5, mew=0, zorder=2)
            boot = [rng.choice(late, len(late)).mean() for _ in range(2000)]
            xs.append(10 ** centre); ys.append(late.mean()); lo.append(np.percentile(boot, 2.5)); hi.append(np.percentile(boot, 97.5))
        ys, lo, hi = map(np.array, (ys, lo, hi))
        ax.plot(xs, ys, '-', color=col, lw=1.2, zorder=4)
        ax.errorbar(xs, ys, yerr=[ys - lo, hi - ys], fmt=mk, color=col, ms=3.6, mec='white', mew=0.6, elinewidth=0.9,
                    capsize=0, zorder=5, label=label)
    ax.set_xscale('log')
    ax.set_xticks([1, 3, 7, 19, 32])
    ax.set_xticklabels(['1', '3', '7', '19', '32'])
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.set(xlim=(0.72, 45), ylim=(-0.05, 1.05), xlabel='wrong notes at the start', ylabel='P(answers wrong), agents 11-30')
    ax.legend(frameon=False, fontsize=5.4, loc='center left', bbox_to_anchor=(0.0, 0.62), handletextpad=0.3)
    return ax


def main():
    use()
    fig = plt.figure(figsize=(5.0, 2.4))   # two candidates, each the width of one panel of the 3 x 1 figure
    gs = fig.add_gridspec(1, 2, wspace=0.62)
    option2(fig, gs[0])
    option3(fig, gs[1])
    fig.text(0.27, 1.0, 'option 2', ha='center', fontsize=7, color=INK)
    fig.text(0.76, 1.0, 'option 3', ha='center', fontsize=7, color=INK)
    out = ROOT.parent / 'figures' / 'mock_fig4c_options.png'
    fig.savefig(out, dpi=240, bbox_inches='tight')
    print('wrote', out)


if __name__ == '__main__':
    main()
