"""SI figure: the chains of Fig. 4c by chain. For each model and starting page, the share of agents 11 to 30 that
answer wrong in every chain (dots, packed side by side; gpt-oss-20b to the left of each starting page, GPT-6.1 Sol to
the right) and its mean over the 20 chains with a 95% bootstrap interval. gpt-oss-20b also has chains from 19 notes.
Writes paper/figs/figS_chains.pdf and figures/figS_chains.png.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / 'src'))
from mock_fig4c_options import MODELS, chains, packed  # noqa: E402
from style import panel, plt, use  # noqa: E402


def main():
    use()
    fig, ax = plt.subplots(1, 1, figsize=(3.4, 2.6))
    rng = np.random.default_rng(4)
    for name, col, mk, label, _ in MODELS:
        C = chains(name)
        side = -1 if name == 'gpt-oss-20b' else 1
        xs, ys, lo, hi = [], [], [], []
        for n in sorted(C):
            late = C[n][:, 10:].mean(1)
            px, py = packed(late, np.log10(n), side)
            ax.plot(px, py, mk, color=col, ms=2.0, alpha=0.5, mew=0, zorder=2)
            boot = [rng.choice(late, len(late)).mean() for _ in range(2000)]
            xs.append(10 ** (np.log10(n) + side * 0.025)); ys.append(late.mean())
            lo.append(np.percentile(boot, 2.5)); hi.append(np.percentile(boot, 97.5))
            print(f'{label:12} start {n:2d}: {len(late)} chains, agents 11-30 wrong {late.mean():.2f} '
                  f'[{lo[-1]:.2f}, {hi[-1]:.2f}], all wrong in {int((late == 1).sum())}, none wrong in {int((late == 0).sum())}')
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
    ax.legend(frameon=False, fontsize=6, loc='center left', bbox_to_anchor=(0.0, 0.62), handletextpad=0.3)
    fig.savefig(ROOT.parent / 'figures' / 'figS_chains.png', dpi=220, bbox_inches='tight')
    fig.savefig(ROOT.parent / 'paper' / 'figs' / 'figS_chains.pdf', bbox_inches='tight')
    print('wrote figS_chains')


if __name__ == '__main__':
    main()
