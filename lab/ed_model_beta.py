"""S-shaped copying in the copy model: simulations (model_beta.py) against the linear theory with an effective
prior strength s_eff = 1 - (1 - s) f'(x*), x* the stable equilibrium share (model_theory.py closed forms with
the gain (1 - s) f'(x*) in place of 1 - s and x*(1 - x*) in place of pi(1 - pi)). Where two equilibria are stable
(s_eff < 0 at the mixed point) the final share sits near either, with equal weight.
Writes paper/figs/ed_model_beta.pdf and the preview figures/ed_model_beta.png (run model_beta.py first).
"""
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy import optimize, stats

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / 'src'))
import model as md  # noqa: E402
import model_theory as mt  # noqa: E402
from common import load_cache  # noqa: E402
from style import GREY, INK, MODEL, plt, use  # noqa: E402

NAME_LIGHT = '#c49a6c'
S = np.linspace(0.02, 1, 120)


def fprime(x, b):
    return 1.0 if b == 0 else b * (1 - np.tanh(b * (x - 0.5)) ** 2) / (2 * np.tanh(b / 2))


def equilibria(s, p, b):
    """Stable fixed points of x = s p + (1 - s) f(x) and their gains (1 - s) f'(x)."""
    g = lambda x: s * p + (1 - s) * md.respond(x, b) - x
    if b == 0:
        return [(p, 1 - s)]
    xs = np.linspace(1e-4, 1 - 1e-4, 1999)
    roots = [optimize.brentq(g, xs[i], xs[i + 1]) for i in range(1998) if g(xs[i]) * g(xs[i + 1]) < 0]
    return [(x, (1 - s) * fprime(x, b)) for x in roots if (1 - s) * fprime(x, b) < 1]


def theory(s, b, N, V, tau, B):
    mus, sds, vols = [], [], []
    for p in md.DIRECTIONS:
        for x, gain in equilibria(s, p, b):
            n_eq = len(equilibria(s, p, b))
            for _ in range(2 // n_eq):
                se = 1 - gain
                mus.append(x - p)
                sds.append(np.sqrt(max(x * (1 - x), 1e-6) * (1 + gain ** 2 * V) / (N * se ** 2)))
                vols.append(mt.closed_volatility(se, N, B, V * (gain / (1 - s)) ** 2 if s < 1 else 0, tau)
                            if gain > 0 else 1.0)
    return mt.folded_quantiles(mus, sds, (0.5,))[0], np.median(vols)


def main():
    use()
    mod = load_cache('model.pkl')
    sims = {0.0: mod['main'], **pickle.load(open(ROOT.parent / 'cache' / 'model_beta.pkl', 'rb'))}
    gc = mt.graph_constants(mod['par'])
    fig, axes = plt.subplots(2, 2, figsize=(5.4, 4.4))
    plt.subplots_adjust(wspace=0.38, hspace=0.42)
    shades = {0.0: 0.35, 1.68: 0.65, 3.9: 1.0}
    for r, (kind, col, B) in enumerate((('form', MODEL, 50), ('name', NAME_LIGHT, 100))):
        g = gc[kind]
        for b, sw in sims.items():
            a = shades[b]
            vm = md.curve(sw, kind, 'vol')[0]
            dm = md.distance_curve(sw, kind)[0]
            axes[r, 0].plot(md.S_GRID, vm, 'o', color=col, alpha=a, ms=3, mec=INK, mew=0.3)
            axes[r, 1].plot(md.S_GRID, dm, 'o', color=col, alpha=a, ms=3, mec=INK, mew=0.3, label=f'|β| = {b:g}')
            th = np.array([theory(s, b, g['N'], g['V'], g['tau'], B) for s in S])
            axes[r, 0].plot(S, th[:, 1], '-', color=INK, alpha=a, lw=0.9)
            axes[r, 1].plot(S, th[:, 0], '-', color=INK, alpha=a, lw=0.9)
            print(kind, b, 'vol sim ', np.round(vm, 2))
            print(kind, b, 'dist sim', np.round(dm, 3))
            print(kind, b, 'dist th ', np.round([theory(s, b, g['N'], g['V'], g['tau'], B)[0] for s in md.S_GRID[1:]], 3))
            crit = [s for s in S if (1 - s) * fprime(0.5, b) > 1]
            if crit and b:
                print(kind, b, f'bistable at pi = 1/2 for s < {max(crit):.2f}')
        axes[r, 0].set(ylabel=f'volatility ({kind}s)', xlim=(-0.03, 1.03), ylim=(0, 7))
        axes[r, 0].axhline(1, color=GREY, lw=0.5, ls=':')
        axes[r, 1].set(ylabel=f'distance from the prior ({kind}s)', yscale='log', ylim=(8e-4, 1.5), xlim=(-0.03, 1.03))
    for ax in axes[1]:
        ax.set_xlabel('prior strength, $s$')
    axes[0, 1].legend(frameon=False, fontsize=5.4, title='simulation (dots), theory (lines)', title_fontsize=5.4)
    out = ROOT.parent / 'figures' / 'ed_model_beta.png'
    fig.savefig(out, dpi=220, bbox_inches='tight')
    fig.savefig(ROOT.parent / 'paper' / 'figs' / 'ed_model_beta.pdf', bbox_inches='tight')
    print('wrote', out)


if __name__ == '__main__':
    main()
