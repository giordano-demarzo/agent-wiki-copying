"""Figure 3 of the paper (fig3_model): one model reproduces the collective, with its analytical results
(model_theory.py) dashed, and Fig. S (consistency).
(a) handles per page, record and model (median and 5-95% over 200 runs);
(b, c) volatility over time and distance from the prior along the prior-strength
    axis s = mu_A + mu_B: model curves (median and 10-90% over runs and prior directions) for a convention
    read from the page and a name feature read from the feed, and the empirical conventions, name
    features and months at their measured s, with binned means. The consistency within a page along the
    same axis is written as figS_consistency for the SI.
"""
import numpy as np

import model as md
import model_theory as mt
from common import Report, ccdf, load_cache
from style import CLASS_COLOR, GREY, INK, MODEL, NAME_BROWN, NAME_LIGHT, SLATE, panel, plt, save, use

# names in this figure: a light brown, so that amber keeps meaning GPT-6.1 Sol across the figures

EDGES = [0, 0.1, 0.25, 0.45, 0.65, 0.85, 1.05]
MARK = {'coined': ('o', CLASS_COLOR['coined'][0]), 'semantic': ('^', CLASS_COLOR['semantic'][0]),
        'habit': ('v', CLASS_COLOR['habit'][0]), 'feature': ('s', NAME_LIGHT), 'month': ('p', GREY)}
LABEL = {'coined': 'coined names', 'semantic': 'near-synonyms', 'habit': 'habits of style',
         'feature': 'name features', 'month': 'months in names'}
FORMS, NAMES = {'coined', 'semantic', 'habit'}, {'feature', 'month'}


def points():
    """The empirical points: conventions, name features and months."""
    pts = [dict(e, kind=e['cls']) for e in load_cache('forms.pkl')['axis']]
    names = load_cache('names.pkl')
    for name, s in names['stats'].items():
        kind = 'feature' if name in names['features'] else ('month' if s['cls'] == 'month' else None)
        if kind is None:
            continue
        mu = s['muA'] + s['muB']
        pts.append(dict(name=name, kind=kind, s=s['s'], gap=s['gap'], vol=s['vol'],
                        dev=max(abs(s['f'] - s['muA'] / mu), 1e-3) if mu > 0 else np.nan))
    return pts


def binned(pts, kinds, stat, log):
    """Means (geometric if log) in bins of s, with standard errors; a bin with a
    single point is drawn as that point."""
    xs, ys, lo, hi = [], [], [], []
    for a, b in zip(EDGES[:-1], EDGES[1:]):
        sel = [p for p in pts if p['kind'] in kinds and a <= p['s'] < b and np.isfinite(p[stat])]
        if not sel:
            continue
        y = np.array([p[stat] for p in sel])
        y = np.log(y) if log else y
        m, e = y.mean(), (y.std(ddof=1) / np.sqrt(len(y)) if len(y) > 1 else 0.0)
        xs.append(np.mean([p['s'] for p in sel]))
        if log:
            ys.append(np.exp(m))
            lo.append(np.exp(m) - np.exp(m - e))
            hi.append(np.exp(m + e) - np.exp(m))
        else:
            ys.append(m)
            lo.append(e)
            hi.append(e)
    return np.array(xs), np.array(ys), np.array(lo), np.array(hi)


def axis_panel(ax, stat, letter, ylabel, log, sw, pts, say, median=True):
    """One outcome along the prior-strength axis: model curves for a convention
    and a name feature, the empirical points and their binned means."""
    panel(ax, letter)
    for kind, col in (('form', MODEL), ('name', NAME_BROWN)):
        m, lo, hi = md.distance_curve(sw, kind) if stat == 'dev' else md.curve(sw, kind, stat)
        if median:
            ax.plot(md.S_GRID, m, '-', color=col, lw=1.3)
        ax.fill_between(md.S_GRID, np.maximum(lo, 1e-4) if log else lo, hi, color=col, alpha=0.18, lw=0)
    for p in pts:
        if np.isfinite(p[stat]):
            mk, col = MARK[p['kind']]
            ax.plot(p['s'], p[stat], mk, color=col, ms=3.4, mec=INK, mew=0.3, alpha=0.35)
    for kinds, col, mk in ((FORMS, MODEL, 'o'), (NAMES, NAME_BROWN, 's')):
        xs, ys, lo, hi = binned(pts, kinds, stat, log)
        ax.errorbar(xs, ys, yerr=[lo, hi], fmt=mk, color=col, ms=4.0, lw=1.2, capsize=2, mfc='white',
                    mew=1.1, zorder=6)
        say(f'({letter}) {stat}, binned {"forms" if kinds == FORMS else "names"}: '
            + ' '.join(f's={x:.2f}:{v:.3g}' for x, v in zip(xs, ys)))
    ax.set_xlabel('prior strength, $s$')
    ax.set_ylabel(ylabel)
    ax.set_xlim(-0.03, 1.03)
    if stat != 'dev':
        ax.axhline(1, color=GREY, lw=0.7, ls=':')
    if stat == 'gap':
        ax.set_yscale('log')
        ax.set_ylim(0.5, 20)
    if stat == 'dev':
        ax.set_yscale('log')
        ax.set_ylim(8e-4, 1.5)


def curve_legend(ax, loc='upper right'):
    ax.plot([], [], '-', color=MODEL, lw=1.3, label='conventions, theory')
    ax.plot([], [], '-', color=NAME_BROWN, lw=1.3, label='names, theory')
    ax.fill_between([], [], [], color=GREY, alpha=0.35, lw=0, label='simulations (10-90%)')
    ax.plot([], [], 'o', color=MODEL, ms=4.0, mfc='white', mew=1.1, label='conventions, record')
    ax.plot([], [], 's', color=NAME_BROWN, ms=4.0, mfc='white', mew=1.1, label='names, record')
    ax.legend(frameon=False, fontsize=5.4, loc=loc, handlelength=1.6)


def symbol_legend(fig, ax, y=-0.17):
    handles = [ax.plot([], [], mk, color=col, ms=3.4, mec=INK, mew=0.3, alpha=0.7, label=LABEL[k])[0]
               for k, (mk, col) in MARK.items()]
    fig.legend(handles=handles, frameon=False, fontsize=5.8, loc='lower center', bbox_to_anchor=(0.5, y),
               ncol=5, handlelength=1.2, columnspacing=1.6)


def main():
    use()
    say = Report('Figure 4')
    mod = load_cache('model.pkl')
    pages = load_cache('pages.pkl')
    sw = mod['main']
    pts = points()
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.5))
    plt.subplots_adjust(wspace=0.45)

    ax = axes[0]
    panel(ax, 'a')
    u, y = ccdf(pages['inputs']['handles_per_page'])
    ax.loglog(u, y, '-', color=INK, lw=1.1, drawstyle='steps-post', label='record')
    cc = mod['pages']['ccdf']
    grid = np.arange(1, cc.shape[1] + 1)
    med = np.median(cc, 0)
    ok = med > 0
    ax.fill_between(grid[ok], np.maximum(np.percentile(cc, 5, 0)[ok], 1e-4), np.percentile(cc, 95, 0)[ok],
                    color=GREY, alpha=0.45, lw=0, label='copy model, simulations')
    k, b = mt.borel_ccdf(mod['par']['c'])
    ax.loglog(k, b, '-', color=SLATE, lw=1.3, label='copy model, theory')
    ax.set_xlim(0.9, 200)
    ax.set_ylim(1e-3, 1.1)
    ax.set_xlabel('handles per page, $k$')
    ax.set_ylabel('P($K \\geq k$)')
    ax.legend(frameon=False, fontsize=5.6, loc='lower left')

    axis_panel(axes[1], 'vol', 'b', 'volatility over time', False, sw, pts, say, median=False)
    axis_panel(axes[2], 'dev', 'c', 'distance from the prior', True, sw, pts, say, median=False)
    gc = mt.graph_constants(mod['par'])
    # the theory: solid where the linear (Gaussian) approximation holds, dotted below s = 0.1
    for kind, col, B in (('form', MODEL, 50), ('name', NAME_BROWN, 100)):
        g = gc[kind]
        say(f'theory, {kind}: N = {g["N"]:.0f}, V = {g["V"]:.2f}, tau = {g["tau"]:.1f}')
        for S, ls in ((np.linspace(0.1, 1, 150), '-'), (np.linspace(0.04, 0.1, 20), ':')):
            axes[1].plot(S, [mt.closed_volatility(s, g['N'], B, g['V'], g['tau']) for s in S], ls, color=col, lw=1.3)
            axes[2].plot(S, [mt.closed_distance(s, g['N'], g['V'], q=(0.5,))[0] for s in S], ls, color=col, lw=1.3)
    curve_legend(axes[1])
    symbol_legend(fig, axes[1])
    counts = {stat: {k: sum(1 for p in pts if p['kind'] in ks and np.isfinite(p[stat]))
                     for k, ks in (('forms', FORMS), ('names', NAMES))} for stat in ('gap', 'vol', 'dev')}
    say(f'points: {counts}')
    save(fig, 'fig3_model')

    # the consistency within a page, along the same axis, for the SI
    fig, ax = plt.subplots(1, 1, figsize=(3.4, 2.5))
    axis_panel(ax, 'gap', '', 'consistency within a page', True, sw, pts, say)
    curve_legend(ax)
    symbol_legend(fig, ax, y=-0.2)
    save(fig, 'figS_consistency')
    say.write()


if __name__ == '__main__':
    main()
