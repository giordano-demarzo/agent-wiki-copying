"""Figure 4: one model reproduces the collective.
(a) handles per page, record and model (median and 5-95% over 200 runs);
(b-d) consistency within a page, volatility over time and distance from the prior along the prior-strength
    axis s = mu_A + mu_B: model curves (median and 10-90% over runs and prior directions) for a convention
    read from the page and a name feature read from the agents in view, and the empirical conventions, name
    features and months at their measured s, with binned means.
"""
import numpy as np

import model as md
from common import Report, ccdf, load_cache
from style import AMBER, CLASS_COLOR, GREY, INK, MODEL, NAME_BROWN, panel, plt, save, use

EDGES = [0, 0.1, 0.25, 0.45, 0.65, 0.85, 1.05]
MARK = {'coined': ('o', CLASS_COLOR['coined'][0]), 'semantic': ('^', CLASS_COLOR['semantic'][0]),
        'habit': ('v', CLASS_COLOR['habit'][0]), 'feature': ('s', AMBER), 'month': ('p', GREY)}
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


def main():
    use()
    say = Report('Figure 4')
    mod = load_cache('model.pkl')
    pages = load_cache('pages.pkl')
    sw = mod['main']
    pts = points()
    fig, axes = plt.subplots(1, 4, figsize=(7.1, 2.5))
    plt.subplots_adjust(wspace=0.58)

    ax = axes[0]
    panel(ax, 'a')
    u, y = ccdf(pages['inputs']['handles_per_page'])
    ax.loglog(u, y, '-', color=INK, lw=1.1, drawstyle='steps-post', label='empirical')
    cc = mod['pages']['ccdf']
    grid = np.arange(1, cc.shape[1] + 1)
    med = np.median(cc, 0)
    ok = med > 0
    ax.loglog(grid[ok], med[ok], '-', color=MODEL, lw=1.2, label='model')
    ax.fill_between(grid[ok], np.maximum(np.percentile(cc, 5, 0)[ok], 1e-4), np.percentile(cc, 95, 0)[ok],
                    color=MODEL, alpha=0.25, lw=0)
    ax.set_xlim(0.9, 200)
    ax.set_ylim(1e-3, 1.1)
    ax.set_xlabel('handles per page, $k$')
    ax.set_ylabel('P($K \\geq k$)')
    ax.legend(frameon=False, fontsize=5.6, loc='lower left')

    for ax, stat, letter, ylabel, log in ((axes[1], 'gap', 'b', 'consistency within a page', True),
                                          (axes[2], 'vol', 'c', 'volatility over time', False),
                                          (axes[3], 'dev', 'd', 'distance from the prior', True)):
        panel(ax, letter)
        for kind, col in (('form', MODEL), ('name', AMBER)):
            m, lo, hi = md.distance_curve(sw, kind) if stat == 'dev' else md.curve(sw, kind, stat)
            ax.plot(md.S_GRID, m, '-', color=col, lw=1.3)
            ax.fill_between(md.S_GRID, np.maximum(lo, 1e-4) if log else lo, hi, color=col, alpha=0.18, lw=0)
        for p in pts:
            if np.isfinite(p[stat]):
                mk, col = MARK[p['kind']]
                ax.plot(p['s'], p[stat], mk, color=col, ms=3.4, mec=INK, mew=0.3, alpha=0.35)
        for kinds, col, mk in ((FORMS, INK, 'o'), (NAMES, NAME_BROWN, 's')):
            xs, ys, lo, hi = binned(pts, kinds, stat, log)
            ax.errorbar(xs, ys, yerr=[lo, hi], fmt=mk + '-', color=col, ms=3.8, lw=1.2, capsize=2, mfc='white',
                        mew=1.1, zorder=6)
            say(f'({letter}) binned {"forms" if kinds == FORMS else "names"}: '
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
    ax = axes[1]
    ax.plot([], [], '-', color=MODEL, lw=1.3, label='model, forms')
    ax.plot([], [], '-', color=AMBER, lw=1.3, label='model, names')
    ax.plot([], [], 'o-', color=INK, ms=3.8, mfc='white', mew=1.1, label='data, forms')
    ax.plot([], [], 's-', color=NAME_BROWN, ms=3.8, mfc='white', mew=1.1, label='data, names')
    ax.legend(frameon=False, fontsize=5.4, loc='upper right', handlelength=1.6)
    handles = [axes[1].plot([], [], mk, color=col, ms=3.4, mec=INK, mew=0.3, alpha=0.7, label=LABEL[k])[0]
               for k, (mk, col) in MARK.items()]
    fig.legend(handles=handles, frameon=False, fontsize=5.8, loc='lower center', bbox_to_anchor=(0.5, -0.17),
               ncol=5, handlelength=1.2, columnspacing=1.6)
    counts = {stat: {k: sum(1 for p in pts if p['kind'] in ks and np.isfinite(p[stat]))
                     for k, ks in (('forms', FORMS), ('names', NAMES))} for stat in ('gap', 'vol', 'dev')}
    say(f'points: {counts}')
    save(fig, 'fig4_model')
    say.write()


if __name__ == '__main__':
    main()
