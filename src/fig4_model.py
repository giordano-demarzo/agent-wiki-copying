"""Figure 3 of the paper (fig3_model): one model reproduces the collective, and can be solved.
(a) handles per page, record and model (median and 5-95% over 200 runs), with the Borel distribution;
(b) the distance from the prior along the prior-strength axis s = mu_A + mu_B, normalized by the
    fluctuation of N independent choices, sqrt(pi (1 - pi) / N), on log-log axes: the 1/s law of the
    Poisson branching argument (line, the root mean square; band, where single realizations fall), the
    simulations on the quarter-decade grid S_LOG (bullets, RMS over runs and directions) and the record
    (conventions, name features and months at their measured s, with RMS in quarter-decade bins). The exact moments of model_theory.py are not drawn.
Also written: figS_consistency (the consistency within a page along the same axis) and
figS_volatility (the volatility over time along the same axis), both for the SI.
"""
import numpy as np

import model as md
import model_theory as mt   # the Borel distribution of panel a
from common import Report, ccdf, load_cache
from style import CLASS_COLOR, GREY, INK, MODEL, NAME_BROWN, NAME_LIGHT, SLATE, panel, plt, save, use

# names in this figure: a light brown, so that amber keeps meaning GPT-6.1 Sol across the figures

EDGES = [0, 0.1, 0.25, 0.45, 0.65, 0.85, 1.05]
EDGES_LOG = list(10 ** np.arange(-1.75, 0.26, 0.25))   # quarter-decade bins for the log axis of Fig. 3b
MARK = {'coined': ('o', CLASS_COLOR['coined'][0]), 'semantic': ('^', CLASS_COLOR['semantic'][0]),
        'habit': ('v', CLASS_COLOR['habit'][0]), 'feature': ('s', NAME_LIGHT), 'month': ('p', GREY)}
LABEL = {'coined': 'coined names', 'semantic': 'near-synonyms', 'habit': 'habits of style',
         'feature': 'name features', 'month': 'months in names'}
FORMS, NAMES = {'coined', 'semantic', 'habit'}, {'feature', 'month'}


def normalized(f, mu_a, mu_b, n):
    """|f - pi| in units of the fluctuation of n independent choices at the prior share
    pi = mu_A / (mu_A + mu_B). A prior the fit puts at zero arrives here already
    replaced by its 95% upper bound (conventions.floor_mu), so pi is never 0 or 1."""
    if not (mu_a + mu_b > 0 and min(mu_a, mu_b) > 0):
        return np.nan
    pi = mu_a / (mu_a + mu_b)
    return abs(f - pi) / np.sqrt(pi * (1 - pi) / n)


def points(n_population):
    """The empirical points: conventions, name features and months."""
    forms = load_cache('forms.pkl')
    pts = []
    for e in forms['axis']:
        pi = e['muA'] / (e['muA'] + e['muB']) if e['muA'] + e['muB'] > 0 else np.nan
        pts.append(dict(e, kind=e['cls'], floored=forms['rows'][e['name']]['floored'],
                        devn=normalized(e['f'], e['muA_floor'], e['muB_floor'], forms['rows'][e['name']]['uses'])))
    names = load_cache('names.pkl')
    for name, s in names['stats'].items():
        kind = 'feature' if name in names['features'] else ('month' if s['cls'] == 'month' else None)
        if kind is None:
            continue
        mu = s['muA'] + s['muB']
        pi = s['muA'] / mu if mu > 0 else np.nan
        pts.append(dict(name=name, kind=kind, s=s['s'], s_floor=s['s'], gap=s['gap'], vol=s['vol'],
                        dev=max(abs(s['f'] - pi), 1e-3) if mu > 0 else np.nan,
                        devn=normalized(s['f'], s['muA'], s['muB'], n_population)))
    return pts


def binned(pts, kinds, stat, log, rms=False, edges=EDGES, skey='s'):
    """Means (geometric if log, root mean square if rms) in bins of s, with
    standard errors; a bin with a single point is drawn as that point. The bin
    is placed at the geometric mean of the s of its points when the bins are
    logarithmic."""
    xs, ys, lo, hi = [], [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        sel = [p for p in pts if p['kind'] in kinds and a <= p[skey] < b and np.isfinite(p[stat])]
        if not sel:
            continue
        y = np.array([p[stat] for p in sel])
        if rms:
            r = np.sqrt(np.mean(y ** 2))
            e = (np.std(y ** 2, ddof=1) / np.sqrt(len(y)) / (2 * r)) if len(y) > 1 else 0.0   # delta method
            xs.append(np.exp(np.mean(np.log([p[skey] for p in sel]))) if edges is EDGES_LOG else np.mean([p[skey] for p in sel]))
            ys.append(r)
            lo.append(min(e, r * 0.95))
            hi.append(e)
            continue
        y = np.log(y) if log else y
        m, e = y.mean(), (y.std(ddof=1) / np.sqrt(len(y)) if len(y) > 1 else 0.0)
        xs.append(np.mean([p[skey] for p in sel]))
        if log:
            ys.append(np.exp(m))
            lo.append(np.exp(m) - np.exp(m - e))
            hi.append(np.exp(m + e) - np.exp(m))
        else:
            ys.append(m)
            lo.append(e)
            hi.append(e)
    return np.array(xs), np.array(ys), np.array(lo), np.array(hi)


def bin_counts(pts, kinds, stat, edges, skey):
    """The number of points in each non-empty bin of `binned`."""
    n = [sum(1 for p in pts if p['kind'] in kinds and a <= p[skey] < b and np.isfinite(p[stat]))
         for a, b in zip(edges[:-1], edges[1:])]
    return [k for k in n if k]


def record_points(ax, pts, stat, log, say, letter, rms=False, edges=EDGES, skey='s', drop_floored=False):
    """The empirical points, faded, and their binned means, open. A convention
    with a prior fitted at zero is drawn in outline at the 95% upper bound of
    that prior: its direction pi is not identified, and with `drop_floored` it
    is left out of the bins (the bins with it are reported)."""
    for p in pts:
        if np.isfinite(p[stat]):
            mk, col = MARK[p['kind']]
            if p.get('floored'):
                ax.plot(p[skey], p[stat], mk, color=col, ms=3.4, mfc='none', mec=col, mew=0.8, alpha=0.8)
            else:
                ax.plot(p[skey], p[stat], mk, color=col, ms=3.4, mec=INK, mew=0.3, alpha=0.35)
    for kinds, col, mk in ((FORMS, MODEL, 'o'), (NAMES, NAME_BROWN, 's')):
        if drop_floored and any(p.get('floored') for p in pts if p['kind'] in kinds):
            xs, ys, _, _ = binned(pts, kinds, stat, log, rms, edges, skey)
            say(f'({letter}) {stat}, binned {"forms" if kinds == FORMS else "names"} with the priors at zero: '
                + ' '.join(f's={x:.2f}:{v:.3g}' for x, v in zip(xs, ys)))
        xs, ys, lo, hi = binned([p for p in pts if not (drop_floored and p.get('floored'))], kinds, stat, log, rms, edges, skey)
        ax.errorbar(xs, ys, yerr=[lo, hi], fmt=mk, color=col, ms=4.0, lw=1.2, capsize=2, mfc='white',
                    mew=1.1, zorder=6)
        kept = [p for p in pts if not (drop_floored and p.get('floored'))]
        say(f'({letter}) {stat}, binned {"forms" if kinds == FORMS else "names"}: '
            + ' '.join(f's={x:.2f}:{v:.3g}' for x, v in zip(xs, ys))
            + f' (points per bin {bin_counts(kept, kinds, stat, edges, skey)})')


def axis_panel(ax, stat, letter, ylabel, log, sw, pts, say, median=True):
    """One outcome along the prior-strength axis, from the simulations: bands for a
    convention and a name feature, the empirical points and their binned means."""
    panel(ax, letter)
    for kind, col in (('form', MODEL), ('name', NAME_BROWN)):
        m, lo, hi = md.distance_curve(sw, kind) if stat == 'dev' else md.curve(sw, kind, stat)
        if median:
            ax.plot(md.S_GRID, m, '-', color=col, lw=1.3)
        ax.fill_between(md.S_GRID, np.maximum(lo, 1e-4) if log else lo, hi, color=col, alpha=0.18, lw=0)
    record_points(ax, pts, stat, log, say, letter)
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


# quantiles of |Z| for a standard Gaussian Z: where single realizations fall around the RMS
Z10, Z90 = 0.1257, 1.6449


def distance_axes(ax):
    """The 1/s law of the Poisson branching argument (line, the root mean
    square; band, the 10th to 90th percentile of single realizations for a
    Gaussian final share) on the log-log axes of Fig. 3b."""
    g = np.logspace(np.log10(0.015), np.log10(1.15), 300)
    ax.fill_between(g, Z10 / g, Z90 / g, color=SLATE, alpha=0.2, lw=0)
    ax.plot(g, 1 / g, '-', color=SLATE, lw=1.3, zorder=4)
    ax.set_xlabel('prior strength, $s$')
    ax.set_ylabel('distance from the prior,\nnormalized')
    ax.set_xscale('log')
    ax.set_xticks([0.02, 0.1, 1])
    ax.set_xticklabels(['0.02', '0.1', '1'])
    ax.set_xlim(0.015, 1.15)      # after the ticks, which would otherwise widen the view
    ax.set_yscale('log')
    ax.set_ylim(0.02, 150)


def distance_panel(ax, letter, sw, n_model, pts, say):
    """The normalized distance from the prior against the 1/s law of the Poisson
    branching argument (Eq. var of the paper, no heterogeneity term): the line is
    the root mean square, the band the 10th to 90th percentile of single
    realizations for a Gaussian final share; bullets are the RMS over the
    simulations per kind, and the record its points and binned RMS."""
    panel(ax, letter)
    distance_axes(ax)
    for kind, col in (('form', MODEL), ('name', NAME_BROWN)):
        sims = []
        for s in md.S_LOG:
            v = np.concatenate([np.abs(sw[(kind, s, p)]['final'] - p) / np.sqrt(p * (1 - p) / n_model[kind])
                                for p in md.DIRECTIONS])
            sims.append(np.sqrt(np.mean(v ** 2)))
        ax.plot(md.S_LOG, sims, 'o', color=col, ms=3.0, mec=col, zorder=5)
        say(f'({letter}) {kind}: simulation RMS on the quarter-decade grid: '
            + ' '.join(f's={s:g}:{v:.2f}' for s, v in zip(md.S_LOG, sims)) + ' (the 1/s law: 1/s)')
    record_points(ax, pts, 'devn', True, say, letter, rms=True, edges=EDGES_LOG, skey='s_floor', drop_floored=True)


def prior_checks(say):
    """Extended Data Fig. 4: the record of Fig. 3b (a) with the priors fitted on first uses only
    and the distance normalized by the effective number of independent writers,
    (b) under the other convention selection rules of Table S(thresholds), and
    (c) the model's own conventions put through the record pipeline
    (si_recovery.py): binned at the refitted s against the true-parameter RMS."""
    d = load_cache('si_prior.pkl')
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.5))
    plt.subplots_adjust(wspace=0.4)
    ax = axes[0]
    panel(ax, 'a')
    distance_axes(ax)
    pts = [dict(kind=c['cls'], s_floor=c['s_first_floor'], devn=c['devn_first'], floored=c['floored_first'])
           for c in d['conventions'].values()]
    record_points(ax, pts, 'devn', True, say, 'a', rms=True, edges=EDGES_LOG, skey='s_floor', drop_floored=True)
    ax.set_title('first uses, independent writers', fontsize=7)
    ax = axes[1]
    panel(ax, 'b')
    distance_axes(ax)
    marks = iter('sD^v<>p')
    for rule in d['grid']:
        pts = [dict(p, kind='coined', s_floor=p['s']) for p in rule['points'] if not p['floored']]
        xs, ys, lo, hi = binned(pts, {'coined'}, 'devn', True, True, EDGES_LOG, 's_floor')
        lab = f'{rule["uses"]} uses, {rule["minority"]:g} minority, {rule["pages"]} pages'
        if rule['main']:
            ax.plot(xs, ys, 'o-', color=MODEL, ms=4.0, mfc='white', mew=1.1, lw=0.9, zorder=6, label=lab + ' (paper)')
        else:
            ax.plot(xs, ys, next(marks) + '-', color=GREY, ms=3.0, lw=0.6, alpha=0.8, label=lab)
        say(f'(b) rule {lab}: {len(pts)} conventions, binned ' + ' '.join(f's={x:.2f}:{v:.3g}' for x, v in zip(xs, ys))
            + f' (per bin {bin_counts(pts, {"coined"}, "devn", EDGES_LOG, "s_floor")})')
    ax.legend(frameon=False, fontsize=4.0, loc='lower left', handlelength=1.6)
    ax.set_title('other selection rules', fontsize=7)
    ax = axes[2]
    panel(ax, 'c')
    distance_axes(ax)
    rec = load_cache('si_recovery.pkl')['rows']
    true = [np.sqrt(np.mean([r['devn_true'] ** 2 for r in rec if r['s'] == s])) for s in md.S_LOG]
    ax.plot(md.S_LOG, true, 'o', color=MODEL, ms=3.0, zorder=5, label='true parameters')
    pts = [dict(kind='coined', s_floor=r['s_hat'], devn=r['devn_hat']) for r in rec
           if r['kept'] and not r['floored'] and np.isfinite(r['devn_hat'])]
    xs, ys, lo, hi = binned(pts, {'coined'}, 'devn', True, True, EDGES_LOG, 's_floor')
    ax.errorbar(xs, ys, yerr=[lo, hi], fmt='o', color=MODEL, ms=4.0, lw=1.2, capsize=2, mfc='white', mew=1.1, zorder=6,
                label='refitted as the record')
    say('(c) model refitted as the record, binned at the refitted s: '
        + ' '.join(f's={x:.3f}:{v:.3g} (1/s {1 / x:.3g}, ratio {v * x:.2f})' for x, v in zip(xs, ys))
        + f' (per bin {bin_counts(pts, {"coined"}, "devn", EDGES_LOG, "s_floor")})')
    ax.legend(frameon=False, fontsize=4.8, loc='lower left', handlelength=1.6)
    ax.set_title('the model, refitted', fontsize=7)
    plt.subplots_adjust(wspace=0.5)
    save(fig, 'ed_prior_fit')


def curve_legend(ax, loc='upper right', theory=True):
    if theory:
        ax.plot([], [], '-', color=SLATE, lw=1.3, label='theory, $1/s$')
        ax.fill_between([], [], [], color=SLATE, alpha=0.2, lw=0, label='theory, 10-90% of realizations')
        ax.plot([], [], 'o', color=MODEL, ms=3.0, label='conventions, simulations')
        ax.plot([], [], 'o', color=NAME_BROWN, ms=3.0, label='names, simulations')
    else:
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
    say = Report('Figure 3')
    mod = load_cache('model.pkl')
    pages = load_cache('pages.pkl')
    par, sw = mod['par'], mod['main']
    n_model = {'form': round(par['n_agents'] * par['edits'] * par['usage']), 'name': par['n_agents']}
    pts = points(par['n_agents'])
    say(f'normalization: N = {n_model["form"]} uses for a convention, {n_model["name"]} agents for a name feature '
        f'in the model; the record uses of each convention and {par["n_agents"]} handles in the record')
    floored = [(p['name'], p['s'], p['s_floor']) for p in pts if p['kind'] in FORMS and abs(p['s_floor'] - p['s']) > 1e-9]
    say('record conventions with a prior at zero, placed at the 95% upper bound of that prior: '
        + ', '.join(f'{n} (s {a:.3f} -> {b:.3f})' for n, a, b in floored))

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.1))
    plt.subplots_adjust(wspace=0.4)

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
    k, b = mt.borel_ccdf(par['c'])
    ax.loglog(k, b, '-', color=SLATE, lw=1.3, label='copy model, theory')
    ax.set_xlim(0.9, 200)
    ax.set_ylim(1e-3, 1.1)
    ax.set_xlabel('handles per page, $k$')
    ax.set_ylabel('P($K \\geq k$)')
    ax.legend(frameon=False, fontsize=5.6, loc='lower left')

    distance_panel(axes[1], 'b', mod['fig3'], n_model, pts, say)
    curve_legend(axes[1], loc='lower left')
    symbol_legend(fig, axes[1])
    counts = {stat: {k: sum(1 for p in pts if p['kind'] in ks and np.isfinite(p[stat]))
                     for k, ks in (('forms', FORMS), ('names', NAMES))} for stat in ('gap', 'vol', 'dev', 'devn')}
    say(f'points: {counts}')
    save(fig, 'fig3_model')

    # the consistency within a page and the volatility over time, along the same axis, for the SI
    for stat, name, ylabel, log in (('gap', 'figS_consistency', 'consistency within a page', True),
                                    ('vol', 'figS_volatility', 'volatility over time', False)):
        fig, ax = plt.subplots(1, 1, figsize=(3.4, 2.5))
        axis_panel(ax, stat, '', ylabel, log, sw, pts, say)
        curve_legend(ax, theory=False)
        symbol_legend(fig, ax, y=-0.2)
        save(fig, name)
    prior_checks(say)
    say.write()


if __name__ == '__main__':
    main()
