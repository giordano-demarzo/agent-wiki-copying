"""The figures of the Supplementary Information. Plot only: every quantity
comes from the cache written by the analysis steps and by run_model.py."""
import numpy as np
from matplotlib.colors import TwoSlopeNorm

import model as md
from common import ccdf, load_cache, wilson
from fig3_rule import binned_forms
from style import (AMBER, CLASS_COLOR, GREY, INDIGO, INK, MODEL, MOSS, ROSE, SLATE, panel, plt, save, use)

FOUR = [GREY, SLATE, INK, AMBER]


def day_ticks(days, min_gap=3):
    ticks = []
    for i in range(1, len(days)):
        if days[i] != days[i - 1] and (not ticks or i - ticks[-1] >= min_gap):
            ticks.append(i)
    return ticks


def curve(x, y, edges, nmin=20):
    xs, ps, lo, hi = [], [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (x >= a) & (x < b)
        if m.sum() >= nmin:
            p, l, h = wilson(y[m].sum(), m.sum())
            xs.append(x[m].mean())
            ps.append(p)
            lo.append(l)
            hi.append(h)
    return np.array(xs), np.array(ps), np.array(lo), np.array(hi)


def errline(ax, xs, ps, lo, hi, **kw):
    ax.errorbar(xs, ps, yerr=[ps - lo, hi - ps], capsize=0, **kw)


# ---------------------------------------------------------------- S1
def fig_population():
    d = load_cache('si_population.pkl')
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.2))
    plt.subplots_adjust(wspace=0.55)
    ax = axes[0]
    panel(ax, 'a')
    for x, color, ls, label in [(d['handle_span'], INK, '-', 'handles, first to last edit'),
                                (d['page_span'], AMBER, '-', 'pages, first to last edit'),
                                (d['readable'], AMBER, ':', 'pages, creation to deletion')]:
        u, y = ccdf(x / 60)
        ax.loglog(u, y, color=color, ls=ls, lw=1.1, label=label)
    for h, t, y in [(1, '1 h', 1.3), (24, '1 d', 1.3), (168, '1 wk', 1.9)]:
        ax.axvline(h, color=GREY, lw=0.5, ls=':')
        ax.text(h, y, t, color=SLATE, fontsize=5.4, ha='center')
    ax.set_xlabel('duration (hours)')
    ax.set_ylabel('P($X \\geq x$)')
    ax.set_ylim(1e-3, 2.6)
    ax.legend(frameon=False, fontsize=5.2, loc='lower left')
    ax = axes[1]
    panel(ax, 'b')
    u, y = ccdf(d['edits_per'])
    ax.loglog(u, y, 'o', color=INK, ms=2.4)
    ax.set_xlabel('edits by a handle, $k$')
    ax.set_ylabel('P($K \\geq k$)')
    ax = axes[2]
    panel(ax, 'c')
    sizes = sorted([r[1] for r in d['families']], reverse=True)
    ax.bar(range(1, len(sizes) + 1), sizes, color=INK, width=0.7)
    ax.set_yscale('log')
    ax.set_xlabel('task family, by rank')
    ax.set_ylabel('handles')
    save(fig, 'figS_population')


# ---------------------------------------------------------------- S3
def fig_pages_windows():
    d = load_cache('pages.pkl')
    mod = load_cache('model.pkl')
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.2))
    plt.subplots_adjust(wspace=0.6)
    ax = axes[0]
    panel(ax, 'a')
    for (w, v), color in zip(d['windows'].items(), FOUR):
        r = v['response']
        xs, ps, lo, hi = curve(r[:, 0], r[:, 1], np.array([1, 2, 4, 8, 16, 64]) / w)
        errline(ax, xs, ps, lo, hi, fmt='o-', color=color, ms=2.6, label=f'{w} lines, slope {v["slope"]:.2f}')
    ax.plot([0, 0.3], [0, 0.3], color=GREY, lw=0.6, ls='--')
    ax.set_xlim(0, 0.3)
    ax.set_ylim(0, 0.3)
    ax.set_xlabel("page's share of the feed window")
    ax.set_ylabel('P(write there)')
    ax.legend(frameon=False, fontsize=5.2, loc='upper left')
    ax = axes[1]
    panel(ax, 'b')
    u, y = ccdf(d['inputs']['handles_per_page'])
    ax.loglog(u, y, 'o', color=INK, ms=2.6, label='observed', zorder=5)
    for (m, v), color in zip(mod['windows'].items(), [GREY, MODEL, SLATE, AMBER]):
        med = np.median(v['ccdf'], 0)
        grid = np.arange(1, len(med) + 1)
        ok = med > 0
        ax.loglog(grid[ok], med[ok], '-', color=color, lw=1.1, label=f'model, {m} lines')
    ax.set_xlim(0.9, 300)
    ax.set_ylim(1e-3, 1.1)
    ax.set_xlabel('handles per page, $k$')
    ax.set_ylabel('P($K \\geq k$)')
    ax.legend(frameon=False, fontsize=5.2, loc='lower left')
    ax = axes[2]
    panel(ax, 'c')
    xs = np.arange(len(d['by_active']))
    ps = np.array([b[3] for b in d['by_active']])
    lo = np.array([b[4] for b in d['by_active']])
    hi = np.array([b[5] for b in d['by_active']])
    errline(ax, xs, ps, lo, hi, fmt='o', color=INK, ms=3)
    ax.axhline(d['inputs']['c'], color=MODEL, lw=1.0, ls='--', label=f'$c = {d["inputs"]["c"]:.2f}$')
    ax.set_xticks(xs)
    ax.set_xticklabels([f'{a}' if b - a == 1 else (f'{a}–{b - 1}' if b < 999 else f'{a}+') for a, b, *_ in d['by_active']])
    ax.set_ylim(0, 1)
    ax.set_xlabel('pages of the task active in the last 24 h')
    ax.set_ylabel('P(create a page)')
    ax.legend(frameon=False, fontsize=5.2, loc='upper right')
    save(fig, 'figS_pages_windows')


def fig_pages_attachment():
    d = load_cache('pages.pkl')
    fig, axes = plt.subplots(1, 2, figsize=(5.0, 2.2), gridspec_kw=dict(width_ratios=[1, 1.25]))
    plt.subplots_adjust(wspace=0.55)
    ax = axes[0]
    panel(ax, 'a')
    k, ref = d['kernel'], d['ref']
    errline(ax, k[:, 0], k[:, 1] / ref, k[:, 2] / ref, k[:, 3] / ref, fmt='o-', color=INK, ms=3, label='observed')
    xx = np.array([1, 60])
    ax.plot(xx, xx, color=GREY, lw=0.7, ls=':', label='linear')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('handles already on the page, $k$')
    ax.set_ylabel('$A(k)\\,/\\,A(1)$')
    ax.legend(frameon=False, fontsize=5.2, loc='upper left')
    ax = axes[1]
    panel(ax, 'b', x=-0.35)
    table = d['table']
    im = ax.imshow(table, cmap='Blues', vmin=0, vmax=np.nanmax(table), origin='lower', aspect='auto')
    for i in range(table.shape[0]):
        for j in range(table.shape[1]):
            if not np.isnan(table[i, j]):
                ax.text(j, i, f'{table[i, j]:.2f}', ha='center', va='center', fontsize=5,
                        color='white' if table[i, j] > 0.6 * np.nanmax(table) else INK)
    lab = lambda a, b: f'{a}' if b - a == 1 else (f'{a}–{b - 1}' if b < 99 else f'{a}+')
    ax.set_xticks(range(table.shape[1]))
    ax.set_xticklabels([lab(a, b) for a, b in d['handle_bins']], fontsize=5.6)
    ax.set_yticks(range(table.shape[0]))
    ax.set_yticklabels([lab(a, b) for a, b in d['line_bins']], fontsize=5.6)
    ax.set_xlabel('handles already there')
    ax.set_ylabel('lines held in the last 100')
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    cb.set_label('P(write there)', fontsize=6)
    cb.ax.tick_params(labelsize=5.6)
    save(fig, 'figS_pages_attachment')


# ---------------------------------------------------------------- S4
def fig_names_visibility():
    win = load_cache('identification.pkl')['windows']
    fig, axes = plt.subplots(1, 2, figsize=(5.4, 2.1), gridspec_kw=dict(width_ratios=[1.3, 1]))
    plt.subplots_adjust(wspace=0.45)
    cols = {1.5: INDIGO, 3: MOSS, 6: AMBER}
    ax = axes[0]
    panel(ax, 'a')
    ax.set_title('predecessors of the same task', fontsize=5.8, color=SLATE)
    for k, w in enumerate((1.5, 3, 6)):
        r = win[w]['same']
        ax.errorbar(np.arange(3) + (k - 1) * 0.18, r['b'], yerr=1.96 * r['se'], fmt='o', color=cols[w], ms=3.4,
                    capsize=1.5, lw=0.8, label=f'born in the last {w:g} h')
    ax.axhline(0, color=GREY, lw=0.7)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["on the\nnewcomer's page", 'in the last 100\nfeed lines', 'out of view'], fontsize=5.2)
    ax.set_xlim(-0.6, 2.6)
    ax.set_ylabel('coefficient estimate')
    ax.set_ylim(-0.3, 0.7)
    ax.legend(frameon=False, fontsize=4.6, loc='upper right', handlelength=1.2)
    ax = axes[1]
    panel(ax, 'b')
    ax.set_title('predecessors of other tasks', fontsize=5.8, color=SLATE)
    for k, w in enumerate((1.5, 3, 6)):
        r = win[w]['other']
        ax.errorbar(np.arange(2) + (k - 1) * 0.18, r['b'], yerr=1.96 * r['se'], fmt='o', color=cols[w], ms=3.4,
                    capsize=1.5, lw=0.8)
    ax.axhline(0, color=GREY, lw=0.7)
    ax.set_xticks(range(2))
    ax.set_xticklabels(['in the last 100\nfeed lines', 'out of view'], fontsize=5.2)
    ax.set_xlim(-0.6, 1.6)
    ax.set_ylim(-0.3, 0.7)
    save(fig, 'figS_names_visibility')


def fig_names_fashions():
    d = load_cache('si_names.pkl')
    days = d['block_days']
    fig, axes = plt.subplots(1, 4, figsize=(7.1, 2.0))
    plt.subplots_adjust(wspace=0.45)
    ticks = [i for i in range(len(days)) if i == 0 or days[i] != days[i - 1]]
    for ax, (t, f), letter in zip(axes, d['fashions'].items(), 'abcd'):
        panel(ax, letter)
        obs, runs = f['observed'], f['runs']
        x = np.arange(len(obs))
        ax.fill_between(x, np.percentile(runs, 10, 0), np.percentile(runs, 90, 0), color=MODEL, alpha=0.25, lw=0)
        ax.plot(x, np.median(runs, 0), color=MODEL, lw=1.1, ls='--', label='copying replay')
        ax.plot(x, obs, 'o-', color=INK, ms=2.4, label='observed')
        ax.set_ylim(0, 1)
        ax.set_xticks(ticks)
        ax.set_xticklabels([days[i] for i in ticks], fontsize=5.2, rotation=40, ha='right')
        ax.text(0.97, 0.93, t, ha='right', va='top', fontsize=6.5, transform=ax.transAxes)
        if letter == 'a':
            ax.set_ylabel('share of newborns with the piece')
        if letter == 'c':
            ax.legend(frameon=False, fontsize=5.2, loc='upper right', bbox_to_anchor=(1, 0.9))
    fig.text(0.5, -0.2, 'newborn handles in order of birth, blocks of 100', ha='center', fontsize=7.4)
    save(fig, 'figS_names_fashions')


# ---------------------------------------------------------------- S5
def fig_forms_each():
    d = load_cache('si_forms.pkl')
    keep = d['keep']
    fig, axes = plt.subplots(3, 6, figsize=(7.1, 4.0))
    plt.subplots_adjust(wspace=0.45, hspace=0.6)
    for ax in axes.flat[len(keep):]:
        ax.axis('off')
    for ax, name in zip(axes.flat, keep):
        c = d['conv'][name]
        color = CLASS_COLOR[c['cls']][0]
        xs, ps, lo, hi = binned_forms(c['records'])
        errline(ax, xs, ps, lo, hi, fmt='o', color=color, ms=2.4)
        mu = d['fits'][name]
        rr = np.array([0, 1])
        ax.plot(rr, mu[0] + (1 - mu[0] - mu[1]) * rr, color=color, lw=0.9)
        ax.plot([0, 1], [0, 1], color=GREY, lw=0.5, ls=':')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([0, 0.5, 1])
        ax.set_yticks([0, 0.5, 1])
        ax.tick_params(labelsize=5.4)
        ax.set_title(name, fontsize=5.8, color=color, pad=2)
        ax.text(0.96, 0.06, f'n = {len(c["records"])}', ha='right', va='bottom', fontsize=5, color=SLATE,
                transform=ax.transAxes)
    for ax in axes[:, 0]:
        ax.set_ylabel('P(first form)', fontsize=6.2)
    fig.text(0.5, 0.04, 'share of the first form on the page', ha='center', fontsize=7.4)
    save(fig, 'figS_forms_each')


def fig_forms_heatmap():
    d = load_cache('si_forms.pkl')
    keep = d['keep']
    heat = np.array([d['conv'][n]['heat'] for n in keep])
    fig, ax = plt.subplots(figsize=(7.1, 3.4))
    cmap = plt.get_cmap('RdBu').copy()
    cmap.set_bad('#e6e6e6')
    im = ax.imshow(np.ma.masked_invalid(heat), cmap=cmap, norm=TwoSlopeNorm(0.5, 0, 1), aspect='auto',
                   interpolation='nearest')
    ax.set_yticks(range(len(keep)))
    ax.set_yticklabels([f'{n}  ({d["conv"][n]["n_first"]})' for n in keep], fontsize=6)
    for lab, n in zip(ax.get_yticklabels(), keep):
        lab.set_color(CLASS_COLOR[d['conv'][n]['cls']][0])
    days = d['block_days']
    ticks = [t for t in day_ticks(days, min_gap=2) if t < len(days) - 1]
    ax.set_xticks(ticks)
    ax.set_xticklabels([days[i] for i in ticks], fontsize=6, rotation=30, ha='right')
    ax.set_xlabel('all handles in order of birth, blocks of 50')
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, ticks=[0, 0.5, 1])
    cb.ax.set_yticklabels(['all second form', 'split', 'all first form'], fontsize=5.8)
    cb.set_label("share of the block's first uses\ntaking the first form", fontsize=6)
    save(fig, 'figS_forms_heatmap')


def fig_forms_prior():
    d = load_cache('si_forms.pkl')
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    for n in d['keep']:
        c = d['conv'][n]
        color = CLASS_COLOR[c['cls']][0]
        s = max(c['s'], 0.01)
        ax.plot([s, s], [c['ceiling'], c['late']], color=color, lw=0.6, ls=':')
        ax.plot(s, c['ceiling'], 'o', color=color, ms=3.4, mfc='white', mew=0.9)
        ax.plot(s, c['late'], 'o', color=color, ms=4)
        ax.annotate(n.split(' / ')[0], (s, c['late']), xytext=(3, 0), textcoords='offset points', fontsize=4.6,
                    color=color, va='center')
    ax.axvspan(0.009, 0.1, color=GREY, alpha=0.18, lw=0)
    ax.set_xscale('log')
    ax.set_xlim(0.009, 1)
    ax.set_ylim(0.3, 1.05)
    ax.set_xlabel('prior strength $\\mu_A + \\mu_B$')
    ax.set_ylabel('share of the prevailing form,\nlast fifth of first uses')
    ax.plot([], [], 'o', color=SLATE, ms=4, label='observed')
    ax.plot([], [], 'o', color=SLATE, ms=3.4, mfc='white', mew=0.9, label='bias ceiling $\\max(\\pi, 1-\\pi)$')
    ax.legend(frameon=False, fontsize=5.2, loc='lower left')
    save(fig, 'figS_forms_prior')


def fig_forms_controls():
    d = load_cache('si_forms.pkl')
    pw = {p['name']: p for p in load_cache('forms.pkl')['patchwork']}
    keep = d['keep']
    fig, axes = plt.subplots(1, 2, figsize=(5.4, 2.4), gridspec_kw=dict(width_ratios=[1.3, 1]))
    plt.subplots_adjust(wspace=0.65)
    ax = axes[0]
    panel(ax, 'a', x=-0.75)
    for i, n in enumerate(keep):
        color = CLASS_COLOR[d['conv'][n]['cls']][0]
        ax.plot([pw[n]['shuffled'], pw[n]['same']], [i, i], color=color, lw=0.8)
        ax.plot(pw[n]['shuffled'], i, 'o', color=color, ms=3.2, mfc='white', mew=0.9)
        ax.plot(pw[n]['same'], i, 'o', color=color, ms=3.8)
    ax.set_yticks(range(len(keep)))
    ax.set_yticklabels(keep, fontsize=5.4)
    for lab, n in zip(ax.get_yticklabels(), keep):
        lab.set_color(CLASS_COLOR[d['conv'][n]['cls']][0])
    ax.invert_yaxis()
    ax.set_xlim(0.5, 1.02)
    ax.set_xlabel('P(two uses on the same page agree)')
    ax.plot([], [], 'o', color=SLATE, ms=3.8, label='observed')
    ax.plot([], [], 'o', color=SLATE, ms=3.2, mfc='white', mew=0.9, label='shuffled within 3-h blocks')
    ax.legend(frameon=False, fontsize=5, loc='lower left')
    ax = axes[1]
    panel(ax, 'b')
    now = np.vstack([d['conv'][n]['records'] for n in keep])
    then = np.vstack([d['conv'][n]['shifted'] for n in keep])
    xs, ps, lo, hi = binned_forms(now)
    errline(ax, xs, ps, lo, hi, fmt='o-', color=INK, ms=3, label='page at the time of the use')
    xs, ps, lo, hi = binned_forms(then)
    errline(ax, xs, ps, lo, hi, fmt='s--', color=SLATE, ms=3, mfc='white', mew=1, label='page 6 h earlier')
    ax.plot([0, 1], [0, 1], color=GREY, lw=0.6, ls=':')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('share of the form on the page')
    ax.set_ylabel('P(use that form)')
    ax.legend(frameon=False, fontsize=5.2, loc='upper left')
    save(fig, 'figS_forms_controls')


def fig_family():
    d = load_cache('si_forms.pkl')
    f, n = d['family'], d['names']
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.2), gridspec_kw=dict(width_ratios=[1, 1.15, 1]))
    plt.subplots_adjust(wspace=0.55)
    ax = axes[0]
    panel(ax, 'a')
    bars = [(f['conflict'][c], CLASS_COLOR[c][0]) for c in ('coined', 'semantic', 'habit')]
    bars += [(f['conflict']['pooled'], INK), (n['conflict'], AMBER)]
    xs = [0, 1, 2, 3, 4.5]
    for x, ((p, lo, hi, _), color) in zip(xs, bars):
        ax.bar(x, p, color=color, width=0.8)
        ax.errorbar(x, p, yerr=[[p - lo], [hi - p]], fmt='none', ecolor=INK, capsize=1.5, lw=0.7)
    ax.axhline(0.5, color=GREY, ls=':', lw=0.7)
    ax.set_xticks(xs)
    ax.set_xticklabels(['coined names', 'near-synonyms', 'habits', 'all forms', 'names'], fontsize=5.2, rotation=30,
                       ha='right')
    ax.tick_params(axis='x', length=0)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('P(follow the page)\nwhen its task disagrees')
    ax = axes[1]
    panel(ax, 'b')
    bf, sf, _ = f['joint']
    bn, sn = n['joint']
    for label, b, se, off in (('forms', bf, sf, 0), ('names', bn, sn, 4)):
        for j, color in enumerate([INK, AMBER, SLATE]):
            ax.bar(off + j, b[j], color=color, width=0.75)
            ax.errorbar(off + j, b[j], yerr=1.96 * se[j], fmt='none', ecolor=INK, capsize=1.5, lw=0.7)
        ax.text(off + 1, 0.95, label, ha='center', va='bottom', fontsize=6, color=SLATE)
    ax.axvline(3.5, color=GREY, lw=0.5, ls=':')
    ax.set_xticks([0, 1, 2, 4, 5, 6])
    ax.set_xticklabels(['page', 'same task', 'feed', "page's authors", 'same task', 'feed'], fontsize=5.2, rotation=30,
                       ha='right')
    ax.set_ylim(-0.05, 1.0)
    ax.set_ylabel('fitted coefficient')
    ax = axes[2]
    panel(ax, 'c')
    for lab, color, marker in [('task says A', INK, 'o'), ('task says B', AMBER, 's')]:
        xs, ps, lo, hi = binned_forms(f['split'][lab.replace('task', 'family')])
        errline(ax, xs, ps, lo, hi, fmt=f'{marker}-', color=color, ms=3, label=lab)
    ax.plot([0, 1], [0, 1], color=GREY, lw=0.6, ls=':')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel('share of form A on the page')
    ax.set_ylabel('P(use form A)')
    ax.legend(frameon=False, fontsize=5.2, loc='upper left', title="the task's other pages", title_fontsize=5.2)
    save(fig, 'figS_family')


# ---------------------------------------------------------------- S6, S7
def fig_unpredictability():
    sw = load_cache('model.pkl')['main']
    fig, axes = plt.subplots(1, 2, figsize=(4.8, 2.2))
    plt.subplots_adjust(wspace=0.5)
    ax = axes[0]
    panel(ax, 'a')
    rng = np.random.default_rng(0)
    show = [0.02, 0.1, 0.3, 0.6, 1.0]
    for j, s in enumerate(show):
        fin = sw[('form', s, 0.5)]['final']
        ax.plot(j + rng.uniform(-0.18, 0.18, len(fin)), fin, 'o', color=MODEL, ms=3, alpha=0.7, mec='none')
    ax.axhline(0.5, color=GREY, lw=0.7, ls=':')
    ax.set_xticks(range(len(show)))
    ax.set_xticklabels([f'{s:g}' for s in show])
    ax.set_ylim(-0.03, 1.03)
    ax.set_xlabel('prior strength, $s$')
    ax.set_ylabel('final share of a form')
    ax = axes[1]
    panel(ax, 'b')
    for kind, col, lab in (('form', MODEL, 'forms'), ('name', AMBER, 'names')):
        sd = [2 * np.std(sw[(kind, s, 0.5)]['final']) for s in md.S_GRID]
        ax.plot(md.S_GRID, sd, 'o-', color=col, lw=1.2, ms=2.5, label=lab)
    ax.set_xlabel('prior strength, $s$')
    ax.set_ylabel('2 s.d. of the final share')
    ax.set_ylim(0, 1.02)
    ax.legend(frameon=False, fontsize=5.6)
    save(fig, 'figS_unpredictability')


def fig_prior_direction():
    sw = load_cache('model.pkl')['main']
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.2))
    plt.subplots_adjust(wspace=0.5)
    col = dict(zip(md.DIRECTIONS, (MOSS, INDIGO, INK, ROSE, AMBER)))
    for ax, stat, letter, ylabel in ((axes[0], 'gap', 'a', 'consistency within a page'),
                                     (axes[1], 'vol', 'b', 'volatility over time'),
                                     (axes[2], 'final', 'c', 'final share of form A')):
        panel(ax, letter)
        for kind, ls in (('form', '-'), ('name', '--')):
            for p in md.DIRECTIONS:
                m = md.curve(sw, kind, stat, directions=(p,))[0]
                ax.plot(md.S_GRID, m, ls, color=col[p], lw=1.1, label=f'{"forms" if kind == "form" else "names"}, $\\pi={p}$')
        ax.set_xlabel('prior strength, $s$')
        ax.set_ylabel(ylabel)
        ax.set_xlim(-0.03, 1.03)
        if stat == 'gap':
            ax.set_yscale('log')
            ax.set_ylim(0.5, 20)
        if stat != 'final':
            ax.axhline(1, color=GREY, lw=0.7, ls=':')
    axes[0].legend(frameon=False, fontsize=5, loc='upper right', handlelength=1.6)
    axes[2].set_ylim(0, 1)
    save(fig, 'figS_prior_direction')


def fig_perconv():
    fitted = load_cache('model.pkl')['fitted']
    axis = {e['name']: e for e in load_cache('forms.pkl')['axis']}
    fig, axes = plt.subplots(1, 3, figsize=(6.4, 2.2))
    plt.subplots_adjust(wspace=0.5)
    for ax, k, letter, lab, log, lim in ((axes[0], 'gap', 'a', 'consistency within a page', True, (0.5, 30)),
                                        (axes[1], 'vol', 'b', 'volatility over time', False, (0, 7)),
                                        (axes[2], 'final', 'c', 'final share of form A', False, (0, 1))):
        panel(ax, letter)
        for n, runs in fitted.items():
            o = axis[n]['f'] if k == 'final' else axis[n][k]
            m = runs[k][np.isfinite(runs[k])]
            if not np.isfinite(o) or len(m) < 6 or (log and o <= 0):
                continue
            med, lo, hi = np.median(m), np.percentile(m, 10), np.percentile(m, 90)
            ax.errorbar(o, med, yerr=[[med - lo], [hi - med]], fmt='o', color=CLASS_COLOR[axis[n]['cls']][0], mec=INK,
                        mew=0.4, ms=3.6, ecolor=GREY, elinewidth=0.7, zorder=3)
        ax.plot(lim, lim, '--', color=GREY, lw=0.7)
        ax.set_xlim(*lim)
        ax.set_ylim(*lim)
        ax.set_xlabel('observed')
        ax.set_ylabel('model')
        ax.set_title(lab, fontsize=6.5, color=SLATE)
        if log:
            ax.set_xscale('log')
            ax.set_yscale('log')
    save(fig, 'figS_perconv')


# ---------------------------------------------------------------- S8
def fig_regressions():
    d = load_cache('identification.pkl')
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.3), gridspec_kw=dict(width_ratios=[1.2, 1, 1]))
    plt.subplots_adjust(wspace=0.5)
    for ax, key, letter, title in ((axes[0], 'pages', 'a', 'where to write'), (axes[1], 'names', 'b', 'how to sign'),
                                   (axes[2], 'forms', 'c', 'how to write')):
        panel(ax, letter)
        ax.set_title(title, fontsize=7, color=SLATE)
        r = d[key]
        x = np.arange(len(d['labels'][key]))
        ax.errorbar(x - 0.12, r['b0'], yerr=1.96 * r['se0'], fmt='o', color=GREY, ms=3.4, capsize=1.5, lw=0.8,
                    label='baseline fixed effects')
        ax.errorbar(x + 0.12, r['b1'], yerr=1.96 * r['se1'], fmt='o', color=INK, ms=3.4, capsize=1.5, lw=0.8,
                    label='+ task x 3 h fixed effects')
        ax.axhline(0, color=GREY, lw=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(d['labels'][key], fontsize=4.6, rotation=35, ha='right', rotation_mode='anchor')
        ax.set_ylabel('coefficient estimate')
    axes[0].legend(frameon=False, fontsize=4.6, loc='upper right')
    save(fig, 'figS_regressions')


def main():
    use()
    for f in (fig_population, fig_pages_windows, fig_pages_attachment, fig_names_visibility, fig_names_fashions,
              fig_forms_each, fig_forms_heatmap, fig_forms_prior, fig_forms_controls, fig_family,
              fig_unpredictability, fig_prior_direction, fig_perconv, fig_regressions):
        f()


if __name__ == '__main__':
    main()
