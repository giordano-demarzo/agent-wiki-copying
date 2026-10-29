"""The Extended Data figures drawn from the record and the model. Plot only:
every quantity comes from the cache written by the analysis steps.

    ED1  ed_attention_patchwork  attention concentrates; the patchwork against two permutation nulls
    ED2  ed_exposure_task        exposure against shared task conditions: conflicts, contrasts, windows
    ED3  ed_perconv              each convention against the model run at its own fitted priors
ED4 (the prior fit and the 1/s law) is drawn by fig4_model.py, ED5 (checking in two models) by
lab/ed_checking_models.py.
"""
import collections
import datetime

import matplotlib.dates as mdates
import numpy as np

from common import Dataset, Report, T, contrast, load_cache, save_cache
from style import CLASS_COLOR, GREY, INK, NAME_BROWN, NAME_DARK, NAME_LIGHT, SLATE, TEAL, panel, plt, save, use

H = 6
TOP = 5
MIN_EDITS = 30


def attention(ds):
    """Share of a block's task edits on its TOP busiest pages, and the share if
    the edits were spread evenly over the pages written on in the block."""
    task = [r for r in ds.revs if ds.is_task(r)]
    times = np.array([T(r['time']) for r in task])
    t0, t1 = datetime.datetime(2026, 6, 15), datetime.datetime(2026, 6, 23)
    edges = [t0 + datetime.timedelta(hours=H * i) for i in range(int((t1 - t0).total_seconds() // 3600 // H) + 1)]
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = [r for r, t in zip(task, times) if a <= t < b]
        if len(m) < MIN_EDITS:
            continue
        c = np.array(sorted(collections.Counter(r['page_id'] for r in m).values(), reverse=True), float)
        out.append(dict(mid=a + (b - a) / 2, edits=len(m), pages=len(c), top=c[:TOP].sum() / len(m),
                        even=min(TOP, len(c)) / len(c)))
    return out, len(task), sum(1 for t in times if t0 <= t < t1)


def ed1(say):
    ds = Dataset()
    blocks, n_task, n_in = attention(ds)
    pw = load_cache('ed_patchwork.pkl')
    fig, axes = plt.subplots(1, 2, figsize=(6.2, 2.9), gridspec_kw=dict(width_ratios=[1, 1.25]))
    plt.subplots_adjust(wspace=0.75)

    ax = axes[0]
    panel(ax, 'a')
    xs = [b['mid'] for b in blocks]
    ax.plot(xs, [b['top'] for b in blocks], 'o-', color=INK, ms=2.6, label=f'{TOP} busiest pages of the block')
    ax.plot(xs, [b['even'] for b in blocks], ':', color=SLATE, lw=1.1, label='the same edits spread evenly')
    ax.set_ylim(0, 1)
    ax.set_ylabel('share of the task edits of the block')
    ax.xaxis.set_major_locator(mdates.DayLocator(bymonthday=[16, 18, 20, 22]))
    ax.set_xlim(datetime.datetime(2026, 6, 15, 12), datetime.datetime(2026, 6, 22, 12))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d Jun'))
    ax.tick_params(axis='x', labelsize=5.8)
    ax.legend(frameon=False, fontsize=5.4, loc='upper center', bbox_to_anchor=(0.5, -0.14), handlelength=1.6)
    edits = [b['edits'] for b in blocks]
    pages = [b['pages'] for b in blocks]
    late = [b for b in blocks if b['mid'] >= datetime.datetime(2026, 6, 18)]
    save_cache('fig2.pkl', dict(top_late=(min(b['top'] for b in late), max(b['top'] for b in late)),
                                even_late=(min(b['even'] for b in late), max(b['even'] for b in late)),
                                n_blocks=len(blocks), edits_in=sum(edits), edits_period=n_in, edits=(min(edits), max(edits)),
                                pages=(min(pages), max(pages)), under_top=sum(p < TOP for p in pages)))
    say(f'ED1a: {len(blocks)} blocks of {H} h with at least {MIN_EDITS} task edits ({sum(edits)} of the {n_in} task edits '
        f'between 15 and 23 June; {n_task} task edits in all); edits per block {min(edits)}-{max(edits)} (median '
        f'{np.median(edits):.0f}), pages written on {min(pages)}-{max(pages)} (median {np.median(pages):.0f}); blocks with '
        f'fewer than {TOP} pages {sum(p < TOP for p in pages)}; top-{TOP} share median {np.median([b["top"] for b in blocks]):.2f}, '
        f'from 18 June {min(b["top"] for b in late):.2f}-{max(b["top"] for b in late):.2f} against '
        f'{min(b["even"] for b in late):.2f}-{max(b["even"] for b in late):.2f} if spread evenly')

    ax = axes[1]
    panel(ax, 'b', x=-0.55)
    rows = sorted(pw['rows'], key=lambda r: r['same'] - r['task']['median'])
    for j, r in enumerate(rows):
        col = CLASS_COLOR[r['cls']][0]
        ax.plot([r['diff'], r['same']], [j, j], '-', color=col, lw=0.9, alpha=0.5)
        ax.plot([r['time']['lo'], r['time']['hi']], [j + 0.22] * 2, '-', color=GREY, lw=2.2, solid_capstyle='butt')
        ax.plot([r['task']['lo'], r['task']['hi']], [j - 0.22] * 2, '-', color=SLATE, lw=2.2, solid_capstyle='butt')
        ax.errorbar(r['same'], j, xerr=[[r['same'] - r['same_lo']], [r['same_hi'] - r['same']]], fmt='o', color=col,
                    ms=3.8, elinewidth=0.6, capsize=0, zorder=3)
        ax.plot(r['diff'], j, 'o', color=col, ms=3, mfc='white', mew=0.9, zorder=3)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r['name'] for r in rows], fontsize=5.2)
    ax.tick_params(axis='y', length=0)
    ax.set_xlim(0.4, 1.02)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_xlabel('P(two uses take the same form)')
    h = [ax.plot([], [], 'o', color=SLATE, ms=3.8)[0], ax.plot([], [], 'o', color=SLATE, ms=3, mfc='white', mew=0.9)[0],
         ax.plot([], [], '-', color=GREY, lw=2.2)[0], ax.plot([], [], '-', color=SLATE, lw=2.2)[0]]
    labs = ['same page', 'different pages', 'same page, shuffled within 3 h', 'same page, shuffled within task and 3 h']
    for col, lab in CLASS_COLOR.values():
        h.append(ax.plot([], [], 's', color=col, ms=3.5)[0])
        labs.append(lab)
    ax.legend(h, labs, frameon=False, fontsize=5.2, loc='upper center', bbox_to_anchor=(0.4, -0.14), ncol=2,
              handlelength=1.4, columnspacing=1.0)
    save(fig, 'ed_attention_patchwork')


def ed2(say):
    d = load_cache('identification.pkl')
    f = load_cache('si_forms.pkl')
    fam, nam = f['family'], f['names']
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.5), gridspec_kw=dict(width_ratios=[0.85, 1.2, 0.9]))
    plt.subplots_adjust(wspace=0.75)

    # (a) conflicts between the page and the other pages of the same task
    ax = axes[0]
    panel(ax, 'a')
    bars = [(fam['conflict'][c], CLASS_COLOR[c][0]) for c in ('coined', 'semantic', 'habit')]
    bars += [(fam['conflict']['pooled'], INK), (nam['conflict'], NAME_BROWN)]
    xs = [0, 1, 2, 3, 4.5]
    for x, ((p, lo, hi, n), color) in zip(xs, bars):
        ax.bar(x, p, color=color, width=0.8)
        ax.errorbar(x, p, yerr=[[p - lo], [hi - p]], fmt='none', ecolor=INK, capsize=1.5, lw=0.7)
        ax.text(x, 0.03, f'{n}', ha='center', va='bottom', fontsize=4.8, color='white')
    ax.axhline(0.5, color=GREY, ls=':', lw=0.7)
    ax.set_xticks(xs)
    ax.set_xticklabels(['coined names', 'near-synonyms', 'habits', 'all forms', 'names'], fontsize=5.4, rotation=35,
                       ha='right', rotation_mode='anchor')
    ax.tick_params(axis='x', length=0)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel('P(follows the page)\nwhen the rest of the task disagrees')
    say('ED2a: ' + ', '.join(f'{lab} {p:.2f} [{lo:.2f}, {hi:.2f}] (n={n})' for lab, (p, lo, hi, n) in
                             zip(('coined', 'near-synonyms', 'habits', 'all forms', 'names'), [b for b, _ in bars])))

    # (b) contrasts with the full clustered covariance
    ax = axes[1]
    panel(ax, 'b', x=-0.5)
    n_, f_ = d['names'], d['forms']
    items = [
        ('names', 'feed $-$ out of view', 'feature, task$\\times$3 h', contrast(n_['b1'], n_['cov1'], [0, 1, -1, 0])),
        ('names', 'feed $-$ out of view', '+ feature$\\times$task', contrast(n_['b2'], n_['cov2'], [0, 1, -1, 0])),
        ('names', 'page $-$ out of view', 'feature, task$\\times$3 h', contrast(n_['b1'], n_['cov1'], [1, 0, -1, 0])),
        ('names', 'page $-$ out of view', '+ feature$\\times$task', contrast(n_['b2'], n_['cov2'], [1, 0, -1, 0])),
        ('names', "page $-$ task's other pages", 'joint fit', nam['contrast']),
        ('forms', 'page $-$ out of view', 'convention, task$\\times$3 h', contrast(f_['b1'], f_['cov1'], [1, 0, -1, 0])),
        ('forms', 'page $-$ out of view', 'convention$\\times$task$\\times$3 h', contrast(f_['bx'], f_['covx'], [1, 0, -1, 0])),
        ('forms', 'feed $-$ out of view', 'convention, task$\\times$3 h', contrast(f_['b1'], f_['cov1'], [0, 1, -1, 0])),
        ('forms', 'feed $-$ out of view', 'convention$\\times$task$\\times$3 h', contrast(f_['bx'], f_['covx'], [0, 1, -1, 0])),
        ('forms', "page $-$ task's other pages", 'joint fit', fam['contrast']),
    ]
    # one row per contrast; filled: the specification of Table 1, open: stricter intercepts
    rows = [('names', 'feed $-$ out of view'), ('names', 'page $-$ out of view'), ('names', "page $-$ task's\nother pages"),
             ('forms', 'page $-$ out of view'), ('forms', 'feed $-$ out of view'), ('forms', "page $-$ task's\nother pages")]
    yrow = {r: y for r, y in zip(rows, [6.4, 5.4, 4.4, 2.4, 1.4, 0.4])}
    for who, what, spec, c in items:
        key = (who, what if 'other pages' not in what else what.replace("task's other", "task's\nother"))
        strict = spec.startswith('+') or spec.startswith('convention$')
        y = yrow[key] + (-0.18 if strict else (0.18 if any(k == key for k in yrow) and 'joint' not in spec else 0))
        col = NAME_BROWN if who == 'names' else INK
        ax.errorbar(c['est'], y, xerr=1.96 * c['se'], fmt='s' if strict else 'o', color=col, mfc='white' if strict else col,
                    ms=3.4, mew=0.9, elinewidth=0.8, capsize=0)
        say(f'ED2b: {who:5} {what:28} [{spec}] {c["est"]:+.2f} [{c["lo"]:+.2f}, {c["hi"]:+.2f}] p {c["p"]:.2g}')
    ax.axvline(0, color=GREY, lw=0.7)
    ax.set_yticks(list(yrow.values()))
    ax.set_yticklabels([w for _, w in yrow], fontsize=5.4)
    for t, (who, _) in zip(ax.get_yticklabels(), yrow):
        t.set_color(NAME_BROWN if who == 'names' else INK)
    ax.tick_params(axis='y', length=0)
    ax.set_ylim(-0.3, 7.1)
    ax.set_xlabel('difference between coefficients')
    ax.set_xlim(-0.25, 1.0)
    ax.text(0.98, 6.9, 'names', ha='right', va='center', fontsize=6, color=NAME_BROWN)
    ax.text(0.98, 2.9, 'forms', ha='right', va='center', fontsize=6, color=INK)
    h = [ax.plot([], [], 'o', color=SLATE, ms=3.4)[0], ax.plot([], [], 's', color=SLATE, mfc='white', ms=3.4)[0],
         ax.plot([], [], 'o', color=SLATE, ms=3.4)[0]]
    ax.legend(h[:2], ['intercepts of Table 1', 'stricter intercepts'], frameon=False, fontsize=5.0, loc='center left',
              bbox_to_anchor=(0.0, 0.47), handletextpad=0.3)

    # (c) the window of the task-mates, names
    ax = axes[2]
    panel(ax, 'c', x=-0.42)
    win = d['windows']
    ws = sorted(win)
    for k, (lab, c, col) in enumerate((('feed $-$ out of view', [0, 1, -1, 0], NAME_DARK),
                                       ('page $-$ out of view', [1, 0, -1, 0], NAME_LIGHT))):
        cs = [contrast(win[w]['same']['b'], win[w]['same']['cov'], c) for w in ws]
        x = np.arange(len(ws)) + (k - 0.5) * 0.2
        ax.errorbar(x, [c_['est'] for c_ in cs], yerr=[1.96 * c_['se'] for c_ in cs], fmt='o', color=col, ms=3.2,
                    capsize=0, elinewidth=0.8, label=lab)
        say(f'ED2c: names {lab}: ' + ', '.join(f'{w:g} h {c_["est"]:+.2f} [{c_["lo"]:+.2f}, {c_["hi"]:+.2f}] '
                                               f'(n={win[w]["same"]["n"]})' for w, c_ in zip(ws, cs)))
    ax.axhline(0, color=GREY, lw=0.7)
    ax.set_xticks(range(len(ws)))
    ax.set_xticklabels([f'{w:g}' for w in ws])
    ax.set_xlabel('task-mates born in the last (h)')
    ax.set_ylabel('difference between coefficients')
    ax.set_ylim(-0.3, 0.7)
    ax.legend(frameon=False, fontsize=5.2, loc='upper right', handlelength=1.2)
    save(fig, 'ed_exposure_task')


def ed3(say):
    fitted = load_cache('model.pkl')['fitted']
    axis = {e['name']: e for e in load_cache('forms.pkl')['axis']}
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.35))
    plt.subplots_adjust(wspace=0.5)
    for ax, k, letter, lab, log, lim in ((axes[0], 'gap', 'a', 'consistency within a page', True, (0.5, 30)),
                                        (axes[1], 'vol', 'b', 'volatility over time, first uses', False, (0, 7)),
                                        (axes[2], 'final', 'c', 'final share of form A', False, (0, 1))):
        panel(ax, letter)
        inside, n_pts, obs, med_ = 0, 0, [], []
        for n, runs in fitted.items():
            o = axis[n]['f'] if k == 'final' else axis[n][k]
            m = runs[k][np.isfinite(runs[k])]
            if not np.isfinite(o) or len(m) < 6 or (log and o <= 0):
                continue
            med, lo, hi = np.median(m), np.percentile(m, 10), np.percentile(m, 90)
            ax.errorbar(o, med, yerr=[[med - lo], [hi - med]], fmt='o', color=CLASS_COLOR[axis[n]['cls']][0], mec=INK,
                        mew=0.4, ms=3.6, ecolor=GREY, elinewidth=0.7, zorder=3)
            n_pts += 1
            inside += lo <= o <= hi
            obs.append(o)
            med_.append(med)
            if not lo <= o <= hi and (k != 'final' or abs(o - med) > 0.25):
                ax.annotate(n.split(' / ')[0], (o, med), xytext=(3, -6), textcoords='offset points', fontsize=4.6,
                            color=SLATE)
        obs, med_ = np.array(obs), np.array(med_)
        r = np.corrcoef(np.log(obs), np.log(med_))[0, 1] if log else np.corrcoef(obs, med_)[0, 1]
        say(f'ED3{letter} {lab}: {n_pts} conventions, observed inside the 10-90% band of its runs {inside}; correlation '
            f'{"of the logarithms " if log else ""}{r:.2f}; mean observed {np.exp(np.mean(np.log(obs))) if log else obs.mean():.2f}, '
            f'mean model median {np.exp(np.mean(np.log(med_))) if log else med_.mean():.2f}{" (geometric)" if log else ""}')
        ax.plot(lim, lim, '--', color=GREY, lw=0.7)
        ax.set_xlim(*lim)
        ax.set_ylim(*lim)
        ax.set_xlabel('observed')
        ax.set_ylabel('model')
        ax.set_title(lab, fontsize=6.5, color=SLATE, pad=8)
        if log:
            ax.set_xscale('log')
            ax.set_yscale('log')
    for col, lab in CLASS_COLOR.values():
        axes[2].plot([], [], 'o', color=col, mec=INK, mew=0.4, ms=3.6, label=lab)
    axes[2].legend(frameon=False, fontsize=5.2, loc='upper left', handletextpad=0.2)
    save(fig, 'ed_perconv')


def main():
    use()
    say = Report('Extended Data figures 1-3')
    ed1(say)
    ed2(say)
    ed3(say)
    say.write()


if __name__ == '__main__':
    main()
