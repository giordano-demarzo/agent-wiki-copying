"""Fig. 4a and 4b for every laboratory model: Extended Data (ed_models_checking: pages of the record's notes, designs A
and R) and Supplementary Information (figS_models_synthetic: synthetic template notes, designs P and B).

Rows: (a-c) pages of N of the record's notes, all giving the wrong (padded) value (Fig. 4a); (d-f) pages of N template
notes, all wrong (agreement without the record's content); (g-i) pages of 20 template notes, a fraction of them right
(Fig. 4b); (j-l) pages of 20 of the record's notes, a fraction of them turned right (the record's content, mixed).
Columns: the probability that the agent opens the source, that it answers right, and that it answers wrong although it
opened the source. The record after the first correction is drawn in a and b as in Fig. 4a.
gpt-oss-20b: the runs of Fig. 4 (real_support, panel_a, check_probe); GPT-6.1 Sol: panel_a (4a only); the other
models: replicate.py (designs A and B). Models whose runs are missing or still running are drawn with what there is.
Points are proportions with 95% Wilson intervals (100 agents per point in a-c, 125 in d-e).
Writes paper/figs/figS_models_checking.pdf and figures/figS_models_checking.png.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / 'src'))
from common import wilson  # noqa: E402
from fig4_checking import record_rows  # noqa: E402
from plot_support import rows  # noqa: E402
from style import GREY, INK, LAB_MODELS, OSS, SOL, panel, plt, use  # noqa: E402

A = ROOT / 'data' / 'answers'
C4 = ('Czech Republic', 'Hungary', 'Poland', 'Slovak Republic')
NS = (0, 1, 3, 7, 11, 19, 32)
MODELS = LAB_MODELS


def load(path):
    return [r for r in map(json.loads, open(path)) if not r.get('error')] if path.exists() else []


def design_a(model):
    """N -> agents on all-wrong pages of the record's notes."""
    if model == 'gpt-oss-20b':
        out = {0: [r for r in load(A / 'check_probe_allwrong_n0_gpt-oss-20b.jsonl') if r['country'] in C4]}
        out.update({n: g for n, g in rows()['record'].items()})
        for r in load(A / 'panel_a_gpt-oss-20b.jsonl'):
            out.setdefault(r['n'], []).append(r)
        return out
    out = {}
    src = A / ('panel_a_gpt-6.1-sol.jsonl' if model == 'gpt-6.1-sol' else f'replicate_{model}.jsonl')
    for r in load(src):
        if r.get('design', 'A') == 'A':
            out.setdefault(r['n'], []).append(r)
    return out


def design_b(model):
    """fraction right -> agents on mixed pages of 20 template notes."""
    rs = load(A / 'check_probe_v2_n20_gpt-oss-20b.jsonl') if model == 'gpt-oss-20b' else \
        [r for r in load(A / f'replicate_{model}.jsonl') if r['design'] == 'B']
    out = {}
    for r in rs:
        out.setdefault(round(r['frac_right'], 1), []).append(r)
    return out


def design(model, d, key):
    """replicate.py designs P (N -> template notes, all wrong) and R (fraction right -> the record's notes)."""
    out = {}
    for r in load(A / f'replicate_{model}.jsonl'):
        if r['design'] == d:
            out.setdefault(round(r[key], 1) if key == 'frac_right' else r[key], []).append(r)
    return out


def curve(ax, xs, groups, key, x_shift=1.0, cond=None, **kw):
    xs, ps, lo, hi = list(xs), [], [], []
    keep = []
    for x, g in zip(xs, groups):
        g = [r for r in g if cond(r)] if cond else g
        if len(g) < 5:
            continue
        p, l, h = wilson(sum(bool(r[key]) for r in g), len(g))
        keep.append(x); ps.append(p); lo.append(p - l); hi.append(h - p)
    if keep:
        ax.errorbar(np.array(keep) * x_shift, ps, yerr=[np.maximum(lo, 0), np.maximum(hi, 0)], capsize=0, ms=3.2,
                    lw=1.1, elinewidth=0.7, **kw)


def log_axis(ax):
    ax.set_xscale('log')
    ax.set_xticks([1, 2, 4, 8, 12, 20, 33])
    ax.set_xticklabels(['0', '1', '3', '7', '11', '19', '32'])
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.set(xlim=(0.85, 40), ylim=(-0.02, 1.04), xlabel='agreeing wrong notes')


ROWS_ALL = {   # design: x axis, row title
    'A': ('n', "notes of the record, all wrong (as Fig. 4a)"),
    'R': ('frac_right', "20 notes of the record, a fraction turned right (as Fig. 4b)"),
    'P': ('n', 'synthetic template notes, all wrong'),
    'B': ('frac_right', '20 synthetic template notes, a fraction right'),
}
VERSIONS = {   # the figure, its rows, and whether the record is drawn in the first row
    'ed': ('ed_models_checking', 'AR', True),       # Extended Data: the notes of the record
    'si': ('figS_models_synthetic', 'PB', False),   # Supplementary Information: the synthetic control
}
COLS = [('checked', None, 'P(opens the source)'), ('is_right', None, 'P(answers right)'),
        ('is_wrong', lambda r: r['checked'], 'P(wrong | opened source)')]


def groups(name, d):
    if d == 'A':
        return design_a(name)
    if d == 'B':
        return design_b(name)
    g = design(name, d, 'n' if d == 'P' else 'frac_right')
    if d == 'P' and g:
        g[0] = design_a(name).get(0, [])          # the empty page is shared with A
    return g


def main(version='ed'):
    use()
    fname, rows, with_record = VERSIONS[version]
    ROWS = [(d, *ROWS_ALL[d]) for d in rows]
    fig, axes = plt.subplots(len(ROWS), 3, figsize=(7.1, 2.3 * len(ROWS) + 0.4), squeeze=False)
    plt.subplots_adjust(wspace=0.42, hspace=0.62)
    shifts = np.linspace(0.93, 1.07, len(MODELS))
    R = record_rows()
    bins = [(0, 1), (1, 2), (2, 5), (5, 40)]
    sel = [[x for x in R if x['h'] >= 0 and x['agree'] in ('none', 'wrong') and lo <= x['n'] < hi] for lo, hi in bins]
    xr = np.array([np.mean([x['n'] for x in g]) for g in sel]) + 1
    letters = iter('abcdefghijkl')
    for row, (d, key, title) in enumerate(ROWS):
        for (name, label, col, mk), sh in zip(MODELS, shifts):
            g = groups(name, d)
            if not g:
                continue
            xs = sorted(x for x in g if (key != 'n' or x in NS))
            for c, (k, cond, _) in enumerate(COLS):
                kw = dict(fmt=mk + '-', color=col, label=label)
                if key == 'n':
                    curve(axes[row, c], np.array(xs) + 1, [g[x] for x in xs], k, sh, cond=cond, **kw)
                else:
                    curve(axes[row, c], np.array(xs) + (sh - 1) * 0.3, [g[x] for x in xs], k, cond=cond, **kw)
            print(f'{d} {label:16} checked ' + ' '.join(f'{x}:{np.mean([r["checked"] for r in g[x]]):.2f}' for x in xs)
                  + ' | right ' + ' '.join(f'{x}:{np.mean([r["is_right"] for r in g[x]]):.2f}' for x in xs))
        for c, (_, _, ylab) in enumerate(COLS):
            ax = axes[row, c]
            panel(ax, next(letters))
            if key == 'n':
                log_axis(ax)
            else:
                ax.set(xlim=(-0.05, 1.05), ylim=(-0.02, 1.04), xlabel='fraction of right notes')
            ax.set_ylabel(ylab)
        axes[row, 0].text(-0.42, 1.22, title, transform=axes[row, 0].transAxes, fontsize=7.2, fontweight='bold')
    for ax, k in (((axes[0, 0], 'src'), (axes[0, 1], 'right')) if with_record else ()):
        ax.plot(xr / 1.08, [np.mean([x[k] for x in g]) for g in sel], 'o--', color=INK, ms=3.6, lw=0.9, mfc='white',
                mew=0.9, zorder=6)
    h = [axes[0, 0].errorbar([], [], yerr=[], fmt=mk + '-', color=col, ms=3.2, lw=1.1, label=label)
         for _, label, col, mk in MODELS]
    if with_record:
        h.append(axes[0, 0].plot([], [], 'o--', color=INK, ms=3.6, mfc='white', mew=0.9,
                                 label='record after the first correction (a: note cites a source; b: right value)')[0])
    fig.legend(handles=h, frameon=False, fontsize=6.4, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 0.0))
    fig.savefig(ROOT.parent / 'figures' / f'{fname}.png', dpi=220, bbox_inches='tight')
    fig.savefig(ROOT.parent / 'paper' / 'figs' / f'{fname}.pdf', bbox_inches='tight')
    print('wrote', fname)


if __name__ == '__main__':
    for v in (sys.argv[1:] or ('ed', 'si')):
        main(v)
