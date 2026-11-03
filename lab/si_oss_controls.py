"""SI figure: what drives checking in gpt-oss-20b, and how stable it is.

    (a) P(opens the source) against the number of notes of the record on the page, when all of them give the wrong
        value (the runs of Fig. 4a) and when half of them were turned right (design M of replicate.py, pages of 2, 4,
        8, 12, 20 and 32 notes): checking falls with the length of the page, not with agreement.
    (b) P(opens the source) against the fraction of right notes on pages of 20 notes of the record (design R, as
        Fig. 4b): the run of Fig. 4b, its repeat on the same pages, and the same pages at medium reasoning effort.
    (c) P(opens the source) on pages of the record's wrong notes (as Fig. 4a): the runs of Fig. 4a and a repeat on the
        same pages and seeds (replicate.py, design A).
100 agents per point, 95% Wilson intervals. Writes paper/figs/figS_oss_controls.pdf and figures/figS_oss_controls.png.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / 'src'))
import si_models_checking as S  # noqa: E402
from common import wilson  # noqa: E402
from style import GREY, INK, OSS, SLATE, panel, plt, use  # noqa: E402

A = ROOT / 'data' / 'answers'


def load(name, design):
    return [r for r in map(json.loads, open(A / name)) if r['design'] == design and not r.get('error')
            and r.get('answer') is not None]


def curve(ax, xs, groups, key='checked', **kw):
    ps = [wilson(sum(bool(r[key]) for r in g), len(g)) for g in groups]
    p = np.array([q[0] for q in ps])
    ax.errorbar(xs, p, yerr=[np.maximum(p - [q[1] for q in ps], 0), np.maximum([q[2] for q in ps] - p, 0)], capsize=0,
                ms=3.0, lw=1.1, elinewidth=0.7, **kw)
    return p


def log_axis(ax, ticks):
    ax.set_xscale('log')
    ax.set_xticks([t + 1 for t in ticks])
    ax.set_xticklabels([str(t) for t in ticks])
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.xaxis.set_minor_locator(plt.NullLocator())


def main():
    use()
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.4))
    plt.subplots_adjust(wspace=0.42)
    a_orig = S.design_a('gpt-oss-20b')

    # (a) length against agreement
    ax = axes[0]
    panel(ax, 'a')
    ns = [0, 1, 3, 7, 11, 19, 32]
    p = curve(ax, np.array(ns) + 1, [a_orig[n] for n in ns], fmt='o-', color=OSS, label='all wrong')
    M = load('replicate_gpt-oss-20b.jsonl', 'M')
    nm = sorted({r['n'] for r in M})
    q = curve(ax, np.array(nm) + 1, [[r for r in M if r['n'] == n] for n in nm], fmt='o--', color=OSS, mfc='white', mew=0.9,
              label='half right')
    print('(a) all wrong ' + ' '.join(f'{n}:{v:.2f}' for n, v in zip(ns, p)) + ' | half right ' +
          ' '.join(f'{n}:{v:.2f}' for n, v in zip(nm, q)))
    log_axis(ax, [0, 1, 3, 7, 11, 19, 32])
    ax.set(xlim=(0.85, 40), ylim=(0, 1.04), xlabel='notes of the record on the page', ylabel='P(opens the source)')
    ax.legend(frameon=False, fontsize=5.6, loc='lower left')

    # (b) the composition of the page: the run of Fig. 4b, its repeat, medium effort
    ax = axes[1]
    panel(ax, 'b')
    for name, fmt, kw, label in (('replicate_gpt-oss-20b.jsonl', 'o-', {}, 'Fig. 4b'),
                                 ('replicate_gpt-oss-20b_repeat.jsonl', 'o--', dict(mfc='white', mew=0.9), 'repeat'),
                                 ('replicate_gpt-oss-20b_effort-medium.jsonl', 's-', dict(alpha=0.6), 'medium effort')):
        R = load(name, 'R')
        fr = sorted({round(r['frac_right'], 1) for r in R})
        v = curve(ax, fr, [[r for r in R if round(r['frac_right'], 1) == f] for f in fr], fmt=fmt,
                  color=OSS if 'medium' not in label else SLATE, label=label, **kw)
        print(f'(b) {label:13} ' + ' '.join(f'{f}:{x:.2f}' for f, x in zip(fr, v)))
    ax.set(xlim=(-0.04, 1.04), ylim=(0, 1.04), xlabel='fraction of right notes on the page', ylabel='P(opens the source)')
    ax.legend(frameon=False, fontsize=5.6, loc='lower center')

    # (c) a repeat of Fig. 4a
    ax = axes[2]
    panel(ax, 'c')
    rep = load('replicate_gpt-oss-20b.jsonl', 'A')
    curve(ax, np.array(ns) * 1.0 + 1, [a_orig[n] for n in ns], fmt='o-', color=OSS, label='Fig. 4a')
    v = curve(ax, (np.array(ns) + 1) * 1.07, [[r for r in rep if r['n'] == n] for n in ns], fmt='o--', color=OSS,
              mfc='white', mew=0.9, label='repeat')
    print('(c) repeat ' + ' '.join(f'{n}:{x:.2f}' for n, x in zip(ns, v)))
    log_axis(ax, [0, 1, 3, 7, 11, 19, 32])
    ax.set(xlim=(0.85, 40), ylim=(0, 1.04), xlabel='agreeing wrong notes', ylabel='P(opens the source)')
    ax.legend(frameon=False, fontsize=5.6, loc='lower left')
    fig.savefig(ROOT.parent / 'figures' / 'figS_oss_controls.png', dpi=220, bbox_inches='tight')
    fig.savefig(ROOT.parent / 'paper' / 'figs' / 'figS_oss_controls.pdf', bbox_inches='tight')
    print('wrote figS_oss_controls')


if __name__ == '__main__':
    main()
