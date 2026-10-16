"""Figure 4: agreement switches checking off, and a wrong consensus protects itself (gpt-oss-20b in the laboratory;
the other models in the SI).

    (a) Record (OECD Equity, after the first correction; pages with 0, 1, 2-3 and 5+ agreeing wrong uses, at their
        means): the note gives a source or evidence, and gives the right value. Laboratory (gpt-oss-20b, pages of the
        record's own notes giving the wrong value, four OECD questions, 100 agents per point): the agent opens the
        accurate source, and answers right. 95% Wilson intervals.
    (b) Laboratory: P(opens the source) against the fraction of right answers on a page of 20 notes (125 agents per
        point); pages of 5, 10 and 40 notes faint.
    (c) Laboratory, the collective: chains of 30 agents on one page that starts with 1, 7, 19 or 32 of the record's
        notes giving the wrong value, each agent with the fetch tool and writing a note with its answer; P(answers
        wrong) by position (20 chains per curve, smoothed over three agents, 95% band).
Writes paper/figs/fig4_checking.pdf and the preview figures/fig4_checking.png.
"""
import collections
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / 'src'))
import answers_oecd as ao  # noqa: E402
import conventions as cv  # noqa: E402
from common import T, Dataset, wilson  # noqa: E402
from plot_support import NS, rows  # noqa: E402
from style import AMBER, GREY, INK, MODEL as TEAL, OSS, panel, plt, use  # noqa: E402

ALERT = T('2026-06-20T03:41:17').timestamp()
SRC = (r'xlsx|workbook|power ?bi|\bPBI\b|tooltip|querydata|sdmx|\braw\b|parsed|extracted|intercept|verified|aria|render'
       r'|9\.9124|16\.376|14\.587|9\.694|23\.130')
C4 = ('Czech Republic', 'Hungary', 'Poland', 'Slovak Republic')


def eb(ax, x, groups, key, **kw):
    ps = [wilson(sum(g), len(g)) for g in groups]
    ax.errorbar(x, [p[0] for p in ps], yerr=[[max(p[0] - p[1], 0) for p in ps], [max(p[2] - p[0], 0) for p in ps]],
                capsize=0, **kw)
    return [p[0] for p in ps]


def record_rows():
    ds = Dataset()
    idx, form = cv.sequence(ds.revs, ao.NAME)
    page = collections.defaultdict(lambda: [0, 0])
    out = []
    for i, f in zip(idx, form):
        r = ds.revs[i]
        a, b = page[r['page_id']]
        n = a + b
        out.append(dict(state='empty' if n == 0 else ('unanimous' if a == 0 or b == 0 else 'mixed'), n=n,
                        agree='none' if n == 0 else ('wrong' if b == 0 else ('right' if a == 0 else 'mixed')), right=f == 1,
                        src=bool(re.search(SRC, r['added'], re.I)), h=(T(r['time']).timestamp() - ALERT) / 3600))
        page[r['page_id']][0 if f == 0 else 1] += 1
    return out


def main():
    use()
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.4))
    plt.subplots_adjust(wspace=0.42)
    R = record_rows()

    # (a) agreement on a wrong value switches checking off, and the right answer with it
    ax = axes[0]
    panel(ax, 'a')
    D = rows()
    base = [r for r in map(json.loads, open(ROOT / 'data' / 'answers' / 'check_probe_allwrong_n0_gpt-oss-20b.jsonl')) if r['country'] in C4]
    extra = [r for r in map(json.loads, open(ROOT / 'data' / 'answers' / 'panel_a_gpt-oss-20b.jsonl')) if not r.get('error')]
    lab = {0: base, **{n: D['record'][n] for n in NS}, **{n: [r for r in extra if r['n'] == n] for n in (2, 3, 11)}}
    ns = sorted(lab)
    xl = (np.array(ns) + 1) * 1.04
    eb(ax, xl, [[r['checked'] for r in lab[n]] for n in ns], None, fmt='o-', color=OSS, ms=2.8, lw=1.0, elinewidth=0.7)
    eb(ax, xl, [[r['is_right'] for r in lab[n]] for n in ns], None, fmt='o:', color=OSS, ms=2.8, lw=1.0, elinewidth=0.7,
       mfc='white', mew=0.9)
    bins = [(0, 1), (1, 2), (2, 5), (5, 40)]
    sel = [[x for x in R if x['h'] >= 0 and x['agree'] in ('none', 'wrong') and lo <= x['n'] < hi] for lo, hi in bins]
    xb = (np.array([np.mean([x['n'] for x in g]) for g in sel]) + 1) / 1.04
    eb(ax, xb, [[x['src'] for x in g] for g in sel], None, fmt='o-', color=INK, ms=3.6, lw=1.2, zorder=5)
    eb(ax, xb, [[x['right'] for x in g] for g in sel], None, fmt='o:', color=INK, ms=3.6, lw=1.2, mfc='white', mew=1,
       zorder=5)
    for lab_, k in (('source', 'src'), ('right', 'right')):
        print(f'(a) record {lab_:6} ' + ' '.join(f'{lo}-{hi - 1}:{np.mean([x[k] for x in g]):.2f}(n={len(g)}, mean {np.mean([x["n"] for x in g]):.1f})'
                                                for (lo, hi), g in zip(bins, sel)))
    for lab_, k in (('checked', 'checked'), ('right', 'is_right')):
        print(f'(a) lab {lab_:7} ' + ' '.join(f'{n}:{np.mean([r[k] for r in lab[n]]):.2f}' for n in ns))
    ax.set_xscale('log')
    ax.set_xticks([1, 2, 3, 4, 8, 12, 20, 33])
    ax.set_xticklabels(['0', '1', '2', '3', '7', '11', '19', '32'])
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.set(xlim=(0.85, 40), ylim=(0, 1.04), xlabel='notes agreeing on the wrong value', ylabel='probability')
    h = [ax.plot([], [], 'o', color=INK, ms=3.6)[0], ax.plot([], [], 'o', color=OSS, ms=2.8)[0],
         ax.plot([], [], '-', color=GREY, lw=1.1)[0], ax.plot([], [], ':', color=GREY, lw=1.1)[0]]
    ax.legend(h, ['record', 'gpt-oss-20b (lab)', 'checks or cites the source', 'gives the right value'], frameon=False, fontsize=5.4,
              loc='lower left', handlelength=1.6, ncol=1)

    # (b) laboratory: disagreement triggers checking
    ax = axes[1]
    panel(ax, 'b')
    for n in (5, 10, 40, 20):
        P = [r for r in map(json.loads, open(ROOT / 'data' / 'answers' / f'check_probe_v2_n{n}_gpt-oss-20b.jsonl')) if not r.get('error')]
        fr = sorted({r['frac_right'] for r in P})
        if n == 20:
            eb(ax, fr, [[r['checked'] for r in P if r['frac_right'] == f] for f in fr], None, fmt='o-', color=OSS, ms=2.8,
               lw=1.1, elinewidth=0.7, label='20 notes')
        else:
            ax.plot(fr, [np.mean([r['checked'] for r in P if r['frac_right'] == f]) for f in fr], '-', color=OSS, lw=0.6,
                    alpha=0.3, label='5, 10, 40 notes' if n == 5 else None)
    ax.set(xlim=(-0.04, 1.04), ylim=(0, 1.04), xlabel='fraction of right answers on the page', ylabel='P(checks the source)')
    ax.legend(frameon=False, fontsize=5.4, loc='lower center')

    # (c) the collective: chains that start from a wrong consensus of the record's notes
    ax = axes[2]
    panel(ax, 'c')
    ch = collections.defaultdict(dict)
    for l in open(ROOT / 'data' / 'answers' / 'chains_gpt-oss-20b.jsonl'):
        r = json.loads(l)
        if r['scenario'].startswith('recstart'):
            ch[(r['scenario'], r['country'], r['chain'])][r['pos']] = r
    pos = np.arange(1, 31)
    for j, n in enumerate((1, 7, 19, 32)):
        vs = [v for (s, _, _), v in ch.items() if s == f'recstart{n}_tool_ans']
        W = np.array([[v[p]['is_wrong'] for p in range(30)] for v in vs], float)
        S = np.array([np.convolve(np.pad(w, 1, mode='edge'), np.ones(3) / 3, mode='valid') for w in W])
        m, e = S.mean(0), 1.96 * S.std(0, ddof=1) / np.sqrt(len(S))
        a = 0.3 + 0.7 * j / 3
        ax.fill_between(pos, np.clip(m - e, 0, 1), np.clip(m + e, 0, 1), color=OSS, alpha=0.12 * a, lw=0)
        ax.plot(pos, m, '-', color=OSS, alpha=a, lw=1.0 + 0.4 * j, label=f'{n}')
        print(f'(c) start {n:2d}: agents 11-30 wrong {W[:, 10:].mean():.2f}; chains never right {np.mean([w.sum() == 30 for w in W]):.2f}')
    ax.set(xlim=(1, 30), ylim=(0, 1.04), xticks=[1, 10, 20, 30], xlabel='agent in the chain', ylabel='P(answers wrong)')
    ax.legend(frameon=False, fontsize=5.4, loc='upper right', title='wrong notes at the start', title_fontsize=5.4)

    out = ROOT.parent / 'figures' / 'fig4_checking.png'
    fig.savefig(out, dpi=220, bbox_inches='tight')
    fig.savefig(ROOT.parent / 'paper' / 'figs' / 'fig4_checking.pdf', bbox_inches='tight')
    print('wrote', out)


if __name__ == '__main__':
    main()
