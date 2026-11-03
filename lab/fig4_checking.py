"""Figure 4: agreement switches checking off, and a wrong consensus protects itself. Every page of the laboratory
is made of the record's own notes; gpt-oss-20b and GPT-6.1 Sol (the other models in the SI).

    (a) P(checks the source) against the number of notes agreeing on the wrong value. Record (OECD Equity, after the
        first correction; pages with 0, 1, 2-3 and 5+ agreeing wrong uses, at their means): the note cites a source or
        gives evidence. Laboratory (pages of N of the record's notes giving the wrong value, four OECD questions, 100
        agents per point): the agent opens the accurate source; P(answers right) follows it closely (SI). 95% Wilson
        intervals.
    (b) Laboratory: P(opens the source) (solid) and P(answers right) (dotted, drawn slightly to the right) against the
        fraction of right notes on a page of 20 of the record's notes, a fraction of them turned right by writing the
        right value in place of the wrong one (design R of replicate.py; 100 agents per point).
    (c) Laboratory, the collective: chains of 30 agents on one page that starts with 1, 3, 7 or 32 of the record's
        notes giving the wrong value, each agent with the fetch tool and appending its own note; P(answers wrong) by
        position, gpt-oss-20b above and GPT-6.1 Sol below (20 chains per curve, smoothed over three agents, 95% band).
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
from style import AMBER, GREY, INK, MODEL as TEAL, OSS, SLATE, SOL, panel, plt, use  # noqa: E402

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


STARTS = (1, 3, 7, 32)
SHADE = (0.30, 0.55, 0.78, 1.0)            # the shade of each starting page, from the fewest notes up


def chain_panels(fig, cell):
    """Panel c: two short stacked axes, P(answers wrong) by position in the chain for each starting page."""
    sub = cell.subgridspec(2, 1, hspace=0.22)
    pos = np.arange(1, 31)
    axes = []
    for k, (name, col, label) in enumerate((('gpt-oss-20b', OSS, 'gpt-oss-20b'), ('gpt-6.1-sol', SOL, 'GPT-6.1 Sol'))):
        ax = fig.add_subplot(sub[k])
        axes.append(ax)
        ch = collections.defaultdict(dict)
        for l in open(ROOT / 'data' / 'answers' / f'chains_{name}.jsonl'):
            r = json.loads(l)
            if r['scenario'].startswith('recstart') and not r.get('error'):
                ch[(r['scenario'], r['country'], r['chain'])][r['pos']] = r
        ends = []
        for j, n in enumerate(STARTS):
            W = np.array([[v[p]['is_wrong'] for p in range(30)] for (s_, _, _), v in ch.items()
                          if s_ == f'recstart{n}_tool_ans' and len(v) == 30], float)
            S = np.array([np.convolve(np.pad(w, 1, mode='edge'), np.ones(3) / 3, mode='valid') for w in W])
            m, e = S.mean(0), 1.96 * S.std(0, ddof=1) / np.sqrt(len(S))
            ax.fill_between(pos, np.clip(m - e, 0, 1), np.clip(m + e, 0, 1), color=col, alpha=0.07 + 0.08 * SHADE[j], lw=0)
            ax.plot(pos, m, '-', color=col, alpha=SHADE[j], lw=0.9 + 0.45 * j, solid_capstyle='round')
            ends.append(m[-1])
            print(f'(c) {label:12} start {n:2d}: {len(W)} chains, agents 11-30 wrong {W[:, 10:].mean():.2f}, '
                  f'chains never right {np.mean(W.sum(1) == 30):.2f}')
        ys = np.array(ends, float)                   # labels in the order of the starting pages, 0.15 apart
        for j in range(1, len(ys)):
            ys[j] = max(ys[j], ys[j - 1] + 0.15)
        if ys[-1] > 1.05:
            ys -= ys[-1] - 1.05
            for j in range(len(ys) - 2, -1, -1):
                ys[j] = min(ys[j], ys[j + 1] - 0.15)
        for j, (n, y0, y) in enumerate(zip(STARTS, ends, ys)):
            ax.plot([30.2, 32.0], [y0, y], '-', color=col, alpha=SHADE[j], lw=0.5, clip_on=False)
            ax.text(32.4, y, str(n), fontsize=5.8, color=col, alpha=max(SHADE[j], 0.55), va='center', clip_on=False,
                    fontweight='bold' if j == len(STARTS) - 1 else 'normal')
        ax.text(2.0, 1.0 if k == 0 else 0.58, label, fontsize=6.0, color=col, va='top', fontweight='bold')
        ax.set(xlim=(1, 30), ylim=(-0.07, 1.07), xticks=[1, 10, 20, 30], yticks=[0, 0.5, 1])
        ax.set_yticklabels(['0', '0.5', '1'])
    axes[0].set_xticklabels([])
    axes[1].set_xlabel('agent in the chain')
    axes[0].text(1.11, 0.93, 'wrong notes\nat the start', transform=axes[0].transAxes, fontsize=5.0, color=SLATE,
                 ha='center', va='bottom', linespacing=1.0)
    fig.text(axes[1].get_position().x0 - 0.045, 0.5, 'P(answers wrong)', rotation=90, va='center', ha='center')
    panel(axes[0], 'c', x=-0.34)


def main():
    use()
    fig = plt.figure(figsize=(7.1, 2.4))
    gs = fig.add_gridspec(1, 3, wspace=0.42)
    axes = [fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), gs[2]]
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
    eb(ax, xl, [[r['checked'] for r in lab[n]] for n in ns], None, fmt='o-', color=OSS, ms=2.8, lw=1.0, elinewidth=0.7, zorder=4)
    sol = [r for r in map(json.loads, open(ROOT / 'data' / 'answers' / 'panel_a_gpt-6.1-sol.jsonl')) if not r.get('error')]
    sns = sorted({r['n'] for r in sol})
    xs_ = (np.array(sns) + 1) * 1.10
    eb(ax, xs_, [[r['checked'] for r in sol if r['n'] == n] for n in sns], None, fmt='D-', color=SOL, ms=2.6, lw=1.0,
       elinewidth=0.7, zorder=4)
    bins = [(0, 1), (1, 2), (2, 5), (5, 40)]
    sel = [[x for x in R if x['h'] >= 0 and x['agree'] in ('none', 'wrong') and lo <= x['n'] < hi] for lo, hi in bins]
    xb = (np.array([np.mean([x['n'] for x in g]) for g in sel]) + 1) / 1.04
    eb(ax, xb, [[x['src'] for x in g] for g in sel], None, fmt='o-', color=INK, ms=3.6, lw=1.2, zorder=5)
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
    ax.set(xlim=(0.85, 40), ylim=(0, 1.04), xlabel='notes agreeing on the wrong value', ylabel='P(checks the source)')
    h = [ax.plot([], [], 'o-', color=INK, ms=3.6)[0], ax.plot([], [], 'o-', color=OSS, ms=2.8)[0],
         ax.plot([], [], 'D-', color=SOL, ms=2.6)[0]]
    ax.legend(h, ['record', 'gpt-oss-20b', 'GPT-6.1 Sol'], frameon=False, fontsize=5.4,
              loc='lower left', handlelength=1.8)

    # (b) laboratory: disagreement triggers checking, on pages of the record's notes
    ax = axes[1]
    panel(ax, 'b')
    for name, col, mk, label in (('gpt-oss-20b', OSS, 'o', 'gpt-oss-20b'), ('gpt-6.1-sol', SOL, 'D', 'GPT-6.1 Sol')):
        P = [r for r in map(json.loads, open(ROOT / 'data' / 'answers' / f'replicate_{name}.jsonl'))
             if r['design'] == 'R' and not r.get('error') and r.get('answer') is not None]
        fr = sorted({round(r['frac_right'], 1) for r in P})
        eb(ax, fr, [[r['checked'] for r in P if round(r['frac_right'], 1) == f] for f in fr], None, fmt=mk + '-', color=col,
           ms=2.8, lw=1.1, elinewidth=0.7, zorder=4)
        eb(ax, np.array(fr) + 0.018, [[r['is_right'] for r in P if round(r['frac_right'], 1) == f] for f in fr], None,
           fmt=mk + ':', color=col, ms=2.8, lw=1.0, elinewidth=0.7, mfc='white', mew=0.9, zorder=3)
        print(f'(b) {label:12} checked ' + ' '.join(f'{f}:{np.mean([r["checked"] for r in P if round(r["frac_right"], 1) == f]):.2f}' for f in fr))
    ax.set(xlim=(-0.04, 1.04), ylim=(0, 1.04), xlabel='fraction of right notes on the page', ylabel='probability')
    h = [ax.plot([], [], 'o', color=OSS, ms=2.8)[0], ax.plot([], [], 'D', color=SOL, ms=2.6)[0],
         ax.plot([], [], '-', color=GREY, lw=1.1)[0], ax.plot([], [], ':', color=GREY, lw=1.1)[0]]
    ax.legend(h, ['gpt-oss-20b', 'GPT-6.1 Sol', 'checks the source', 'answers right'], frameon=False, fontsize=5.2,
              loc='lower center', handlelength=1.6)

    # (c) the collective: chains of 30 agents that start from 1, 3, 7 or 32 of the record's wrong notes; P(answers
    # wrong) by position, gpt-oss-20b above and GPT-6.1 Sol below, every curve labelled with its starting page
    chain_panels(fig, axes[2])

    out = ROOT.parent / 'figures' / 'fig4_checking.png'
    fig.savefig(out, dpi=220, bbox_inches='tight')
    fig.savefig(ROOT.parent / 'paper' / 'figs' / 'fig4_checking.pdf', bbox_inches='tight')
    print('wrote', out)


if __name__ == '__main__':
    main()
