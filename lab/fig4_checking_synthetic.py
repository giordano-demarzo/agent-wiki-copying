"""Figure 4, variant with synthetic notes in a and b (template notes; c as in Fig. 4): agreement switches checking off, and a wrong consensus protects itself. Every page of the laboratory
is made of the record's own notes; gpt-oss-20b and GPT-6.1 Sol (the other models in the SI).

    (a) Record (OECD Equity, after the first correction; pages with 0, 1, 2-3 and 5+ agreeing wrong uses, at their
        means): the note gives a source or evidence, and gives the right value. Laboratory (pages of N of the record's
        notes giving the wrong value, four OECD questions, 100 agents per point): the agent opens the accurate source,
        and answers right. 95% Wilson intervals.
    (b) Laboratory: P(opens the source) against the fraction of right notes on a page of 20 of the record's notes, a
        fraction of them turned right by writing the right value in place of the wrong one (design R of replicate.py;
        100 agents per point).
    (c) Laboratory, the collective: chains of 30 agents on one page that starts with N of the record's notes giving
        the wrong value (gpt-oss-20b: N = 1, 7, 19, 32; GPT-6.1 Sol: N = 1, 3, 7, 32), each agent with the fetch tool
        and appending its own note; the share of agents 11 to 30 that answer wrong (20 chains per point, 95% bootstrap
        intervals over chains). The time courses are in the SI.
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
from style import AMBER, GREY, INK, MODEL as TEAL, OSS, SOL, panel, plt, use  # noqa: E402

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


def load_design(name, d):
    return [r for r in map(json.loads, open(ROOT / 'data' / 'answers' / f'replicate_{name}.jsonl'))
            if r['design'] == d and not r.get('error') and r.get('answer') is not None]


def main():
    use()
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.4))
    plt.subplots_adjust(wspace=0.42)
    R = record_rows()

    # (a) agreement on a wrong value, synthetic notes: template notes all giving the wrong value (design P)
    ax = axes[0]
    panel(ax, 'a')
    A0 = {name: [r for r in load_design(name, 'A') if r['n'] == 0] for name in ('gpt-oss-20b', 'gpt-6.1-sol')}
    A0['gpt-oss-20b'] = [r for r in map(json.loads, open(ROOT / 'data' / 'answers' / 'check_probe_allwrong_n0_gpt-oss-20b.jsonl'))
                         if r['country'] in C4]
    A0['gpt-6.1-sol'] = [r for r in map(json.loads, open(ROOT / 'data' / 'answers' / 'panel_a_gpt-6.1-sol.jsonl')) if r['n'] == 0]
    for name, col, mk, shift in (('gpt-oss-20b', OSS, 'o', 1.04), ('gpt-6.1-sol', SOL, 'D', 1.10)):
        P = load_design(name, 'P')
        g = {0: A0[name], **{n: [r for r in P if r['n'] == n] for n in sorted({r['n'] for r in P})}}
        ns = sorted(g)
        xl = (np.array(ns) + 1) * shift
        eb(ax, xl, [[r['checked'] for r in g[n]] for n in ns], None, fmt=mk + '-', color=col, ms=2.8, lw=1.0, elinewidth=0.7,
           zorder=4)
        eb(ax, xl * 1.07, [[r['is_right'] for r in g[n]] for n in ns], None, fmt=mk + ':', color=col, ms=2.8, lw=1.0,
           elinewidth=0.7, mfc='white', mew=0.9, zorder=3)
        print(f'(a) {name:12} checked ' + ' '.join(f'{n}:{np.mean([r["checked"] for r in g[n]]):.2f}' for n in ns))
    bins = [(0, 1), (1, 2), (2, 5), (5, 40)]
    sel = [[x for x in R if x['h'] >= 0 and x['agree'] in ('none', 'wrong') and lo <= x['n'] < hi] for lo, hi in bins]
    xb = (np.array([np.mean([x['n'] for x in g]) for g in sel]) + 1) / 1.04
    eb(ax, xb, [[x['src'] for x in g] for g in sel], None, fmt='o-', color=INK, ms=3.6, lw=1.2, zorder=5)
    eb(ax, xb, [[x['right'] for x in g] for g in sel], None, fmt='o:', color=INK, ms=3.6, lw=1.2, mfc='white', mew=1, zorder=5)
    ax.set_xscale('log')
    ax.set_xticks([1, 2, 4, 8, 12, 20, 33])
    ax.set_xticklabels(['0', '1', '3', '7', '11', '19', '32'])
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.set(xlim=(0.85, 40), ylim=(0, 1.04), xlabel='notes agreeing on the wrong value', ylabel='probability')
    h = [ax.plot([], [], 'o', color=INK, ms=3.6)[0], ax.plot([], [], 'o', color=OSS, ms=2.8)[0],
         ax.plot([], [], 'D', color=SOL, ms=2.6)[0],
         ax.plot([], [], '-', color=GREY, lw=1.1)[0], ax.plot([], [], ':', color=GREY, lw=1.1)[0]]
    ax.legend(h, ['record', 'gpt-oss-20b', 'GPT-6.1 Sol', 'checks or cites the source', 'gives the right value'], frameon=False,
              fontsize=5.2, loc='lower left', handlelength=1.6, ncol=1)

    # (b) disagreement, synthetic notes: 20 template notes, a fraction of them right (design B)
    ax = axes[1]
    panel(ax, 'b')
    B = {'gpt-oss-20b': [r for r in map(json.loads, open(ROOT / 'data' / 'answers' / 'check_probe_v2_n20_gpt-oss-20b.jsonl'))
                         if not r.get('error')],
         'gpt-6.1-sol': load_design('gpt-6.1-sol', 'B')}
    for name, col, mk, label in (('gpt-oss-20b', OSS, 'o', 'gpt-oss-20b'), ('gpt-6.1-sol', SOL, 'D', 'GPT-6.1 Sol')):
        P = B[name]
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

    # (c) the collective: chains of 30 agents that start from a wrong consensus of the record's notes; the share of
    # agents 11 to 30 that answer wrong, against the number of wrong notes at the start (95% bootstrap over chains)
    ax = axes[2]
    panel(ax, 'c')
    rng = np.random.default_rng(3)
    for name, col, mk, shift, label in (('gpt-oss-20b', OSS, 'o', 0.97, 'gpt-oss-20b'), ('gpt-6.1-sol', SOL, 'D', 1.03, 'GPT-6.1 Sol')):
        ch = collections.defaultdict(dict)
        for l in open(ROOT / 'data' / 'answers' / f'chains_{name}.jsonl'):
            r = json.loads(l)
            if r['scenario'].startswith('recstart') and not r.get('error'):
                ch[(r['scenario'], r['country'], r['chain'])][r['pos']] = r
        starts = sorted({int(re.match(r'recstart(\d+)', s).group(1)) for s, _, _ in ch})
        xs, ys, lo, hi = [], [], [], []
        for n in starts:
            late = np.array([np.mean([v[p]['is_wrong'] for p in range(10, 30)]) for (s, _, _), v in ch.items()
                             if s == f'recstart{n}_tool_ans' and len(v) == 30])
            boot = [rng.choice(late, len(late)).mean() for _ in range(2000)]
            xs.append(n); ys.append(late.mean()); lo.append(np.percentile(boot, 2.5)); hi.append(np.percentile(boot, 97.5))
            print(f'(c) {label:12} start {n:2d}: agents 11-30 wrong {late.mean():.2f} [{lo[-1]:.2f}, {hi[-1]:.2f}], '
                  f'chains locked wrong {np.mean(late == 1):.2f} ({len(late)} chains)')
        ys, lo, hi = map(np.array, (ys, lo, hi))
        ax.errorbar(np.array(xs) * shift, ys, yerr=[ys - lo, hi - ys], fmt=mk + '-', color=col, ms=3.2, lw=1.1, elinewidth=0.7,
                    capsize=0, label=label)
    ax.set_xscale('log')
    ax.set_xticks([1, 3, 7, 19, 32])
    ax.set_xticklabels(['1', '3', '7', '19', '32'])
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.set(xlim=(0.8, 40), ylim=(-0.02, 1.04), xlabel='wrong notes at the start',
           ylabel='P(wrong), agents 11 to 30')
    ax.legend(frameon=False, fontsize=5.4, loc='upper left')

    out = ROOT.parent / 'figures' / 'fig4_checking_synthetic.png'
    fig.savefig(out, dpi=220, bbox_inches='tight')
    fig.savefig(ROOT.parent / 'paper' / 'figs' / 'fig4_checking_synthetic.pdf', bbox_inches='tight')
    print('wrote', out)


if __name__ == '__main__':
    main()
