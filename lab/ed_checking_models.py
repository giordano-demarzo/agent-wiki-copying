"""Extended Data figure: Fig. 4a with GPT-6.1 Sol, and the error phase of the record.

    (a) As Fig. 4a, with GPT-6.1 Sol (amber; 100 agents per point, 2 and 3 notes pooled) added to gpt-oss-20b
        (vermilion) and the record after the first correction (black).
    (b) Checking relative to the empty page against the agreeing wrong notes: the record before the first
        correction (note gives a source; bins 0, 1-3, 4+) and both laboratory models; 95% bootstrap intervals.
Writes paper/figs/ed_checking_models.pdf and the preview figures/ed_checking_models.png.
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
from style import AMBER, GREY, INK, MODEL as TEAL, OSS, ROSE, SOL, panel, plt, use  # noqa: E402

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
    fig, axes = plt.subplots(1, 2, figsize=(5.4, 2.5))
    plt.subplots_adjust(wspace=0.4)
    R = record_rows()
    # (a) agreement suppresses checking, and the right answer goes with it: laboratory and record
    ax = axes[0]
    panel(ax, 'a')
    D = rows()
    base = [r for r in map(json.loads, open(ROOT / 'data' / 'answers' / 'check_probe_allwrong_n0_gpt-oss-20b.jsonl')) if r['country'] in C4]
    labg = [base] + [D['record'][n] for n in NS]
    xl = np.array((0,) + NS) + 1
    eb(ax, xl, [[r['checked'] for r in g] for g in labg], None, fmt='s-', color=OSS, ms=3.4, lw=1.3, label='gpt-oss-20b: opens the source')
    eb(ax, xl, [[r['is_right'] for r in g] for g in labg], None, fmt='s:', color=OSS, ms=3.4, lw=1.1, mfc='white', mew=1,
       label='gpt-oss-20b: answers right')
    # GPT-6.1 Sol at the record's points (0, 1, 2-3 pooled, 11) and two beyond the record (19, 32)
    sol = [r for r in map(json.loads, open(ROOT / 'data' / 'answers' / 'panel_a_gpt-6.1-sol.jsonl')) if not r.get('error')]
    sol_pts = [((0,), 0), ((1,), 1), ((2, 3), 2.3), ((11,), 11), ((19,), 19), ((32,), 32)]
    solg = [[r for r in sol if r['n'] in ns] for ns, _ in sol_pts]
    xs_sol = np.array([x for _, x in sol_pts]) + 1
    eb(ax, xs_sol * 1.04, [[r['checked'] for r in g] for g in solg], None, fmt='^-', color=SOL, ms=3.4, lw=1.3,
       label='GPT-6.1 Sol: opens the source')
    eb(ax, xs_sol * 1.04, [[r['is_right'] for r in g] for g in solg], None, fmt='^:', color=SOL, ms=3.4, lw=1.1, mfc='white', mew=1,
       label='GPT-6.1 Sol: answers right')
    bins = [(0, 1), (1, 2), (2, 5), (5, 40)]
    xb = np.array([0, 1, 2.3, 11]) + 1
    sel = [[x for x in R if x['h'] >= 0 and x['agree'] in ('none', 'wrong') and lo <= x['n'] < hi] for lo, hi in bins]
    eb(ax, xb, [[x['src'] for x in g] for g in sel], None, fmt='o-', color=INK, ms=3.4, lw=1.3, label='record: note gives a source')
    eb(ax, xb, [[x['right'] for x in g] for g in sel], None, fmt='o:', color=INK, ms=3.4, lw=1.1, mfc='white', mew=1,
       label='record: gives the right value')
    for lab_, k in (('source', 'src'), ('right', 'right')):
        print(f'(a) record {lab_:6} ' + ' '.join(f'{lo}-{hi - 1}:{np.mean([x[k] for x in g]):.2f}(n={len(g)})' for (lo, hi), g in zip(bins, sel)))
    print('(a) lab checked ' + ' '.join(f'{n}:{np.mean([r["checked"] for r in g]):.2f}' for n, g in zip((0,) + NS, labg)),
          '| right ' + ' '.join(f'{n}:{np.mean([r["is_right"] for r in g]):.2f}' for n, g in zip((0,) + NS, labg)))
    ax.set_xscale('log')
    ax.set_xticks([1, 2, 3.3, 12, 20, 33])
    ax.set_xticklabels(['0', '1', '2-3', '11', '19', '32'])
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.set(xlim=(0.85, 40), ylim=(0, 1.32), xlabel='notes agreeing on the wrong value', ylabel='probability')
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.legend(frameon=False, fontsize=4.6, loc='upper right', ncol=1, handlelength=1.6)

    # (a) agreement on a wrong value switches checking off: record before the first correction and laboratory,
    # each relative to its own empty page
    ax = axes[1]
    panel(ax, 'b')
    rng = np.random.default_rng(1)

    def rel(ax, x, groups, label, **kw):
        g0 = np.array(groups[0], float)
        y = [np.mean(g) / g0.mean() for g in groups]
        bs = np.array([[rng.choice(g, len(g)).mean() for g in map(np.array, groups)] for _ in range(2000)], float)
        bs = bs / np.maximum(bs[:, :1], 1e-9)
        lo, hi = np.percentile(bs, 2.5, 0), np.percentile(bs, 97.5, 0)
        ax.errorbar(x, y, yerr=[np.maximum(y - lo, 0), np.maximum(hi - y, 0)], capsize=0, label=label, **kw)
        print(f'(a) {label:44}', ' '.join(f'{xx - 1:5.1f}:{v:.2f}' for xx, v in zip(x, y)))

    base = [r for r in map(json.loads, open(ROOT / 'data' / 'answers' / 'check_probe_allwrong_n0_gpt-oss-20b.jsonl')) if r['country'] in C4]
    D = rows()
    oss = [r for r in map(json.loads, open(ROOT / 'data' / 'answers' / 'panel_a_gpt-oss-20b.jsonl')) if not r.get('error')]
    oss_g = {0: base, **{n: D['record'][n] for n in NS}, **{n: [r for r in oss if r['n'] == n] for n in (2, 3, 11)}}
    on = (0, 1, 2, 3, 7, 11, 19, 32)
    rel(ax, np.array(on) + 1, [[r['checked'] for r in oss_g[n]] for n in on],
        f'gpt-oss-20b (empty page {np.mean([r["checked"] for r in base]):.2f})', fmt='s-', color=OSS, ms=3.2, lw=1.2)
    sol = [r for r in map(json.loads, open(ROOT / 'data' / 'answers' / 'panel_a_gpt-6.1-sol.jsonl')) if not r.get('error')]
    sn = (0, 1, 2, 3, 7, 11, 19, 32)
    rel(ax, (np.array(sn) + 1) * 1.04, [[r['checked'] for r in sol if r['n'] == n] for n in sn],
        f'GPT-6.1 Sol ({np.mean([r["checked"] for r in sol if r["n"] == 0]):.2f})', fmt='^-', color=SOL, ms=3.4, lw=1.2)
    pre = [x for x in R if x['h'] < 0]          # before the first correction every use gives the wrong value
    bins = [(0, 1), (1, 4), (4, 40)]
    sel = [[x for x in pre if lo <= x['n'] < hi] for lo, hi in bins]
    xb = np.array([np.mean([x['n'] for x in g]) for g in sel]) + 1
    rel(ax, xb / 1.04, [[x['src'] for x in g] for g in sel], f'record: note gives a source ({np.mean([x["src"] for x in sel[0]]):.2f})',
        fmt='o-', color=INK, ms=3.6, lw=1.4)
    print('(a) record n per bin', [len(g) for g in sel])
    ax.axhline(1, color=GREY, lw=0.5, ls=':')
    ax.set_xscale('log')
    ax.set_xticks([1, 2, 4, 8, 12, 20, 33])
    ax.set_xticklabels(['0', '1', '3', '7', '11', '19', '32'])
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.set(xlim=(0.85, 40), ylim=(0, 1.45), xlabel='notes agreeing on the wrong value',
           ylabel='checking, relative to the empty page')
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0, 1.25])
    ax.legend(frameon=False, fontsize=4.6, loc='upper right', handlelength=1.6)

    fig.savefig(ROOT.parent / 'figures' / 'ed_checking_models.png', dpi=220, bbox_inches='tight')
    fig.savefig(ROOT.parent / 'paper' / 'figs' / 'ed_checking_models.pdf', bbox_inches='tight')
    print('wrote ed_checking_models')


if __name__ == '__main__':
    main()
