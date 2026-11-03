"""Extended Data figure: checking in the record before the first correction against the two laboratory models of
Fig. 4a. Checking relative to the empty page against the number of agreeing wrong notes: the record before the first
correction (the note cites a source; bins 0, 1-3 and 4+) and gpt-oss-20b and GPT-6.1 Sol (the agent opens the source);
95% bootstrap intervals of the ratio.
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
    fig, ax0 = plt.subplots(1, 1, figsize=(2.9, 2.5))
    axes = [None, ax0]
    R = record_rows()
    # (a) agreement on a wrong value switches checking off: record before the first correction and laboratory,
    # each relative to its own empty page
    ax = axes[1]

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
        f'gpt-oss-20b: opens the workbook (empty page {np.mean([r["checked"] for r in base]):.2f})', fmt='o-', color=OSS, ms=3.2, lw=1.2)
    sol = [r for r in map(json.loads, open(ROOT / 'data' / 'answers' / 'panel_a_gpt-6.1-sol.jsonl')) if not r.get('error')]
    sn = (0, 1, 2, 3, 7, 11, 19, 32)
    rel(ax, (np.array(sn) + 1) * 1.04, [[r['checked'] for r in sol if r['n'] == n] for n in sn],
        f'GPT-6.1 Sol: opens the workbook ({np.mean([r["checked"] for r in sol if r["n"] == 0]):.2f})', fmt='D-', color=SOL, ms=3.4, lw=1.2)
    pre = [x for x in R if x['h'] < 0]          # before the first correction every use gives the wrong value
    bins = [(0, 1), (1, 4), (4, 40)]
    sel = [[x for x in pre if lo <= x['n'] < hi] for lo, hi in bins]
    xb = np.array([np.mean([x['n'] for x in g]) for g in sel]) + 1
    rel(ax, xb / 1.04, [[x['src'] for x in g] for g in sel], f'record, before the correction: note cites a source ({np.mean([x["src"] for x in sel[0]]):.2f})',
        fmt='o-', color=INK, ms=3.6, lw=1.4)
    print('(a) record n per bin', [len(g) for g in sel])
    ax.axhline(1, color=GREY, lw=0.5, ls=':')
    ax.set_xscale('log')
    ax.set_xticks([1, 2, 4, 8, 12, 20, 33])
    ax.set_xticklabels(['0', '1', '3', '7', '11', '19', '32'])
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.set(xlim=(0.85, 40), ylim=(0, 1.3), xlabel='notes agreeing on the wrong value',
           ylabel='checking, relative to the empty page')
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.legend(frameon=False, fontsize=4.6, loc='upper right', handlelength=1.6, bbox_to_anchor=(1.0, 1.08))

    fig.savefig(ROOT.parent / 'figures' / 'ed_checking_models.png', dpi=220, bbox_inches='tight')
    fig.savefig(ROOT.parent / 'paper' / 'figs' / 'ed_checking_models.pdf', bbox_inches='tight')
    print('wrote ed_checking_models')


if __name__ == '__main__':
    main()
