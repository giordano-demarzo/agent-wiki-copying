"""Checking against the support for a wrong value. P(agent opens the accurate source) and P(answers wrong) on an
all-wrong page of N notes, for three kinds of notes: plain templates (check_probe.py allwrong), real notes from the
record before the first correction, and templates claiming a definitive source (real_support.py). gpt-oss-20b,
Czech Republic, Hungary, Poland, Slovak Republic, 100 agents per point. Writes lab/plots/support_checking.png.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / 'src'))
from common import wilson  # noqa: E402
from style import AMBER, GREY, INK, MODEL as TEAL, ROSE, panel, plt, use  # noqa: E402

NS = (1, 7, 19, 32)
C4 = ('Czech Republic', 'Hungary', 'Poland', 'Slovak Republic')


def rows():
    out = {'plain': {}, 'record': {}, 'definitive': {}}
    for n in NS:
        out['plain'][n] = [r for r in map(json.loads, open(ROOT / 'data' / 'answers' / f'check_probe_allwrong_n{n}_gpt-oss-20b.jsonl'))
                           if r['country'] in C4 and not r.get('error')]
    for r in map(json.loads, open(ROOT / 'data' / 'answers' / 'real_support_gpt-oss-20b.jsonl')):
        if not r.get('error'):
            out[r['style']].setdefault(r['n'], []).append(r)
    return out


def main():
    use()
    D = rows()
    base = [r for r in map(json.loads, open(ROOT / 'data' / 'answers' / 'check_probe_allwrong_n0_gpt-oss-20b.jsonl')) if r['country'] in C4]
    fig, axes = plt.subplots(1, 2, figsize=(5.4, 2.5), sharey=True)
    plt.subplots_adjust(wspace=0.12)
    for ax, key, letter, title in ((axes[0], 'checked', 'a', 'P(agent opens the source)'), (axes[1], 'is_wrong', 'b', 'P(agent answers wrong)')):
        panel(ax, letter)
        for style, col, mk, lab in (('plain', GREY, 'o', 'plain notes'), ('record', INK, 's', "the record's own notes"),
                                    ('definitive', ROSE, '^', '"definitive workbook" notes')):
            ns = [0] + list(NS)
            vals = [base] + [D[style][n] for n in NS]
            ps = [wilson(sum(r[key] for r in g), len(g)) for g in vals]
            x = np.array(ns) + 1
            ax.errorbar(x, [p[0] for p in ps], yerr=[[max(p[0] - p[1], 0) for p in ps], [max(p[2] - p[0], 0) for p in ps]],
                        fmt=mk + '-', color=col, ms=3.6, lw=1.1, capsize=0, label=lab)
        ax.set_xscale('log')
        ax.set_xticks([1, 2, 8, 20, 33])
        ax.set_xticklabels(['0', '1', '7', '19', '32'])
        ax.set(xlim=(0.85, 40), ylim=(-0.03, 1.03), xlabel='wrong notes on the page')
        ax.set_title(title, fontsize=6.4)
    axes[0].set_ylabel('probability')
    axes[0].legend(frameon=False, fontsize=5.2, loc='lower left')
    out = ROOT.parent / 'figures' / 'support_checking.png'
    fig.savefig(out, dpi=220, bbox_inches='tight')
    print('wrote', out)


if __name__ == '__main__':
    main()
