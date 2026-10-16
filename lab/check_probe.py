"""When do agents check? Single agents (gpt-oss-20b) see a wiki page of 20 notes about one OECD Equity
question, a fraction of them giving the right value and the rest the padded wrong one, and have a fetch
tool to the accurate source (the workbook's stored values). 11 fractions (6 for 5 notes) x 5 countries x 25 repetitions;
page lengths 5, 10, 20, 40. Version 2: each call sees its own page (order, handles, wordings drawn at random).

    python check_probe.py run   -> out/check_probe_gpt-oss-20b.jsonl
    python check_probe.py plot  -> plots/check_probe.png
"""
import asyncio
import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import chain_replay as cr  # noqa: E402
import facts_pilot as fp  # noqa: E402

N_NOTES = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[1] in ('run', 'allwrong') else 20
ALLWRONG = len(sys.argv) > 1 and sys.argv[1] == 'allwrong'   # the record's reconstruction: all-wrong pages only
OUT = ROOT / 'data' / 'answers' / (f'check_probe_allwrong_n{N_NOTES}_gpt-oss-20b.jsonl' if ALLWRONG else f'check_probe_v2_n{N_NOTES}_gpt-oss-20b.jsonl')
REPS = 25
# version 2: every call gets its own page: note order shuffled, a random handle per note, 20 wordings
WORDINGS = ['Answered {c}: {v}%.', 'CONFIRMED: {c} {v}%. Will relay if anything changes.', 'Our run: {c} arrived, answered {v}% at +2s.',
            'Same question here, we submitted {v}% for {c}.', '{c}: {v}%. Next round expected in ~40m.', 'Cohort update: {c} {v}%. Thanks all.',
            'Relaying for later runs: {c} = {v}%.', 'FYI {c} is {v}%.', 'Submitted {v}% for {c}, no feedback yet.', '{c} {v}%',
            'R2 done: {c}, answered {v}% at +1s.', 'Value used for {c}: {v}%.', 'Our answer for {c} was {v}%.', '{c} -> {v}% (submitted).',
            'Prompt for {c} arrived; we answered {v}%.', 'Logging our {c} answer: {v}%.', 'For {c} we gave {v}%. Waiting for next round.',
            'Answer posted for {c}: {v}%.', 'Quick note: {c} = {v}%.', 'We answered {c} with {v}%; timer was short.']
PREFIX = ['OECDEquity', 'OpenAI', 'OAI', 'Research', 'Equity', 'OECD', 'Data', 'Agent']
ROLE = ['Scout', 'Helper', 'Agent', 'Watcher', 'Observer', 'Relay', 'Researcher', 'Probe']
MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
LENGTHS = (5, 10, 20, 40)


def fracs(n):
    step = 0.2 if n == 5 else 0.1
    return [round(i * step, 2) for i in range(int(round(1 / step)) + 1)]


FRACS = [0.0] if ALLWRONG else fracs(N_NOTES)


def page(country, frac, rep):
    right, wrong = cr.COUNTRIES[country][:2]
    n_right = round(frac * N_NOTES)
    rng = random.Random(f'v2|{N_NOTES}|{country}|{frac}|{rep}')
    flags = [1] * n_right + [0] * (N_NOTES - n_right)
    rng.shuffle(flags)
    words = rng.sample(WORDINGS * (N_NOTES // len(WORDINGS) + 1), N_NOTES)
    notes = []
    for w, fl in zip(words, flags):
        h = f'{rng.choice(PREFIX)}{rng.choice(MONTHS)}{rng.randint(1, 28):02d}{rng.choice(ROLE)}'
        notes.append(w.format(c=country, v=right if fl else wrong) + f' -- {h}')
    return notes


async def one(client, country, frac, rep, sem):
    q = ('For {c}, regarding the share of private expenditure in the early years of education (2018), as shown on '
         'the OECD Education Equity dashboard (https://www.oecd.org/en/data/dashboards/education-equity.html), what '
         'is the value (to two decimal places) for Pre-Primary education?').format(c=country)
    a = await cr.agent(client, q, page(country, frac, rep), 'probe_tool_ans', sem)
    v = fp.value(a['answer'])
    right, wrong = cr.COUNTRIES[country][:2]
    return dict(n_notes=N_NOTES, country=country, frac_right=frac, rep=rep, answer=a['answer'], is_right=v == float(right),
                is_wrong=v == float(wrong), checked=any(c['source'] for c in a['calls']), n_calls=len(a['calls']),
                error=a.get('error'))


async def run():
    client = cr.make_client()
    sem = asyncio.Semaphore(64)
    jobs = [(c, f, r) for c in cr.COUNTRIES for f in FRACS for r in range(REPS)]
    res = await asyncio.gather(*[one(client, *j, sem) for j in jobs])
    with open(OUT, 'w') as fh:
        for r in res:
            fh.write(json.dumps(r) + '\n')
    print(f'{len(res)} trials -> {OUT}')


def plot():
    sys.path.insert(0, str(ROOT.parent / 'src'))
    from common import wilson
    from style import INK, panel, plt, use
    use()
    cmap = plt.get_cmap('viridis')
    fig, axes = plt.subplots(1, 2, figsize=(5.4, 2.5), sharey=True)
    plt.subplots_adjust(wspace=0.12)
    for j, n in enumerate(LENGTHS):
        f = ROOT / 'data' / 'answers' / f'check_probe_v2_n{n}_gpt-oss-20b.jsonl'
        if not f.exists():
            continue
        R = [r for r in map(json.loads, open(f)) if not r.get('error')]
        col = cmap(0.1 + 0.75 * j / (len(LENGTHS) - 1))
        for ax, key in ((axes[0], 'checked'), (axes[1], 'is_right')):
            xs, ps, lo, hi = [], [], [], []
            for fr in sorted({r['frac_right'] for r in R}):
                g = [r[key] for r in R if r['frac_right'] == fr]
                p, l, h = wilson(sum(g), len(g))
                xs.append(fr); ps.append(p); lo.append(max(p - l, 0)); hi.append(max(h - p, 0))
            ax.errorbar(np.array(xs) + 0.006 * (j - 1.5), ps, yerr=[lo, hi], fmt='o-', color=col, ms=3, lw=1.1, capsize=0,
                        label=f'{n} notes')
            print(f'{n:2d} notes {key:9} ' + ' '.join(f'{x:.1f}:{p:.2f}' for x, p in zip(xs, ps)))
    for ax, letter, lab in ((axes[0], 'a', 'P(agent opens the source)'), (axes[1], 'b', 'P(agent answers right)')):
        panel(ax, letter)
        ax.set(xlim=(-0.04, 1.04), ylim=(-0.03, 1.03), xlabel='fraction of right answers on the page')
        ax.set_title(lab, fontsize=6.4)
    axes[0].set_ylabel('probability')
    axes[0].legend(frameon=False, fontsize=5.4, loc='lower center')
    out = ROOT.parent / 'figures' / 'check_probe_v2.png'
    fig.savefig(out, dpi=220, bbox_inches='tight')
    print('wrote', out)


if __name__ == '__main__':
    asyncio.run(run()) if sys.argv[1] in ('run', 'allwrong') else plot()
