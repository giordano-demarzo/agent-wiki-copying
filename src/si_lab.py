"""SI S7: the laboratory experiments of Fig. 2 and Fig. 4, counted from the stored replies.

No model is called. For every experiment: the agents prepared, those with an API
or parsing error, and, for the answer experiments, the agents that opened the
workbook (`checked`, from the tool log), answered right, answered the padded
value, or gave another number. Tables:
  si_table_lab_design.tex    every experiment, its cells, its countries and its counts
  si_table_lab_agreement.tex Fig. 4a and Extended Data: by model and number of agreeing wrong notes
  si_table_lab_country.tex   Fig. 4a by country (gpt-oss-20b) and Fig. 4b, mixed against unanimous pages, by country
  si_table_lab_chains.tex    Fig. 4c: by starting page, with intervals over chains
Writes cache/si_lab.pkl.
"""
import collections
import json
import os

import numpy as np

from common import ROOT, Report, save_cache, wilson, write_table

ANS = os.path.join(ROOT, 'lab', 'data', 'answers')
C4 = ('Czech Republic', 'Hungary', 'Poland', 'Slovak Republic')
SHORT = {'Czech Republic': 'CZ', 'Hungary': 'HU', 'Poland': 'PL', 'Slovak Republic': 'SK', 'Slovenia': 'SI'}
N_BOOT = 2000


def load(name):
    with open(os.path.join(ANS, name + '.jsonl')) as fh:
        return [json.loads(line) for line in fh]


def counts(rows):
    ok = [r for r in rows if not r.get('error')]
    return dict(n=len(rows), errors=len(rows) - len(ok), checked=sum(r['checked'] for r in ok),
                right=sum(r['is_right'] for r in ok), wrong=sum(r['is_wrong'] for r in ok),
                other=sum(not r['is_right'] and not r['is_wrong'] for r in ok), valid=len(ok),
                right_unchecked=sum(r['is_right'] and not r['checked'] for r in ok),
                wrong_checked=sum(r['is_wrong'] and r['checked'] for r in ok))


def pct(k, n):
    return f'{100 * k / n:.0f}' if n else '--'


def copying_counts():
    """The copying experiment of Fig. 2: stimuli by kind and replies by model."""
    trials = load_jsonl(os.path.join(ROOT, 'lab', 'data', 'trials.jsonl'))
    kinds = collections.Counter(t.get('kind') for t in trials)
    out = dict(trials=len(trials), kinds=dict(kinds), models={})
    for model in ('gpt-oss-20b', 'gpt-6.1-sol'):
        path = os.path.join(ROOT, 'lab', 'data', f'replies_{model}.jsonl')
        reps = load_jsonl(path)
        by_kind = collections.Counter(r.get('kind') for r in reps)
        bad = sum(1 for r in reps if not isinstance(r.get('reply'), dict))
        out['models'][model] = dict(replies=len(reps), by_kind=dict(by_kind), errors=bad)
    return out


def load_jsonl(path):
    with open(path) as fh:
        return [json.loads(line) for line in fh]


def main():
    say = Report('SI S7: the laboratory experiments')
    out = {}

    # ---- copying (Fig. 2)
    cp = copying_counts()
    out['copying'] = cp
    say(f'copying: {cp["trials"]} stimuli, by kind {cp["kinds"]}')
    for m, c in cp['models'].items():
        say(f'  {m}: {c["replies"]} replies, by kind {c["by_kind"]}, reply not parsed {c["errors"]}')

    # ---- Fig. 4a: pages of N agreeing wrong notes
    oss = {0: [r for r in load('check_probe_allwrong_n0_gpt-oss-20b') if r['country'] in C4]}
    for r in load('real_support_gpt-oss-20b'):
        if r['style'] == 'record':
            oss.setdefault(r['n'], []).append(r)
    for r in load('panel_a_gpt-oss-20b'):
        oss.setdefault(r['n'], []).append(r)
    sol = collections.defaultdict(list)
    for r in load('panel_a_gpt-6.1-sol'):
        sol[r['n']].append(r)
    agreement = {}
    lines = [r'\begin{tabular}{r rrrrr rrrrr}', r'\toprule',
             r' & \multicolumn{5}{c}{gpt-oss-20b} & \multicolumn{5}{c}{GPT-6.1 Sol} \\',
             r'\cmidrule(lr){2-6}\cmidrule(lr){7-11}',
             r'Wrong notes & agents & error & checks (\%) & right (\%) & wrong (\%) & agents & error & checks (\%) & right (\%) & wrong (\%) \\',
             r'\midrule']
    for n in sorted(set(oss) | set(sol)):
        a, b = counts(oss[n]), counts(sol[n])
        agreement[n] = dict(oss=a, sol=b)
        lines.append(f'{n} & ' + ' & '.join(f'{c["n"]} & {c["errors"]} & {pct(c["checked"], c["valid"])} & '
                                            f'{pct(c["right"], c["valid"])} & {pct(c["wrong"], c["valid"])}' for c in (a, b)) + r' \\')
        say(f'4a N={n:2d}: ' + '; '.join(f'{lab} n {c["n"]} errors {c["errors"]} checked {c["checked"]} right {c["right"]} '
                                          f'wrong {c["wrong"]} other {c["other"]} (right without checking {c["right_unchecked"]}, '
                                          f'wrong after checking {c["wrong_checked"]})' for lab, c in (('oss', a), ('Sol', b))))
    lines += [r'\bottomrule', r'\end{tabular}']
    write_table('si_table_lab_agreement.tex', lines)
    out['agreement'] = agreement
    tot = lambda key, k: sum(agreement[n][key][k] for n in agreement)
    say(f'4a gpt-oss-20b: {tot("oss", "checked")} agents checked, {tot("oss", "wrong_checked")} of them answered the padded value; '
        f'{tot("oss", "right_unchecked")} answered right without checking. GPT-6.1 Sol: {tot("sol", "checked")} checked, '
        f'{tot("sol", "wrong_checked")} wrong after checking, {tot("sol", "right_unchecked")} right without checking')

    # ---- Fig. 4b: pages of N notes with a fraction right
    mixed = {}
    for nn in (5, 10, 20, 40):
        P = [r for r in load(f'check_probe_v2_n{nn}_gpt-oss-20b') if not r.get('error')]
        fr = sorted({r['frac_right'] for r in P})
        cty = sorted({r['country'] for r in P})
        cells = collections.Counter((r['country'], r['frac_right']) for r in P)
        say(f'4b N={nn}: {len(P)} agents, {len(fr)} fractions, countries {", ".join(cty)}, agents per country and fraction '
            f'{min(cells.values())}-{max(cells.values())}; P(checks) ' +
            ' '.join(f'{f:.1f}:{np.mean([r["checked"] for r in P if r["frac_right"] == f]):.2f}' for f in fr))
        per = {}
        for c in cty:
            un = [r['checked'] for r in P if r['country'] == c and r['frac_right'] in (0.0, 1.0)]
            mx = [r['checked'] for r in P if r['country'] == c and 0.3 <= r['frac_right'] <= 0.7]
            per[c] = (np.mean(mx) - np.mean(un), len(mx), len(un), np.mean(mx), np.mean(un))
        un = [r['checked'] for r in P if r['frac_right'] in (0.0, 1.0)]
        mx = [r['checked'] for r in P if 0.3 <= r['frac_right'] <= 0.7]
        rng = np.random.default_rng(nn)
        un_a, mx_a = np.array(un, float), np.array(mx, float)
        boot = [rng.choice(mx_a, len(mx_a)).mean() - rng.choice(un_a, len(un_a)).mean() for _ in range(N_BOOT)]
        mixed[nn] = dict(diff=mx_a.mean() - un_a.mean(), lo=np.percentile(boot, 2.5), hi=np.percentile(boot, 97.5),
                         n_mixed=len(mx), n_unan=len(un), per_country=per, n=len(P), countries=cty,
                         per_cell=(min(cells.values()), max(cells.values())),
                         curve={f: counts([r for r in P if r['frac_right'] == f]) for f in fr})
        say(f'  mixed (30-70% right) minus unanimous pages: {mixed[nn]["diff"]:+.2f} [{mixed[nn]["lo"]:+.2f}, {mixed[nn]["hi"]:+.2f}]; '
            'by country ' + ', '.join(f'{SHORT[c]} {v[0]:+.2f}' for c, v in per.items()))
    out['mixed'] = mixed

    # ---- Fig. 4c: chains
    ch = collections.defaultdict(dict)
    for r in load('chains_gpt-oss-20b'):
        ch[(r['scenario'], r['country'], r['chain'])][r['pos']] = r
    chains = {}
    rng = np.random.default_rng(4)
    for n in (1, 7, 19, 32):
        keys = [k for k in ch if k[0] == f'recstart{n}_tool_ans']
        W = np.array([[ch[k][p]['is_wrong'] for p in range(30)] for k in keys], float)
        C = np.array([[ch[k][p]['checked'] for p in range(30)] for k in keys], float)
        err = sum(ch[k][p].get('error') is not None for k in keys for p in range(30))
        late = W[:, 10:].mean(1)
        boot = [late[rng.integers(len(late), size=len(late))].mean() for _ in range(N_BOOT)]
        chains[n] = dict(chains=len(keys), agents=W.size, errors=err, countries=collections.Counter(k[1] for k in keys),
                         wrong_first10=W[:, :10].mean(), wrong_late=late.mean(), late_lo=np.percentile(boot, 2.5),
                         late_hi=np.percentile(boot, 97.5), never_right=int((W.sum(1) == 30).sum()),
                         final_wrong=W[:, -1].mean(), checked=C.mean(), raw=W.mean(0), late_by_chain=late)
        say(f'4c start {n:2d}: {len(keys)} chains, {W.size} agents, errors {err}; wrong at agents 1-10 {W[:, :10].mean():.2f}, '
            f'11-30 {late.mean():.2f} [{chains[n]["late_lo"]:.2f}, {chains[n]["late_hi"]:.2f}] (bootstrap over chains); '
            f'agent 30 wrong in {int(W[:, -1].sum())} of {len(keys)} chains; never right {chains[n]["never_right"]}; '
            f'check rate {C.mean():.2f}; chains with 11-30 wrong share above one half {int((late > 0.5).sum())}')
    out['chains'] = chains

    # ---- tables
    lines = [r'\begin{tabular}{lp{5.1cm}lrrr}', r'\toprule',
             r'Experiment & Cells & Countries & Prepared & Missing & Used \\', r'\midrule']
    for m, c in cp['models'].items():
        lines.append(f'Fig.~2, {"gpt-oss-20b" if m == "gpt-oss-20b" else "GPT-6.1 Sol"} & stimuli P, P0, W, W0 '
                     f'({cp["kinds"]["P"]}, {cp["kinds"]["P0"]}, {cp["kinds"]["W"]}, {cp["kinds"]["W0"]}) & -- & '
                     f'{cp["trials"]} & {cp["trials"] - c["replies"] + c["errors"]} & {c["replies"] - c["errors"]} \\\\')
    lines.append(f'Fig.~4a, gpt-oss-20b & 0, 1, 2, 3, 7, 11, 19, 32 wrong notes of the record & CZ, HU, PL, SK & '
                 f'{sum(len(v) for v in oss.values()):,} & {sum(counts(v)["errors"] for v in oss.values())} & '
                 f'{sum(counts(v)["valid"] for v in oss.values()):,} \\\\')
    lines.append(f'Fig.~4a, GPT-6.1 Sol & the same & CZ, HU, PL, SK & {sum(len(v) for v in sol.values())} & '
                 f'{sum(counts(v)["errors"] for v in sol.values())} & {sum(counts(v)["valid"] for v in sol.values()):,} \\\\')
    for nn, m in mixed.items():
        lines.append(f'Fig.~4b, {nn} notes & {len(m["curve"])} fractions of right notes & '
                     f'{", ".join(SHORT[c] for c in m["countries"])} & {m["n"]:,} & 0 & {m["n"]:,} \\\\')
    lines.append(f'Fig.~4c, chains & 1, 7, 19, 32 wrong notes at the start; 5 chains of 30 agents per country & '
                 f'CZ, HU, PL, SK & {sum(c["agents"] for c in chains.values()):,} & {sum(c["errors"] for c in chains.values())} & '
                 f'{sum(c["agents"] for c in chains.values()):,} \\\\')
    lines += [r'\bottomrule', r'\end{tabular}']
    write_table('si_table_lab_design.tex', lines)

    lines = [r'\begin{tabular}{l rrrrr}', r'\toprule',
             r' & \multicolumn{5}{c}{Wrong notes on the page} \\', r'\cmidrule(lr){2-6}',
             r'Country & 0 & 1 & 7 & 19 & 32 \\', r'\midrule',
             r'\multicolumn{6}{l}{\emph{Fig.~4a, gpt-oss-20b: agents that open the workbook (\%), 25 per cell}} \\']
    for c in C4:
        lines.append(f'{c} & ' + ' & '.join(pct(sum(r['checked'] for r in oss[n] if r['country'] == c),
                                                  sum(1 for r in oss[n] if r['country'] == c)) for n in (0, 1, 7, 19, 32)) + r' \\')
    lines += [r'\midrule', r'\multicolumn{6}{l}{\emph{Fig.~4b, 20 notes: P(checks) on mixed pages minus unanimous pages}} \\']
    for c, v in mixed[20]['per_country'].items():
        lines.append(f'{c} & \\multicolumn{{5}}{{l}}{{{v[0]:+.2f} \\quad ({v[3]:.2f} on {v[1]} agents against {v[4]:.2f} on {v[2]})}} \\\\')
    lines += [r'\bottomrule', r'\end{tabular}']
    write_table('si_table_lab_country.tex', lines)

    lines = [r'\begin{tabular}{r rrrrrr}', r'\toprule',
             r'Wrong notes & chains & agents & wrong, agents 1--10 & wrong, agents 11--30 & 95\% interval & agent 30 wrong \\',
             r'at the start & & & (\%) & (\%) & over chains & (chains) \\', r'\midrule']
    for n, c in chains.items():
        lines.append(f'{n} & {c["chains"]} & {c["agents"]} & {100 * c["wrong_first10"]:.0f} & {100 * c["wrong_late"]:.0f} & '
                     f'{100 * c["late_lo"]:.0f}--{100 * c["late_hi"]:.0f} & {int(round(c["final_wrong"] * c["chains"]))} of {c["chains"]} \\\\')
    lines += [r'\bottomrule', r'\end{tabular}']
    write_table('si_table_lab_chains.tex', lines)
    save_cache('si_lab.pkl', out)
    say.write()


if __name__ == '__main__':
    main()
