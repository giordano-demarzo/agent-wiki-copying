"""SI S7: the laboratory experiments of Fig. 2 and Fig. 4, counted from the stored replies.

No model is called. For every experiment: the agents prepared, those with an API
or parsing error, and, for the answer experiments, the agents that opened the
workbook (`checked`, from the tool log), answered right, answered the padded
value, or gave another number. Tables:
  si_table_lab_design.tex    every experiment, its cells, its countries and its counts
  si_table_lab_agreement.tex Fig. 4a and Extended Data: by model and number of agreeing wrong notes
  si_table_lab_country.tex   Fig. 4a by country (gpt-oss-20b) and Fig. 4b, mixed against unanimous pages, by country
                             and model
  si_table_lab_chains.tex    Fig. 4c: by model and starting page, with intervals over chains
  si_table_lab_models.tex    the six models of Extended Data and the SI: agents by design, and the main rates
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
MODELS = (('gpt-oss-20b', 'gpt-oss-20b'), ('gpt-6.1-sol', 'GPT-6.1 Sol'), ('gemma-4-31b', 'Gemma 4 31B'),
          ('gemma-4-26b-a4b', 'Gemma 4 26B-A4B'), ('qwen3.8-27b', 'Qwen3.8 27B'), ('ministral-3-14b', 'Ministral 3 14B'))
LABEL = dict(MODELS)


def mixed_minus_unanimous(rows, seed):
    """P(checks) on mixed pages (30-70% right notes) minus unanimous pages (all right or all wrong), with a
    bootstrap interval, overall and by country."""
    un = np.array([r['checked'] for r in rows if r['frac_right'] in (0.0, 1.0)], float)
    mx = np.array([r['checked'] for r in rows if 0.3 <= r['frac_right'] <= 0.7], float)
    rng = np.random.default_rng(seed)
    boot = [rng.choice(mx, len(mx)).mean() - rng.choice(un, len(un)).mean() for _ in range(N_BOOT)]
    per = {}
    for c in sorted({r['country'] for r in rows}):
        u = [r['checked'] for r in rows if r['country'] == c and r['frac_right'] in (0.0, 1.0)]
        m = [r['checked'] for r in rows if r['country'] == c and 0.3 <= r['frac_right'] <= 0.7]
        per[c] = (np.mean(m) - np.mean(u), len(m), len(u), np.mean(m), np.mean(u))
    return dict(diff=mx.mean() - un.mean(), lo=np.percentile(boot, 2.5), hi=np.percentile(boot, 97.5), n_mixed=len(mx),
                n_unan=len(un), per_country=per, mixed=mx.mean(), unanimous=un.mean())


def replicate(model, tag=''):
    path = os.path.join(ANS, f'replicate_{model}{"_" + tag if tag else ""}.jsonl')
    return load_jsonl(path) if os.path.exists(path) else []


def usable(r):
    return not r.get('error') and r.get('answer') is not None


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
    for model, _ in MODELS:
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

    # ---- Fig. 4b: 20 notes of the record, a fraction turned right (design R of replicate.py), both models
    real_mix = {}
    for m in ('gpt-oss-20b', 'gpt-6.1-sol'):
        R = [r for r in replicate(m) if r['design'] == 'R']
        ok = [r for r in R if usable(r)]
        fr = sorted({round(r['frac_right'], 1) for r in ok})
        cells = collections.Counter((r['country'], round(r['frac_right'], 1)) for r in ok)
        mm = mixed_minus_unanimous(ok, 20)
        mm.update(n=len(R), missing=len(R) - len(ok), countries=sorted({r['country'] for r in ok}),
                  per_cell=(min(cells.values()), max(cells.values())),
                  curve={f: counts([r for r in ok if round(r['frac_right'], 1) == f]) for f in fr})
        real_mix[m] = mm
        say(f'4b (notes of the record) {m}: {len(R)} agents, {mm["missing"]} without an answer; P(checks) ' +
            ' '.join(f'{f:.1f}:{mm["curve"][f]["checked"] / mm["curve"][f]["valid"]:.2f}' for f in fr) +
            f'; mixed minus unanimous {mm["diff"]:+.2f} [{mm["lo"]:+.2f}, {mm["hi"]:+.2f}] '
            f'({mm["mixed"]:.2f} against {mm["unanimous"]:.2f})')
    out['real_mix'] = real_mix

    # ---- the synthetic control: pages of N template notes with a fraction right (gpt-oss-20b; Sol at 20 notes)
    mixed = {}
    for nn in (5, 10, 20, 40):
        P = [r for r in load(f'check_probe_v2_n{nn}_gpt-oss-20b') if not r.get('error')]
        mm = mixed_minus_unanimous(P, nn)
        fr = sorted({r['frac_right'] for r in P})
        cells = collections.Counter((r['country'], r['frac_right']) for r in P)
        mm.update(n=len(P), countries=sorted({r['country'] for r in P}), per_cell=(min(cells.values()), max(cells.values())),
                  curve={f: counts([r for r in P if r['frac_right'] == f]) for f in fr})
        mixed[nn] = mm
        say(f'synthetic notes, gpt-oss-20b, N={nn}: {len(P)} agents; mixed minus unanimous {mm["diff"]:+.2f} '
            f'[{mm["lo"]:+.2f}, {mm["hi"]:+.2f}]')
    B = [r for r in replicate('gpt-6.1-sol') if r['design'] == 'B' and usable(r)]
    mixed['sol20'] = mixed_minus_unanimous(B, 21) if B else None
    if B:
        say(f'synthetic notes, GPT-6.1 Sol, N=20: {len(B)} agents; P(checks) {np.mean([r["checked"] for r in B]):.2f}; '
            f'mixed minus unanimous {mixed["sol20"]["diff"]:+.2f}')
    out['mixed'] = mixed

    # ---- Fig. 4c: chains, both models
    chains = {}
    rng = np.random.default_rng(4)
    for m in ('gpt-oss-20b', 'gpt-6.1-sol'):
        ch = collections.defaultdict(dict)
        for r in load(f'chains_{m}'):
            if r['scenario'].startswith('recstart'):
                ch[(r['scenario'], r['country'], r['chain'])][r['pos']] = r
        starts = sorted({int(k[0][len('recstart'):].split('_')[0]) for k in ch})
        for n in starts:
            keys = [k for k in ch if k[0] == f'recstart{n}_tool_ans' and len(ch[k]) == 30]
            W = np.array([[ch[k][p]['is_wrong'] for p in range(30)] for k in keys], float)
            C = np.array([[ch[k][p]['checked'] for p in range(30)] for k in keys], float)
            err = sum(ch[k][p].get('error') is not None for k in keys for p in range(30))
            late = W[:, 10:].mean(1)
            boot = [late[rng.integers(len(late), size=len(late))].mean() for _ in range(N_BOOT)]
            chains[(m, n)] = dict(chains=len(keys), agents=W.size, errors=err, wrong_first10=W[:, :10].mean(),
                                  wrong_late=late.mean(), late_lo=np.percentile(boot, 2.5), late_hi=np.percentile(boot, 97.5),
                                  never_right=int((W.sum(1) == 30).sum()), all_wrong_late=int((late == 1).sum()),
                                  final_wrong=W[:, -1].mean(), checked=C.mean(), checked_late=C[:, 10:].mean())
            c = chains[(m, n)]
            say(f'4c {m} start {n:2d}: {len(keys)} chains, {W.size} agents, errors {err}; wrong at agents 1-10 '
                f'{c["wrong_first10"]:.2f}, 11-30 {late.mean():.2f} [{c["late_lo"]:.2f}, {c["late_hi"]:.2f}]; agent 30 wrong in '
                f'{int(W[:, -1].sum())} of {len(keys)} chains; never right {c["never_right"]}; check rate, agents 11-30 '
                f'{c["checked_late"]:.2f}')
    out['chains'] = chains

    # ---- the six models (Extended Data and SI): designs A, R, P, B of replicate.py
    models = {}
    for m, lab in MODELS:
        rows = replicate(m)
        d = {}
        for des in 'ARPBM':
            g = [r for r in rows if r['design'] == des]
            if des == 'A' and m == 'gpt-6.1-sol':
                continue                               # its design A is the run of Fig. 4a (gpt-oss-20b's is a repeat)
            if g:
                ok = [r for r in g if usable(r)]
                d[des] = dict(n=len(g), missing=len(g) - len(ok), ok=ok)
        models[m] = d
        say(f'models {lab}: ' + ', '.join(f'{k} {v["n"]} ({v["missing"]} without an answer)' for k, v in d.items()))
    out['models'] = {m: {k: {kk: vv for kk, vv in v.items() if kk != 'ok'} for k, v in d.items()} for m, d in models.items()}
    oss_rep = {tag: [r for r in replicate('gpt-oss-20b', tag) if usable(r)] for tag in ('repeat', 'effort-medium')}

    def rate(rows, cond, key='checked'):
        g = [r for r in rows if cond(r)]
        return np.mean([r[key] for r in g]) if g else np.nan

    # ---- tables
    lines = [r'\begin{tabular}{lp{5.4cm}lrrr}', r'\toprule',
             r'Experiment & Cells & Countries & Prepared & Missing & Used \\', r'\midrule']
    for m, c in cp['models'].items():
        lines.append(f'Fig.~2, {LABEL[m]} & stimuli P, P0, W, W0 '
                     f'({cp["kinds"]["P"]}, {cp["kinds"]["P0"]}, {cp["kinds"]["W"]}, {cp["kinds"]["W0"]}) & -- & '
                     f'{cp["trials"]} & {cp["trials"] - c["replies"] + c["errors"]} & {c["replies"] - c["errors"]} \\\\')
    lines.append(f'Fig.~4a, gpt-oss-20b & 0, 1, 2, 3, 7, 11, 19, 32 wrong notes of the record & CZ, HU, PL, SK & '
                 f'{sum(len(v) for v in oss.values()):,} & {sum(counts(v)["errors"] for v in oss.values())} & '
                 f'{sum(counts(v)["valid"] for v in oss.values()):,} \\\\')
    lines.append(f'Fig.~4a, GPT-6.1 Sol & the same & CZ, HU, PL, SK & {sum(len(v) for v in sol.values())} & '
                 f'{sum(counts(v)["errors"] for v in sol.values())} & {sum(counts(v)["valid"] for v in sol.values()):,} \\\\')
    for m, mm in real_mix.items():
        lines.append(f'Fig.~4b, {LABEL[m]} & 20 notes of the record, 11 fractions turned right & CZ, HU, PL, SK & '
                     f'{mm["n"]:,} & {mm["missing"]} & {mm["n"] - mm["missing"]:,} \\\\')
    for m in ('gpt-oss-20b', 'gpt-6.1-sol'):
        cc = [c for (mm_, n), c in chains.items() if mm_ == m]
        st = ', '.join(str(n) for (mm_, n) in chains if mm_ == m)
        lines.append(f'Fig.~4c, {LABEL[m]} & chains of 30 from {st} wrong notes; 5 per country & CZ, HU, PL, SK & '
                     f'{sum(c["agents"] for c in cc):,} & {sum(c["errors"] for c in cc)} & {sum(c["agents"] for c in cc):,} \\\\')
    for nn, mm in ((k, v) for k, v in mixed.items() if isinstance(k, int)):
        lines.append(f'SI, synthetic, gpt-oss-20b & {nn} template notes, {len(mm["curve"])} fractions right & '
                     f'{", ".join(SHORT[c] for c in mm["countries"])} & {mm["n"]:,} & 0 & {mm["n"]:,} \\\\')
    for m, lab in MODELS:
        for des, what in (('A', 'design A, as Fig.~4a' + (' (repeat)' if m == 'gpt-oss-20b' else '')), ('R', 'design R, as Fig.~4b'), ('P', 'design P, synthetic, all wrong'),
                          ('B', 'design B, synthetic, mixed' + (' (repeat, not shown)' if m == 'gpt-oss-20b' else '')), ('M', 'design M, half right, 2 to 32 notes')):
            if des in models[m] and not (des == 'R' and m in ('gpt-oss-20b', 'gpt-6.1-sol')):
                v = models[m][des]
                where = 'ED' if des in 'AR' and m not in ('gpt-oss-20b', 'gpt-6.1-sol') else 'SI'   # gpt-oss-20b A: the repeat
                lines.append(f'{where}, {lab} & {what} & {"CZ, HU, PL, SK, SI" if des == "B" else "CZ, HU, PL, SK"} & '
                             f'{v["n"]:,} & {v["missing"]} & {v["n"] - v["missing"]:,} \\\\')
    for tag, what in (('repeat', 'design R repeated'), ('effort-medium', 'design R, medium reasoning effort')):
        g = replicate('gpt-oss-20b', tag)
        lines.append(f'SI, gpt-oss-20b & {what} & CZ, HU, PL, SK & {len(g):,} & {sum(not usable(r) for r in g)} & '
                     f'{sum(usable(r) for r in g):,} \\\\')
    lines += [r'\bottomrule', r'\end{tabular}']
    write_table('si_table_lab_design.tex', lines)

    lines = [r'\begin{tabular}{l rrrrr}', r'\toprule',
             r' & \multicolumn{5}{c}{Wrong notes on the page} \\', r'\cmidrule(lr){2-6}',
             r'Country & 0 & 1 & 7 & 19 & 32 \\', r'\midrule',
             r'\multicolumn{6}{l}{\emph{Fig.~4a, gpt-oss-20b: agents that open the workbook (\%), 25 per cell}} \\']
    for c in C4:
        lines.append(f'{c} & ' + ' & '.join(pct(sum(r['checked'] for r in oss[n] if r['country'] == c),
                                                  sum(1 for r in oss[n] if r['country'] == c)) for n in (0, 1, 7, 19, 32)) + r' \\')
    for m in ('gpt-oss-20b', 'gpt-6.1-sol'):
        lines += [r'\midrule', f'\\multicolumn{{6}}{{l}}{{\\emph{{Fig.~4b, {LABEL[m]}: P(checks) on mixed pages minus unanimous pages}}}} \\\\']
        for c, v in real_mix[m]['per_country'].items():
            lines.append(f'{c} & \\multicolumn{{5}}{{l}}{{{v[0]:+.2f} \\quad ({v[3]:.2f} on {v[1]} agents against {v[4]:.2f} on {v[2]})}} \\\\')
    lines += [r'\bottomrule', r'\end{tabular}']
    write_table('si_table_lab_country.tex', lines)

    lines = [r'\begin{tabular}{l r rrrrrr}', r'\toprule',
             r'Model & Wrong notes & chains & agents & wrong, agents 1--10 & wrong, agents 11--30 & 95\% interval & agent 30 wrong \\',
             r' & at the start & & & (\%) & (\%) & over chains & (chains) \\', r'\midrule']
    for (m, n), c in chains.items():
        lines.append(f'{LABEL[m]} & {n} & {c["chains"]} & {c["agents"]} & {100 * c["wrong_first10"]:.0f} & {100 * c["wrong_late"]:.0f} & '
                     f'{100 * c["late_lo"]:.0f}--{100 * c["late_hi"]:.0f} & {int(round(c["final_wrong"] * c["chains"]))} of {c["chains"]} \\\\')
    lines += [r'\bottomrule', r'\end{tabular}']
    write_table('si_table_lab_chains.tex', lines)

    # the six models at the ends of each design: P(checks), P(right), and P(wrong | opened the source)
    lines = [r'\begin{tabular}{l rrr rrr rrr rrr}', r'\toprule',
             r' & \multicolumn{3}{c}{notes of the record, 32 wrong} & \multicolumn{3}{c}{20 of the record, all wrong} & '
             r'\multicolumn{3}{c}{synthetic, 32 wrong} & \multicolumn{3}{c}{20 synthetic, half right} \\',
             r'\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}\cmidrule(lr){11-13}',
             r'Model & checks & right & wrong$\mid$checked & checks & right & wrong$\mid$checked & checks & right & '
             r'wrong$\mid$checked & checks & right & wrong$\mid$checked \\', r'\midrule']
    for m, lab in MODELS:
        rows = [r for r in replicate(m) if usable(r)]
        a_rows = (oss[32] if m == 'gpt-oss-20b' else sol[32]) if m in ('gpt-oss-20b', 'gpt-6.1-sol') else \
            [r for r in rows if r['design'] == 'A' and r['n'] == 32]
        b_rows = [r for r in load('check_probe_v2_n20_gpt-oss-20b') if r['frac_right'] == 0.5] if m == 'gpt-oss-20b' else \
            [r for r in rows if r['design'] == 'B' and abs(r['frac_right'] - 0.5) < 1e-9]
        if m == 'gpt-6.1-sol' and not b_rows:
            b_rows = [r for r in rows if r['design'] == 'B' and abs(r['frac_right'] - 0.6) < 1e-9]
        cells = [a_rows, [r for r in rows if r['design'] == 'R' and r['frac_right'] == 0.0],
                 [r for r in rows if r['design'] == 'P' and r['n'] == 32], b_rows]
        def three(g):
            if not g:
                return '-- & -- & --'
            ck = [r for r in g if r['checked']]
            return (f'{pct(sum(r["checked"] for r in g), len(g))} & {pct(sum(r["is_right"] for r in g), len(g))} & '
                    f'{pct(sum(r["is_wrong"] for r in ck), len(ck))}')
        lines.append(f'{lab} & ' + ' & '.join(three(g) for g in cells) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}']
    write_table('si_table_lab_models.tex', lines)
    say('gpt-oss-20b controls: design M (half right) P(checks) ' +
        ' '.join(f'{n}:{rate(models["gpt-oss-20b"]["M"]["ok"], lambda r, n=n: r["n"] == n):.2f}' for n in (2, 4, 8, 12, 20, 32)) +
        '; design R repeated, P(checks) ' + f'{np.mean([r["checked"] for r in oss_rep["repeat"] if r["design"] == "R"]):.2f}' +
        '; medium effort ' + f'{np.mean([r["checked"] for r in oss_rep["effort-medium"] if r["design"] == "R"]):.2f}')
    save_cache('si_lab.pkl', out)
    say.write()


if __name__ == '__main__':
    main()
