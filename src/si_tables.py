"""The tables of the paper and the SI, as LaTeX fragments in paper/tables/."""
import numpy as np

import names as nm
from common import load_cache, tex, wilson, write_table
from conventions import CLASSES, CONV

CLASS_NAME = {'coined': 'coined', 'semantic': 'near-synonym', 'habit': 'habit'}
CLASS_PLURAL = {'coined': 'coined names', 'semantic': 'near-synonyms', 'habit': 'habits of style'}


def pm(b, se, d=2):
    return f'${b:.{d}f}\\pm{1.96 * se:.{d}f}$'


def ci(b, se, d=2):
    return f'[{b - 1.96 * se:.{d}f}, {b + 1.96 * se:.{d}f}]'


def table_regressions():
    """Table 1 of the paper in the standard layout: one panel per choice, the
    regressors as rows with the 95% interval beneath each coefficient, then the
    intercepts, the task x 3 h effects, the observations and R^2."""
    d = load_cache('identification.pkl')
    p, n, f = d['pages'], d['names'], d['forms']
    lines = ['\\begin{tabular}{lcc}', '\\toprule', ' & (1) & (2) \\\\', '\\midrule']
    panels = (('A. Where to write: the candidate page is the one chosen', 'pages', p, 'arrival', f'{p["n"]:,} candidates, {p["n_decisions"]:,} arrivals'),
              ('B. How to sign: the name carries the feature', 'names', n, 'feature', f'{n["n"]:,} newcomer--feature records'),
              ('C. How to write: the first use takes form A', 'forms', f, 'convention', f'{f["n"]:,} first uses'))
    for title, key, r, unit, obs in panels:
        nested = r.get('nested', False)
        lines.append(f'\\multicolumn{{3}}{{l}}{{\\textit{{{title}}}}} \\\\[2pt]')
        for j, lab in enumerate(d['labels'][key]):
            dj = 3 if (key == 'pages' and j >= 2) else 2
            c1 = f'{r["b0"][j]:.{dj}f}'
            c2 = '' if nested else f'{r["b1"][j]:.{dj}f}'
            lines.append(f'{lab} & {c1} & {c2} \\\\')
            lines.append(f' & {ci(r["b0"][j], r["se0"][j], dj)} & {"" if nested else ci(r["b1"][j], r["se1"][j], dj)} \\\\')
        lines.append(f'{unit.capitalize()} intercepts & yes & {"" if nested else "yes"} \\\\')
        lines.append(f'Task $\\times$ 3\\,h intercepts & no & {"nested" if nested else "yes"} \\\\')
        lines.append(f'Observations & \\multicolumn{{2}}{{c}}{{{obs}}} \\\\')
        lines.append(f'$R^2$ & {r["r2_0"]:.2f} & {"" if nested else f"{r[chr(114)+chr(50)+chr(95)+chr(49)]:.2f}"} \\\\')
        if key != 'forms':
            lines.append('\\midrule')
    lines += ['\\bottomrule', '\\end{tabular}']
    write_table('table_regressions.tex', lines)


def table_families():
    rows = load_cache('si_population.pkl')['families']
    lines = ['\\begin{tabular}{lrrrll}', '\\toprule', 'task family & handles & edits & pages & first & last \\\\', '\\midrule']
    for fam, h, e, p, a, b in rows:
        lines.append(f'{tex(fam)} & {h} & {e} & {p} & {a[5:]} & {b[5:]} \\\\')
    lines += ['\\midrule', f'all & {sum(r[1] for r in rows)}$^*$ & {sum(r[2] for r in rows)} & {sum(r[3] for r in rows)} & & \\\\',
              '\\bottomrule', '\\end{tabular}']
    write_table('si_table_families.tex', lines)


def table_pieces():
    stats = load_cache('names.pkl')['stats']
    lines = ['\\begin{tabular}{llrrrrl}', '\\toprule',
             'piece & class & carriers & over-representation & $s$ & consistency & most common family \\\\', '\\midrule']
    for cls in ('generic', 'task-specific', 'month'):
        for s in sorted([s for s in stats.values() if s['cls'] == cls and s['name'] != 'date'], key=lambda s: -s['carriers']):
            gap = f'{s["gap"]:.1f}' if np.isfinite(s['gap']) and s['gap'] > 0 else '--'
            lines.append(f'\\texttt{{{tex(s["name"])}}} & {cls} & {s["carriers"]} & {s["over"]:.1f} & {s["s"]:.2f} & {gap} & '
                         f'{tex(s["top_family"])} \\\\')
    lines += ['\\bottomrule', '\\end{tabular}']
    write_table('si_table_pieces.tex', lines)


def table_conventions():
    rows = load_cache('forms.pkl')['rows']
    lines = ['\\begin{tabular}{llrrrll}', '\\toprule',
             'convention & class & uses & pages & minority & $s$ (95\\% interval) & kept \\\\', '\\midrule']
    order = sorted(rows, key=lambda n: (not rows[n]['kept'], -rows[n]['uses']))
    for n in order:
        r = rows[n]
        s = (f'{sum(r["mu"]):.2f} ({r["s_interval"][1]:.2f}--{r["s_interval"][2]:.2f})' if r['s_interval'] else '--')
        minority = '--' if np.isnan(r['minority']) else f'{r["minority"]:.2f}'
        kept = 'yes' if r['kept'] else 'no (' + ', '.join(r['fails']) + ')'
        lines.append(f'{tex(n)} & {CLASS_NAME[r["cls"]]} & {r["uses"]} & {r["n_pages"]} & {minority} & {s} & {kept} \\\\')
    lines += ['\\bottomrule', '\\end{tabular}']
    write_table('si_table_conventions.tex', lines)


def table_slopes():
    v = load_cache('si_forms.pkl')['variants']
    lines = ['\\begin{tabular}{lrrr}', '\\toprule', 'class & first uses & all uses & page 6 h earlier \\\\', '\\midrule']
    for cls, name in CLASS_PLURAL.items():
        lines.append(f'{name} & {v[cls]["first"]:.2f} (n = {v[cls]["n_first"]}) & {v[cls]["all"]:.2f} (n = {v[cls]["n_all"]}) & '
                     f'{v[cls]["shifted"]:.2f} (n = {v[cls]["n_shifted"]}) \\\\')
    lines += ['\\bottomrule', '\\end{tabular}']
    write_table('si_table_slopes.tex', lines)


def table_robustness():
    """The measurements of the paper in the population and on all handles."""
    f, ff = load_cache('forms.pkl'), load_cache('forms_full.pkl')
    d, df = load_cache('identification.pkl'), load_cache('identification_full.pkl')
    both = lambda fn: f'{fn(f)} & {fn(ff)}'
    lines = ['\\begin{tabular}{lll}', '\\toprule', 'quantity & population (1,201 handles) & all handles (3,099) \\\\',
             '\\midrule', '\\multicolumn{3}{l}{\\emph{Where to write}} \\\\',
             f'decisions, candidates & {d["pages"]["n_decisions"]:,}, {d["pages"]["n"]:,} & identical \\\\',
             '\\multicolumn{3}{l}{\\emph{How to sign}} \\\\',
             f'records & {d["names"]["n"]:,} & {df["names"]["n"]:,} \\\\']
    for j, lab in enumerate(d['labels']['names']):
        lines.append(f'{lab} & {d["names"]["b1"][j]:.2f} $\\pm$ {d["names"]["se1"][j]:.2f} & '
                     f'{df["names"]["b1"][j]:.2f} $\\pm$ {df["names"]["se1"][j]:.2f} \\\\')
    lines.append('\\multicolumn{3}{l}{\\emph{How to write}} \\\\')
    lines.append('conventions kept & ' + both(lambda x: str(len(x['keep']))) + ' \\\\')
    for cls, name in CLASS_PLURAL.items():
        slope = lambda x: f'{np.polyfit(x["by_class"][cls][:, 0], x["by_class"][cls][:, 1], 1)[0]:.2f} (n = {len(x["by_class"][cls])})'
        lines.append(f'response slope, {name} & ' + both(slope) + ' \\\\')
    for test, key in (('page beats feed', 'versus_feed'), ('page beats own past', 'versus_own')):
        for cls, name in CLASS_PLURAL.items():
            def follow(x, cls=cls, key=key):
                v = x[key][cls]
                p, lo, hi = wilson(sum(v), len(v))
                return f'{p:.2f} [{lo:.2f}, {hi:.2f}] (n = {len(v)})'
            lines.append(f'{test}, {name} & ' + both(follow) + ' \\\\')
    lines.append(f'first uses in the regression & {d["forms"]["n"]:,} & {df["forms"]["n"]:,} \\\\')
    for j, lab in enumerate(d['labels']['forms']):
        lines.append(f'{lab} & {d["forms"]["b1"][j]:.2f} $\\pm$ {d["forms"]["se1"][j]:.2f} & '
                     f'{df["forms"]["b1"][j]:.2f} $\\pm$ {df["forms"]["se1"][j]:.2f} \\\\')
    lines += ['\\bottomrule', '\\end{tabular}']
    write_table('si_table_robustness.tex', lines)


def table_identification():
    """The three regressions in full: every specification, with standard errors."""
    d = load_cache('identification.pkl')
    p = d['pages']
    lines = ['\\begin{tabular}{lc}', '\\toprule', 'regressor & arrival intercepts \\\\', '\\midrule']
    for j, lab in enumerate(d['labels']['pages']):
        lines.append(f'{lab} & ${p["b0"][j]:.3f} \\pm {p["se0"][j]:.3f}$ \\\\')
    lines += ['\\bottomrule', '\\end{tabular}']
    write_table('si_table_reg_pages.tex', lines)

    n, w = d['names'], d['windows']
    lines = ['\\begin{tabular}{lccccccc}', '\\toprule',
             'regressor & feature & + task $\\times$ 3\\,h & + feature $\\times$ task & + feature $\\times$ 3\\,h & 1.5 h & 3 h & 6 h \\\\', '\\midrule']
    for j, lab in enumerate(d['labels']['names']):
        cells = [f'${n["b0"][j]:.2f} \\pm {n["se0"][j]:.2f}$', f'${n["b1"][j]:.2f} \\pm {n["se1"][j]:.2f}$',
                 f'${n["b2"][j]:.2f} \\pm {n["se2"][j]:.2f}$', f'${n["b3"][j]:.2f} \\pm {n["se3"][j]:.2f}$']
        cells += [f'${w[h]["same"]["b"][j]:.2f} \\pm {w[h]["same"]["se"][j]:.2f}$' if j < 3 else '' for h in (1.5, 3, 6)]
        lines.append(f'{lab} & ' + ' & '.join(cells) + ' \\\\')
    lines += ['\\bottomrule', '\\end{tabular}']
    write_table('si_table_reg_names.tex', lines)

    f = d['forms']
    lines = ['\\begin{tabular}{lccc}', '\\toprule',
             'regressor & convention intercepts & + task $\\times$ 3\\,h & convention $\\times$ task $\\times$ 3\\,h \\\\',
             '\\midrule']
    for j, lab in enumerate(d['labels']['forms']):
        lines.append(f'{lab} & ${f["b0"][j]:.2f} \\pm {f["se0"][j]:.2f}$ & ${f["b1"][j]:.2f} \\pm {f["se1"][j]:.2f}$ & '
                     f'${f["bx"][j]:.2f} \\pm {f["sex"][j]:.2f}$ \\\\')
    lines += ['\\bottomrule', '\\end{tabular}']
    write_table('si_table_reg_forms.tex', lines)

    # the contrasts the text relies on, tested directly
    lines = ['\\begin{tabular}{llrrl}', '\\toprule', 'choice & contrast & difference & $\\chi^2$ & $p$ \\\\', '\\midrule']
    pw = p['wald']['own task vs other task, share of the feed']
    lines.append(f'where to write & share of the feed, own task vs other task & {p["b1"][0] - p["b1"][1]:.2f} & {pw[0]:.1f} & {pw[1]:.1e} \\\\')
    order = [1, 2, 3]
    for key, r, spec in (('how to sign', n, 'wald'), ('how to sign, feature $\\times$ task intercepts', n, 'wald2'), ('how to write', f, 'wald')):
        b = r['b2'] if spec == 'wald2' else r['b1']
        for (name, (stat, pv)), (i, j) in zip(r[spec].items(), ((0, 2), (1, 2), (1, 3))):
            lines.append(f'{key} & {name} & {b[i] - b[j]:.2f} & {stat:.1f} & {pv:.1e} \\\\')
    lines += ['\\bottomrule', '\\end{tabular}']
    write_table('si_table_wald.tex', lines)

    # standard errors clustered one way and two ways
    lines = ['\\begin{tabular}{llccc}', '\\toprule', 'choice & regressor & coefficient & s.e., handle & s.e., handle and task $\\times$ 3\\,h \\\\', '\\midrule']
    for key, r, dg in (('where to write', p, 3), ('how to sign', n, 2), ('how to write', f, 2)):
        for j, lab in enumerate(d['labels'][{'where to write': 'pages', 'how to sign': 'names', 'how to write': 'forms'}[key]]):
            dj = 2 if (key == 'where to write' and j < 2) else dg
            lines.append(f'{key if j == 0 else ""} & {lab} & {r["b1"][j]:.{dj}f} & {r["se1"][j]:.{dj}f} & {r["se_twoway"][j]:.{dj}f} \\\\')
    lines += ['\\bottomrule', '\\end{tabular}']
    write_table('si_table_twoway.tex', lines)

    # conditional logit for the page choice
    cl = p['clogit']
    lines = ['\\begin{tabular}{lcc}', '\\toprule', 'regressor & linear probability model & conditional logit \\\\', '\\midrule']
    for j, lab in enumerate(d['labels']['pages']):
        dj = 2 if j < 2 else 3
        lines.append(f'{lab} & ${p["b1"][j]:.{dj}f} \\pm {p["se1"][j]:.{dj}f}$ & ${cl["b"][j]:.2f} \\pm {cl["se"][j]:.2f}$ \\\\')
    lines += ['\\midrule', f'$R^2$ & {p["r2_1"]:.2f} & {cl["pseudo_r2"]:.2f} (McFadden) \\\\', '\\bottomrule', '\\end{tabular}']
    write_table('si_table_clogit.tex', lines)


def table_thresholds():
    """SI: the writing results under other selection rules."""
    rows = load_cache('si_thresholds.pkl')
    lines = ['\\begin{tabular}{rrrrrccccc}', '\\toprule',
             'uses & minority & pages & kept & first uses & page & feed, own task & out of view & other tasks & slope \\\\', '\\midrule']
    for r in rows:
        cells = ' & '.join(f'${b:.2f} \\pm {s:.2f}$' for b, s in zip(r['b'], r['se']))
        mark = '$^*$' if r['main'] else ''
        lines.append(f'{r["uses"]}{mark} & {100 * r["minority"]:g}\\% & {r["pages"]} & {r["kept"]} & {r["n"]:,} & {cells} & {r["slope"]:.2f} \\\\')
    lines += ['\\bottomrule', '\\end{tabular}']
    write_table('si_table_thresholds.tex', lines)


def main():
    table_regressions()
    table_thresholds()
    table_families()
    table_pieces()
    table_conventions()
    table_slopes()
    table_robustness()
    table_identification()


if __name__ == '__main__':
    main()
