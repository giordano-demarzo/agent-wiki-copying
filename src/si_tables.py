"""The tables of the paper and the SI, as LaTeX fragments in paper/tables/."""
import numpy as np

import names as nm
from common import load_cache, tex, wilson, write_table
from conventions import CLASSES, CONV

CLASS_NAME = {'coined': 'coined', 'semantic': 'near-synonym', 'habit': 'habit'}
CLASS_PLURAL = {'coined': 'coined names', 'semantic': 'near-synonyms', 'habit': 'habits of style'}


def pm(b, se, d=2):
    return f'${b:.{d}f}\\pm{1.96 * se:.{d}f}$'


def table_regressions():
    """Table 1 of the paper: the three choices, baseline and task x 3 h fixed effects, 95% intervals."""
    d = load_cache('identification.pkl')
    p, n, f = d['pages'], d['names'], d['forms']
    lines = ['\\begin{tabular}{llcc}', '\\toprule', 'Choice & Exposure & Baseline FE & + task $\\times$ 3\\,h FE \\\\',
             '\\midrule']
    heads = {'pages': [f'Where to write', f'({p["n_decisions"]:,} decisions,', f'{p["n"]:,} candidates)', '', ''],
             'names': ['How to sign', f'({n["n"]:,} records)', '', ''],
             'forms': ['How to write', f'({f["n"]:,} first uses)', '', '']}
    for key, r in (('pages', p), ('names', n), ('forms', f)):
        digits = 3 if key == 'pages' else 2
        for j, lab in enumerate(d['labels'][key]):
            dj = 2 if (key == 'pages' and j < 2) else digits
            lines.append(f'{heads[key][j]} & {lab} & {pm(r["b0"][j], r["se0"][j], dj)} & {pm(r["b1"][j], r["se1"][j], dj)} \\\\')
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
    lines = ['\\begin{tabular}{lcc}', '\\toprule', 'regressor & decision FE & + task $\\times$ 3\\,h FE \\\\', '\\midrule']
    for j, lab in enumerate(d['labels']['pages']):
        lines.append(f'{lab} & ${p["b0"][j]:.3f} \\pm {p["se0"][j]:.3f}$ & ${p["b1"][j]:.3f} \\pm {p["se1"][j]:.3f}$ \\\\')
    lines += ['\\bottomrule', '\\end{tabular}']
    write_table('si_table_reg_pages.tex', lines)

    n, w = d['names'], d['windows']
    lines = ['\\begin{tabular}{lccccc}', '\\toprule',
             'regressor & feature FE & + task $\\times$ 3\\,h FE & 1.5 h & 3 h & 6 h \\\\', '\\midrule']
    for j, lab in enumerate(d['labels']['names']):
        cells = [f'${n["b0"][j]:.2f} \\pm {n["se0"][j]:.2f}$', f'${n["b1"][j]:.2f} \\pm {n["se1"][j]:.2f}$']
        cells += [f'${w[h]["same"]["b"][j]:.2f} \\pm {w[h]["same"]["se"][j]:.2f}$' if j < 3 else '' for h in (1.5, 3, 6)]
        lines.append(f'{lab} & ' + ' & '.join(cells) + ' \\\\')
    lines += ['\\bottomrule', '\\end{tabular}']
    write_table('si_table_reg_names.tex', lines)

    f = d['forms']
    lines = ['\\begin{tabular}{lccc}', '\\toprule',
             'regressor & convention FE & + task $\\times$ 3\\,h FE & convention $\\times$ task $\\times$ 3\\,h FE \\\\',
             '\\midrule']
    for j, lab in enumerate(d['labels']['forms']):
        lines.append(f'{lab} & ${f["b0"][j]:.2f} \\pm {f["se0"][j]:.2f}$ & ${f["b1"][j]:.2f} \\pm {f["se1"][j]:.2f}$ & '
                     f'${f["bx"][j]:.2f} \\pm {f["sex"][j]:.2f}$ \\\\')
    lines += ['\\bottomrule', '\\end{tabular}']
    write_table('si_table_reg_forms.tex', lines)


def main():
    table_regressions()
    table_families()
    table_pieces()
    table_conventions()
    table_slopes()
    table_robustness()
    table_identification()


if __name__ == '__main__':
    main()
